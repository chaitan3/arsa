//! Vulkan compute decoder for the lossless tile-delta codec.
//!
//! This is the client mirror of the server's `vkcodec` encoder. One persistent
//! RGBA8 image is both the reference and the frame presented on screen; one
//! host-visible `VkBuffer` receives the coded words directly (no device-side
//! upload copy). A single compute dispatch applies the changed tiles.

use std::ffi::CString;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use rdma_display_codec::{Geometry, Header, HEADER_LEN, INDEX_WORDS};
use rdma_display_vkcore::{ash::vk, HostBuffer, Image, VkCore};

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lossless_dec.spv"));

pub struct VulkanLossless {
    core: Arc<VkCore>,
    geom: Geometry,
    image: Image,
    view: vk::ImageView,
    coded: HostBuffer,
    set_layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    image_layout: vk::ImageLayout,
}

impl VulkanLossless {
    pub fn new(core: Arc<VkCore>, geom: Geometry) -> Result<Self> {
        geom.validate()?;
        ensure!(geom.tile_log2 == 6, "Vulkan decoder expects 64px tiles");
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let image = core
            .create_image(vk::Format::R8G8B8A8_UNORM, geom.width, geom.height, usage, vk::ImageTiling::OPTIMAL)
            .context("lossless decoder image")?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            core.one_shot(|cb| {
                let to_general = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: image.image,
                    subresource_range: range,
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };
                core.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                let clear = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] };
                core.device
                    .cmd_clear_color_image(cb, image.image, vk::ImageLayout::GENERAL, &clear, &[range]);
                let after_clear = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: image.image,
                    subresource_range: range,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE,
                    ..Default::default()
                };
                core.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[after_clear],
                );
            })?;
        }
        let view = core.make_view(
            image.image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
            vk::ComponentSwizzle::IDENTITY,
            vk::ImageLayout::GENERAL,
        )?;
        let coded = core
            .host_buffer(geom.max_coded_len() as u64)
            .context("lossless coded input buffer")?;

        let module = shader_module(&core, SPV)?;
        let set_layout = descriptor_layout(&core)?;
        let layout = pipeline_layout(&core, set_layout)?;
        let pipeline = compute_pipeline(&core, module, layout)?;
        unsafe { core.device.destroy_shader_module(module, None) };
        let (pool, set) = descriptor_set(&core, set_layout, &coded, view)?;

        Ok(Self {
            core,
            geom,
            image,
            view,
            coded,
            set_layout,
            set,
            pool,
            layout,
            pipeline,
            image_layout: vk::ImageLayout::GENERAL,
        })
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.view
    }

    /// Decode one coded frame from the fallback host buffer.
    pub fn decode(&mut self, coded: &[u8]) -> Result<()> {
        self.decode_source(coded, None)
    }

    /// Decode directly from a slot region of the imported receive pool.
    /// `offset` is relative to the imported buffer's VkBuffer; no CPU copy of
    /// the coded bytes is made.
    #[allow(dead_code)]
    pub fn decode_from_pool(
        &mut self,
        coded: &[u8],
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    ) -> Result<()> {
        self.decode_source(coded, Some((buffer, offset, range)))
    }

    /// Record the decode of a slot region from the imported receive pool into
    /// an existing command buffer. No submit and no CPU copy.
    pub fn record_decode_from_pool(
        &mut self,
        cb: vk::CommandBuffer,
        coded: &[u8],
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    ) -> Result<()> {
        self.record(cb, coded, Some((buffer, offset, range)))
    }

    fn decode_source(
        &mut self,
        coded: &[u8],
        source: Option<(vk::Buffer, u64, u64)>,
    ) -> Result<()> {
        let core = self.core.clone();
        core.one_shot_result(|cb| self.record(cb, coded, source))
    }

    fn record(
        &mut self,
        cb: vk::CommandBuffer,
        coded: &[u8],
        source: Option<(vk::Buffer, u64, u64)>,
    ) -> Result<()> {
        let h = Header::parse(coded)?;
        ensure!(
            h.geometry() == self.geom,
            "lossless geometry changed mid-session"
        );
        let cap = h.index_capacity as usize;
        let pbase_words = HEADER_LEN / 4 + cap * INDEX_WORDS;
        let n = HEADER_LEN + cap * INDEX_WORDS * 4 + h.payload_words as usize * 4;
        ensure!(n <= coded.len(), "coded frame truncated");
        if let Some((_, _, range)) = source {
            ensure!(n as u64 <= range, "coded frame {n} > slot range {range}");
        } else {
            ensure!(
                n <= self.coded.size as usize,
                "coded frame {n} > decoder capacity {}",
                self.coded.size
            );
            self.coded.as_mut_slice()[..n].copy_from_slice(&coded[..n]);
        }
        // Keep the descriptor at offset 0 (the imported pool base is page
        // aligned) and address the slot with a word base in the shader. This
        // avoids the storage-buffer dynamic-offset alignment requirement.
        let (src_buffer, slot_base_bytes) = match source {
            Some((b, o, _)) => {
                ensure!(o % 4 == 0, "lossless slot offset not word-aligned");
                (b, o)
            }
            None => (self.coded.buffer, 0),
        };
        let binfo = vk::DescriptorBufferInfo {
            buffer: src_buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &binfo,
            ..Default::default()
        };
        unsafe { self.core.device.update_descriptor_sets(&[write], &[]) };
        if h.count == 0 {
            return Ok(());
        }

        let old = self.image_layout;
        let image_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let image = self.image.image;
        let pipeline = self.pipeline;
        let layout = self.layout;
        let set = self.set;
        let pc = [
            self.geom.width,
            self.geom.height,
            self.geom.stride / 4,
            self.geom.tiles_x() as u32,
            h.count,
            pbase_words as u32,
            (slot_base_bytes / 4) as u32,
        ];
        unsafe {
            if old != vk::ImageLayout::GENERAL {
                let to_general = vk::ImageMemoryBarrier {
                    old_layout: old,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: image_range,
                    src_access_mask: vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::TRANSFER_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE,
                    ..Default::default()
                };
                self.core.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
            self.core.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.core.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            self.core.device.cmd_push_constants(
                cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, pc.len() * 4),
            );
            self.core.device.cmd_dispatch(cb, h.count, 1, 1);
            let to_read = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: image_range,
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            self.core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        }
        self.image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        Ok(())
    }

    /// Headless/`--dump` diagnostic readback into a fresh host buffer. The
    /// streaming path never calls this.
    pub fn read_rgba(&mut self) -> Result<Vec<u8>> {
        let host = self
            .core
            .host_buffer((self.geom.width as u64) * (self.geom.height as u64) * 4)?;
        self.download(&host)?;
        let data = host.as_slice().to_vec();
        unsafe {
            self.core.device.unmap_memory(host.memory);
            self.core.device.destroy_buffer(host.buffer, None);
            self.core.device.free_memory(host.memory, None);
        }
        Ok(data)
    }

    /// Read the decoded image back for headless tests/diagnostics.
    pub fn download(&mut self, host: &HostBuffer) -> Result<()> {
        self.core.download_image(
            self.image.image,
            host,
            self.geom.width,
            self.geom.height,
            self.image_layout,
        )?;
        self.image_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        Ok(())
    }
}

impl Drop for VulkanLossless {
    fn drop(&mut self) {
        unsafe {
            let dev = &self.core.device;
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.layout, None);
            dev.destroy_descriptor_pool(self.pool, None);
            dev.destroy_descriptor_set_layout(self.set_layout, None);
            dev.destroy_image_view(self.view, None);
            dev.unmap_memory(self.coded.memory);
            dev.destroy_buffer(self.coded.buffer, None);
            dev.free_memory(self.coded.memory, None);
        }
        self.core.destroy_image(Image {
            image: self.image.image,
            memory: self.image.memory,
            format: self.image.format,
            width: self.image.width,
            height: self.image.height,
        });
    }
}

fn shader_module(core: &VkCore, spv: &[u8]) -> Result<vk::ShaderModule> {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    unsafe {
        core.device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    code_size: spv.len(),
                    p_code: words.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .context("create lossless decoder shader module")
    }
}

fn descriptor_layout(core: &VkCore) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
    ];
    unsafe {
        core.device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .context("lossless descriptor layout")
    }
}

fn pipeline_layout(core: &VkCore, set_layout: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout> {
    let layouts = [set_layout];
    let range = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: 28,
    }];
    unsafe {
        core.device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: layouts.as_ptr(),
                    push_constant_range_count: 1,
                    p_push_constant_ranges: range.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .context("lossless pipeline layout")
    }
}

fn compute_pipeline(
    core: &VkCore,
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let name = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::COMPUTE,
        module,
        p_name: name.as_ptr(),
        ..Default::default()
    };
    unsafe {
        core.device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo {
                    stage,
                    layout,
                    ..Default::default()
                }],
                None,
            )
            .map_err(|(_, e)| e)
            .context("lossless compute pipeline")
            .map(|p| p[0])
    }
}

fn descriptor_set(
    core: &VkCore,
    set_layout: vk::DescriptorSetLayout,
    coded: &HostBuffer,
    view: vk::ImageView,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        },
    ];
    unsafe {
        let pool = core.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: sizes.len() as u32,
                p_pool_sizes: sizes.as_ptr(),
                ..Default::default()
            },
            None,
        )?;
        let layouts = [set_layout];
        let set = core.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: layouts.as_ptr(),
                ..Default::default()
            },
        )?[0];
        let buffer = vk::DescriptorBufferInfo {
            buffer: coded.buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let image = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        let writes = [
            vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &image,
                ..Default::default()
            },
        ];
        core.device.update_descriptor_sets(&writes, &[]);
        Ok((pool, set))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdma_display_codec::Encoder;

    fn frame(w: u32, h: u32, seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for (i, b) in v.iter_mut().enumerate() {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            *b = (x >> 13) as u8;
        }
        v
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_lossless_decodes_from_imported_slot_pool() {
        let (w, h) = (256u32, 256u32);
        let geom = Geometry::new(w, h, w * 4);
        let slot_len = geom.max_coded_len();
        let stride = (slot_len + 3) & !3;
        let pool = rdma_display_rdma::ClientSlotMemory::new(slot_len, 2, stride - slot_len)
            .expect("slot pool");
        let core = Arc::new(VkCore::new().expect("vulkan core"));
        let imported = core
            .import_host_buffer(pool.base_ptr(), pool.total_len())
            .expect("host pool import");
        let mut dec = VulkanLossless::new(core.clone(), geom).expect("vulkan decoder");
        let mut enc = rdma_display_vkcodec::Encoder::new(geom).expect("vk encoder");
        for i in 0..3u32 {
            let f = frame(w, h, i);
            let n = enc.encode_host(&f).expect("vk encode");
            unsafe { pool.slot_mut(1)[..n].copy_from_slice(&enc.coded()[..n]) };
            let offset = imported.base_offset + pool.slot_capacity() as u64;
            dec.decode_from_pool(
                &pool.slot(1)[..n],
                imported.buffer,
                offset,
                pool.slot_capacity() as u64,
            )
            .expect("vulkan decode from pool");
            let out = dec.read_rgba().expect("readback");
            assert_eq!(&out[..f.len()], &f[..], "frame {i}");
        }
        unsafe {
            core.device.destroy_buffer(imported.buffer, None);
            core.device.free_memory(imported.memory, None);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_lossless_decodes_vkcodec_frames() {
        let (w, h) = (256u32, 256u32);
        let geom = Geometry::new(w, h, w * 4);
        let core = Arc::new(VkCore::new().expect("vulkan core"));
        let mut dec = VulkanLossless::new(core.clone(), geom).expect("vulkan decoder");
        let mut enc = rdma_display_vkcodec::Encoder::new(geom).expect("vk encoder");
        for i in 0..3u32 {
            let f = frame(w, h, i);
            let n = enc.encode_host(&f).expect("vk encode");
            dec.decode(&enc.coded()[..n]).expect("vulkan decode");
            let out = dec.read_rgba().expect("readback");
            assert_eq!(&out[..f.len()], &f[..], "frame {i}");
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_lossless_decodes_cpu_encoded_frames() {
        let (w, h) = (256u32, 256u32);
        let geom = Geometry::new(w, h, w * 4);
        let core = Arc::new(VkCore::new().expect("vulkan core"));
        let mut dec = VulkanLossless::new(core.clone(), geom).expect("vulkan decoder");
        let mut enc = Encoder::new(geom).expect("cpu encoder");
        let mut coded = vec![0u8; geom.max_coded_len()];
        let out = core
            .host_buffer((w as u64) * (h as u64) * 4)
            .expect("readback");
        for i in 0..3u32 {
            let f = frame(w, h, i);
            let n = enc.encode(&f, &mut coded).expect("cpu encode");
            dec.decode(&coded[..n]).expect("vulkan decode");
            dec.download(&out).expect("download");
            assert_eq!(&out.as_slice()[..f.len()], &f[..], "frame {i}");
        }
    }
}
