//! Vulkan compute lossless encoder for the server.
//!
//! The tile-delta algorithm is shared with the client's GLES decoder
//! (`codec-gpu`); the shader body lives once in
//! `shaders/enc_common.glsl` and is concatenated into the SSBO-input and
//! scanout-image variants at build time. The coded header is written by a
//! second one-invocation dispatch, so the CPU never touches the payload.
//!
//! The coded output can be an external host-visible or device buffer
//! (`set_output`; the production server paths), or a lazily allocated internal
//! host buffer for the no-external fallback and the host-input tests. The CPU
//! `upload` buffer is likewise allocated only when `encode_host` is used.

use std::ffi::CString;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use rdma_display_codec::{Geometry, HEADER_LEN, INDEX_WORDS};
use rdma_display_vkcore::{ash::vk, HostBuffer, VkCore};

const SSBO_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/enc_ssbo.spv"));
const IMAGE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/enc_image.spv"));
const HEADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/enc_header.spv"));

/// A Vulkan lossless encoder for one geometry.
pub struct Encoder {
    core: Arc<VkCore>,
    geom: Geometry,
    first: bool,
    pbase_words: u32,
    cur: vk::Buffer,
    cur_mem: vk::DeviceMemory,
    reference: vk::Buffer,
    ref_mem: vk::DeviceMemory,
    /// internal coded output; allocated only when no external output is bound
    coded: Option<HostBuffer>,
    /// CPU upload buffer; allocated only for `encode_host`
    upload: Option<HostBuffer>,
    /// tiny valid buffer for the initial coded/header descriptors
    placeholder: HostBuffer,
    counter: HostBuffer,
    ssbo_pipe: vk::Pipeline,
    ssbo_layout: vk::PipelineLayout,
    ssbo_set_layout: vk::DescriptorSetLayout,
    ssbo_set: vk::DescriptorSet,
    image_pipe: vk::Pipeline,
    image_layout: vk::PipelineLayout,
    image_set_layout: vk::DescriptorSetLayout,
    image_set: vk::DescriptorSet,
    header_pipe: vk::Pipeline,
    header_layout: vk::PipelineLayout,
    header_set_layout: vk::DescriptorSetLayout,
    header_set: vk::DescriptorSet,
    desc_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    /// current imported scanout view (owned by the shared import cache)
    scanout_view: Option<vk::ImageView>,
    /// external coded-output buffer (the staging slot); None = internal `coded`
    output: Option<(vk::Buffer, usize)>,
}

impl Encoder {
    pub fn new(geom: Geometry) -> Result<Self> {
        Self::with_core(Arc::new(VkCore::new()?), geom)
    }

    pub fn with_core(core: Arc<VkCore>, geom: Geometry) -> Result<Self> {
        ensure!(geom.width > 0 && geom.height > 0, "zero geometry");
        ensure!(
            geom.tile_log2 == 6,
            "vkcodec shaders are compiled for 64 px tiles (tile_log2=6)"
        );
        let frame_bytes = geom.frame_bytes() as u64;
        let pbase_words = ((HEADER_LEN + geom.tile_count() * INDEX_WORDS * 4) / 4) as u32;

        let (cur, cur_mem) = device_buffer(&core, frame_bytes)?;
        let (reference, ref_mem) = device_buffer(&core, frame_bytes)?;
        // The first frame is full, but the shader still reads `ref` while
        // measuring run counts, so start from a defined zero buffer.
        core.one_shot(|cb| unsafe {
            core.device.cmd_fill_buffer(cb, reference, 0, frame_bytes, 0);
        })?;

        // Do not allocate the full coded/upload buffers here. The server
        // binds an external output (GPU or host staging) for the production
        // path; the internal `coded` is only needed by the fallback/tests.
        // The descriptor sets still need a valid coded buffer, so bind a tiny
        // placeholder until set_output() or the lazy coded allocation.
        let placeholder = core.host_buffer(4)?;
        let counter = core.host_buffer(8)?;

        let ssbo_mod = shader_module(&core, SSBO_SPV)?;
        let image_mod = shader_module(&core, IMAGE_SPV)?;
        let header_mod = shader_module(&core, HEADER_SPV)?;

        let storage = vk::DescriptorType::STORAGE_BUFFER;
        let ssbo_set_layout = set_layout(
            &core,
            &[(0, storage), (1, storage), (2, storage), (3, storage)],
        )?;
        let image_set_layout = set_layout(
            &core,
            &[
                (0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                (1, storage),
                (2, storage),
                (3, storage),
            ],
        )?;
        let header_set_layout = set_layout(&core, &[(0, storage), (1, storage)])?;

        let ssbo_layout = pipeline_layout(&core, ssbo_set_layout, 7 * 4)?;
        let image_layout = pipeline_layout(&core, image_set_layout, 7 * 4)?;
        let header_layout = pipeline_layout(&core, header_set_layout, 6 * 4)?;

        let ssbo_pipe = compute_pipeline(&core, ssbo_mod, ssbo_layout)?;
        let image_pipe = compute_pipeline(&core, image_mod, image_layout)?;
        let header_pipe = compute_pipeline(&core, header_mod, header_layout)?;
        unsafe {
            core.device.destroy_shader_module(ssbo_mod, None);
            core.device.destroy_shader_module(image_mod, None);
            core.device.destroy_shader_module(header_mod, None);
        }

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: storage,
                descriptor_count: 12,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
        ];
        let desc_pool = unsafe {
            core.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo {
                    max_sets: 3,
                    pool_size_count: pool_sizes.len() as u32,
                    p_pool_sizes: pool_sizes.as_ptr(),
                    ..Default::default()
                },
                None,
            )?
        };
        let layouts = [ssbo_set_layout, image_set_layout, header_set_layout];
        let sets = unsafe {
            core.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool: desc_pool,
                descriptor_set_count: layouts.len() as u32,
                p_set_layouts: layouts.as_ptr(),
                ..Default::default()
            })?
        };
        let (ssbo_set, image_set, header_set) = (sets[0], sets[1], sets[2]);

        update_buffers(
            &core,
            ssbo_set,
            0,
            &[cur, reference, placeholder.buffer, counter.buffer],
        );
        // binding 0 is the combined image sampler, filled by set_scanout
        update_buffers(
            &core,
            image_set,
            1,
            &[reference, placeholder.buffer, counter.buffer],
        );
        update_buffers(&core, header_set, 0, &[placeholder.buffer, counter.buffer]);

        let sampler = unsafe {
            core.device.create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    ..Default::default()
                },
                None,
            )?
        };

        Ok(Self {
            core,
            geom,
            first: true,
            pbase_words,
            cur,
            cur_mem,
            reference,
            ref_mem,
            coded: None,
            upload: None,
            placeholder,
            counter,
            ssbo_pipe,
            ssbo_layout,
            ssbo_set_layout,
            ssbo_set,
            image_pipe,
            image_layout,
            image_set_layout,
            image_set,
            header_pipe,
            header_layout,
            header_set_layout,
            header_set,
            desc_pool,
            sampler,
            scanout_view: None,
            output: None,
        })
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    pub fn core(&self) -> &Arc<VkCore> {
        &self.core
    }

    /// Force the next frame to be full (a reference reset).
    pub fn reset(&mut self) {
        self.first = true;
    }

    /// Encode a host frame (BGRA/XRGB bytes) and return the coded length.
    /// The coded bytes are in [`coded`](Self::coded) only when no external
    /// output is bound; otherwise read the buffer passed to `set_output`.
    pub fn encode_host(&mut self, frame: &[u8]) -> Result<usize> {
        ensure!(
            frame.len() == self.geom.frame_bytes(),
            "frame {} != geometry {}",
            frame.len(),
            self.geom.frame_bytes()
        );
        self.ensure_output()?;
        if self.upload.is_none() {
            self.upload = Some(
                self.core
                    .host_buffer(self.geom.frame_bytes() as u64)
                    .context("allocate host upload buffer")?,
            );
        }
        let upload_buf = self.upload.as_mut().unwrap();
        upload_buf.as_mut_slice()[..frame.len()].copy_from_slice(frame);
        let full = self.first;
        let upload = upload_buf.buffer;
        let cur = self.cur;
        let frame_bytes = self.geom.frame_bytes() as u64;
        self.core.one_shot(|cb| unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: frame_bytes,
            };
            self.core
                .device
                .cmd_copy_buffer(cb, upload, cur, &[region]);
            self.barrier(cb, true);
        })?;
        let n = self.dispatch(full, false)?;
        self.first = false;
        Ok(n)
    }

    /// Bind an external coded-output buffer (the host-visible staging slot).
    /// `capacity` is its size in bytes. The compute `coded` SSBO and the header
    /// dispatch write directly into it, so the server never copies the coded
    /// payload.
    pub fn set_output(&mut self, buffer: vk::Buffer, capacity: usize) -> Result<()> {
        let need = self
            .coded
            .as_ref()
            .map(|c| c.size as usize)
            .unwrap_or_else(|| self.geom.max_coded_len());
        ensure!(
            capacity >= need,
            "external coded buffer {capacity} < {need}"
        );
        if self.output == Some((buffer, capacity)) {
            return Ok(());
        }
        // An external output makes the internal coded buffer unnecessary.
        // Drop it if a previous internal encode allocated it.
        if self.coded.is_some() {
            if let Some(c) = self.coded.take() {
                unsafe {
                    self.core.device.destroy_buffer(c.buffer, None);
                    self.core.device.free_memory(c.memory, None);
                }
            }
        }
        update_buffers(&self.core, self.ssbo_set, 2, &[buffer]);
        update_buffers(&self.core, self.image_set, 2, &[buffer]);
        update_buffers(&self.core, self.header_set, 0, &[buffer]);
        self.output = Some((buffer, capacity));
        Ok(())
    }

    /// Use a scanout image view (owned by the shared import cache; the caller
    /// must have transitioned it to `SHADER_READ_ONLY_OPTIMAL`).
    pub fn set_scanout_image(&mut self, view: vk::ImageView) -> Result<()> {
        let info = vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.image_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &info,
            ..Default::default()
        };
        unsafe { self.core.device.update_descriptor_sets(&[write], &[]); }
        self.scanout_view = Some(view);
        Ok(())
    }

    /// Encode from the scanout set by [`set_scanout_image`](Self::set_scanout_image).
    pub fn encode_scanout(&mut self) -> Result<usize> {
        if self.scanout_view.is_none() {
            bail!("encode_scanout before set_scanout_image");
        }
        self.ensure_output()?;
        let full = self.first;
        let n = self.dispatch(full, true)?;
        self.first = false;
        Ok(n)
    }

    /// Changed tiles in the last encode (0 = an idle delta).
    pub fn last_count(&self) -> u32 {
        let c = self.counter.as_slice();
        u32::from_ne_bytes(c[0..4].try_into().unwrap())
    }

    /// The internal coded bytes (host-visible; valid until the next encode).
    /// Empty when an external output is bound and no internal coded buffer was
    /// needed.
    pub fn coded(&self) -> &[u8] {
        self.coded
            .as_ref()
            .map(|c| c.as_slice())
            .unwrap_or(&[])
    }

    /// The mapped internal coded-buffer pointer for host-MR registration.
    /// Null when the internal coded buffer was not allocated.
    pub fn coded_ptr(&self) -> *const u8 {
        self.coded
            .as_ref()
            .map(|c| c.ptr as *const u8)
            .unwrap_or(std::ptr::null())
    }

    pub fn coded_capacity(&self) -> usize {
        self.coded.as_ref().map(|c| c.size as usize).unwrap_or(0)
    }

    /// Allocate the internal coded output on first use when no external output
    /// is bound. The production server paths always bind an external output,
    /// so they never allocate it.
    fn ensure_output(&mut self) -> Result<()> {
        if self.output.is_some() || self.coded.is_some() {
            return Ok(());
        }
        let coded = self
            .core
            .host_buffer(self.geom.max_coded_len() as u64)
            .context("allocate internal coded buffer")?;
        update_buffers(&self.core, self.ssbo_set, 2, &[coded.buffer]);
        update_buffers(&self.core, self.image_set, 2, &[coded.buffer]);
        update_buffers(&self.core, self.header_set, 0, &[coded.buffer]);
        self.coded = Some(coded);
        Ok(())
    }

    fn dispatch(&self, full: bool, image: bool) -> Result<usize> {
        let geom = self.geom;
        let (pipe, layout, set) = if image {
            (self.image_pipe, self.image_layout, self.image_set)
        } else {
            (self.ssbo_pipe, self.ssbo_layout, self.ssbo_set)
        };
        let pc = [
            geom.width,
            geom.height,
            geom.stride / 4,
            geom.tiles_x() as u32,
            full as u32,
            self.pbase_words,
            1, // update the reference in-pass
        ];
        let hpc = [
            geom.width,
            geom.height,
            geom.stride,
            geom.tile_log2 as u32,
            geom.tile_count() as u32,
            (!full) as u32,
        ];
        let counter = self.counter.buffer;
        let header_pipe = self.header_pipe;
        let header_layout = self.header_layout;
        let header_set = self.header_set;
        let tiles = geom.tile_count() as u32;
        self.core.one_shot(|cb| unsafe {
            let dev = &self.core.device;
            dev.cmd_fill_buffer(cb, counter, 0, 8, 0);
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                }],
                &[],
                &[],
            );
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
            dev.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            dev.cmd_push_constants(
                cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(pc.as_ptr() as *const u8, pc.len() * 4),
            );
            dev.cmd_dispatch(cb, tiles, 1, 1);
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                }],
                &[],
                &[],
            );
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, header_pipe);
            dev.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                header_layout,
                0,
                &[header_set],
                &[],
            );
            dev.cmd_push_constants(
                cb,
                header_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(hpc.as_ptr() as *const u8, hpc.len() * 4),
            );
            dev.cmd_dispatch(cb, 1, 1, 1);
            // Make the coded writes visible to an external consumer (the NIC
            // reading the exported dma-buf). For host staging this is a no-op.
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::MEMORY_READ
                        | vk::AccessFlags::MEMORY_WRITE,
                    ..Default::default()
                }],
                &[],
                &[],
            );
        })?;

        let c = self.counter.as_slice();
        let alloc = u32::from_ne_bytes(c[4..8].try_into().unwrap());
        let len = (self.pbase_words as usize + alloc as usize) * 4;
        let cap = self
            .output
            .map(|(_, c)| c)
            .or_else(|| self.coded.as_ref().map(|c| c.size as usize))
            .unwrap_or(0);
        ensure!(
            cap > 0 && len <= cap,
            "coded length {len} exceeds buffer {cap}"
        );
        Ok(len)
    }

    fn barrier(&self, cb: vk::CommandBuffer, transfer_to_compute: bool) {
        let (src, dst, sa, da) = if transfer_to_compute {
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            )
        } else {
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            )
        };
        unsafe {
            self.core.device.cmd_pipeline_barrier(
                cb,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: sa,
                    dst_access_mask: da,
                    ..Default::default()
                }],
                &[],
                &[],
            );
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            let dev = &self.core.device;
            dev.destroy_sampler(self.sampler, None);
            dev.destroy_descriptor_pool(self.desc_pool, None);
            dev.destroy_pipeline(self.ssbo_pipe, None);
            dev.destroy_pipeline(self.image_pipe, None);
            dev.destroy_pipeline(self.header_pipe, None);
            dev.destroy_pipeline_layout(self.ssbo_layout, None);
            dev.destroy_pipeline_layout(self.image_layout, None);
            dev.destroy_pipeline_layout(self.header_layout, None);
            dev.destroy_descriptor_set_layout(self.ssbo_set_layout, None);
            dev.destroy_descriptor_set_layout(self.image_set_layout, None);
            dev.destroy_descriptor_set_layout(self.header_set_layout, None);
            dev.destroy_buffer(self.cur, None);
            dev.free_memory(self.cur_mem, None);
            dev.destroy_buffer(self.reference, None);
            dev.free_memory(self.ref_mem, None);
            for b in [Some(&self.placeholder), Some(&self.counter)]
                .into_iter()
                .flatten()
                .chain(self.coded.iter())
                .chain(self.upload.iter())
            {
                dev.destroy_buffer(b.buffer, None);
                dev.free_memory(b.memory, None);
            }
        }
    }
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPIR-V not word aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn shader_module(core: &VkCore, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let words = spirv_words(bytes);
    unsafe {
        Ok(core.device.create_shader_module(
            &vk::ShaderModuleCreateInfo {
                code_size: words.len() * 4,
                p_code: words.as_ptr(),
                ..Default::default()
            },
            None,
        )?)
    }
}

fn device_buffer(core: &VkCore, size: u64) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    unsafe {
        let buffer = core.device.create_buffer(
            &vk::BufferCreateInfo {
                size,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            None,
        )?;
        let req = core.device.get_buffer_memory_requirements(buffer);
        let mt = core.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let memory = core.device.allocate_memory(
            &vk::MemoryAllocateInfo {
                allocation_size: req.size,
                memory_type_index: mt,
                ..Default::default()
            },
            None,
        )?;
        core.device.bind_buffer_memory(buffer, memory, 0)?;
        Ok((buffer, memory))
    }
}

fn set_layout(
    core: &VkCore,
    bindings: &[(u32, vk::DescriptorType)],
) -> Result<vk::DescriptorSetLayout> {
    let b: Vec<vk::DescriptorSetLayoutBinding> = bindings
        .iter()
        .map(|(binding, ty)| vk::DescriptorSetLayoutBinding {
            binding: *binding,
            descriptor_type: *ty,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        })
        .collect();
    unsafe {
        Ok(core.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo {
                binding_count: b.len() as u32,
                p_bindings: b.as_ptr(),
                ..Default::default()
            },
            None,
        )?)
    }
}

fn pipeline_layout(
    core: &VkCore,
    set_layout: vk::DescriptorSetLayout,
    push_size: u32,
) -> Result<vk::PipelineLayout> {
    let range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: push_size,
    };
    unsafe {
        Ok(core.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &range,
                ..Default::default()
            },
            None,
        )?)
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
    let ci = vk::ComputePipelineCreateInfo {
        stage,
        layout,
        ..Default::default()
    };
    unsafe {
        core.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[ci], None)
            .map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines: {e:?}"))
            .map(|v| v[0])
    }
}

fn update_buffers(
    core: &VkCore,
    set: vk::DescriptorSet,
    first_binding: u32,
    buffers: &[vk::Buffer],
) {
    let infos: Vec<vk::DescriptorBufferInfo> = buffers
        .iter()
        .map(|buffer| vk::DescriptorBufferInfo {
            buffer: *buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: first_binding + i as u32,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: info,
            ..Default::default()
        })
        .collect();
    unsafe { core.device.update_descriptor_sets(&writes, &[]); }
}
