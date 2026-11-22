//! Compare the raw client's two GPU-side paths at 4K:
//!   * direct: compute reads the host-visible receive VkBuffer
//!   * copy:   GPU copies the slot to a VRAM image, then samples that image
//! This is an offscreen compute benchmark (no compositor), reporting GPU
//! timestamps. Run:
//!   cargo run --release -p rdma-display-client --bin raw-host-bench

use std::ffi::CString;

use anyhow::{Context, Result};
use rdma_display_vkcore::ash::vk;
use rdma_display_vkcore::VkCore;

const DIRECT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raw_direct.spv"));
const COPY_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raw_copy.spv"));

const W: u32 = 3840;
const H: u32 = 2160;
const WARMUP: usize = 5;
const RUNS: usize = 30;

struct Timed {
    query: vk::QueryPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    period: f32,
}

impl Timed {
    fn new(core: &VkCore) -> Result<Self> {
        let period = unsafe {
            core.instance
                .get_physical_device_properties(core.physical_device)
                .limits
                .timestamp_period
        };
        unsafe {
            let query = core.device.create_query_pool(
                &vk::QueryPoolCreateInfo {
                    query_type: vk::QueryType::TIMESTAMP,
                    query_count: 2,
                    ..Default::default()
                },
                None,
            )?;
            let cb = core.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: core.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0];
            let fence = core.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(Self { query, cb, fence, period })
        }
    }

    fn record(&self, core: &VkCore, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        unsafe {
            core.device.reset_fences(&[self.fence])?;
            core.device.begin_command_buffer(
                self.cb,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            core.device.cmd_reset_query_pool(self.cb, self.query, 0, 2);
            core.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::TOP_OF_PIPE, self.query, 0);
            f(self.cb);
            core.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query, 1);
            core.device.end_command_buffer(self.cb)?;
            core.device.queue_submit(
                core.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &self.cb,
                    ..Default::default()
                }],
                self.fence,
            )?;
            core.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            let mut q = [0u64; 2];
            core.device.get_query_pool_results(
                self.query,
                0,
                &mut q,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(())
    }

    fn elapsed_ms(&self, core: &VkCore) -> Result<f64> {
        let mut q = [0u64; 2];
        unsafe {
            core.device.get_query_pool_results(
                self.query,
                0,
                &mut q,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok((q[1].saturating_sub(q[0])) as f64 * self.period as f64 / 1e6)
    }
}

fn shader(core: &VkCore, spv: &[u8]) -> Result<vk::ShaderModule> {
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
            .context("shader")
    }
}

fn set_layout(core: &VkCore, bindings: &[vk::DescriptorType]) -> Result<vk::DescriptorSetLayout> {
    let b: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(i, ty)| vk::DescriptorSetLayoutBinding {
            binding: i as u32,
            descriptor_type: *ty,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        })
        .collect();
    unsafe {
        core.device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo {
                    binding_count: b.len() as u32,
                    p_bindings: b.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .context("set layout")
    }
}

fn pipeline(
    core: &VkCore,
    spv: &[u8],
    set: vk::DescriptorSetLayout,
) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    let module = shader(core, spv)?;
    let push = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: 12,
    }];
    let layout = unsafe {
        core.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &set,
                push_constant_range_count: 1,
                p_push_constant_ranges: push.as_ptr(),
                ..Default::default()
            },
            None,
        )?
    };
    let name = CString::new("main").unwrap();
    let pipe = unsafe {
        core.device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo {
                    stage: vk::PipelineShaderStageCreateInfo {
                        stage: vk::ShaderStageFlags::COMPUTE,
                        module,
                        p_name: name.as_ptr(),
                        ..Default::default()
                    },
                    layout,
                    ..Default::default()
                }],
                None,
            )
            .map_err(|(_, e)| e)?
    }[0];
    unsafe { core.device.destroy_shader_module(module, None) };
    Ok((pipe, layout))
}

fn main() -> Result<()> {
    let core = VkCore::new().context("Vulkan core")?;
    let frame_bytes = (W as usize) * (H as usize) * 4;
    let src_img = core.create_image(
        vk::Format::R8G8B8A8_UNORM,
        W,
        H,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageTiling::OPTIMAL,
    )?;
    let src_view = core.make_view(
        src_img.image,
        src_img.format,
        vk::ImageAspectFlags::COLOR,
        vk::ComponentSwizzle::IDENTITY,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;
    let out = core.create_image(
        vk::Format::R8G8B8A8_UNORM,
        W,
        H,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageTiling::OPTIMAL,
    )?;
    let out_view = core.make_view(
        out.image,
        out.format,
        vk::ImageAspectFlags::COLOR,
        vk::ComponentSwizzle::IDENTITY,
        vk::ImageLayout::GENERAL,
    )?;
    core.transition(
        out.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
        vk::ImageAspectFlags::COLOR,
    )?;
    let src_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    let mut host = vec![0u8; frame_bytes];
    for (i, b) in host.iter_mut().enumerate() {
        *b = (i * 31 + 7) as u8;
    }
    let imp = core
        .import_host_buffer(host.as_mut_ptr(), host.len())
        .context("host import (VK_EXT_external_memory_host)")?;
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

    let direct_set_layout = set_layout(
        &core,
        &[vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_IMAGE],
    )?;
    let copy_set_layout = set_layout(
        &core,
        &[vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::DescriptorType::STORAGE_IMAGE],
    )?;
    let (direct_pipe, direct_layout) = pipeline(&core, DIRECT_SPV, direct_set_layout)?;
    let (copy_pipe, copy_layout) = pipeline(&core, COPY_SPV, copy_set_layout)?;
    let pool_sizes = [
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 2 },
    ];
    let pool = unsafe {
        core.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo {
                max_sets: 2,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            },
            None,
        )?
    };
    let layouts = [direct_set_layout, copy_set_layout];
    let sets = unsafe {
        core.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: 2,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        })?
    };
    let direct_set = sets[0];
    let copy_set = sets[1];
    let buf_info = vk::DescriptorBufferInfo {
        buffer: imp.buffer,
        offset: imp.base_offset,
        range: frame_bytes as u64,
    };
    let direct_img = vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: out_view,
        image_layout: vk::ImageLayout::GENERAL,
    };
    let copy_img = vk::DescriptorImageInfo {
        sampler,
        image_view: src_view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let copy_dst = vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: out_view,
        image_layout: vk::ImageLayout::GENERAL,
    };
    let writes = [
        vk::WriteDescriptorSet {
            dst_set: direct_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buf_info,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: direct_set,
            dst_binding: 1,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &direct_img,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: copy_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &copy_img,
            ..Default::default()
        },
        vk::WriteDescriptorSet {
            dst_set: copy_set,
            dst_binding: 1,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &copy_dst,
            ..Default::default()
        },
    ];
    unsafe { core.device.update_descriptor_sets(&writes, &[]) };

    let timed = Timed::new(&core)?;
    let pc = [W, H, W];
    let pc_bytes = unsafe { std::slice::from_raw_parts(pc.as_ptr() as *const u8, 12) };
    let mut direct_ms = Vec::new();
    let mut copy_ms = Vec::new();
    for i in 0..(WARMUP + RUNS) {
        timed.record(&core, |cb| unsafe {
            core.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, direct_pipe);
            core.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, direct_layout, 0, &[direct_set], &[]);
            core.device.cmd_push_constants(cb, direct_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            core.device.cmd_dispatch(cb, W.div_ceil(8), H.div_ceil(8), 1);
        })?;
        let d = timed.elapsed_ms(&core)?;
        if i >= WARMUP { direct_ms.push(d); }

        timed.record(&core, |cb| unsafe {
            let to_dst = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: src_img.image,
                subresource_range: src_range,
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_dst],
            );
            let copy = vk::BufferImageCopy {
                buffer_offset: imp.base_offset,
                buffer_row_length: W,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D { width: W, height: H, depth: 1 },
                ..Default::default()
            };
            core.device.cmd_copy_buffer_to_image(
                cb,
                imp.buffer,
                src_img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_read = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: src_img.image,
                subresource_range: src_range,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_read],
            );
            core.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, copy_pipe);
            core.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, copy_layout, 0, &[copy_set], &[]);
            core.device.cmd_push_constants(cb, copy_layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
            core.device.cmd_dispatch(cb, W.div_ceil(8), H.div_ceil(8), 1);
        })?;
        let c = timed.elapsed_ms(&core)?;
        if i >= WARMUP { copy_ms.push(c); }
    }
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2], v.iter().sum::<f64>() / v.len() as f64)
    };
    let (dmin, dmed, dmean) = stats(&mut direct_ms);
    let (cmin, cmed, cmean) = stats(&mut copy_ms);
    println!("4K raw path GPU time (ms):");
    println!("  direct host-buffer sampling: min {dmin:.3}  median {dmed:.3}  mean {dmean:.3}");
    println!("  copy-to-image + sample:      min {cmin:.3}  median {cmed:.3}  mean {cmean:.3}");
    println!("  direct/copy median ratio:    {:.2}x", dmed / cmed);
    // Keep the imported host allocation alive through the last GPU wait.
    drop(imp);
    drop(host);
    Ok(())
}
