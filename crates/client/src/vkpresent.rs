//! Vulkan Wayland presenter: swapchain, raw/lossless image upload/decode
//! target and PyroWave YUV planes, all on the shared `VkCore` device.
//!
//! There is no EGL/GL here. The lossless coded words enter through a single
//! host-visible `VkBuffer` in `VulkanLossless`, and the raw path copies the
//! received slot into one persistent host-visible buffer before a GPU copy.
//! Lossy PyroWave decode planes are sampled directly.

use std::ffi::CString;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rdma_display_codec::{Codec, Geometry};
use rdma_display_rdma::ClientSlotMemory;
use rdma_display_vkcore::ash::khr::{surface, swapchain};
use rdma_display_vkcore::ash::vk;
use rdma_display_vkcore::{HostBuffer, Image, ImportedHostBuffer, VkCore};

use crate::lossy::LossyDecoder;
use crate::vklossless::VulkanLossless;

const VIDEO_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/video.spv"));
const VIDEO_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/video_frag.spv"));
const YUV_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/yuv_frag.spv"));
const OVERLAY_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay_frag.spv"));

struct RawTarget {
    image: Image,
    view: vk::ImageView,
    /// source buffer/offset staged for the next present command buffer
    pending: Option<(vk::Buffer, u64)>,
    layout: vk::ImageLayout,
}

struct OverlayTarget {
    image: Image,
    view: vk::ImageView,
    host: HostBuffer,
    layout: vk::ImageLayout,
    w: u32,
    h: u32,
}

pub struct Renderer {
    core: Arc<VkCore>,
    surface_loader: surface::Instance,
    swapchain_loader: swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,

    command: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    fence: vk::Fence,

    codec: Codec,
    frame_w: u32,
    frame_h: u32,
    frame_stride: u32,
    filter_mode: u32,

    video_pipeline: vk::Pipeline,
    yuv_pipeline: vk::Pipeline,
    overlay_pipeline: vk::Pipeline,
    video_layout: vk::PipelineLayout,
    yuv_layout: vk::PipelineLayout,
    overlay_layout: vk::PipelineLayout,
    video_set_layout: vk::DescriptorSetLayout,
    yuv_set_layout: vk::DescriptorSetLayout,
    overlay_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    video_set: vk::DescriptorSet,
    yuv_set: vk::DescriptorSet,
    overlay_set: vk::DescriptorSet,
    sampler_nearest: vk::Sampler,
    sampler_linear: vk::Sampler,

    video_view: vk::ImageView,
    yuv_views: [vk::ImageView; 3],
    last_filter: u32,
    raw: Option<RawTarget>,
    lossless: Option<VulkanLossless>,
    lossy: Option<LossyDecoder>,
    overlay: Option<OverlayTarget>,
    slot_pool: Option<Arc<ClientSlotMemory>>,
    host_import: Option<ImportedHostBuffer>,
    pending_lossless: Vec<usize>,
}

impl Renderer {
    pub fn new(
        wl_display: *mut std::ffi::c_void,
        wl_surface: *mut std::ffi::c_void,
        codec: Codec,
        frame_w: u32,
        frame_h: u32,
        frame_stride: u32,
        swapchain_w: u32,
        swapchain_h: u32,
        chroma: u8,
        metrics: bool,
        filter_mode: u32,
    ) -> Result<Self> {
        let (core, surface) = VkCore::new_client(wl_display, wl_surface)
            .context("create Vulkan Wayland instance/surface")?;
        let core = Arc::new(core);
        let surface_loader = surface::Instance::new(&core.entry, &core.instance);
        let swapchain_loader = swapchain::Device::new(&core.instance, &core.device);
        let (swapchain, format, extent, images, views) = make_swapchain(
            &core,
            &surface_loader,
            surface,
            vk::SwapchainKHR::null(),
            Some((swapchain_w.max(1), swapchain_h.max(1))),
        )?;
        log::info!(
            "[vk] swapchain {}x{} format {:?} ({} images)",
            extent.width,
            extent.height,
            format,
            views.len()
        );

        let video_set_layout = set_layout(&core, 1)?;
        let yuv_set_layout = set_layout(&core, 3)?;
        let overlay_set_layout = set_layout(&core, 1)?;
        let video_layout = pipeline_layout(&core, video_set_layout, 20)?;
        let yuv_layout = pipeline_layout(&core, yuv_set_layout, 20)?;
        let overlay_layout = pipeline_layout(&core, overlay_set_layout, 24)?;
        let (video_pipeline, yuv_pipeline, overlay_pipeline, format) = create_pipelines(
            &core,
            format,
            video_layout,
            yuv_layout,
            overlay_layout,
        )?;

        let sampler_nearest = sampler(&core, vk::Filter::NEAREST)?;
        let sampler_linear = sampler(&core, vk::Filter::LINEAR)?;
        let (descriptor_pool, sets) = descriptor_sets(
            &core,
            video_set_layout,
            yuv_set_layout,
            overlay_set_layout,
        )?;
        let video_set = sets[0];
        let yuv_set = sets[1];
        let overlay_set = sets[2];

        let command = unsafe {
            core.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: core.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0]
        };
        let sem = |core: &VkCore| unsafe {
            core.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(anyhow::Error::from)
        };
        let image_available = sem(&core)?;
        let render_finished = sem(&core)?;
        let fence = unsafe {
            core.device.create_fence(
                &vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
                None,
            )?
        };

        let mut r = Self {
            core,
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            format,
            extent,
            images,
            views,
            command,
            image_available,
            render_finished,
            fence,
            codec,
            frame_w,
            frame_h,
            frame_stride,
            filter_mode,
            video_pipeline,
            yuv_pipeline,
            overlay_pipeline,
            video_layout,
            yuv_layout,
            overlay_layout,
            video_set_layout,
            yuv_set_layout,
            overlay_set_layout,
            descriptor_pool,
            video_set,
            yuv_set,
            overlay_set,
            sampler_nearest,
            sampler_linear,
            video_view: vk::ImageView::null(),
            yuv_views: [vk::ImageView::null(); 3],
            last_filter: u32::MAX,
            raw: None,
            lossless: None,
            lossy: None,
            overlay: None,
            slot_pool: None,
            host_import: None,
            pending_lossless: Vec::new(),
        };
        r.create_codec_resources(chroma, metrics)?;
        r.update_video_set(r.effective_filter());
        Ok(r)
    }

    fn create_codec_resources(&mut self, chroma: u8, metrics: bool) -> Result<()> {
        match self.codec {
            Codec::Raw => {
                let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
                let image = self.core.create_image(
                    vk::Format::R8G8B8A8_UNORM,
                    self.frame_w,
                    self.frame_h,
                    usage,
                    vk::ImageTiling::OPTIMAL,
                )?;
                let view = self.core.make_view(
                    image.image,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageAspectFlags::COLOR,
                    vk::ComponentSwizzle::IDENTITY,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?;
                self.video_view = view;
                self.raw = Some(RawTarget {
                    image,
                    view,
                    pending: None,
                    layout: vk::ImageLayout::UNDEFINED,
                });
            }
            Codec::Lossless => {
                let geom = Geometry::new(self.frame_w, self.frame_h, self.frame_stride);
                let dec = VulkanLossless::new(self.core.clone(), geom)?;
                self.video_view = dec.image_view();
                self.lossless = Some(dec);
            }
            Codec::Lossy => {
                let dec = LossyDecoder::new(
                    self.core.clone(),
                    self.frame_w,
                    self.frame_h,
                    chroma,
                )?;
                self.yuv_views = dec.plane_views();
                self.lossy = Some(dec);
            }
        }
        if metrics {
            let w = crate::metrics::OVERLAY_W;
            let h = crate::metrics::OVERLAY_H;
            let image = self.core.create_image(
                vk::Format::R8G8B8A8_UNORM,
                w,
                h,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageTiling::OPTIMAL,
            )?;
            let view = self.core.make_view(
                image.image,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageAspectFlags::COLOR,
                vk::ComponentSwizzle::IDENTITY,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )?;
            let host = self.core.host_buffer((w as u64) * (h as u64) * 4)?;
            self.overlay = Some(OverlayTarget {
                image,
                view,
                host,
                layout: vk::ImageLayout::UNDEFINED,
                w,
                h,
            });
            let info = vk::DescriptorImageInfo {
                sampler: self.sampler_nearest,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let write = vk::WriteDescriptorSet {
                dst_set: self.overlay_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &info,
                ..Default::default()
            };
            unsafe { self.core.device.update_descriptor_sets(&[write], &[]) };
        }
        Ok(())
    }

    /// Import the single stable receive pool once. Raw and lossless consume
    /// slot regions directly from this host-visible VkBuffer; a failure is
    /// fatal instead of silently adding a CPU copy to the hot path.
    pub fn attach_host_pool(&mut self, pool: Arc<ClientSlotMemory>) -> Result<()> {
        self.slot_pool = Some(pool.clone());
        if self.raw.is_some() || self.lossless.is_some() {
            let imp = self
                .core
                .import_host_buffer(pool.base_ptr(), pool.total_len())
                .with_context(|| {
                    format!(
                        "import client receive pool ({} bytes, {} slots);                          raw/lossless require VK_EXT_external_memory_host",
                        pool.total_len(),
                        pool.slots()
                    )
                })?;
            self.host_import = Some(imp);
            log::info!(
                "[vk] host receive pool imported once: {} slots x {} bytes",
                pool.slots(),
                pool.slot_len()
            );
        }
        Ok(())
    }

    /// Queue a received lossless delta. The actual compute dispatch is
    /// recorded into the next present command buffer, so all deltas received
    /// in one drain share a single GPU submission instead of one CPU wait each.
    pub fn decode_lossless(&mut self, slot: usize) -> Result<()> {
        self.host_import
            .as_ref()
            .context("lossless requires the imported receive pool")?;
        self.pending_lossless.push(slot);
        Ok(())
    }

    fn record_pending_lossless(&mut self, cb: vk::CommandBuffer) -> Result<()> {
        if self.pending_lossless.is_empty() {
            return Ok(());
        }
        let slots = std::mem::take(&mut self.pending_lossless);
        let pool = self
            .slot_pool
            .as_ref()
            .context("lossless slot pool missing")?
            .clone();
        let imp = self
            .host_import
            .as_ref()
            .context("lossless requires the imported receive pool")?;
        let dec = self
            .lossless
            .as_mut()
            .context("lossless decoder not active")?;
        for slot in slots {
            let coded = pool.slot(slot);
            let offset = imp.base_offset + (slot * pool.slot_capacity()) as u64;
            dec.record_decode_from_pool(
                cb,
                coded,
                imp.buffer,
                offset,
                pool.slot_capacity() as u64,
            )?;
        }
        Ok(())
    }

    pub fn decode_lossy(&mut self, coded: &[u8]) -> Result<()> {
        let d = self.lossy.as_mut().context("lossy decoder not active")?;
        d.decode_async(coded)?;
        let views = d.plane_views();
        if views != self.yuv_views {
            self.yuv_views = views;
            self.update_yuv_set(self.effective_filter());
        }
        Ok(())
    }

    /// Stage the newest raw slot for the next present command buffer. The GPU
    /// reads the slot region directly from the imported receive pool.
    pub fn upload_raw(&mut self, slot: usize, bytes: &[u8]) -> Result<()> {
        let raw = self.raw.as_mut().context("raw target not active")?;
        let n = (self.frame_stride as usize) * (self.frame_h as usize);
        if bytes.len() < n {
            bail!("raw frame {} < expected {n}", bytes.len());
        }
        let imp = self
            .host_import
            .as_ref()
            .context("raw requires the imported receive pool")?;
        let pool = self
            .slot_pool
            .as_ref()
            .context("raw requires the receive slot pool")?;
        let off = imp.base_offset + (slot * pool.slot_capacity()) as u64;
        raw.pending = Some((imp.buffer, off));
        Ok(())
    }

    /// Record the staged raw slot→image copy in the present command buffer.
    fn record_raw_upload(&mut self, cb: vk::CommandBuffer) {
        let Some(raw) = self.raw.as_mut() else { return };
        let Some((src, src_offset)) = raw.pending.take() else { return };
        let image = raw.image.image;
        let old = raw.layout;
        let range = color_range();
        unsafe {
            let to_dst = barrier(
                image,
                old,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                range,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            self.core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_dst],
            );
            let copy = vk::BufferImageCopy {
                buffer_offset: src_offset,
                buffer_row_length: self.frame_stride / 4,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D {
                    width: self.frame_w,
                    height: self.frame_h,
                    depth: 1,
                },
                ..Default::default()
            };
            self.core.device.cmd_copy_buffer_to_image(
                cb,
                src,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_read = barrier(
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                range,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
            self.core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_read],
            );
        }
        raw.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }

    /// Wait for the most recently submitted render command buffer. Used to
    /// retire a receive slot whose data was read by that command buffer.
    pub fn wait_frame(&self) -> Result<()> {
        unsafe {
            self.core
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        Ok(())
    }

    pub fn resize(&mut self, w: u32, h: u32) -> Result<()> {
        if w == 0 || h == 0 || (w == self.extent.width && h == self.extent.height) {
            return Ok(());
        }
        self.recreate_swapchain(Some((w, h)))
    }

    fn recreate_swapchain(&mut self, target: Option<(u32, u32)>) -> Result<()> {
        // Wayland surfaces usually report currentExtent = u32::MAX; keep our
        // known size when the caller did not provide a new one.
        let target = target.or(Some((self.extent.width, self.extent.height)));
        unsafe { self.core.device.device_wait_idle()? };
        let old_swap = self.swapchain;
        let old_views = std::mem::take(&mut self.views);
        let (swapchain, format, extent, images, views) = match make_swapchain(
            &self.core,
            &self.surface_loader,
            self.surface,
            old_swap,
            target,
        ) {
            Ok(v) => v,
            Err(e) => {
                self.views = old_views;
                return Err(e);
            }
        };
        unsafe {
            for v in old_views {
                self.core.device.destroy_image_view(v, None);
            }
            if old_swap != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(old_swap, None);
            }
        }
        self.swapchain = swapchain;
        self.extent = extent;
        self.images = images;
        self.views = views;
        if format != self.format {
            self.format = format;
            let (vp, yp, op, _) = create_pipelines(
                &self.core,
                format,
                self.video_layout,
                self.yuv_layout,
                self.overlay_layout,
            )?;
            unsafe {
                self.core.device.destroy_pipeline(self.video_pipeline, None);
                self.core.device.destroy_pipeline(self.yuv_pipeline, None);
                self.core.device.destroy_pipeline(self.overlay_pipeline, None);
            }
            self.video_pipeline = vp;
            self.yuv_pipeline = yp;
            self.overlay_pipeline = op;
        }
        log::info!("[vk] swapchain recreated {}x{}", self.extent.width, self.extent.height);
        Ok(())
    }

    fn effective_filter(&self) -> u32 {
        match self.filter_mode {
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            _ => {
                let scl = (self.extent.width as f32 / self.frame_w.max(1) as f32)
                    .min(self.extent.height as f32 / self.frame_h.max(1) as f32);
                if (scl - scl.round()).abs() < 1e-3 {
                    0
                } else if scl < 1.0 {
                    3
                } else {
                    1
                }
            }
        }
    }

    fn sampler_for(&self, filter: u32) -> vk::Sampler {
        if filter == 1 { self.sampler_linear } else { self.sampler_nearest }
    }

    fn update_video_set(&mut self, filter: u32) {
        if self.video_view == vk::ImageView::null() {
            return;
        }
        let info = vk::DescriptorImageInfo {
            sampler: self.sampler_for(filter),
            image_view: self.video_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let write = vk::WriteDescriptorSet {
            dst_set: self.video_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &info,
            ..Default::default()
        };
        unsafe { self.core.device.update_descriptor_sets(&[write], &[]) };
    }

    fn update_yuv_set(&mut self, filter: u32) {
        if self.yuv_views[0] == vk::ImageView::null() {
            return;
        }
        let infos: Vec<vk::DescriptorImageInfo> = self
            .yuv_views
            .iter()
            .map(|v| vk::DescriptorImageInfo {
                sampler: self.sampler_for(filter),
                image_view: *v,
                image_layout: vk::ImageLayout::GENERAL,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet {
                dst_set: self.yuv_set,
                dst_binding: i as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: info,
                ..Default::default()
            })
            .collect();
        unsafe { self.core.device.update_descriptor_sets(&writes, &[]) };
    }

    fn upload_overlay(&mut self, bytes: &[u8], w: u32, h: u32) -> Result<()> {
        let ov = self.overlay.as_mut().context("overlay not active")?;
        if w != ov.w || h != ov.h {
            bail!("overlay {}x{} != {}x{}", w, h, ov.w, ov.h);
        }
        ov.host.as_mut_slice().copy_from_slice(bytes);
        let old = ov.layout;
        let image = ov.image.image;
        let host = ov.host.buffer;
        self.core.one_shot(|cb| unsafe {
            let range = color_range();
            let to_dst = barrier(image, old, vk::ImageLayout::TRANSFER_DST_OPTIMAL, range, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
            self.core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_dst],
            );
            let copy = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                ..Default::default()
            };
            self.core.device.cmd_copy_buffer_to_image(
                cb, host, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy],
            );
            let to_read = barrier(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, range, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ);
            self.core.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_read],
            );
        })?;
        ov.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        Ok(())
    }

    /// Present the current video source. Returns the client-side render start
    /// and present-submit timestamps (microseconds).
    pub fn present(&mut self, overlay: Option<(&[u8], u32, u32)>) -> Result<(u64, u64)> {
        if let Some((bytes, w, h)) = overlay {
            self.upload_overlay(bytes, w, h)?;
        }
        let filter = self.effective_filter();
        if filter != self.last_filter {
            self.update_video_set(filter);
            self.update_yuv_set(filter);
            self.last_filter = filter;
        }

        let (index, mut suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok(v) => v,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(None)?;
                unsafe {
                    self.swapchain_loader.acquire_next_image(
                        self.swapchain,
                        u64::MAX,
                        self.image_available,
                        vk::Fence::null(),
                    )?
                }
            }
            Err(e) => return Err(e.into()),
        };
        let _ = &mut suboptimal;
        let t4 = rdma_display_proto::unix_now_us();
        unsafe {
            self.core.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            if let Some(d) = self.lossy.as_mut() {
                // The previous render submission has consumed this decode
                // semaphore; its sync object can now be destroyed.
                d.release_in_flight();
            }
            self.core.device.reset_fences(&[self.fence])?;
            self.core.device.reset_command_buffer(
                self.command,
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.core.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            self.record_pending_lossless(self.command)?;
            let range = color_range();
            let to_color = barrier(
                self.images[index as usize],
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                range,
                vk::AccessFlags::empty(),
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
            self.core.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_color],
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            };
            let attachment = vk::RenderingAttachmentInfo {
                image_view: self.views[index as usize],
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: clear,
                ..Default::default()
            };
            let rendering = vk::RenderingInfo {
                render_area: scissor,
                layer_count: 1,
                color_attachment_count: 1,
                p_color_attachments: &attachment,
                ..Default::default()
            };
            if self.codec == Codec::Raw {
                self.record_raw_upload(self.command);
            }
            if self.codec == Codec::Lossy {
                if let Some(d) = self.lossy.as_ref() {
                    d.record_sample_barrier(self.command);
                }
            }
            self.core.device.cmd_begin_rendering(self.command, &rendering);
            self.core.device.cmd_set_viewport(self.command, 0, &[viewport]);
            self.core.device.cmd_set_scissor(self.command, 0, &[scissor]);

            let mut pc = [0u8; 20];
            let fs: [f32; 4] = [
                self.extent.width as f32,
                self.extent.height as f32,
                self.frame_w as f32,
                self.frame_h as f32,
            ];
            pc[..16].copy_from_slice(std::slice::from_raw_parts(fs.as_ptr() as *const u8, 16));
            pc[16..20].copy_from_slice(&(filter as i32).to_le_bytes());
            match self.codec {
                Codec::Raw | Codec::Lossless => {
                    self.core.device.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::GRAPHICS, self.video_pipeline);
                    self.core.device.cmd_bind_descriptor_sets(self.command, vk::PipelineBindPoint::GRAPHICS, self.video_layout, 0, &[self.video_set], &[]);
                    self.core.device.cmd_push_constants(self.command, self.video_layout, vk::ShaderStageFlags::FRAGMENT, 0, &pc);
                    self.core.device.cmd_draw(self.command, 3, 1, 0, 0);
                }
                Codec::Lossy => {
                    self.core.device.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::GRAPHICS, self.yuv_pipeline);
                    self.core.device.cmd_bind_descriptor_sets(self.command, vk::PipelineBindPoint::GRAPHICS, self.yuv_layout, 0, &[self.yuv_set], &[]);
                    self.core.device.cmd_push_constants(self.command, self.yuv_layout, vk::ShaderStageFlags::FRAGMENT, 0, &pc);
                    self.core.device.cmd_draw(self.command, 3, 1, 0, 0);
                }
            }

            if let Some(ov) = &self.overlay {
                if ov.layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                    let mut opc = [0f32; 6];
                    opc[0] = 0.0;
                    opc[1] = 0.0;
                    opc[2] = ov.w as f32;
                    opc[3] = ov.h as f32;
                    opc[4] = self.extent.width as f32;
                    opc[5] = self.extent.height as f32;
                    let bytes = std::slice::from_raw_parts(opc.as_ptr() as *const u8, 24);
                    self.core.device.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::GRAPHICS, self.overlay_pipeline);
                    self.core.device.cmd_bind_descriptor_sets(self.command, vk::PipelineBindPoint::GRAPHICS, self.overlay_layout, 0, &[self.overlay_set], &[]);
                    self.core.device.cmd_push_constants(self.command, self.overlay_layout, vk::ShaderStageFlags::FRAGMENT, 0, bytes);
                    self.core.device.cmd_draw(self.command, 3, 1, 0, 0);
                }
            }
            self.core.device.cmd_end_rendering(self.command);
            let to_present = barrier(
                self.images[index as usize],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                range,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::empty(),
            );
            self.core.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[], &[], &[to_present],
            );
            self.core.device.end_command_buffer(self.command)?;

            let lossy_sem = self
                .lossy
                .as_ref()
                .and_then(|d| d.pending_semaphore());
            let mut wait_sems = vec![self.image_available];
            let mut stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            // Timeline values are ignored for the binary image-available
            // semaphore; the PyroWave decode semaphore uses value 1.
            let mut wait_values = vec![0u64];
            if let Some(sem) = lossy_sem {
                wait_sems.push(sem);
                stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
                wait_values.push(1);
            }
            let signal = [self.render_finished];
            let signal_values = [0u64];
            let timeline = vk::TimelineSemaphoreSubmitInfo {
                wait_semaphore_value_count: wait_values.len() as u32,
                p_wait_semaphore_values: wait_values.as_ptr(),
                signal_semaphore_value_count: 1,
                p_signal_semaphore_values: signal_values.as_ptr(),
                ..Default::default()
            };
            let cbs = [self.command];
            let submit = vk::SubmitInfo {
                p_next: &timeline as *const _ as *const std::ffi::c_void,
                wait_semaphore_count: wait_sems.len() as u32,
                p_wait_semaphores: wait_sems.as_ptr(),
                p_wait_dst_stage_mask: stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: cbs.as_ptr(),
                signal_semaphore_count: 1,
                p_signal_semaphores: signal.as_ptr(),
                ..Default::default()
            };
            self.core.device.queue_submit(self.core.queue, &[submit], self.fence)?;
            if lossy_sem.is_some() {
                if let Some(d) = self.lossy.as_mut() {
                    d.mark_submitted();
                }
            }
            // Do NOT wait for the render fence here: queue_present already
            // waits on render_finished, and waiting on the CPU would only add
            // a full GPU-render round trip. The next frame waits for this fence
            // before reusing the command buffer.
            let waits = [self.render_finished];
            let swapchains = [self.swapchain];
            let indices = [index];
            let present = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: waits.as_ptr(),
                swapchain_count: 1,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: indices.as_ptr(),
                ..Default::default()
            };
            match self.swapchain_loader.queue_present(self.core.queue, &present) {
                Ok(false) => {}
                Ok(true) => suboptimal = true,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => suboptimal = true,
                Err(e) => return Err(e.into()),
            }
        }
        let t5 = rdma_display_proto::unix_now_us();
        if suboptimal {
            if let Err(e) = self.recreate_swapchain(None) {
                log::warn!("[vk] swapchain recreate after present failed: {e:#}");
            }
        }
        Ok((t4, t5))
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.core.device.device_wait_idle();
            if let Some(r) = self.raw.take() {
                self.core.device.destroy_image_view(r.view, None);
                self.core.destroy_image(r.image);
            }
            if let Some(o) = self.overlay.take() {
                self.core.device.destroy_image_view(o.view, None);
                self.core.device.unmap_memory(o.host.memory);
                self.core.device.destroy_buffer(o.host.buffer, None);
                self.core.device.free_memory(o.host.memory, None);
                self.core.destroy_image(o.image);
            }
            self.lossy.take();
            self.lossless.take();
            if let Some(imp) = self.host_import.take() {
                self.core.device.destroy_buffer(imp.buffer, None);
                self.core.device.free_memory(imp.memory, None);
            }
            self.core.device.destroy_pipeline(self.video_pipeline, None);
            self.core.device.destroy_pipeline(self.yuv_pipeline, None);
            self.core.device.destroy_pipeline(self.overlay_pipeline, None);
            self.core.device.destroy_pipeline_layout(self.video_layout, None);
            self.core.device.destroy_pipeline_layout(self.yuv_layout, None);
            self.core.device.destroy_pipeline_layout(self.overlay_layout, None);
            self.core.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.core.device.destroy_descriptor_set_layout(self.video_set_layout, None);
            self.core.device.destroy_descriptor_set_layout(self.yuv_set_layout, None);
            self.core.device.destroy_descriptor_set_layout(self.overlay_set_layout, None);
            self.core.device.destroy_sampler(self.sampler_nearest, None);
            self.core.device.destroy_sampler(self.sampler_linear, None);
            self.core.device.destroy_semaphore(self.image_available, None);
            self.core.device.destroy_semaphore(self.render_finished, None);
            self.core.device.destroy_fence(self.fence, None);
            self.core.device.free_command_buffers(self.core.command_pool, &[self.command]);
            for v in &self.views {
                self.core.device.destroy_image_view(*v, None);
            }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

fn make_swapchain(
    core: &VkCore,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    old: vk::SwapchainKHR,
    target: Option<(u32, u32)>,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>, Vec<vk::ImageView>)> {
    unsafe {
        let caps = surface_loader.get_physical_device_surface_capabilities(core.physical_device, surface)?;
        let formats = surface_loader.get_physical_device_surface_formats(core.physical_device, surface)?;
        if formats.is_empty() {
            bail!("Vulkan surface reports no formats");
        }
        let pick = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
            .unwrap_or(&formats[0]);
        let format = pick.format;
        let color_space = pick.color_space;
        let modes = surface_loader.get_physical_device_surface_present_modes(core.physical_device, surface)?;
        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        let (mut w, mut h) = match target {
            Some((w, h)) if w > 0 && h > 0 => (w, h),
            _ => (caps.current_extent.width, caps.current_extent.height),
        };
        if w == u32::MAX || w == 0 {
            w = 1280.max(caps.min_image_extent.width);
        }
        if h == u32::MAX || h == 0 {
            h = 720.max(caps.min_image_extent.height);
        }
        w = w.clamp(caps.min_image_extent.width, caps.max_image_extent.width);
        h = h.clamp(caps.min_image_extent.height, caps.max_image_extent.height);
        let mut count = caps.min_image_count.saturating_add(1);
        if caps.max_image_count > 0 {
            count = count.min(caps.max_image_count);
        }
        let alpha = if caps.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };
        let ci = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: count,
            image_format: format,
            image_color_space: color_space,
            image_extent: vk::Extent2D { width: w, height: h },
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: caps.current_transform,
            composite_alpha: alpha,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: old,
            ..Default::default()
        };
        let loader = swapchain::Device::new(&core.instance, &core.device);
        let swapchain = loader.create_swapchain(&ci, None)?;
        let images = loader.get_swapchain_images(swapchain)?;
        let mut views = Vec::with_capacity(images.len());
        for image in &images {
            views.push(core.device.create_image_view(
                &vk::ImageViewCreateInfo {
                    image: *image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            )?);
        }
        Ok((swapchain, format, vk::Extent2D { width: w, height: h }, images, views))
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn barrier(
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier {
        old_layout: old,
        new_layout: new,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: range,
        src_access_mask: src,
        dst_access_mask: dst,
        ..Default::default()
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
            .context("create shader module")
    }
}

fn set_layout(core: &VkCore, count: u32) -> Result<vk::DescriptorSetLayout> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..count)
        .map(|b| vk::DescriptorSetLayoutBinding {
            binding: b,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        })
        .collect();
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
            .context("descriptor set layout")
    }
}

fn pipeline_layout(
    core: &VkCore,
    set_layout: vk::DescriptorSetLayout,
    push_size: u32,
) -> Result<vk::PipelineLayout> {
    let sets = [set_layout];
    let ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: push_size,
    }];
    unsafe {
        core.device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo {
                    set_layout_count: 1,
                    p_set_layouts: sets.as_ptr(),
                    push_constant_range_count: 1,
                    p_push_constant_ranges: ranges.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .context("pipeline layout")
    }
}

fn descriptor_sets(
    core: &VkCore,
    video: vk::DescriptorSetLayout,
    yuv: vk::DescriptorSetLayout,
    overlay: vk::DescriptorSetLayout,
) -> Result<(vk::DescriptorPool, [vk::DescriptorSet; 3])> {
    let sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 5,
    }];
    unsafe {
        let pool = core.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo {
                max_sets: 3,
                pool_size_count: 1,
                p_pool_sizes: sizes.as_ptr(),
                ..Default::default()
            },
            None,
        )?;
        let layouts = [video, yuv, overlay];
        let sets = core.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 3,
                p_set_layouts: layouts.as_ptr(),
                ..Default::default()
            },
        )?;
        Ok((pool, [sets[0], sets[1], sets[2]]))
    }
}

fn create_pipelines(
    core: &VkCore,
    format: vk::Format,
    video_layout: vk::PipelineLayout,
    yuv_layout: vk::PipelineLayout,
    overlay_layout: vk::PipelineLayout,
) -> Result<(vk::Pipeline, vk::Pipeline, vk::Pipeline, vk::Format)> {
    let vs = shader_module(core, VIDEO_SPV)?;
    let video_fs = shader_module(core, VIDEO_FRAG_SPV)?;
    let yuv_fs = shader_module(core, YUV_FRAG_SPV)?;
    let overlay_fs = shader_module(core, OVERLAY_FRAG_SPV)?;
    let video = make_pipeline(core, format, video_layout, vs, video_fs, false)?;
    let yuv = make_pipeline(core, format, yuv_layout, vs, yuv_fs, false)?;
    let overlay = make_pipeline(core, format, overlay_layout, vs, overlay_fs, true)?;
    unsafe {
        core.device.destroy_shader_module(vs, None);
        core.device.destroy_shader_module(video_fs, None);
        core.device.destroy_shader_module(yuv_fs, None);
        core.device.destroy_shader_module(overlay_fs, None);
    }
    Ok((video, yuv, overlay, format))
}

fn make_pipeline(
    core: &VkCore,
    format: vk::Format,
    layout: vk::PipelineLayout,
    vs: vk::ShaderModule,
    fs: vk::ShaderModule,
    blend: bool,
) -> Result<vk::Pipeline> {
    let name = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: vs,
            p_name: name.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: fs,
            p_name: name.as_ptr(),
            ..Default::default()
        },
    ];
    let vertex = vk::PipelineVertexInputStateCreateInfo::default();
    let input = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    let viewport = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count: 1,
        ..Default::default()
    };
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states: dynamic_states.as_ptr(),
        ..Default::default()
    };
    let raster = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        ..Default::default()
    };
    let multisample = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: if blend { vk::TRUE } else { vk::FALSE },
        src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    let color = vk::PipelineColorBlendStateCreateInfo {
        attachment_count: 1,
        p_attachments: &attachment,
        ..Default::default()
    };
    let formats = [format];
    let rendering = vk::PipelineRenderingCreateInfo {
        color_attachment_count: 1,
        p_color_attachment_formats: formats.as_ptr(),
        ..Default::default()
    };
    let ci = vk::GraphicsPipelineCreateInfo {
        p_next: &rendering as *const _ as *const std::ffi::c_void,
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex,
        p_input_assembly_state: &input,
        p_viewport_state: &viewport,
        p_rasterization_state: &raster,
        p_multisample_state: &multisample,
        p_color_blend_state: &color,
        p_dynamic_state: &dynamic,
        layout,
        ..Default::default()
    };
    unsafe {
        core.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[ci], None)
            .map_err(|(_, e)| e)
            .context("create graphics pipeline")
            .map(|p| p[0])
    }
}

fn sampler(core: &VkCore, filter: vk::Filter) -> Result<vk::Sampler> {
    unsafe {
        core.device
            .create_sampler(
                &vk::SamplerCreateInfo {
                    mag_filter: filter,
                    min_filter: filter,
                    mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    max_lod: 0.0,
                    ..Default::default()
                },
                None,
            )
            .context("create sampler")
    }
}
