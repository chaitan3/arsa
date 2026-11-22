//! Shared Vulkan device + dma-buf import cache, and the raw (readback)
//! materializer.
//!
//! One `Gpu` is created per server process and shared by the raw materializer,
//! the lossless encoder and the lossy materializer. A scanout dma-buf is
//! imported once per buffer identity and kept in the cache until the
//! compositor frees it; all three consumers use the same `VkImage`.

use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use rdma_display_capture::{CaptureEvent, DmabufScanout, FrameFormat, FrameMaterializer};
use rdma_display_proto::MatTimes;
use rdma_display_vkcore::{ash::vk, HostBuffer, Image, VkCore};

type Key = (u64, u32, u32, u32, u32, u64, u32);

pub struct Gpu {
    pub core: Arc<VkCore>,
    cache: Mutex<HashMap<Key, Arc<CachedImage>>>,
}

pub struct CachedImage {
    core: Arc<VkCore>,
    img: Image,
    view: vk::ImageView,
    pub format: vk::Format,
    layout: Mutex<vk::ImageLayout>,
}

impl CachedImage {
    pub fn image(&self) -> vk::Image {
        self.img.image
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Transition the image if needed. Consumers run on the server thread;
    /// the layout stabilises after the first frame per codec.
    pub fn ensure_layout(&self, new: vk::ImageLayout) -> Result<()> {
        let mut cur = self.layout.lock().unwrap();
        if *cur == new {
            return Ok(());
        }
        self.core
            .transition(self.img.image, *cur, new, vk::ImageAspectFlags::COLOR)?;
        *cur = new;
        Ok(())
    }
}

impl Drop for CachedImage {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_image_view(self.view, None);
        }
        let img = unsafe { std::ptr::read(&self.img) };
        self.core.destroy_image(img);
    }
}

impl Gpu {
    pub fn new() -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            core: Arc::new(VkCore::new().context("create the shared Vulkan context")?),
            cache: Mutex::new(HashMap::new()),
        }))
    }

    /// Drop all imported scanout images. Called when a session ends so an
    /// idle server does not retain a compositor's 4K scanout buffers (up to
    /// eight full-frame images in the cache).
    pub fn clear_cache(&self) -> Result<()> {
        unsafe { self.core.device.device_wait_idle()? };
        self.cache.lock().unwrap().clear();
        Ok(())
    }

    /// Import (or look up) the scanout dma-buf. `buf_id == 0` (no stable
    /// identity, e.g. QEMU) bypasses the cache.
    pub fn import(&self, d: &DmabufScanout) -> Result<Arc<CachedImage>> {
        if d.fds.len() != 1 {
            bail!(
                "Vulkan materializer needs a single-plane scanout (got {})",
                d.fds.len()
            );
        }
        let fmt = &d.format;
        let stride = d.strides.first().copied().unwrap_or(fmt.stride);
        let offset = d.offsets.first().copied().unwrap_or(0);
        let format = crate::lossy::fourcc_to_vk(fmt.fourcc).with_context(|| {
            format!(
                "unsupported scanout fourcc {:#010x} for Vulkan",
                fmt.fourcc
            )
        })?;
        let key = (
            d.buf_id,
            fmt.width,
            fmt.height,
            stride,
            offset,
            fmt.modifier,
            fmt.fourcc,
        );
        if d.buf_id != 0 {
            if let Some(img) = self.cache.lock().unwrap().get(&key) {
                return Ok(img.clone());
            }
        }
        let img = self.core.import_dmabuf(
            d.fds[0].as_raw_fd(),
            fmt.width,
            fmt.height,
            format,
            fmt.modifier,
            offset,
            stride,
        )?;
        let view = self.core.make_view(
            img.image,
            format,
            vk::ImageAspectFlags::COLOR,
            vk::ComponentSwizzle::IDENTITY,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;
        let cached = Arc::new(CachedImage {
            core: self.core.clone(),
            img,
            view,
            format,
            layout: Mutex::new(vk::ImageLayout::UNDEFINED),
        });
        if d.buf_id != 0 {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, cached.clone());
            // The compositor cycles a handful of buffers. Drop entries no
            // consumer holds; if a flood still fills the map, clear it.
            if cache.len() > 8 {
                cache.retain(|_, v| Arc::strong_count(v) > 1);
            }
            if cache.len() > 8 {
                cache.clear();
            }
        }
        Ok(cached)
    }
}

/// The per-session host-visible staging buffers. Shared by the raw
/// materializer, the lossless encoder and the transport registration.
pub struct VkStaging {
    core: Arc<VkCore>,
    pub buffers: Vec<HostBuffer>,
    /// postable payload length (the MR length; the buffers have +16 tailroom)
    pub len: usize,
}

impl VkStaging {
    /// The staging buffer whose CPU address is `ptr`, if any.
    pub fn buffer_matching(&self, ptr: *mut u8) -> Option<vk::Buffer> {
        self.buffers
            .iter()
            .find(|b| b.ptr == ptr)
            .map(|b| b.buffer)
    }

    /// The staging buffer for `slot` only when `ptr` is that buffer's CPU
    /// address. This keeps direct GPU output bound to the memory the transport
    /// is actually about to post (RDMA slot or the TCP send buffer); a
    /// mismatched pointer falls back to the explicit coded copy.
    pub fn buffer_for(&self, slot: usize, ptr: *mut u8) -> Option<vk::Buffer> {
        self.buffers
            .get(slot)
            .filter(|b| b.ptr == ptr)
            .map(|b| b.buffer)
    }
}

impl Drop for VkStaging {
    fn drop(&mut self) {
        unsafe {
            // never destroy memory the GPU may still be reading
            let _ = self.core.device.device_wait_idle();
            for b in self.buffers.drain(..) {
                self.core.device.destroy_buffer(b.buffer, None);
                self.core.device.free_memory(b.memory, None);
            }
        }
    }
}

pub type SharedStaging = Arc<Mutex<Option<Arc<VkStaging>>>>;

/// Raw materializer: imports the scanout into Vulkan and copies it to
/// host-visible staging buffers (the PBO equivalent). The RDMA path uses the
/// async `issue_frame`/`finish_frame` pair so the copy overlaps the previous
/// frame; TCP uses the synchronous `copy_frame`.
pub struct VkMaterializer {
    gpu: Arc<Gpu>,
    staging: SharedStaging,
    image: Option<Arc<CachedImage>>,
    fmt: Option<FrameFormat>,
    frame_bytes: usize,
    scratch: Option<HostBuffer>,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    dirty: bool,
}

unsafe impl Send for VkMaterializer {}

impl VkMaterializer {
    pub fn new(gpu: Arc<Gpu>, staging: SharedStaging) -> Self {
        Self {
            gpu,
            staging,
            image: None,
            fmt: None,
            frame_bytes: 0,
            scratch: None,
            cmds: Vec::new(),
            fences: Vec::new(),
            dirty: false,
        }
    }

    fn record_copy(&self, cb: vk::CommandBuffer, img: &CachedImage, dst: vk::Buffer) {
        let fmt = self.fmt.as_ref().expect("scanout format");
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: fmt.stride / 4,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: fmt.width,
                height: fmt.height,
                depth: 1,
            },
            ..Default::default()
        };
        unsafe {
            self.gpu.core.device.cmd_copy_image_to_buffer(
                cb,
                img.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                &[region],
            );
        }
    }

    fn copy_image(&self, img: &CachedImage, dst: vk::Buffer) -> Result<()> {
        let core = &self.gpu.core;
        core.one_shot(|cb| self.record_copy(cb, img, dst))
    }

    fn destroy_staging(&mut self) {
        *self.staging.lock().unwrap() = None;
        unsafe {
            let dev = &self.gpu.core.device;
            if let Some(b) = self.scratch.take() {
                dev.destroy_buffer(b.buffer, None);
                dev.free_memory(b.memory, None);
            }
            for f in self.fences.drain(..) {
                dev.destroy_fence(f, None);
            }
            if !self.cmds.is_empty() {
                dev.free_command_buffers(self.gpu.core.command_pool, &self.cmds);
                self.cmds.clear();
            }
        }
    }
}

impl FrameMaterializer for VkMaterializer {
    fn name(&self) -> &'static str {
        "vk"
    }

    fn prepare(&mut self, fmt: &FrameFormat, dmabuf: Option<&DmabufScanout>) -> Result<()> {
        let d = dmabuf.context("vk materializer requires a dmabuf scanout")?;
        let img = self.gpu.import(d)?;
        if !matches!(
            img.format,
            vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
        ) {
            bail!(
                "scanout fourcc {:#010x} is not 8-bit BGRA/RGBA",
                fmt.fourcc
            );
        }
        self.image = Some(img);
        self.fmt = Some(fmt.clone());
        self.frame_bytes = fmt.frame_bytes() as usize;
        self.dirty = true;
        if self.staging.lock().unwrap().is_none() && self.scratch.is_none() {
            self.scratch = Some(
                self.gpu
                    .core
                    .host_buffer(self.frame_bytes as u64)
                    .context("allocate the readback scratch buffer")?,
            );
        }
        Ok(())
    }

    fn copy_frame(&mut self, out: &mut [u8]) -> Result<(bool, MatTimes)> {
        if !self.dirty {
            return Ok((false, MatTimes::ZERO));
        }
        let Some(img) = self.image.as_ref() else {
            return Ok((false, MatTimes::ZERO));
        };
        img.ensure_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
        // TCP with the external staging buffer: copy straight into the memory
        // that will be sealed and written, with no scratch + memcpy.
        let staging = self.staging.lock().unwrap().clone();
        if let Some(staging) = staging.as_ref() {
            if let Some(dst) = staging.buffer_matching(out.as_ptr() as *mut u8) {
                self.copy_image(img, dst)?;
                self.dirty = false;
                return Ok((true, MatTimes::ZERO));
            }
        }
        if self.scratch.is_none() {
            self.scratch = Some(self.gpu.core.host_buffer(self.frame_bytes as u64)?);
        }
        let buf = self.scratch.as_ref().unwrap();
        self.copy_image(img, buf.buffer)?;
        let n = self.frame_bytes.min(out.len());
        out[..n].copy_from_slice(&buf.as_slice()[..n]);
        self.dirty = false;
        Ok((true, MatTimes::ZERO))
    }

    fn bind_staging(&mut self, n: usize, slot_len: usize) -> Option<Vec<(*mut u8, usize)>> {
        if let Some(s) = self.staging.lock().unwrap().as_ref() {
            if s.buffers.len() == n && s.len >= slot_len {
                return Some(s.buffers.iter().map(|b| (b.ptr, slot_len)).collect());
            }
        }
        self.destroy_staging();
        let mut host = Vec::with_capacity(n);
        for _ in 0..n {
            // +16 tailroom for the in-place AES-GCM tag on the TCP path
            match self.gpu.core.host_buffer(slot_len as u64 + 16) {
                Ok(b) => host.push(b),
                Err(e) => {
                    log::warn!("[vk] staging buffer alloc failed: {e:#}");
                    return None;
                }
            }
        }
        let staging = Arc::new(VkStaging {
            core: self.gpu.core.clone(),
            buffers: host,
            len: slot_len,
        });
        *self.staging.lock().unwrap() = Some(staging.clone());
        let cmds = match unsafe {
            self.gpu.core.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo {
                    command_pool: self.gpu.core.command_pool,
                    level: vk::CommandBufferLevel::PRIMARY,
                    command_buffer_count: n as u32,
                    ..Default::default()
                },
            )
        } {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[vk] staging command buffers failed: {e}");
                *self.staging.lock().unwrap() = None;
                return None;
            }
        };
        let mut fences = Vec::with_capacity(n);
        for _ in 0..n {
            match unsafe {
                self.gpu
                    .core
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            } {
                Ok(f) => fences.push(f),
                Err(e) => {
                    log::warn!("[vk] staging fence failed: {e}");
                    for f in fences {
                        unsafe { self.gpu.core.device.destroy_fence(f, None) };
                    }
                    unsafe {
                        self.gpu.core.device.free_command_buffers(self.gpu.core.command_pool, &cmds);
                    }
                    *self.staging.lock().unwrap() = None;
                    return None;
                }
            }
        }
        self.cmds = cmds;
        self.fences = fences;
        Some(staging.buffers.iter().map(|b| (b.ptr, slot_len)).collect())
    }

    fn issue_frame(&mut self, slot: usize, out: &mut [u8]) -> Result<bool> {
        let staging = self.staging.lock().unwrap().clone();
        let Some(staging) = staging else {
            return Ok(self.copy_frame(out)?.0);
        };
        if slot >= staging.buffers.len() || slot >= self.cmds.len() || slot >= self.fences.len() {
            return Ok(self.copy_frame(out)?.0);
        }
        if !self.dirty {
            return Ok(false);
        }
        let Some(img) = self.image.as_ref() else {
            return Ok(false);
        };
        img.ensure_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
        let cb = self.cmds[slot];
        let fence = self.fences[slot];
        let core = &self.gpu.core;
        unsafe {
            core.device.reset_fences(&[fence])?;
            core.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )?;
            self.record_copy(cb, img, staging.buffers[slot].buffer);
            core.device.end_command_buffer(cb)?;
            let cbs = [cb];
            core.device.queue_submit(
                core.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: cbs.as_ptr(),
                    ..Default::default()
                }],
                fence,
            )?;
        }
        self.dirty = false;
        Ok(true)
    }

    fn finish_frame(&mut self, slot: usize, _out: &mut [u8]) -> Result<(bool, MatTimes)> {
        if let Some(f) = self.fences.get(slot) {
            unsafe {
                self.gpu
                    .core
                    .device
                    .wait_for_fences(&[*f], true, u64::MAX)?;
            }
        }
        Ok((true, MatTimes::ZERO))
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn handle_event(&mut self, _ev: CaptureEvent) -> Result<bool> {
        Ok(false)
    }

    fn effective_format(&self) -> Option<FrameFormat> {
        self.fmt.clone()
    }

    fn drop_frames(&mut self) {
        if !self.fences.is_empty() {
            unsafe {
                let _ = self
                    .gpu
                    .core
                    .device
                    .wait_for_fences(&self.fences, true, u64::MAX);
                let _ = self.gpu.core.device.reset_fences(&self.fences);
            }
        }
        self.dirty = true;
    }

    fn release(&mut self) {
        self.destroy_staging();
        self.image = None;
        self.fmt = None;
        self.frame_bytes = 0;
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::FromRawFd;

    #[test]
    fn raw_materializer_copies_a_scanout() {
        let gpu = Gpu::new().expect("Vulkan GPU");
        let (img, fd, pitch) = gpu
            .core
            .create_exportable_image(
                vk::Format::B8G8R8A8_UNORM,
                64,
                64,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .expect("exportable image");
        let stride = pitch as usize;
        let frame_bytes = stride * 64;
        let mut pattern = vec![0u8; frame_bytes];
        for y in 0..64usize {
            for x in 0..64usize {
                let o = y * stride + x * 4;
                pattern[o] = (x * 3) as u8;
                pattern[o + 1] = (y * 5) as u8;
                pattern[o + 2] = (x ^ y) as u8;
                pattern[o + 3] = 0xff;
            }
        }
        let mut host = gpu.core.host_buffer(frame_bytes as u64).expect("host buffer");
        host.as_mut_slice()[..frame_bytes].copy_from_slice(&pattern);
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        gpu.core
            .one_shot(|cb| unsafe {
                let to_dst = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    image: img.image,
                    subresource_range: range,
                    src_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };
                gpu.core.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_dst],
                );
                let copy = vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D {
                        width: 64,
                        height: 64,
                        depth: 1,
                    },
                    ..Default::default()
                };
                gpu.core.device.cmd_copy_buffer_to_image(
                    cb,
                    host.buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );
                let to_read = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: img.image,
                    subresource_range: range,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                };
                gpu.core.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_read],
                );
            })
            .expect("upload scanout");

        let fmt = FrameFormat {
            width: 64,
            height: 64,
            stride: pitch,
            fourcc: 0x3432_5258, // XR24
            modifier: 0,
            y0_top: true,
            num_planes: 1,
        };
        let d = DmabufScanout {
            fb_id: 0,
            buf_id: 1,
            fds: vec![unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) }],
            offsets: vec![0],
            strides: vec![pitch],
            format: fmt.clone(),
        };
        let staging: SharedStaging = Arc::new(Mutex::new(None));
        let mut mat = VkMaterializer::new(gpu.clone(), staging);
        mat.prepare(&fmt, Some(&d)).expect("prepare");
        let mut out = vec![0u8; frame_bytes];
        let (changed, _) = mat.copy_frame(&mut out).expect("copy");
        assert!(changed);
        assert_eq!(out, pattern, "raw Vulkan materializer mismatch");
    }
}
