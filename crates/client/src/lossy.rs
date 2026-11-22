//! Client-side PyroWave decoder for `--codec lossy`.
//!
//! The server sends a framed frame: `[u32 n][u32 sizes...][packet bytes]`.
//! We push each packet and decode on the GPU into Y/U/V plane images on the
//! client's shared Vulkan device. The Vulkan presenter samples those planes
//! directly and does YUV->RGB in its fragment shader. There is no dma-buf
//! export/import, no GL round trip and no full-frame readback on the streaming
//! path.
//!
//! (`read_rgba` is the headless/`--dump` debug path: it downloads and converts
//! on the CPU so the content can be inspected; it is never used when
//! presenting.)

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rdma_display_vkcore::ash::vk;
use rdma_display_vkcore::pyrowave::{
    Chroma, Pyrowave, PyrowaveDecoder, PyrowaveDevice, PyrowaveGpuBuffers, PyrowaveImageView,
    PyrowaveSyncObject,
};
use rdma_display_vkcore::{HostBuffer, Image, VkCore};

pub struct LossyDecoder {
    vk: Arc<VkCore>,
    pw: Pyrowave,
    dev: PyrowaveDevice,
    dec: PyrowaveDecoder,
    w: u32,
    h: u32,
    chroma444: bool,
    /// R8 decode planes sampled by the Vulkan presenter
    planes: Vec<Image>,
    views: [vk::ImageView; 3],
    /// current layout of the three decode planes
    layout: vk::ImageLayout,
    /// async decode waiting for the next present submission
    pending_sync: Option<PyrowaveSyncObject>,
    pending_semaphore: vk::Semaphore,
    /// async decode already submitted to the GPU; released after its fence
    in_flight_sync: Option<PyrowaveSyncObject>,
    /// headless/--dump debug readback, allocated lazily on first use
    readback: Option<LossyReadback>,
}

struct LossyReadback {
    ybuf: HostBuffer,
    ubuf: HostBuffer,
    vbuf: HostBuffer,
    bgra: Vec<u8>,
}

impl LossyDecoder {
    /// Create the decoder on the client's shared Vulkan device. The R8 decode
    /// planes stay on that device and are sampled directly by the presenter;
    /// no dma-buf export/import or GL round trip.
    pub fn new(vk: Arc<VkCore>, w: u32, h: u32, chroma: u8) -> Result<Self> {
        let chroma444 = chroma == 1;
        let pw = Pyrowave::load(std::env::var("PYROWAVE_LIB").ok().as_deref())?;
        let dev = pw.create_shared_device(&vk)?;
        let chroma = if chroma444 { Chroma::Subsampling444 } else { Chroma::Subsampling420 };
        let dec = pw.decoder_new(dev, w, h, chroma)?;

        let (cw, ch) = if chroma444 { (w, h) } else { ((w + 1) / 2, (h + 1) / 2) };
        let dims = [(w, h), (cw, ch), (cw, ch)];
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let mut planes = Vec::new();
        let mut views = [vk::ImageView::null(); 3];
        for (i, &(pw_, ph_)) in dims.iter().enumerate() {
            let im = vk
                .create_image(vk::Format::R8_UNORM, pw_, ph_, usage, vk::ImageTiling::OPTIMAL)
                .context("lossy: decode plane")?;
            vk.transition(
                im.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            )?;
            views[i] = vk.make_view(
                im.image,
                vk::Format::R8_UNORM,
                vk::ImageAspectFlags::COLOR,
                vk::ComponentSwizzle::IDENTITY,
                vk::ImageLayout::GENERAL,
            )?;
            planes.push(im);
        }
        log::info!(
            "[lossy] client decoder ready {}x{} chroma {} (Vulkan planes, no dma-buf export)",
            w,
            h,
            if chroma444 { "4:4:4" } else { "4:2:0" },
        );
        Ok(Self {
            vk,
            pw,
            dev,
            dec,
            w,
            h,
            chroma444,
            planes,
            views,
            layout: vk::ImageLayout::GENERAL,
            pending_sync: None,
            pending_semaphore: vk::Semaphore::null(),
            in_flight_sync: None,
            readback: None,
        })
    }

    fn plane_view(&self, i: usize) -> PyrowaveImageView {
        let im = &self.planes[i];
        PyrowaveImageView {
            image: im.image,
            width: im.width,
            height: im.height,
            image_format: vk::Format::R8_UNORM,
            view_format: vk::Format::R8_UNORM,
            mip_level: 0,
            layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            swizzle: vk::ComponentSwizzle::IDENTITY,
            layout: vk::ImageLayout::GENERAL,
        }
    }

    /// The three R8 plane views for the Vulkan presenter's YUV pipeline.
    pub fn plane_views(&self) -> [vk::ImageView; 3] {
        self.views
    }

    /// Record the barrier that makes PyroWave's GPU writes visible to the
    /// presenter fragment shader. The planes stay in GENERAL, so this is not a
    /// layout transition and does not need a separate queue submission.
    pub fn record_sample_barrier(&self, cb: vk::CommandBuffer) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        for im in &self.planes {
            let b = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: im.image,
                subresource_range: range,
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };
            unsafe {
                self.vk.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[b],
                );
            }
        }
    }

    fn prepare_decode(&mut self, coded: &[u8]) -> Result<()> {
        if self.layout != vk::ImageLayout::GENERAL {
            for im in &self.planes {
                self.vk.transition(
                    im.image,
                    self.layout,
                    vk::ImageLayout::GENERAL,
                    vk::ImageAspectFlags::COLOR,
                )?;
            }
            self.layout = vk::ImageLayout::GENERAL;
        }
        if coded.len() < 4 {
            bail!("lossy: short frame");
        }
        let n = u32::from_le_bytes(coded[0..4].try_into().unwrap()) as usize;
        if coded.len() < 4 + 4 * n {
            bail!("lossy: short packet table");
        }
        let mut off = 4 + 4 * n;
        unsafe { (self.pw.decoder_clear)(self.dec) };
        for i in 0..n {
            let sz = u32::from_le_bytes(coded[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
            if off + sz > coded.len() {
                bail!("lossy: packet {i} overruns frame");
            }
            self.pw.push(self.dec, &coded[off..off + sz])?;
            off += sz;
        }
        Ok(())
    }

    /// Decode synchronously (headless stats/dump).
    pub fn decode(&mut self, coded: &[u8]) -> Result<()> {
        self.prepare_decode(coded)?;
        let bufs = PyrowaveGpuBuffers {
            planes: [self.plane_view(0), self.plane_view(1), self.plane_view(2)],
        };
        self.pw.decode_gpu_synced(self.dev, self.dec, &bufs)?;
        Ok(())
    }

    /// Queue a GPU decode and return without waiting. The presenter adds the
    /// returned timeline semaphore to its own render submit, so the decode
    /// pipeline and the YUV draw share one GPU wait instead of a CPU wait.
    pub fn decode_async(&mut self, coded: &[u8]) -> Result<()> {
        if self.pending_sync.is_some() {
            bail!("lossy: previous async decode was not submitted");
        }
        self.prepare_decode(coded)?;
        let bufs = PyrowaveGpuBuffers {
            planes: [self.plane_view(0), self.plane_view(1), self.plane_view(2)],
        };
        let sync = self.pw.decode_gpu_async(self.dev, self.dec, &bufs)?;
        self.pending_semaphore = self.pw.sync_semaphore(sync);
        self.pending_sync = Some(sync);
        Ok(())
    }

    pub fn pending_semaphore(&self) -> Option<vk::Semaphore> {
        if self.pending_semaphore == vk::Semaphore::null() {
            None
        } else {
            Some(self.pending_semaphore)
        }
    }

    pub fn mark_submitted(&mut self) {
        if let Some(sync) = self.pending_sync.take() {
            self.in_flight_sync = Some(sync);
            self.pending_semaphore = vk::Semaphore::null();
        }
    }

    /// Called after the render fence that waited on the decode semaphore.
    pub fn release_in_flight(&mut self) {
        if let Some(sync) = self.in_flight_sync.take() {
            self.pw.sync_destroy(sync);
        }
    }

    /// Headless/`--dump` only: download the planes and convert to BGRA on the
    /// CPU so the content can be inspected. Never used when presenting.
    pub fn read_rgba(&mut self) -> Result<&[u8]> {
        let (w, h) = (self.w as usize, self.h as usize);
        let (cw, ch) = if self.chroma444 {
            (w, h)
        } else {
            ((w + 1) / 2, (h + 1) / 2)
        };
        if self.readback.is_none() {
            self.readback = Some(LossyReadback {
                ybuf: self.vk.host_buffer((w as u64) * (h as u64))?,
                ubuf: self.vk.host_buffer((cw as u64) * (ch as u64))?,
                vbuf: self.vk.host_buffer((cw as u64) * (ch as u64))?,
                bgra: vec![0u8; w * h * 4],
            });
        }
        let old = self.layout;
        let rb = self.readback.as_mut().unwrap();
        self.vk.download_r8(self.planes[0].image, &rb.ybuf, self.w, self.h, old)?;
        self.vk.download_r8(self.planes[1].image, &rb.ubuf, cw as u32, ch as u32, old)?;
        self.vk.download_r8(self.planes[2].image, &rb.vbuf, cw as u32, ch as u32, old)?;
        self.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        let yp = rb.ybuf.as_slice();
        let up = rb.ubuf.as_slice();
        let vp = rb.vbuf.as_slice();
        let out = &mut rb.bgra;
        for j in 0..h {
            for i in 0..w {
                let y = yp[j * w + i] as f32;
                let (ci, cj) = if self.chroma444 { (i, j) } else { (i / 2, j / 2) };
                let u = up[cj * cw + ci] as f32 - 128.0;
                let v = vp[cj * cw + ci] as f32 - 128.0;
                // BT.709 full-range
                let r = (y + 1.5748 * v).clamp(0.0, 255.0) as u8;
                let g = (y - 0.1873 * u - 0.4681 * v).clamp(0.0, 255.0) as u8;
                let b = (y + 1.8556 * u).clamp(0.0, 255.0) as u8;
                let o = (j * w + i) * 4;
                out[o] = b;
                out[o + 1] = g;
                out[o + 2] = r;
                out[o + 3] = 255;
            }
        }
        Ok(out)
    }
}

impl Drop for LossyDecoder {
    fn drop(&mut self) {
        unsafe {
            // No decode/draw may still touch the planes or decoder while we
            // tear down the PyroWave decoder and its shared device.
            let _ = self.vk.device.device_wait_idle();
            if let Some(sync) = self.pending_sync.take() {
                let _ = (self.pw.sync_object_cpu_wait)(sync, 1, 1_000_000_000);
                self.pw.sync_destroy(sync);
            }
            if let Some(sync) = self.in_flight_sync.take() {
                self.pw.sync_destroy(sync);
            }
            // The decoder references the plane views; destroy it first, then
            // the images it decoded into, then the PyroWave device itself.
            (self.pw.decoder_destroy)(self.dec);
            for v in self.views {
                if v != vk::ImageView::null() {
                    self.vk.device.destroy_image_view(v, None);
                }
            }
            for im in self.planes.drain(..) {
                self.vk.destroy_image(im);
            }
            (self.pw.device_destroy)(self.dev);
            if let Some(rb) = self.readback.take() {
                for b in [&rb.ybuf, &rb.ubuf, &rb.vbuf] {
                    self.vk.device.unmap_memory(b.memory);
                    self.vk.device.destroy_buffer(b.buffer, None);
                    self.vk.device.free_memory(b.memory, None);
                }
            }
        }
    }
}
