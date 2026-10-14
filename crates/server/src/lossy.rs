//! Lossy (PyroWave) frame materializer.
//!
//! Imports the KMS scanout dma-buf into Vulkan (no GL, no readback), encodes it
//! with PyroWave's GPU pipeline (`encode_gpu_scaled_synchronous`, which also
//! does RGB->YCbCr + scale), and packetizes the bitstream straight into the
//! RDMA staging slot. `coded_len()` reports the length for the RDMA post.

use std::os::fd::{AsRawFd, OwnedFd};

use anyhow::{bail, Context, Result};
use rdma_display_capture::materializer::FrameMaterializer;
use rdma_display_capture::{CaptureEvent, DmabufScanout, FrameFormat};
use rdma_display_proto::{MatKind, MatTimes};
use rdma_display_vkcore::ash::vk;
use rdma_display_vkcore::pyrowave::{
    Chroma, Pyrowave, PyrowaveDevice, PyrowaveEncoder, PyrowaveImageView, PyrowaveRateControl,
    PyrowaveScaledEncodeInfo,
};
use rdma_display_vkcore::{Image, VkCore};

const PACKET_BOUNDARY: usize = 4096;
const PACKET_PADDING: usize = 2000;

/// Map a DRM fourcc to the matching Vulkan format for the scanout content.
fn fourcc_to_vk(fourcc: u32) -> Option<vk::Format> {
    Some(match fourcc {
        // B,G,R,[A/X] in memory
        0x3432_5258 | 0x3432_5241 | 0x3432_4142 | 0x3432_5842 => vk::Format::B8G8R8A8_UNORM,
        // R,G,B,[A/X] in memory
        0x3432_4258 | 0x3432_4241 | 0x3432_4152 | 0x3432_5852 => vk::Format::R8G8B8A8_UNORM,
        // 10-bit 2101010/1010102
        0x3033_5258 | 0x3033_5241 => vk::Format::A2R10G10B10_UNORM_PACK32,
        0x3033_4258 | 0x3033_4241 => vk::Format::A2B10G10R10_UNORM_PACK32,
        _ => return None,
    })
}

/// Identifies an imported scanout buffer: the DRM fb id plus the properties
/// that determine the import layout. The compositor cycles a handful of
/// buffers, so keying the import on this turns the per-frame Vulkan import
/// (and its `device_wait_idle`) into a lookup — mirrors the lossless codec's
/// `SrcImport` cache in `codec-gpu`.
type ImportKey = (u32, u32, u32, u64, u32, u32, u32);

fn import_key(fmt: &FrameFormat, d: &DmabufScanout) -> ImportKey {
    (
        d.fb_id,
        fmt.width,
        fmt.height,
        fmt.modifier,
        d.strides.first().copied().unwrap_or(fmt.stride),
        d.offsets.first().copied().unwrap_or(0),
        fmt.fourcc,
    )
}

/// Maximum number of imported scanout buffers kept alive (the compositor
/// swaps between a handful; 4 covers triple/double buffering plus a spare).
const IMPORT_CACHE: usize = 4;

pub struct LossyMaterializer {
    budget: usize,
    chroma444: bool,
    vk: VkCore,
    pw: Pyrowave,
    dev: PyrowaveDevice,
    enc: Option<PyrowaveEncoder>,
    enc_geom: Option<(u32, u32, u32)>,
    fmt: Option<FrameFormat>,
    fd: Option<OwnedFd>,
    modifier: u64,
    stride: u32,
    offset: u32,
    /// currently selected import (what `copy_frame` samples)
    cur: Option<(ImportKey, Image)>,
    /// other recently imported scanout buffers, reused when the compositor
    /// flips back to one (key == `cur` is always excluded)
    imports: Vec<(ImportKey, Image)>,
    dirty: bool,
    last_len: usize,
    last_mat: MatTimes,
    /// encode scratch (packet-based bitstream before framing)
    temp: Vec<u8>,
}

unsafe impl Send for LossyMaterializer {}

impl LossyMaterializer {
    pub fn new(budget: usize, chroma444: bool) -> Result<Self> {
        let vk = VkCore::new().context("Vulkan core for lossy capture")?;
        let pw = Pyrowave::load(std::env::var("PYROWAVE_LIB").ok().as_deref())?;
        let dev = pw.create_shared_device(&vk)?;
        let chroma = if chroma444 { Chroma::Subsampling444 } else { Chroma::Subsampling420 };
        log::info!(
            "[lossy] PyroWave encoder ready (budget {} KiB, chroma {})",
            budget / 1024,
            if chroma444 { "4:4:4" } else { "4:2:0" }
        );
        let _ = chroma;
        Ok(Self {
            budget,
            chroma444,
            vk,
            pw,
            dev,
            enc: None,
            enc_geom: None,
            fmt: None,
            fd: None,
            modifier: 0,
            stride: 0,
            offset: 0,
            cur: None,
            imports: Vec::new(),
            dirty: false,
            last_len: 0,
            last_mat: MatTimes::ZERO,
            temp: vec![0u8; budget + (1 << 20)],
        })
    }

    fn ensure_encoder(&mut self, w: u32, h: u32) -> Result<()> {
        let chroma = if self.chroma444 { Chroma::Subsampling444 } else { Chroma::Subsampling420 };
        let key = (w, h, chroma as u32);
        if self.enc_geom == Some(key) {
            return Ok(());
        }
        if let Some(e) = self.enc.take() {
            unsafe { (self.pw.encoder_destroy)(e) };
        }
        self.enc = Some(self.pw.encoder_new(self.dev, w, h, chroma)?);
        self.enc_geom = Some(key);
        Ok(())
    }

    /// A fresh import: record the plane geometry for `d`, then import.
    fn import_scanout(&mut self, fmt: &FrameFormat, d: &DmabufScanout) -> Result<Image> {
        self.fd = Some(d.fds[0].try_clone().context("dup scanout fd")?);
        self.modifier = fmt.modifier;
        self.stride = d.strides[0];
        self.offset = d.offsets[0];
        self.import(fmt)
    }

    fn import(&mut self, fmt: &FrameFormat) -> Result<Image> {
        let fd = self.fd.as_ref().context("no dmabuf fd")?;
        let vkfmt = fourcc_to_vk(fmt.fourcc)
            .with_context(|| format!("lossy: unsupported fourcc {:#010x} '{}'", fmt.fourcc, rdma_display_capture::fourcc_name(fmt.fourcc)))?;
        let img = self.vk.import_dmabuf(
            fd.as_raw_fd(),
            fmt.width,
            fmt.height,
            vkfmt,
            self.modifier,
            self.offset,
            self.stride,
        )?;
        self.vk.transition(
            img.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        )?;
        Ok(img)
    }
}

impl FrameMaterializer for LossyMaterializer {
    fn name(&self) -> &'static str {
        "lossy-pyrowave"
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn prepare(&mut self, fmt: &FrameFormat, dmabuf: Option<&DmabufScanout>) -> Result<()> {
        let d = dmabuf.context("lossy materializer requires a dmabuf scanout (use --source drm or qemu gl=on)")?;
        if d.fds.len() != 1 || fmt.num_planes != 1 {
            bail!("lossy: multi-plane scanout not supported (planes={})", d.fds.len());
        }
        self.ensure_encoder(fmt.width, fmt.height)?;
        // Reuse a cached import when the compositor flips back to a buffer we
        // have already imported; otherwise import it once and cache it. The
        // import (and its `device_wait_idle`) used to run every frame.
        let key = import_key(fmt, d);
        if d.fb_id == 0 {
            // no stable buffer id (e.g. QEMU D-Bus scanouts): re-import every
            // frame rather than risk reusing the previous image for a new buf
            let img = self.import_scanout(fmt, d)?;
            if let Some((_, old)) = self.cur.replace((key, img)) {
                self.vk.device_wait_idle()?;
                self.vk.destroy_image(old);
            }
        } else if self.cur.as_ref().map(|(k, _)| *k) != Some(key) {
            if let Some(pos) = self.imports.iter().position(|(k, _)| *k == key) {
                let hit = self.imports.remove(pos);
                if let Some(old) = self.cur.replace(hit) {
                    self.imports.push(old);
                }
            } else {
                let img = self.import_scanout(fmt, d)?;
                if let Some(old) = self.cur.replace((key, img)) {
                    self.imports.push(old);
                }
            }
            // bound the cache. Destroying is safe without a wait: the encode
            // is synchronous and has already returned, so nothing references
            // the evicted image.
            while self.imports.len() > IMPORT_CACHE {
                let (_, e) = self.imports.remove(0);
                self.vk.destroy_image(e);
            }
        }
        self.fmt = Some(fmt.clone());
        self.dirty = true;
        Ok(())
    }

    fn copy_frame(&mut self, out: &mut [u8]) -> Result<(bool, MatTimes)> {
        if !self.dirty {
            return Ok((false, MatTimes::ZERO));
        }
        let fmt = self.fmt.clone().context("lossy: no scanout")?;
        let img = &self.cur.as_ref().context("lossy: no imported image")?.1;
        let enc = self.enc.context("lossy: no encoder")?;
        let vkfmt = fourcc_to_vk(fmt.fourcc).context("lossy: unsupported fourcc")?;
        let view = PyrowaveImageView {
            image: img.image,
            width: fmt.width,
            height: fmt.height,
            image_format: vkfmt,
            view_format: vkfmt,
            mip_level: 0,
            layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            swizzle: vk::ComponentSwizzle::IDENTITY,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let info = PyrowaveScaledEncodeInfo {
            view,
            input_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            output_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            intermediate_plane_format: vk::Format::R8_UNORM,
            ycbcr_chroma_midpoint: 0.5,
            force_linear_filtering: false,
            skip_dither: false,
            crop_rect: std::ptr::null(),
        };
        let rate = PyrowaveRateControl {
            maximum_bitstream_size: self.budget,
        };
        let t = std::time::Instant::now();
        // encode into scratch, then frame as [u32 n][u32 sizes...][packet bytes]
        // so the client can push each packet without guessing boundaries.
        let frame = self
            .pw
            .encode_to(enc, &info, &rate, PACKET_BOUNDARY, PACKET_PADDING, &mut self.temp)?;
        let encode_us = t.elapsed().as_micros() as u64;
        let n = frame.packets.len();
        let hdr = 4 + 4 * n;
        anyhow::ensure!(hdr + frame.len <= out.len(), "lossy: slot too small ({} + {} > {})", hdr, frame.len, out.len());
        let t = std::time::Instant::now();
        out[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        let mut off = hdr;
        for (i, (po, ps)) in frame.packets.iter().enumerate() {
            out[4 + 4 * i..8 + 4 * i].copy_from_slice(&(*ps as u32).to_le_bytes());
            out[off..off + ps].copy_from_slice(&self.temp[*po..*po + *ps]);
            off += ps;
        }
        let packetize_us = t.elapsed().as_micros() as u64;
        self.last_len = off;
        self.last_mat = MatTimes::new(
            MatKind::Lossy,
            [encode_us, packetize_us, 0, 0, 0, 0],
        );
        self.dirty = false;
        Ok((true, self.last_mat))
    }

    fn coded_len(&self) -> Option<usize> {
        Some(self.last_len)
    }

    fn handle_event(&mut self, ev: CaptureEvent) -> Result<bool> {
        match ev {
            CaptureEvent::Scanout(..) | CaptureEvent::DmabufFrame(..) => {
                self.dirty = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn effective_format(&self) -> Option<FrameFormat> {
        self.fmt.clone()
    }
}
