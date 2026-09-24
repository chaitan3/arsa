//! Client-side PyroWave decoder for `--codec lossy`.
//!
//! The server sends a framed frame: `[u32 n][u32 sizes...][packet bytes]`.
//! We push each packet, decode on the GPU into Y/U/V plane images, and hand
//! those planes to the presenter as **exported dma-bufs** — the GL presenter
//! imports them and does the YUV->RGB conversion on the GPU (see
//! `eglwin::import_yuv`/`present_yuv`). No full-frame readback, no CPU colour
//! conversion on the streaming path.
//!
//! (`read_rgba` is the headless/`--dump` debug path: it downloads and converts
//! on the CPU so the content can be inspected; it is never used when
//! presenting.)

use anyhow::{bail, Context, Result};
use rdma_display_vkcore::ash::vk;
use rdma_display_vkcore::pyrowave::{
    Chroma, Pyrowave, PyrowaveDecoder, PyrowaveDevice, PyrowaveGpuBuffers, PyrowaveImageView,
};
use rdma_display_vkcore::{HostBuffer, Image, VkCore};

pub struct LossyDecoder {
    vk: VkCore,
    pw: Pyrowave,
    dev: PyrowaveDevice,
    dec: PyrowaveDecoder,
    w: u32,
    h: u32,
    chroma444: bool,
    /// linear R8 decode planes, exported as dma-bufs for GL
    planes: Vec<Image>,
    plane_fds: [i32; 3],
    plane_pitch: [u32; 3],
    plane_w: [u32; 3],
    plane_h: [u32; 3],
    // headless/--dump debug readback
    ybuf: HostBuffer,
    ubuf: HostBuffer,
    vbuf: HostBuffer,
    bgra: Vec<u8>,
}

impl LossyDecoder {
    pub fn new(w: u32, h: u32, chroma: u8) -> Result<Self> {
        let chroma444 = chroma == 1;
        let vk = VkCore::new().context("lossy client: Vulkan core")?;
        let pw = Pyrowave::load(std::env::var("PYROWAVE_LIB").ok().as_deref())?;
        let dev = pw.create_shared_device(&vk)?;
        let chroma = if chroma444 { Chroma::Subsampling444 } else { Chroma::Subsampling420 };
        let dec = pw.decoder_new(dev, w, h, chroma)?;

        let (cw, ch) = if chroma444 { (w, h) } else { ((w + 1) / 2, (h + 1) / 2) };
        let dims = [(w, h), (cw, ch), (cw, ch)];
        let mut planes = Vec::new();
        let mut plane_fds = [0i32; 3];
        let mut plane_pitch = [0u32; 3];
        let mut plane_w = [0u32; 3];
        let mut plane_h = [0u32; 3];
        for &(pw_, ph_) in &dims {
            let (im, fd, pitch) = vk
                .create_exportable_image(
                    vk::Format::R8_UNORM,
                    pw_,
                    ph_,
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .context("lossy: exportable decode plane (does this client's Vulkan export dma-bufs?)")?;
            let i = planes.len();
            plane_fds[i] = fd;
            plane_pitch[i] = pitch;
            plane_w[i] = pw_;
            plane_h[i] = ph_;
            planes.push(im);
        }
        let ybuf = vk.host_buffer((w as u64) * (h as u64))?;
        let ubuf = vk.host_buffer((cw as u64) * (ch as u64))?;
        let vbuf = vk.host_buffer((cw as u64) * (ch as u64))?;
        eprintln!(
            "[lossy] client decoder ready {}x{} chroma {} (decode planes exported for GPU YUV->RGB)",
            w,
            h,
            if chroma444 { "4:4:4" } else { "4:2:0" }
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
            plane_fds,
            plane_pitch,
            plane_w,
            plane_h,
            ybuf,
            ubuf,
            vbuf,
            bgra: vec![0u8; (w as usize) * (h as usize) * 4],
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

    /// The three decode planes as `(fd, width, height, row_pitch)` for the GL
    /// presenter to import once (`eglwin::import_yuv`).
    pub fn plane_descs(&self) -> [(i32, u32, u32, u32); 3] {
        [
            (self.plane_fds[0], self.plane_w[0], self.plane_h[0], self.plane_pitch[0]),
            (self.plane_fds[1], self.plane_w[1], self.plane_h[1], self.plane_pitch[1]),
            (self.plane_fds[2], self.plane_w[2], self.plane_h[2], self.plane_pitch[2]),
        ]
    }

    /// Decode on the GPU into the exported planes. No host round trip.
    pub fn decode(&mut self, coded: &[u8]) -> Result<()> {
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
        let bufs = PyrowaveGpuBuffers {
            planes: [self.plane_view(0), self.plane_view(1), self.plane_view(2)],
        };
        self.pw.decode_gpu_synced(self.dev, self.dec, &bufs)?;
        Ok(())
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
        self.vk
            .download_r8(self.planes[0].image, &self.ybuf, self.w, self.h, vk::ImageLayout::GENERAL)?;
        self.vk
            .download_r8(self.planes[1].image, &self.ubuf, cw as u32, ch as u32, vk::ImageLayout::GENERAL)?;
        self.vk
            .download_r8(self.planes[2].image, &self.vbuf, cw as u32, ch as u32, vk::ImageLayout::GENERAL)?;
        let yp = self.ybuf.as_slice();
        let up = self.ubuf.as_slice();
        let vp = self.vbuf.as_slice();
        let out = &mut self.bgra;
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
