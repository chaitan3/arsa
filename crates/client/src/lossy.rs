//! Client-side PyroWave decoder for `--codec lossy`.
//!
//! The server sends a framed frame: `[u32 n][u32 sizes...][packet bytes]`.
//! We push each packet, decode on the GPU into Y/U/V plane images, read the
//! planes back and convert YCbCr->BGRA for the existing GL presenter.
//!
//! (The readback + CPU colour conversion is the bring-up path; a later step
//! exports the decoded planes as dma-bufs and converts on the GPU.)

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
    planes: Vec<Image>,
    ybuf: HostBuffer,
    ubuf: HostBuffer,
    vbuf: HostBuffer,
    bgra: Vec<u8>,
    frames: u64,
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
        let mut planes = Vec::new();
        for i in 0..3 {
            let (pw_, ph_) = if i == 0 { (w, h) } else { (cw, ch) };
            let im = vk.create_image(
                vk::Format::R8_UNORM,
                pw_,
                ph_,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                vk::ImageTiling::OPTIMAL,
            )?;
            vk.transition(
                im.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            )?;
            planes.push(im);
        }
        let ybuf = vk.host_buffer((w as u64) * (h as u64))?;
        let ubuf = vk.host_buffer((cw as u64) * (ch as u64))?;
        let vbuf = vk.host_buffer((cw as u64) * (ch as u64))?;
        eprintln!(
            "[lossy] client decoder ready {}x{} chroma {}",
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
            ybuf,
            ubuf,
            vbuf,
            bgra: vec![0u8; (w as usize) * (h as usize) * 4],
            frames: 0,
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

    /// Decode a framed frame and return tightly packed BGRA (wire order).
    pub fn decode(&mut self, coded: &[u8]) -> Result<&[u8]> {
        if coded.len() < 4 {
            bail!("lossy: short frame");
        }
        let n = u32::from_le_bytes(coded[0..4].try_into().unwrap()) as usize;
        if coded.len() < 4 + 4 * n {
            bail!("lossy: short packet table");
        }
        let mut off = 4 + 4 * n;
        let t_push = std::time::Instant::now();
        unsafe { (self.pw.decoder_clear)(self.dec) };
        for i in 0..n {
            let sz = u32::from_le_bytes(coded[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
            if off + sz > coded.len() {
                bail!("lossy: packet {i} overruns frame");
            }
            self.pw.push(self.dec, &coded[off..off + sz])?;
            off += sz;
        }
        let push_us = t_push.elapsed().as_micros() as u64;
        let bufs = PyrowaveGpuBuffers {
            planes: [self.plane_view(0), self.plane_view(1), self.plane_view(2)],
        };
        let t_dec = std::time::Instant::now();
        self.pw.decode_gpu_synced(self.dev, self.dec, &bufs)?;
        let dec_us = t_dec.elapsed().as_micros() as u64;

        let (w, h) = (self.w as usize, self.h as usize);
        let (cw, ch) = if self.chroma444 {
            (w, h)
        } else {
            ((w + 1) / 2, (h + 1) / 2)
        };
        let t_dl = std::time::Instant::now();
        self.vk
            .download_r8(self.planes[0].image, &self.ybuf, self.w, self.h, vk::ImageLayout::GENERAL)?;
        self.vk
            .download_r8(self.planes[1].image, &self.ubuf, cw as u32, ch as u32, vk::ImageLayout::GENERAL)?;
        self.vk
            .download_r8(self.planes[2].image, &self.vbuf, cw as u32, ch as u32, vk::ImageLayout::GENERAL)?;
        let dl_us = t_dl.elapsed().as_micros() as u64;

        let t_cv = std::time::Instant::now();
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
        let cv_us = t_cv.elapsed().as_micros() as u64;
        self.frames += 1;
        if self.frames % 30 == 1 {
            eprintln!(
                "[lossy] decode us/frame: push={push_us} gpu={dec_us} download={dl_us} convert={cv_us} total={}",
                push_us + dec_us + dl_us + cv_us
            );
        }
        Ok(out)
    }
}
