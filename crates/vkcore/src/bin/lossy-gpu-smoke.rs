//! GPU (Vulkan compute) PyroWave round trip: encode a synthetic RGBA image via
//! the shared device, packetize, decode into YUV planes (synchronously), and
//! read back. Exercises the real GPU path (no dma-buf), so it runs in the
//! sandbox. CPU-decode of the same bitstream is a correctness cross-check.

use std::time::Instant;

use anyhow::Result;
use ash::vk;
use rdma_display_vkcore::{pyrowave::*, VkCore};

const W: u32 = 1280;
const H: u32 = 720;

fn plane_view(im: &rdma_display_vkcore::Image) -> PyrowaveImageView {
    PyrowaveImageView {
        image: im.image,
        width: im.width,
        height: im.height,
        image_format: im.format,
        view_format: im.format,
        mip_level: 0,
        layer: 0,
        aspect: vk::ImageAspectFlags::COLOR,
        swizzle: vk::ComponentSwizzle::IDENTITY,
        layout: vk::ImageLayout::GENERAL,
    }
}

fn mean_std(v: &[u8]) -> (f64, f64) {
    let mean = v.iter().map(|b| *b as f64).sum::<f64>() / v.len() as f64;
    let var = v
        .iter()
        .map(|b| {
            let d = *b as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / v.len() as f64;
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    let vk = VkCore::new()?;
    let pw = Pyrowave::load(std::env::var("PYROWAVE_LIB").ok().as_deref())?;
    let dev = pw.create_shared_device(&vk)?;

    // input RGBA8 image
    let img = vk.create_image(
        vk::Format::R8G8B8A8_UNORM,
        W,
        H,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageTiling::OPTIMAL,
    )?;
    let mut hb = vk.host_buffer((W as u64) * (H as u64) * 4)?;
    {
        let s = hb.as_mut_slice();
        for j in 0..H {
            for i in 0..W {
                let o = (j as usize * W as usize + i as usize) * 4;
                let g = (((i * 255 / W) + (j * 255 / H)) / 2) as u8;
                s[o] = g;
                s[o + 1] = g;
                s[o + 2] = g;
                s[o + 3] = 255;
            }
        }
        for j in (H / 4)..(H / 4 + H / 6) {
            for i in (W / 3)..(W / 3 + W / 5) {
                let o = (j as usize * W as usize + i as usize) * 4;
                s[o] = 235;
                s[o + 1] = 235;
                s[o + 2] = 235;
            }
        }
    }
    vk.upload_image(img.image, &hb, W, H)?;

    // encode (GPU, synchronous)
    let enc = pw.encoder_new(dev, W, H, Chroma::Subsampling444)?;
    let info = PyrowaveScaledEncodeInfo {
        view: PyrowaveImageView {
            image: img.image,
            width: W,
            height: H,
            image_format: vk::Format::R8G8B8A8_UNORM,
            view_format: vk::Format::R8G8B8A8_UNORM,
            mip_level: 0,
            layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            swizzle: vk::ComponentSwizzle::IDENTITY,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        input_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        output_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        intermediate_plane_format: vk::Format::R8_UNORM,
        ycbcr_chroma_midpoint: 0.5,
        force_linear_filtering: false,
        skip_dither: false,
        crop_rect: std::ptr::null(),
    };
    let budget = 400 * 1024usize;
    let rate = PyrowaveRateControl {
        maximum_bitstream_size: budget,
    };
    let mut bitstream = vec![0u8; budget + 8192];
    let t = Instant::now();
    let frame = pw.encode_to(enc, &info, &rate, 4096, 2000, &mut bitstream)?;
    let enc_ms = t.elapsed().as_secs_f64() * 1e3;

    // decode into 3 R8 planes (444)
    let dec = pw.decoder_new(dev, W, H, Chroma::Subsampling444)?;
    let mut planes = Vec::new();
    for _ in 0..3 {
        let im = vk.create_image(
            vk::Format::R8_UNORM,
            W,
            H,
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
    let bufs = PyrowaveGpuBuffers {
        planes: [
            plane_view(&planes[0]),
            plane_view(&planes[1]),
            plane_view(&planes[2]),
        ],
    };

    let t = Instant::now();
    for (off, size) in &frame.packets {
        pw.push(dec, &bitstream[*off..*off + *size])?;
    }
    let ready = unsafe { (pw.decoder_decode_is_ready)(dec, false) };
    pw.decode_gpu_synced(dev, dec, &bufs)?;
    let dec_ms = t.elapsed().as_secs_f64() * 1e3;

    // read back luma
    let ybuf = vk.host_buffer((W as u64) * (H as u64))?;
    vk.download_r8(planes[0].image, &ybuf, W, H, vk::ImageLayout::GENERAL)?;
    let (mean, std) = mean_std(ybuf.as_slice());

    // cross-check via CPU decode of the same bitstream
    let dec2 = pw.decoder_new(dev, W, H, Chroma::Subsampling444)?;
    let (mut oy, mut ou, mut ov) = (
        vec![0u8; (W * H) as usize],
        vec![0u8; (W * H) as usize],
        vec![0u8; (W * H) as usize],
    );
    let cpubuf = PyrowaveCpuBuffer {
        data: [
            oy.as_mut_ptr() as *mut std::ffi::c_void,
            ou.as_mut_ptr() as *mut std::ffi::c_void,
            ov.as_mut_ptr() as *mut std::ffi::c_void,
        ],
        row_stride_in_bytes: [W as usize, W as usize, W as usize],
        plane_size_in_bytes: [(W * H) as usize, (W * H) as usize, (W * H) as usize],
        width: W as i32,
        height: H as i32,
        format: CpuBufferFormat::Yuv444p,
    };
    for (off, size) in &frame.packets {
        pw.push(dec2, &bitstream[*off..*off + *size])?;
    }
    let cpu_ready = unsafe { (pw.decoder_decode_is_ready)(dec2, false) };
    let rc = unsafe { (pw.decoder_decode_cpu_buffer_synchronous)(dec2, &cpubuf) };
    let (cmean, cstd) = mean_std(&oy);

    let raw = W as usize * H as usize * 4;
    println!(
        "[lossy-gpu] {W}x{H} 444 | coded {} B ({} pkts) | {}x vs RGBA | enc {:.2} ms | dec {:.2} ms | ready={ready}",
        frame.len,
        frame.packets.len(),
        raw / frame.len.max(1),
        enc_ms,
        dec_ms,
    );
    println!(
        "[lossy-gpu] gpu-decode luma mean {mean:.1} std {std:.1} | cpu-decode ({rc:?}, ready={cpu_ready}) mean {cmean:.1} std {cstd:.1}"
    );

    unsafe {
        (pw.decoder_destroy)(dec2);
        (pw.decoder_destroy)(dec);
        (pw.encoder_destroy)(enc);
        (pw.device_destroy)(dev);
    }
    Ok(())
}
