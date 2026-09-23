//! End-to-end PyroWave round trip on the CPU (bring-up): encode a synthetic
//! 720p YUV frame to a bitstream inside a byte budget, decode it back, and
//! report coded size, encode/decode time and PSNR. Validates the C bindings
//! and the codec without needing the dma-buf/Vulkan interop path.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{bail, Result};
use rdma_display_vkcore::pyrowave::*;

const W: i32 = 1280;
const H: i32 = 720;

fn check(r: PyrowaveResult, what: &str) -> Result<()> {
    if r != PyrowaveResult::Success {
        bail!("{what}: {r:?}");
    }
    Ok(())
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut sse = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        sse += d * d;
    }
    let mse = sse / a.len() as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn main() -> Result<()> {
    let pw = Pyrowave::load(std::env::var("PYROWAVE_LIB").ok().as_deref())?;

    let mut dev: PyrowaveDevice = std::ptr::null_mut();
    check(unsafe { (pw.create_default_device)(&mut dev) }, "create_default_device")?;

    let mut enc: PyrowaveEncoder = std::ptr::null_mut();
    let einfo = PyrowaveEncoderCreateInfo {
        device: dev,
        width: W,
        height: H,
        chroma: Chroma::Subsampling420,
    };
    check(unsafe { (pw.encoder_create)(&einfo, &mut enc) }, "encoder_create")?;

    let mut dec: PyrowaveDecoder = std::ptr::null_mut();
    let dinfo = PyrowaveDecoderCreateInfo {
        device: dev,
        width: W,
        height: H,
        chroma: Chroma::Subsampling420,
        fragment_path: false,
    };
    check(unsafe { (pw.decoder_create)(&dinfo, &mut dec) }, "decoder_create")?;

    // synthetic frame: gradient background + a bright block (desktop-ish)
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (((i * 255 / w) as u32 + (j * 255 / h) as u32) / 2) as u8;
        }
    }
    for j in (h / 4)..(h / 4 + h / 6) {
        for i in (w / 3)..(w / 3 + w / 5) {
            y[j * w + i] = 235;
        }
    }
    let mut u = vec![128u8; cw * ch];
    let mut v = vec![128u8; cw * ch];
    let in_buf = PyrowaveCpuBuffer {
        data: [y.as_mut_ptr() as *mut c_void, u.as_mut_ptr() as *mut c_void, v.as_mut_ptr() as *mut c_void],
        row_stride_in_bytes: [w, cw, cw],
        plane_size_in_bytes: [w * h, cw * ch, cw * ch],
        width: W,
        height: H,
        format: CpuBufferFormat::Yuv420p,
    };

    const BUDGET: usize = 300 * 1024;
    let rate = PyrowaveRateControl { maximum_bitstream_size: BUDGET };
    let mut bitstream = vec![0u8; BUDGET + 4096];

    let t = Instant::now();
    let frame = pw.encode_cpu(enc, &in_buf, &rate, 4 * 1024, 2000, &mut bitstream)?;
    let enc_ms = t.elapsed().as_secs_f64() * 1e3;

    // decode: push every packet, then decode into fresh planes
    let mut oy = vec![0u8; w * h];
    let mut ou = vec![0u8; cw * ch];
    let mut ov = vec![0u8; cw * ch];
    let out_buf = PyrowaveCpuBuffer {
        data: [oy.as_mut_ptr() as *mut c_void, ou.as_mut_ptr() as *mut c_void, ov.as_mut_ptr() as *mut c_void],
        row_stride_in_bytes: [w, cw, cw],
        plane_size_in_bytes: [w * h, cw * ch, cw * ch],
        width: W,
        height: H,
        format: CpuBufferFormat::Yuv420p,
    };
    let t = Instant::now();
    for (off, size) in &frame.packets {
        pw.push(dec, &bitstream[*off..*off + *size])?;
    }
    let ready = unsafe { (pw.decoder_decode_is_ready)(dec, false) };
    check(
        unsafe { (pw.decoder_decode_cpu_buffer_synchronous)(dec, &out_buf) },
        "decode_cpu_buffer",
    )?;
    let dec_ms = t.elapsed().as_secs_f64() * 1e3;

    // chroma is 420 (plane data only present at 1/4 res), luma is the signal
    let raw = w * h * 4; // 4 bytes/pixel RGBA equivalent
    println!(
        "[lossy-smoke] {W}x{H} 420 | budget {} KiB | coded {} B ({} packets) | {}x vs raw RGBA | enc {:.2} ms | dec {:.2} ms | ready={ready} | luma PSNR {:.1} dB",
        BUDGET / 1024,
        frame.len,
        frame.packets.len(),
        raw / frame.len.max(1),
        enc_ms,
        dec_ms,
        psnr(&y, &oy),
    );

    unsafe {
        (pw.decoder_destroy)(dec);
        (pw.encoder_destroy)(enc);
        (pw.device_destroy)(dev);
    }
    Ok(())
}
