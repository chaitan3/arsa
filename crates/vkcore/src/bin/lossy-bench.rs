//! Warm PyroWave GPU encode/decode benchmark at display resolution. Reuses one
//! encoder/decoder and one uploaded input image across iterations so the
//! numbers are steady-state (the smoke test is a cold single shot).
//!
//! Compares three ways to wait for the decode:
//!  - `decode_gpu_synced`: a fresh timeline-semaphore sync object per call
//!    (what the client does today)
//!  - reuse: one semaphore, monotonically increasing signal values
//!  - `decode_gpu` + `device_wait_idle`
//! and checks that all three reconstruct the same luma.
//!
//! Env: W, H, CHROMA (0=4:2:0 default, 1=4:4:4), BUDGET, ITERS.

use std::time::Instant;

use anyhow::Result;
use ash::vk;
use rdma_display_vkcore::{pyrowave::*, VkCore};

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

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
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
    let w = env_u32("W", 3840);
    let h = env_u32("H", 2160);
    let chroma444 = env_u32("CHROMA", 0) == 1;
    let budget = env_u32("BUDGET", 1_000_000) as usize;
    let iters = env_u32("ITERS", 30) as usize;

    let vk = VkCore::new()?;
    let pw = Pyrowave::load(std::env::var("PYROWAVE_LIB").ok().as_deref())?;
    let dev = pw.create_shared_device(&vk)?;

    // A desktop-ish frame: gradient wallpaper, solid panels, "text" lines.
    let img = vk.create_image(
        vk::Format::R8G8B8A8_UNORM,
        w,
        h,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageTiling::OPTIMAL,
    )?;
    let mut hb = vk.host_buffer((w as u64) * (h as u64) * 4)?;
    {
        let s = hb.as_mut_slice();
        for j in 0..h as usize {
            for i in 0..w as usize {
                let o = (j * w as usize + i) * 4;
                let a = (i * 255) / w as usize;
                let b = (j * 255) / h as usize;
                s[o] = a as u8;
                s[o + 1] = ((a + b) / 2) as u8;
                s[o + 2] = b as u8;
                s[o + 3] = 255;
            }
        }
        let rect = |s: &mut [u8], x: usize, y: usize, rw: usize, rh: usize, c: u8| {
            for j in y..(y + rh).min(h as usize) {
                for i in x..(x + rw).min(w as usize) {
                    let o = (j * w as usize + i) * 4;
                    s[o] = c;
                    s[o + 1] = c;
                    s[o + 2] = c;
                }
            }
        };
        rect(s, 120, 90, 1500, 1000, 0x28);
        rect(s, 1700, 200, 1400, 1200, 0x22);
        let mut y = 160;
        while y < 1000 {
            let n = 200 + ((y * 37) % 900);
            rect(s, 160, y, n, 14, 0xd0);
            y += 28;
        }
    }
    vk.upload_image(img.image, &hb, w, h)?;

    let chroma = if chroma444 {
        Chroma::Subsampling444
    } else {
        Chroma::Subsampling420
    };
    let enc = pw.encoder_new(dev, w, h, chroma)?;
    let view = PyrowaveImageView {
        image: img.image,
        width: w,
        height: h,
        image_format: vk::Format::R8G8B8A8_UNORM,
        view_format: vk::Format::R8G8B8A8_UNORM,
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
        maximum_bitstream_size: budget,
    };
    let mut bitstream = vec![0u8; budget + 8192];

    // decoder + decode planes
    let dec = pw.decoder_new(dev, w, h, chroma)?;
    let (cw, ch) = if chroma444 {
        (w, h)
    } else {
        ((w + 1) / 2, (h + 1) / 2)
    };
    let dims = [(w, h), (cw, ch), (cw, ch)];
    let export_planes = env_u32("EXPORT", 0) == 1;
    let mut planes = Vec::new();
    for &(pw_, ph_) in &dims {
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::SAMPLED;
        let im = if export_planes {
            // the client's GPU path: linear, dma-buf-exportable planes
            let (im, _fd, _pitch) =
                vk.create_exportable_image(vk::Format::R8_UNORM, pw_, ph_, usage)?;
            im
        } else {
            vk.create_image(
                vk::Format::R8_UNORM,
                pw_,
                ph_,
                usage,
                vk::ImageTiling::OPTIMAL,
            )?
        };
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
    let luma = vk.host_buffer((w as u64) * (h as u64))?;

    let encode_once = |bitstream: &mut [u8]| -> Result<EncodedFrame> {
        pw.encode_to(enc, &info, &rate, 4096, 2000, bitstream)
    };

    // --- variant 1: a fresh sync object per call (client today) --------------
    let mut synced_us = Vec::new();
    let mut enc_us = Vec::new();
    let mut coded_len = 0usize;
    let mut npkts = 0usize;
    for it in 0..iters {
        let t = Instant::now();
        let frame = encode_once(&mut bitstream)?;
        enc_us.push(t.elapsed().as_secs_f64() * 1e3);
        coded_len = frame.len;
        npkts = frame.packets.len();
        unsafe { (pw.decoder_clear)(dec) };
        for (off, size) in &frame.packets {
            pw.push(dec, &bitstream[*off..*off + *size])?;
        }
        let t = Instant::now();
        pw.decode_gpu_synced(dev, dec, &bufs)?;
        synced_us.push(t.elapsed().as_secs_f64() * 1e3);
        if it == 0 {
            synced_us.clear(); // discard cold
            enc_us.clear();
        }
    }
    vk.download_r8(planes[0].image, &luma, w, h, vk::ImageLayout::GENERAL)?;
    let synced_luma = mean_std(luma.as_slice());

    // --- variant 2: reuse one semaphore, increasing values -------------------
    let cinfo = PyrowaveSyncObjectCreateInfo {
        device: dev,
        external_handle: usize::MAX,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
        semaphore_type: vk::SemaphoreType::TIMELINE,
        import_flags: vk::SemaphoreImportFlags::TEMPORARY,
    };
    let mut sync: PyrowaveSyncObject = std::ptr::null_mut();
    let r = unsafe { (pw.sync_object_create)(&cinfo, &mut sync) };
    assert_eq!(r, PyrowaveResult::Success);
    let sem = unsafe { (pw.sync_object_get_semaphore)(sync) };
    let mut reuse_us = Vec::new();
    let mut v = 1u64;
    for it in 0..iters {
        let frame = encode_once(&mut bitstream)?;
        unsafe { (pw.decoder_clear)(dec) };
        for (off, size) in &frame.packets {
            pw.push(dec, &bitstream[*off..*off + *size])?;
        }
        let release = PyrowaveGpuSyncOperation {
            images: std::ptr::null(),
            num_images: 0,
            sync: PyrowaveSyncPoint { semaphore: sem, value: v },
        };
        let t = Instant::now();
        let r = unsafe { (pw.decoder_decode_gpu_buffer)(dec, std::ptr::null(), &release, &bufs) };
        assert_eq!(r, PyrowaveResult::Success);
        let r = unsafe { (pw.sync_object_cpu_wait)(sync, v, 1_000_000_000) };
        assert_eq!(r, PyrowaveResult::Success, "reuse wait v={v} failed");
        reuse_us.push(t.elapsed().as_secs_f64() * 1e3);
        v += 1;
        if it == 0 {
            reuse_us.clear();
        }
    }
    unsafe { (pw.sync_object_destroy)(sync) };
    vk.download_r8(planes[0].image, &luma, w, h, vk::ImageLayout::GENERAL)?;
    let reuse_luma = mean_std(luma.as_slice());

    // --- variant 3: decode_gpu + device_wait_idle ----------------------------
    // WRONG for correctness: with no release sync PyroWave never flushes
    // Granite's frame context, so the decode does not run; device_wait_idle
    // returns immediately and a real consumer (GL importing the planes) reads
    // stale/uninitialized data. Kept only to show why it looks "fast".
    let mut dev_us = Vec::new();
    for it in 0..iters {
        let frame = encode_once(&mut bitstream)?;
        unsafe { (pw.decoder_clear)(dec) };
        for (off, size) in &frame.packets {
            pw.push(dec, &bitstream[*off..*off + *size])?;
        }
        let t = Instant::now();
        pw.decode_gpu(dec, &bufs)?;
        vk.device_wait_idle()?;
        dev_us.push(t.elapsed().as_secs_f64() * 1e3);
        if it == 0 {
            dev_us.clear();
        }
    }
    vk.download_r8(planes[0].image, &luma, w, h, vk::ImageLayout::GENERAL)?;
    let dev_luma = mean_std(luma.as_slice());

    // --- variant 4: queue only, no per-frame wait ----------------------------
    let mut queue_us = Vec::new();
    for it in 0..iters {
        let frame = encode_once(&mut bitstream)?;
        unsafe { (pw.decoder_clear)(dec) };
        for (off, size) in &frame.packets {
            pw.push(dec, &bitstream[*off..*off + *size])?;
        }
        let t = Instant::now();
        pw.decode_gpu(dec, &bufs)?;
        queue_us.push(t.elapsed().as_secs_f64() * 1e3);
        if it == 0 {
            queue_us.clear();
        }
    }
    // one wait at the end: if the luma is right, queued decodes did complete
    vk.device_wait_idle()?;
    vk.download_r8(planes[0].image, &luma, w, h, vk::ImageLayout::GENERAL)?;
    let queue_luma = mean_std(luma.as_slice());

    // --- costs ---------------------------------------------------------------
    let mut sync_us = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        let mut s: PyrowaveSyncObject = std::ptr::null_mut();
        let r = unsafe { (pw.sync_object_create)(&cinfo, &mut s) };
        assert_eq!(r, PyrowaveResult::Success);
        unsafe { (pw.sync_object_destroy)(s) };
        sync_us.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut dst = vec![0u8; coded_len];
    let t = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&bitstream[..coded_len]);
    }
    let cpy_us = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // client CPU-fallback path (`read_rgba`): the 3-plane download alone. This
    // is a floor for the CPU path; if it already exceeds the observed `upl`,
    // the client is on the GPU YUV path, not this one.
    let ub = vk.host_buffer((cw as u64) * (ch as u64))?;
    let vb = vk.host_buffer((cw as u64) * (ch as u64))?;
    let cpu_iters = 5usize;
    let t = Instant::now();
    for _ in 0..cpu_iters {
        vk.download_r8(planes[0].image, &luma, w, h, vk::ImageLayout::GENERAL)?;
        vk.download_r8(planes[1].image, &ub, cw, ch, vk::ImageLayout::GENERAL)?;
        vk.download_r8(planes[2].image, &vb, cw, ch, vk::ImageLayout::GENERAL)?;
    }
    let dl_us = t.elapsed().as_secs_f64() * 1e3 / cpu_iters as f64;

    let raw = w as usize * h as usize * 4;
    println!(
        "[lossy-bench] {}x{} {} budget {} KiB | planes {} | coded {} B ({} pkts, {}x) | iters {}",
        w,
        h,
        if chroma444 { "444" } else { "420" },
        budget / 1024,
        if export_planes { "linear/exported" } else { "optimal" },
        coded_len,
        npkts,
        raw / coded_len.max(1),
        iters
    );
    println!(
        "[lossy-bench] encode p50 {:.2} ms (incl. packetize) | decode wait: fresh-sync p50 {:.2} ms | reused-sync p50 {:.2} ms | device-wait p50 {:.2} ms (WRONG: no flush) | queue-only p50 {:.2} ms",
        median(&mut enc_us),
        median(&mut synced_us),
        median(&mut reuse_us),
        median(&mut dev_us),
        median(&mut queue_us),
    );
    println!(
        "[lossy-bench] fresh sync create+destroy p50 {:.3} ms | bitstream copy {:.3} ms ({} KiB)",
        median(&mut sync_us),
        cpy_us,
        coded_len / 1024
    );
    println!(
        "[lossy-bench] luma mean/std: fresh-sync {:.1}/{:.1} | reused-sync {:.1}/{:.1} | device-wait {:.1}/{:.1} | queue-only {:.1}/{:.1}",
        synced_luma.0, synced_luma.1, reuse_luma.0, reuse_luma.1, dev_luma.0, dev_luma.1, queue_luma.0, queue_luma.1
    );
    println!(
        "[lossy-bench] CPU fallback floor: 3-plane download {:.2} ms/frame (if this already exceeds `upl`, the client is on the GPU path)",
        dl_us,
    );

    unsafe {
        (pw.decoder_destroy)(dec);
        (pw.encoder_destroy)(enc);
        (pw.device_destroy)(dev);
    }
    Ok(())
}
