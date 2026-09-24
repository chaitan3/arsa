//! codecbench: CPU vs GPU (GLES compute) lossless codec on simulated 4K
//! desktop frames.
//!
//! Two passes: a correctness pass (GPU output must be bit-identical to the
//! source and decodable by the CPU reference) and a clean timing pass with no
//! validation readbacks.

use std::time::Instant;

use anyhow::{bail, Result};
use rdma_display_codec::{crc32, Decoder, Encoder, Geometry, Header};
use rdma_display_codec_gpu::{GpuCodec, GpuVideoDecoder};

const W: u32 = 3840;
const H: u32 = 2160;

fn words_bytes(w: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }
}

fn rect(f: &mut [u32], x: i32, y: i32, rw: i32, rh: i32, color: u32) {
    let (x0, y0) = (x.max(0), y.max(0));
    let (x1, y1) = ((x + rw).min(W as i32), (y + rh).min(H as i32));
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for yy in y0..y1 {
        let row = yy as usize * W as usize;
        for xx in x0..x1 {
            f[row + xx as usize] = color;
        }
    }
}

/// A static-ish desktop: gradient wallpaper, window panels, "text" lines.
fn base_frame() -> Vec<u32> {
    let mut f = vec![0u32; (W * H) as usize];
    for y in 0..H {
        let row = y as usize * W as usize;
        for x in 0..W {
            let a = x * 255 / W;
            let b = y * 255 / H;
            f[row + x as usize] = 0xFF00_0000 | (a << 8) | b;
        }
    }
    rect(&mut f, 120, 90, 1500, 1000, 0xFF20_2028);
    rect(&mut f, 120, 90, 1500, 40, 0xFF3A_3A46);
    rect(&mut f, 1700, 200, 1400, 1200, 0xFF18_1C22);
    rect(&mut f, 1700, 200, 1400, 40, 0xFF3A_3A46);
    let mut y = 160;
    while y < 1000 {
        let len = 200 + ((y * 37) % 900);
        rect(&mut f, 160, y, len, 14, 0xFFC8_C8D0);
        y += 28;
    }
    f
}

fn cursor(f: &mut [u32], x: i32, y: i32) {
    rect(f, x, y, 40, 40, 0xFFFF_FFFF);
    rect(f, x, y, 40, 5, 0xFF00_0000);
    rect(f, x, y, 5, 40, 0xFF00_0000);
}

fn menu(f: &mut [u32], x: i32, y: i32) {
    rect(f, x, y, 460, 400, 0xFF2E_2E38);
    rect(f, x, y, 460, 4, 0xFF60_6070);
    for i in 0..6 {
        rect(f, x + 20, y + 40 + i * 52, 400, 18, 0xFFD0_D0D8);
    }
}

/// 10 frames: a cursor sweeping, a menu opening (frame 4), a hover highlight
/// (frame 6), the menu closing (frame 8).
fn scenario() -> Vec<Vec<u32>> {
    let base = base_frame();
    let cursor_pos = [
        (100, 100),
        (150, 112),
        (200, 124),
        (250, 136),
        (300, 168),
        (330, 180),
        (360, 196),
        (400, 208),
        (520, 240),
        (560, 252),
    ];
    let mut frames = Vec::new();
    for (i, (cx, cy)) in cursor_pos.iter().enumerate() {
        let mut f = base.clone();
        if (4..=7).contains(&i) {
            menu(&mut f, 1500, 700);
            if i == 6 {
                rect(&mut f, 1520, 740 + 3 * 52, 400, 18, 0xFF50_80C0);
            }
        }
        cursor(&mut f, *cx, *cy);
        frames.push(f);
    }
    frames
}

struct PassStats {
    sizes: Vec<usize>,
    counts: Vec<usize>,
    enc_us: Vec<f64>,
    dec_us: Vec<f64>,
}

fn correctness(geom: Geometry, frames: &[Vec<u32>]) -> Result<PassStats> {
    // CPU
    let mut enc = Encoder::new(geom)?;
    enc.set_crc(false); // streaming path omits the CRC
    let mut dec = Decoder::new(geom)?;
    let mut buf = vec![0u8; geom.max_coded_len()];
    let mut cpu = PassStats { sizes: vec![], counts: vec![], enc_us: vec![], dec_us: vec![] };
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        let n = enc.encode(fb, &mut buf)?;
        if dec.decode(&buf[..n])? != fb {
            bail!("CPU round trip mismatch at frame {i}");
        }
        cpu.counts.push(Header::parse(&buf[..n])?.count as usize);
        cpu.sizes.push(n);
        cpu.enc_us.push(0.0);
        cpu.dec_us.push(0.0);
    }

    // GPU: output must match the source and be CPU-decodable
    let mut gpu = GpuCodec::new(geom)?;
    let mut dec_cross = Decoder::new(geom)?;
    let mut g = PassStats { sizes: vec![], counts: vec![], enc_us: vec![], dec_us: vec![] };
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        gpu.upload(fb)?;
        gpu.sync();
        let n = gpu.encode(i == 0)?;
        let count = Header::parse(gpu.coded(n))?.count as usize;
        gpu.decode(count)?;
        if count != cpu.counts[i] || n != cpu.sizes[i] {
            bail!(
                "GPU counts/size differ at frame {i}: gpu {count}/{n} vs cpu {}/{}",
                cpu.counts[i],
                cpu.sizes[i]
            );
        }
        if gpu.read_frame()? != fb {
            bail!("GPU decode mismatch at frame {i}");
        }
        if dec_cross.decode(gpu.coded(n))? != fb {
            bail!("CPU decode of GPU frame mismatch at frame {i}");
        }
        g.counts.push(count);
        g.sizes.push(n);
        g.enc_us.push(0.0);
        g.dec_us.push(0.0);
    }
    eprintln!("[correctness] all {} frames bit-exact CPU↔GPU (both directions)", frames.len());
    Ok(cpu)
}

fn cpu_timing(geom: Geometry, frames: &[Vec<u32>]) -> Result<PassStats> {
    let mut enc = Encoder::new(geom)?;
    enc.set_crc(false); // streaming path omits the CRC
    let mut dec = Decoder::new(geom)?;
    let mut buf = vec![0u8; geom.max_coded_len()];
    let mut s = PassStats { sizes: vec![], counts: vec![], enc_us: vec![], dec_us: vec![] };
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        let t = Instant::now();
        let n = enc.encode(fb, &mut buf)?;
        s.enc_us.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let out = dec.decode(&buf[..n])?;
        s.dec_us.push(t.elapsed().as_secs_f64() * 1e3);
        debug_assert_eq!(out, fb);
        s.counts.push(Header::parse(&buf[..n])?.count as usize);
        s.sizes.push(n);
        let _ = i;
    }
    Ok(s)
}

fn gpu_timing(geom: Geometry, frames: &[Vec<u32>]) -> Result<PassStats> {
    let mut gpu = GpuCodec::new(geom)?;
    let mut s = PassStats { sizes: vec![], counts: vec![], enc_us: vec![], dec_us: vec![] };
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        gpu.upload(fb)?;
        gpu.sync();
        let t = Instant::now();
        let n = gpu.encode(i == 0)?;
        s.enc_us.push(t.elapsed().as_secs_f64() * 1e3);
        let count = Header::parse(gpu.coded(n))?.count as usize;
        let t = Instant::now();
        gpu.decode(count)?;
        s.dec_us.push(t.elapsed().as_secs_f64() * 1e3);
        s.counts.push(count);
        s.sizes.push(n);
    }
    Ok(s)
}

fn decoder_roundtrip(geom: Geometry, frames: &[Vec<u32>]) -> Result<()> {
    let mut gpu = GpuCodec::new(geom)?;
    let dec = GpuVideoDecoder::attach_current(geom)?;
    let tex = dec.create_texture(W, H)?;
    let mut out = vec![0u8; geom.frame_bytes()];
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        gpu.upload(fb)?;
        gpu.sync();
        let n = gpu.encode(i == 0)?;
        gpu.update_reference()?;
        let count = rdma_display_codec::Header::parse(gpu.coded(n))?.count as usize;
        dec.upload(gpu.coded(n))?;
        dec.decode_to_image(tex, count)?;
        dec.read_texture(tex, &mut out)?;
        if out != fb {
            let off = out.iter().zip(fb.iter()).position(|(a, b)| a != b).unwrap();
            bail!(
                "decoder -> texture mismatch at frame {i}, first byte {off}: got {:#04x} want {:#04x} (out[0..8]={:02x?} fb[0..8]={:02x?})",
                out[off],
                fb[off],
                &out[..8],
                &fb[..8]
            );
        }
    }
    eprintln!(
        "[decoder] all {} frames decoded into a texture bit-exact",
        frames.len()
    );
    Ok(())
}

/// Texture-input encode: the same frames uploaded into the RGBA8 source image
/// and encoded through the image-input compute (the production texture-direct
/// path). Must be bit-identical to the SSBO encode and CPU-decodable.
fn texture_encode(geom: Geometry, frames: &[Vec<u32>]) -> Result<()> {
    let mut gpu = GpuCodec::new(geom)?;
    let mut ssbo = GpuCodec::new(geom)?;
    let mut dec = Decoder::new(geom)?;
    let mut tex_us = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        // each GpuCodec owns its own EGL context: make it current before use
        ssbo.activate()?;
        ssbo.upload(fb)?;
        ssbo.sync();
        let n_ssbo = ssbo.encode(i == 0)?;
        let count_ssbo = Header::parse(ssbo.coded(n_ssbo))?.count as usize;

        gpu.activate()?;
        gpu.upload_source_image(fb)?;
        let t0 = Instant::now();
        let n = gpu.encode_tex(i == 0)?;
        tex_us.push(t0.elapsed().as_secs_f64() * 1e3);
        let count = Header::parse(gpu.coded(n))?.count as usize;
        if count != count_ssbo || n != n_ssbo {
            bail!("texture encode differs at frame {i}: {count}/{n} vs {count_ssbo}/{n_ssbo}");
        }
        if dec.decode(gpu.coded(n))? != fb {
            bail!("CPU decode of texture-encoded frame {i} mismatch");
        }
        // the texture encode writes the reference in-pass; no update_reference
        ssbo.activate()?;
        ssbo.update_reference()?;
    }
    eprintln!(
        "[texture] {} frames: image-input encode bit-identical to SSBO encode; full {:.2} ms, steady delta {:.2} ms",
        frames.len(),
        tex_us[0],
        tex_us[1..].iter().sum::<f64>() / (tex_us.len() - 1) as f64
    );
    Ok(())
}

/// Direct external-texture encode (`ENC_EXT_SRC`, no blit) vs the blit+image
/// path, on a synthetic external texture (EGL_KHR_gl_texture_2D_image). Both
/// must match the SSBO encode exactly.
fn external_encode(geom: Geometry, frames: &[Vec<u32>]) -> Result<()> {
    let mut ext = GpuCodec::new(geom)?;
    let mut tex = GpuCodec::new(geom)?;
    let mut ssbo = GpuCodec::new(geom)?;
    let mut dec = Decoder::new(geom)?;
    if !ext.has_ext_encode() {
        eprintln!("[external] external-texture compute encode unavailable; skipped");
        return Ok(());
    }
    let mut ext_us = Vec::new();
    let mut path_us = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);

        ssbo.activate()?;
        ssbo.upload(fb)?;
        ssbo.sync();
        let n_ssbo = ssbo.encode(i == 0)?;
        let cnt_ssbo = Header::parse(ssbo.coded(n_ssbo))?.count as usize;
        ssbo.update_reference()?;

        ext.activate()?;
        ext.set_scanout_test_frame(fb, false)?;
        ext.sync(); // drain the import; time only the encode
        let t = Instant::now();
        let n = ext.encode_ext(i == 0)?;
        ext_us.push(t.elapsed().as_secs_f64() * 1e3);
        let cnt = Header::parse(ext.coded(n))?.count as usize;
        if n != n_ssbo || cnt != cnt_ssbo {
            bail!("external encode differs at frame {i}: {cnt}/{n} vs {cnt_ssbo}/{n_ssbo}");
        }
        if dec.decode(ext.coded(n))? != fb {
            bail!("CPU decode of external-encoded frame {i} mismatch");
        }
        // ext writes its reference in-pass (see ENC_EXT_SRC)
        tex.activate()?;
        tex.set_scanout_test_frame(fb, false)?;
        tex.sync();
        let t = Instant::now();
        tex.blit_source()?;
        let n2 = tex.encode_tex(i == 0)?;
        path_us.push(t.elapsed().as_secs_f64() * 1e3);
        if n2 != n_ssbo {
            bail!("tex encode differs at frame {i}: {n2} vs {n_ssbo}");
        }
    }
    let d = |v: &[f64]| v[1..].iter().sum::<f64>() / (v.len() - 1) as f64;
    eprintln!(
        "[external] {} frames bit-identical to SSBO encode | full {:.2}/{:.2} ms, steady delta: direct ext {:.2} ms, blit+image {:.2} ms",
        frames.len(),
        ext_us[0],
        path_us[0],
        d(&ext_us),
        d(&path_us)
    );
    Ok(())
}

/// `texelFetch` encode from a plain `sampler2D` (the linear-scanout fast path,
/// where the dmabuf EGLImage binds to `GL_TEXTURE_2D`). Must match the SSBO
/// encode exactly.
fn fetch_encode(geom: Geometry, frames: &[Vec<u32>]) -> Result<()> {
    let mut fetch = GpuCodec::new(geom)?;
    let mut ssbo = GpuCodec::new(geom)?;
    let mut dec = Decoder::new(geom)?;
    let mut us = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let fb = words_bytes(frame);
        ssbo.activate()?;
        ssbo.upload(fb)?;
        ssbo.sync();
        let n_ssbo = ssbo.encode(i == 0)?;
        let c_ssbo = Header::parse(ssbo.coded(n_ssbo))?.count as usize;
        ssbo.update_reference()?;

        fetch.activate()?;
        fetch.set_scanout_test_frame(fb, true)?;
        fetch.sync();
        let t = Instant::now();
        let n = fetch.encode_fetch(i == 0)?;
        us.push(t.elapsed().as_secs_f64() * 1e3);
        let c = Header::parse(fetch.coded(n))?.count as usize;
        if n != n_ssbo || c != c_ssbo {
            bail!("fetch encode differs at frame {i}: {c}/{n} vs {c_ssbo}/{n_ssbo}");
        }
        if dec.decode(fetch.coded(n))? != fb {
            bail!("CPU decode of fetch-encoded frame {i} mismatch");
        }
        // fetch writes its reference in-pass (see ENC_FETCH_SRC)
    }
    eprintln!(
        "[fetch] {} frames bit-identical to SSBO encode | full {:.2} ms, steady delta {:.2} ms (sampler2D texelFetch)",
        frames.len(),
        us[0],
        us[1..].iter().sum::<f64>() / (us.len() - 1) as f64
    );
    Ok(())
}

/// Measures the dmabuf import itself (EGLImage + target bind, no cache): the
/// cost that was previously paid every frame and dominated the encode.
fn import_cost(geom: Geometry, frames: &[Vec<u32>]) -> Result<()> {
    let mut gpu = GpuCodec::new(geom)?;
    let mut us = Vec::new();
    for frame in frames.iter() {
        let fb = words_bytes(frame);
        gpu.activate()?;
        gpu.sync();
        let t = Instant::now();
        gpu.set_scanout_test_frame(fb, true)?;
        gpu.sync();
        us.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut v = us[1..].to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "[import] EGLImage import (per frame, uncached): {:.2} ms median — cached per buffer in production",
        v[v.len() / 2]
    );
    Ok(())
}

/// Allocate a real dma-buf with GBM (linear and NVIDIA 16BX2 block-linear) and
/// time the encoder's *detection read* of it — no compositor, no writer. This
/// isolates "the tiled layout is slow to read" from "someone else is writing
/// the scanout". Run under sudo on a host with /dev/dri/renderD128.
#[cfg(target_os = "linux")]
fn gbm_read_bench(geom: Geometry) -> Result<()> {
    use std::ffi::c_void;
    use std::os::fd::AsRawFd;
    type CreateDevice = unsafe extern "C" fn(i32) -> *mut c_void;
    type DeviceDestroy = unsafe extern "C" fn(*mut c_void);
    type BoCreateMods =
        unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u64, u32) -> *mut c_void;
    type BoCreateMods2 =
        unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u64, u32, u32) -> *mut c_void;
    type BoCreatePlain = unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32) -> *mut c_void;
    type BoI32 = unsafe extern "C" fn(*mut c_void) -> i32;
    type BoU32 = unsafe extern "C" fn(*mut c_void) -> u32;
    type BoOffset = unsafe extern "C" fn(*mut c_void, i32) -> u32;
    type BoU64 = unsafe extern "C" fn(*mut c_void) -> u64;
    type BoDestroy = unsafe extern "C" fn(*mut c_void);

    let file = ["/dev/dri/renderD128", "/dev/dri/card0"]
        .iter()
        .find_map(|p| {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(p)
                .map_err(|e| eprintln!("[gbm] open {p}: {e}"))
                .ok()
        })
        .ok_or_else(|| anyhow::anyhow!("open no DRM node (run under sudo)"))?;
    let fd = file.as_raw_fd();
    let lib = unsafe { libloading::Library::new("libgbm.so.1") }?;
    unsafe {
        let create_device: CreateDevice = *lib.get(b"gbm_create_device")?;
        let device_destroy: DeviceDestroy = *lib.get(b"gbm_device_destroy")?;
        let bo_create: BoCreateMods = *lib.get(b"gbm_bo_create_with_modifiers")?;
        let bo_create2: Option<BoCreateMods2> =
            lib.get(b"gbm_bo_create_with_modifiers2").ok().map(|s| *s);
        let bo_create_plain: BoCreatePlain = *lib.get(b"gbm_bo_create")?;
        let bo_fd: BoI32 = *lib.get(b"gbm_bo_get_fd")?;
        let bo_stride: BoU32 = *lib.get(b"gbm_bo_get_stride")?;
        let bo_offset: BoOffset = *lib.get(b"gbm_bo_get_offset")?;
        let bo_modifier: BoU64 = *lib.get(b"gbm_bo_get_modifier")?;
        let bo_destroy: BoDestroy = *lib.get(b"gbm_bo_destroy")?;

        let dev = create_device(fd);
        if dev.is_null() {
            anyhow::bail!("gbm_create_device failed");
        }
        const XR24: u32 = 0x3432_5258;
        let cases: [(&str, [u64; 1]); 2] = [
            ("linear", [0u64]),
            ("nvidia-16bx2", [0x0300_0000_0060_6014u64]),
        ];
        for (name, mods) in cases {
            // GBM flags: SCANOUT=1, RENDERING=4, LINEAR=16
            let (w, h) = (geom.width, geom.height);
            let mut bo = std::ptr::null_mut();
            if let Some(f) = bo_create2 {
                bo = f(dev, w, h, XR24, mods.as_ptr(), mods.len() as u32, 1 | 4);
            }
            if bo.is_null() {
                bo = bo_create(dev, w, h, XR24, mods.as_ptr(), mods.len() as u32);
            }
            if bo.is_null() && mods[0] == 0 {
                bo = bo_create_plain(dev, w, h, XR24, 4 | 16);
            }
            if bo.is_null() {
                eprintln!("[gbm] {name}: gbm_bo_create failed");
                continue;
            }
            let img = rdma_display_codec_gpu::DmabufImage {
                key: 0,
                fd: bo_fd(bo),
                offset: bo_offset(bo, 0),
                stride: bo_stride(bo),
                fourcc: XR24,
                modifier: bo_modifier(bo),
                width: geom.width,
                height: geom.height,
            };
            let mut gpu = GpuCodec::new(geom)?;
            eprintln!(
                "[gbm] {name:13} fd={:3} stride={:6} offset={:3} modifier=0x{:016x}",
                img.fd,
                img.stride,
                img.offset,
                img.modifier
            );
            if img.fd < 0 {
                eprintln!("[gbm] {name}: gbm_bo_get_fd failed (no exportable buffer)");
                bo_destroy(bo);
                continue;
            }
            if let Err(e) = gpu.set_scanout(&img) {
                eprintln!("[gbm] {name}: import failed: {e:#}");
                bo_destroy(bo);
                continue;
            }
            // Measure the two regimes that matter, on the same imported BO:
            //   full  = every tile coded (a full-damage frame): coded ≈ raw, so
            //           the payload write dominates — the expensive case
            //   delta = k tiles changed: only those tiles' payload leaves the
            //           GPU — the steady-state case
            // A "changed" tile is simulated by poking one word of it in the
            // reference (equivalent to the source changing there, and it avoids
            // writing the tiled BO). The shader resets it in-pass each time.
            //
            // frame 0 is the first encode after import (cold); the following
            // full frames are warm, and only those are comparable to a server
            // steady state.
            gpu.sync();
            let t = Instant::now();
            let n_full = gpu.encode_ext(true)?;
            let cold_full = t.elapsed().as_secs_f64() * 1e3;
            let p50 = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let mut full_v = Vec::new();
            for _ in 0..5 {
                gpu.sync();
                let t = Instant::now();
                gpu.encode_ext(true)?;
                full_v.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let stride_px = geom.width as usize; // bench geometry: stride = width*4
            let tiles_x = geom.tiles_x() as usize;
            fn poke(gpu: &GpuCodec, k: usize, stride_px: usize, tiles_x: usize) {
                for tid in 0..k {
                    let (tx, ty) = (tid % tiles_x, tid / tiles_x);
                    let at = (ty * 64) * stride_px + tx * 64;
                    gpu.debug_poke_reference(at, &[0xdead_beefu32]);
                }
            }
            let mut curve = Vec::new();
            for k in [8usize, 256, 1024, 2040] {
                poke(&gpu, k, stride_px, tiles_x);
                gpu.sync();
                let t = Instant::now();
                let n = gpu.encode_ext(false)?;
                curve.push((k, t.elapsed().as_secs_f64() * 1e3, n));
            }
            eprintln!(
                "[gbm] {name:13} modifier=0x{:016x} stride={:6} | frame0 full {:.2} ms (cold, {} B) | full warm p50 {:.2} ms",
                img.modifier,
                img.stride,
                cold_full,
                n_full,
                p50(&mut full_v),
            );
            for (k, ms, n) in &curve {
                eprintln!("[gbm]     {:4} tiles changed -> {:6.2} ms ({:8} B coded)", k, ms, n);
            }
            bo_destroy(bo);
        }
        device_destroy(dev);
    }
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--gbm") {
        let geom = Geometry::new(W, H, W * 4);
        return gbm_read_bench(geom);
    }
    let geom = Geometry::new(W, H, W * 4);
    let raw = geom.frame_bytes();
    eprintln!(
        "geometry {}x{} stride {} tile {} → {} tiles | raw frame {:.2} MiB | slot {:.2} MiB",
        W,
        H,
        W * 4,
        1 << geom.tile_log2,
        geom.tile_count(),
        raw as f64 / (1024.0 * 1024.0),
        geom.max_coded_len() as f64 / (1024.0 * 1024.0)
    );

    let frames = scenario();
    let _ = correctness(geom, &frames)?;
    decoder_roundtrip(geom, &frames)?;
    texture_encode(geom, &frames)?;
    external_encode(geom, &frames)?;
    fetch_encode(geom, &frames)?;
    import_cost(geom, &frames)?;

    let cpu = cpu_timing(geom, &frames)?;
    let gpu = gpu_timing(geom, &frames)?;

    // CRC cost, for context: it runs over the *coded* payload each frame.
    let big = vec![0u8; raw];
    let t = Instant::now();
    let c = crc32(&big);
    let crc_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[context] crc32 over a full {} MiB payload: {:.1} ms ({:08x})",
        raw / (1024 * 1024),
        crc_ms,
        c
    );

    println!();
    println!(
        "{:>5} {:>6} {:>13} {:>9} {:>11} {:>11} {:>11} {:>11}",
        "frame", "tiles", "coded bytes", "saving", "cpu enc", "gpu enc", "cpu dec", "gpu dec"
    );
    let mut tot_raw = 0u64;
    let mut tot_coded = 0u64;
    let mut sum = [0f64; 4];
    for i in 0..frames.len() {
        let coded = gpu.sizes[i];
        println!(
            "{:>5} {:>6} {:>13} {:>8.1}x {:>9.2}ms {:>9.2}ms {:>9.2}ms {:>9.2}ms",
            i,
            gpu.counts[i],
            coded,
            raw as f64 / coded as f64,
            cpu.enc_us[i],
            gpu.enc_us[i],
            cpu.dec_us[i],
            gpu.dec_us[i],
        );
        tot_raw += raw as u64;
        tot_coded += coded as u64;
        sum[0] += cpu.enc_us[i];
        sum[1] += gpu.enc_us[i];
        sum[2] += cpu.dec_us[i];
        sum[3] += gpu.dec_us[i];
    }
    let n = frames.len() as f64;
    // exclude the full first frame from the steady-state delta average
    let deltas = &gpu.sizes[1..];
    let avg_delta = deltas.iter().sum::<usize>() as f64 / deltas.len() as f64;
    println!();
    println!(
        "10-frame total: raw {:.1} MiB → coded {:.1} MiB  ({:.1}x less over the wire)",
        tot_raw as f64 / 1048576.0,
        tot_coded as f64 / 1048576.0,
        tot_raw as f64 / tot_coded as f64
    );
    println!(
        "steady-state delta (frames 1..9): avg {:.0} B/frame vs {:.1} MiB raw  →  {:.0}x",
        avg_delta,
        raw as f64 / 1048576.0,
        raw as f64 / avg_delta
    );
    println!(
        "avg/frame: cpu enc {:.2}ms  gpu enc {:.2}ms  |  cpu dec {:.2}ms  gpu dec {:.2}ms",
        sum[0] / n,
        sum[1] / n,
        sum[2] / n,
        sum[3] / n
    );
    println!(
        "avg/frame excluding full frame 0: cpu enc {:.2}ms  gpu enc {:.2}ms  |  cpu dec {:.2}ms  gpu dec {:.2}ms",
        (sum[0] - cpu.enc_us[0]) / (n - 1.0),
        (sum[1] - gpu.enc_us[0]) / (n - 1.0),
        (sum[2] - cpu.dec_us[0]) / (n - 1.0),
        (sum[3] - gpu.dec_us[0]) / (n - 1.0),
    );
    Ok(())
}
