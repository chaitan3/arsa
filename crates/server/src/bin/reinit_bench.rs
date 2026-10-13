//! reinit-bench: measure the cost of (re)initializing each part of the
//! capture pipeline — i.e. what a client connect pays when the server only
//! starts the pipeline on demand (the "Sunshine" lifecycle).
//!
//! There is no real client: the bench fakes a connect by constructing each
//! component directly. Two measurement shapes:
//!
//!   * construct/drop components (rdma, egl, gpu codec, uinput, drm probe):
//!     run N iterations and report the distribution. These are cleanly
//!     re-creatable, so N > 1.
//!   * streaming components (drm capture, qemu capture, audio): time from
//!     spawn to the first frame/packet. The capture/audio workers are not
//!     stoppable yet (see Phase 0), so each runs once and is left running;
//!     the process exits right after.
//!
//! Components whose hardware/privileges are missing are skipped with a
//! reason, so the same binary runs on a dev box and under sudo on the host.
//!
//! Usage (the interesting run needs root for DRM/uinput and the user's
//! XDG_RUNTIME_DIR for audio):
//!   cargo build --release -p rdma-display-server --bin reinit-bench
//!   sudo -E ./target/release/reinit-bench --iters 30

use std::time::{Duration, Instant};

use anyhow::Context;
use rdma_display_capture::{
    CaptureEvent, CaptureSource, DrmKmsSource, EglMaterializer, QemuDbusSource,
};

struct Cfg {
    iters: usize,
    width: u32,
    height: u32,
    rendernode: Option<String>,
    drm_card: Option<String>,
    dbus_socket: Option<String>,
    console: u32,
    audio_device: Option<String>,
    audio: bool,
    skip: Vec<String>,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            iters: 20,
            width: 1920,
            height: 1080,
            rendernode: None,
            drm_card: None,
            dbus_socket: None,
            console: 0,
            audio_device: None,
            audio: true,
            skip: Vec::new(),
        }
    }
}

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(2);
        }
    };
    println!(
        "[reinit-bench] construct/drop iters={} | codec geometry {}x{} | faking a client connect",
        cfg.iters, cfg.width, cfg.height
    );
    println!(
        "{:<22} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "component", "n", "min_us", "p50_us", "p90_us", "max_us", "mean_us"
    );

    let skipped = |name: &str| cfg.skip.iter().any(|s| s == name);

    // --- clean construct/drop components -----------------------------------
    if !skipped("rdma") {
        bench_construct("rdma-open+pd", cfg.iters, || {
            let ctx = rdma_display_rdma::open_device(None)?;
            let _pd = ctx.alloc_pd()?;
            Ok(())
        });
    }

    if !skipped("egl") {
        let rn = cfg.rendernode.clone();
        bench_construct("egl-context", cfg.iters, move || {
            let m = EglMaterializer::new(rn.as_deref())?;
            drop(m);
            Ok(())
        });
    }

    if !skipped("egl-phases") {
        let rn = cfg.rendernode.clone();
        match EglMaterializer::new_timed(rn.as_deref()) {
            Ok((m, ph)) => {
                drop(m);
                println!(
                    "  egl phases (us): load={} display={} init={} config={} context={} program={} | total={}",
                    ph.load, ph.display, ph.initialize, ph.config, ph.context, ph.program, ph.total()
                );
            }
            Err(e) => println!("{:<22} skip: {e:#}", "egl-phases"),
        }
    }

    // one context shared by the materializer and the encoder: the design we
    // want on connect. Times the shared-context path end to end.
    if !skipped("shared-session") {
        let geom = rdma_display_codec::Geometry::new(cfg.width, cfg.height, cfg.width * 4);
        match rdma_display_egl::SharedContext::new(cfg.rendernode.as_deref()) {
            Ok(sh) => {
                let sh = std::sync::Arc::new(sh);
                let t = Instant::now();
                let mat = EglMaterializer::new_shared(&sh);
                let mat_us = t.elapsed().as_micros();
                let t = Instant::now();
                let codec = rdma_display_codec_gpu::GpuCodec::new_shared(geom, sh.clone());
                let codec_us = t.elapsed().as_micros();
                match (mat, codec) {
                    (Ok(m), Ok(c)) => {
                        drop(m);
                        drop(c);
                        println!(
                            "  shared-session (us): materializer={mat_us} codec={codec_us} | total={}",
                            mat_us + codec_us
                        );
                    }
                    (m, c) => println!(
                        "{:<22} skip: materializer={:?} codec={:?}",
                        "shared-session",
                        m.err().map(|e| format!("{e:#}")),
                        c.err().map(|e| format!("{e:#}")),
                    ),
                }
            }
            Err(e) => println!("{:<22} skip: {e:#}", "shared-session"),
        }
    }

    if !skipped("codec") {
        let geom = rdma_display_codec::Geometry::new(cfg.width, cfg.height, cfg.width * 4);
        bench_construct("gpu-codec", cfg.iters, move || {
            let c = rdma_display_codec_gpu::GpuCodec::new(geom)?;
            c.sync();
            drop(c);
            Ok(())
        });
    }

    if !skipped("uinput") {
        bench_construct("uinput-mouse", cfg.iters, || {
            let m = rdma_display_capture::input::UinputMouse::open()?;
            drop(m);
            Ok(())
        });
    }

    if !skipped("drm-probe") {
        let card = cfg.drm_card.clone();
        bench_construct("drm-probe", cfg.iters, move || {
            let _fmt = rdma_display_capture::drm::probe_display(card.as_deref())?;
            Ok(())
        });
    }

    // --- streaming components: spawn -> first event (run once) -------------
    if !skipped("drm-stream") {
        bench_drm_stream(&cfg);
    }
    if !skipped("qemu-stream") {
        bench_qemu_stream(&cfg);
    }
    if cfg.audio && !skipped("audio") {
        bench_audio(&cfg);
    }

    println!("[reinit-bench] done");
}

fn bench_construct<F>(label: &str, iters: usize, mut f: F)
where
    F: FnMut() -> anyhow::Result<()>,
{
    let mut us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        match f() {
            Ok(()) => us.push(t.elapsed().as_micros()),
            Err(e) => {
                println!("{label:<22} skip: {e:#}");
                return;
            }
        }
    }
    print_stats(label, &us);
}

fn bench_drm_stream(cfg: &Cfg) {
    let (tx, rx) = std::sync::mpsc::channel();
    let src = Box::new(DrmKmsSource::with_sync(
        cfg.drm_card.clone(),
        rdma_display_capture::CaptureSync::Auto,
    ));
    let t0 = Instant::now();
    let handle = match CaptureSource::spawn(src, tx) {
        Ok(h) => h,
        Err(e) => {
            println!("{:<22} skip: {e:#}", "drm-stream");
            return;
        }
    };
    match wait_first_scanout(&rx, t0) {
        Some(us) => print_single("drm-stream", us),
        None => println!("{:<22} skip: no scanout within 8 s", "drm-stream"),
    }
    drop(handle); // stop + join the capture thread
}

fn bench_qemu_stream(cfg: &Cfg) {
    let (tx, rx) = std::sync::mpsc::channel();
    let src = Box::new(QemuDbusSource::new(cfg.dbus_socket.clone(), cfg.console));
    let handle = match CaptureSource::spawn(src, tx) {
        Ok(h) => h,
        Err(e) => {
            println!("{:<22} skip: {e:#}", "qemu-stream");
            return;
        }
    };
    match wait_any_event(&rx, Duration::from_secs(8)) {
        Some(us) => print_single("qemu-stream", us),
        None => println!("{:<22} skip: no event within 8 s", "qemu-stream"),
    }
    drop(handle);
}

fn bench_audio(cfg: &Cfg) {
    let t = Instant::now();
    match rdma_display_audio::capture::start_capture(cfg.audio_device.as_deref()) {
        Ok(mut audio) => {
            print_single("audio-start", t.elapsed().as_micros());
            let t = Instant::now();
            if audio.mailbox.wait_first(Duration::from_secs(8)) {
                print_single("audio-first-packet", t.elapsed().as_micros());
            } else {
                println!("{:<22} skip: no packet within 8 s", "audio-first-packet");
            }
            audio.stop();
        }
        Err(e) => {
            println!("{:<22} skip: {e:#}", "audio-start");
            println!("{:<22} skip (see above)", "audio-first-packet");
        }
    }
}

fn wait_first_scanout(
    rx: &std::sync::mpsc::Receiver<CaptureEvent>,
    t0: Instant,
) -> Option<u128> {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return None;
        }
        match rx.recv_timeout(remaining) {
            Ok(CaptureEvent::Scanout(..)) => return Some(t0.elapsed().as_micros()),
            Ok(_) => {}
            Err(_) => return None,
        }
    }
}

fn wait_any_event(
    rx: &std::sync::mpsc::Receiver<CaptureEvent>,
    timeout: Duration,
) -> Option<u128> {
    let t = Instant::now();
    match rx.recv_timeout(timeout) {
        Ok(_) => Some(t.elapsed().as_micros()),
        Err(_) => None,
    }
}

fn print_single(label: &str, us: u128) {
    println!(
        "{label:<22} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        1, us, us, us, us, us
    );
}

fn print_stats(label: &str, us: &[u128]) {
    if us.is_empty() {
        println!("{label:<22} skip: no samples");
        return;
    }
    let mut v = us.to_vec();
    v.sort_unstable();
    let pick = |q: f64| v[((v.len() as f64 - 1.0) * q).round() as usize];
    let mean = us.iter().sum::<u128>() as f64 / us.len() as f64;
    println!(
        "{label:<22} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10.0}",
        us.len(),
        v[0],
        pick(0.50),
        pick(0.90),
        v[v.len() - 1],
        mean
    );
}

fn parse_args() -> anyhow::Result<Cfg> {
    let mut cfg = Cfg::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--iters" => cfg.iters = args.next().context("--iters needs a value")?.parse()?,
            "--width" => cfg.width = args.next().context("--width needs a value")?.parse()?,
            "--height" => cfg.height = args.next().context("--height needs a value")?.parse()?,
            "--rendernode" => cfg.rendernode = Some(args.next().context("value")?),
            "--drm-card" => cfg.drm_card = Some(args.next().context("value")?),
            "--dbus-socket" => cfg.dbus_socket = Some(args.next().context("value")?),
            "--console" => cfg.console = args.next().context("value")?.parse()?,
            "--audio-device" => cfg.audio_device = Some(args.next().context("value")?),
            "--no-audio" => cfg.audio = false,
            "--skip" => {
                cfg.skip = args
                    .next()
                    .context("--skip needs a comma list")?
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            "--help" | "-h" => {
                println!(
                    "usage: reinit-bench [--iters N] [--width W] [--height H] \
                     [--rendernode PATH] [--drm-card PATH] [--dbus-socket PATH] \
                     [--console N] [--audio-device NAME] [--no-audio] [--skip a,b,c]\n\
                     components: rdma egl egl-phases shared-session codec uinput \
                     drm-probe drm-stream qemu-stream audio"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    Ok(cfg)
}
