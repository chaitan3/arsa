//! rdma-display server: captures a display source and streams frames to a
//! client over RDMA.
//!
//! Modes (defaults shown):
//!   --mode capture   stream a display source:
//!     --source drm (default)  KMS capture of the host screen (Sunshine-style):
//!             grabs the CRTC's current fb via GETFB2 as a dmabuf each flip;
//!             requires sudo (CAP_SYS_ADMIN) and kernel >= 5.8; egl
//!             materialize only; --input uinput (default)
//!     --source qemu           QEMU dbus display capture (the guest):
//!             --materialize egl    GPU blit+readback (default, gl=on)
//!             --materialize plain  D-Bus host-memory scanouts (gl=off)
//!             --readback pbo       zero-copy staging: GL readback DMA fills
//!                                  persistent-mapped PBOs registered as RDMA
//!                                  slots (default for egl); "direct" = legacy
//!                                  readPixels path
//!             --input qemu         inject into the guest console (default)
//!     --codec lossless (default)|raw|lossy  frame coding (lossless =
//!             tile-delta against the previous frame; raw = unchanged wire;
//!             lossy = PyroWave; see docs/compression.md). Client negotiates.
//!     --capture-sync auto (default) = fence -> vblank -> one-behind
//!     --fps 0 (default) = uncapped, consumer-paced; --audio on by default
//!   --mode testcard   synthetic frames (validates the transport end-to-end)

use std::net::TcpListener;
use std::sync::mpsc;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use rdma_display_proto::{Conn, DisplayMeta};
use rdma_display_capture::{
    CaptureEvent, CaptureSource, CaptureSync, DrmKmsSource, FrameFormat, FrameMaterializer,
    InputSink, QemuDbusInput, QemuDbusSource, ScreenSize,
};

mod gpu_power;
mod lossy;

use gpu_power::GpuPower;

fn main() -> Result<()> {
    let mut mode = "capture".to_string();
    let mut device: Option<String> = None;
    let mut listen: String = "0.0.0.0:4791".to_string();
    let mut slots: usize = 3;
    let mut fps: u64 = 0;
    let mut source = "drm".to_string();
    let mut input: Option<String> = None; // None = paired default for the source
    let mut dbus_socket: Option<String> = None;
    let mut console: u32 = 0;
    let mut materialize: String = "egl".to_string();
    let mut rendernode: Option<String> = None;
    let mut drm_card: Option<String> = None;
    let mut input_test = false;
    let mut readback: String = "pbo".to_string();
    let mut audio = true;
    let mut audio_device: Option<String> = None;
    let mut audio_helper = false;
    let mut capture_sync = CaptureSync::Auto;
    let mut codec = rdma_display_codec::Codec::Lossless;
    let mut lossy_budget: usize = 1_000_000;
    let mut lossy_chroma444 = false;
    let mut transport = rdma_display_rdma::TransportPref::Auto;
    let mut gpu_power = GpuPower::Off;
    // log idle/session RSS+CPU every N seconds (0 = off)
    let mut idle_metrics: u64 = 0;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => mode = args.next().expect("--mode needs a value"),
            "--device" => device = Some(args.next().expect("--device needs a value")),
            "--listen" => listen = args.next().expect("--listen needs a value"),
            "--slots" => slots = args.next().expect("--slots needs a value").parse()?,
            "--fps" => fps = args.next().expect("--fps needs a value").parse()?,

            "--source" => source = args.next().expect("--source needs a value"),
            "--input" => input = Some(args.next().expect("--input needs a value")),
            "--dbus-socket" => {
                dbus_socket = Some(args.next().expect("--dbus-socket needs a value"))
            }
            "--console" => console = args.next().expect("--console needs a value").parse()?,
            "--materialize" => materialize = args.next().expect("--materialize needs a value"),
            "--readback" => readback = args.next().expect("--readback needs a value"),
            "--rendernode" => rendernode = Some(args.next().expect("--rendernode needs a value")),
            "--drm-card" => drm_card = Some(args.next().expect("--drm-card needs a value")),
            "--capture-sync" => {
                let v = args.next().expect("--capture-sync needs a value");
                capture_sync = CaptureSync::parse(&v).with_context(|| {
                    format!(
                        "--capture-sync must be onbehind|vblank|zero|auto|settle[:ms]|fence, got {v:?}"
                    )
                })?;
            }
            "--input-test" => input_test = true,
            "--codec" => {
                let v = args.next().expect("--codec needs a value");
                codec = rdma_display_codec::Codec::parse(&v).with_context(|| {
                    format!("--codec must be raw|lossless|lossy, got {v:?}")
                })?;
                if !codec.implemented() {
                    bail!(
                        "--codec {} is a reserved selector and not implemented yet",
                        codec.name()
                    );
                }
            }
            "--lossy-budget" => {
                lossy_budget = args
                    .next()
                    .expect("--lossy-budget needs a value")
                    .parse()?;
            }
            "--lossy-chroma" => {
                let v = args.next().expect("--lossy-chroma needs a value");
                lossy_chroma444 = match v.as_str() {
                    "420" | "4:2:0" => false,
                    "444" | "4:4:4" => true,
                    other => bail!("--lossy-chroma must be 420|444, got {other:?}"),
                };
            }
            "--transport" => {
                let v = args.next().expect("--transport needs a value");
                transport = rdma_display_rdma::TransportPref::parse(&v)
                    .with_context(|| format!("--transport must be auto|rdma|tcp, got {v:?}"))?;
            }
            "--audio" => audio = true,
            "--no-audio" => audio = false,
            "--audio-device" => audio_device = Some(args.next().expect("--audio-device needs a value")),
            "--audio-helper" => audio_helper = true,
            "--gpu-power" => {
                let v = args.next().expect("--gpu-power needs a value");
                gpu_power = GpuPower::parse(&v)?;
            }
            "--idle-metrics" => {
                idle_metrics = args.next().expect("--idle-metrics needs a value").parse()?
            }
            "--help" | "-h" => {
                println!("usage: rdma-display-server [--mode capture|testcard] [--device mlx5_0] [--listen 0.0.0.0:4791] [--slots 3] [--fps N]\n  \
  defaults: --mode capture --source drm --fps 0 --codec lossless --audio on --capture-sync auto (a bare command is a root KMS capture of the host)\n  \
capture: --source qemu|drm (qemu = QEMU dbus-display, the guest; drm = KMS capture, the host — requires sudo/CAP_SYS_ADMIN, kernel >= 5.8)\n  \
  --input qemu|uinput (default auto: qemu↔qemu-dbus, drm↔uinput)\n  \
  qemu: [--dbus-socket /path/dbus.sock] [--console 0]   drm: [--drm-card /dev/dri/cardN] (default: first card0..card8 with an active scanout)\n  \
  [--materialize egl|plain] [--rendernode /dev/dri/renderD128] [--readback pbo|direct] [--input-test]\n  \
  --materialize egl (default) imports the dmabuf and reads the frame back for raw/lossy and for the CPU/readback lossless fallback; GPU lossless bypasses the readback entirely and encodes straight from the imported dmabuf (no full-frame readback, no re-upload). --readback pbo|direct only affects that readback path.\n  \
  --capture-sync onbehind|vblank|zero|auto|settle[:ms]|fence  (drm) which fb to capture: onbehind = safe, one frame old; vblank = current frame as its vblank retires it (lowest latency; dead on this nvidia-drm stack, falls back); zero = current frame immediately, no sync (testing only, may tear); settle[:ms] = current frame ms after its commit (default 2 ms; usually complete, but frames whose render outlasts the window tear); fence = read the current frame when the compositor's per-frame completion fence signals (tracefs; ~1 ms, tear-free, needs root, falls back); auto = try fence, then vblank, then one-behind (default)\n  \
  --codec raw|lossless|lossy  frame coding for the whole session (default lossless = v2 tile-delta against the previous frame, desktop-friendly; raw = unchanged wire). lossy = PyroWave intra-only GPU codec (needs libpyrowave-shared.so). The client negotiates automatically; build server and client together (lossless v2 is a wire-format version).\n  \
  --lossy-budget BYTES  lossy: max bitstream bytes per frame (default 1000000; slot size sent to the client).\n  \
  --lossy-chroma 420|444  lossy: chroma subsampling (default 420).\n  \
  --gpu-power off|p5|p0|max|<sm>[,<mem>]  floor the NVIDIA GPU clocks with nvidia-smi, still allowing boost above the floor (needs root; warns and continues if unavailable). p5 = the measured fast floor (sm 435 / mem 810 MHz), p0/max = floor at the card's maximum (i.e. pinned), <sm>[,<mem>] = explicit MHz floors. Default off. GeForce cards reject the memory-clock floor.\n  \
  --transport auto|rdma|tcp  video transport (default auto: RDMA/RoCE when available, else TCP). Negotiated with the client; TCP sends length-prefixed coded frames on a TAG_VIDEO connection.\n  \
  --idle-metrics N  log RSS/CPU/threads every N seconds, labelled idle or session (default off).\n  \
  --fps N (capture): pacing cap; 0 = uncapped, consumer-paced by client credits (default 0).\n  \
  --audio | --no-audio [--audio-device my-sink.monitor]  audio side-channel (default on; --no-audio disables): capture the named source's PCM through the pipewire-pulse\n  \
  compat socket (XDG_RUNTIME_DIR must point at the user's session when running under sudo; '@DEFAULT_MONITOR@' = default sink).");
                return Ok(());
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }

    // internal: setuid-down audio helper (spawned by run_capture; runs as
    // the session user, needs no RDMA device, must not touch ibverbs)
    if audio_helper {
        return rdma_display_audio::capture::run_helper(
            audio_device.as_deref().unwrap_or("@DEFAULT_MONITOR@"),
        );
    }

    // best-effort GPU clock floor (--gpu-power): never fatal, so a non-root or
    // non-NVIDIA box still streams on the driver's own power management. On
    // success the floors are released again on SIGINT/SIGTERM (ctrl-c).
    match gpu_power.apply() {
        Ok(applied) => gpu_power::install_cleanup(applied),
        Err(e) => {
            eprintln!("[gpu-power] {e:#}");
            eprintln!("[gpu-power] continuing with the driver's default power management");
        }
    }

    // video transport: auto prefers RDMA/RoCE and falls back to TCP when no
    // device is available (or when forced with --transport)
    let rdma: Option<(ibverbs::Context, ibverbs::ProtectionDomain)> = match transport {
        rdma_display_rdma::TransportPref::Tcp => None,
        _ => match rdma_display_rdma::open_device(device.as_deref()) {
            Ok(ctx) => {
                let pd = ctx.alloc_pd()?;
                Some((ctx, pd))
            }
            Err(e) => {
                if transport == rdma_display_rdma::TransportPref::Rdma {
                    return Err(e).context("--transport rdma: open RDMA device");
                }
                eprintln!("[server] no RDMA device ({e:#}); falling back to TCP video transport");
                None
            }
        },
    };
    let rdma_ref = rdma.as_ref().map(|(c, p)| (c, p));

    match mode.as_str() {
        "testcard" => run_testcard_server(rdma_ref, transport, &listen, slots, fps),
        "capture" => run_capture(
            rdma_ref, transport, &source, input.as_deref(), &dbus_socket, console, &drm_card,
            &materialize, rendernode.as_deref(), &listen, slots, fps, input_test, &readback,
            capture_sync, audio, audio_device.as_deref(), codec, lossy_budget, lossy_chroma444,
            idle_metrics,
        ),
        other => bail!("mode {other:?} not implemented yet"),
    }
}

/// TCP transport: accept (and attach) the client's TAG_VIDEO socket. The
/// client opens it immediately after the codec handshake; poll for it so the
/// blocking send loop can start with a usable socket.
fn wait_for_video(link: &mut rdma_display_rdma::ServerLink, listener: &TcpListener) -> Result<()> {
    use rdma_display_rdma::Transport;
    if link.transport() != Transport::Tcp {
        return Ok(());
    }
    listener.set_nonblocking(true)?;
    let start = std::time::Instant::now();
    while !link.has_video() && start.elapsed() < std::time::Duration::from_secs(5) {
        match listener.accept() {
            Ok((mut s, _)) => {
                let _ = s.set_nodelay(true);
                let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(1)));
                let mut tag = [0u8; 1];
                let ok = std::io::Read::read_exact(&mut s, &mut tag).is_ok();
                let _ = s.set_read_timeout(None);
                if ok && tag[0] == rdma_display_proto::TAG_VIDEO {
                    link.attach_video(s);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            Err(e) => return Err(e).context("accept video socket"),
        }
    }
    listener.set_nonblocking(false)?;
    anyhow::ensure!(link.has_video(), "client never opened its TAG_VIDEO socket");
    Ok(())
}

fn run_testcard_server(
    rdma: Option<(&ibverbs::Context, &ibverbs::ProtectionDomain)>,
    transport: rdma_display_rdma::TransportPref,
    listen: &str,
    slots: usize,
    fps: u64,
) -> Result<()> {
    const W: u32 = 1280;
    const H: u32 = 800;
    let meta = DisplayMeta {
        width: W,
        height: H,
        stride: W * 4,
        fourcc: 0x3432_5258, // 'XR24' XRGB8888
        modifier: 0,
        y0_top: false,
        frame_bytes: W * 4 * H,
        gen: 0,
    };
    let listener = TcpListener::bind(listen)?;
    eprintln!("listening on {listen}, mode=testcard");
    loop {
        let (mut stream, _) = listener.accept()?;
        let _ = stream.set_nodelay(true);
        // explicit first-byte tag routing; a short timeout keeps a silent
        // connection from stalling the loop
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
        let mut tag = [0u8; 1];
        let tag_ok = std::io::Read::read_exact(&mut stream, &mut tag).is_ok();
        let _ = stream.set_read_timeout(None);
        if !tag_ok {
            continue;
        }
        if tag[0] == rdma_display_audio::wire::TAG_AUDIO {
            eprintln!("[server] audio connection in testcard mode: dropped");
            continue;
        }
        if tag[0] == rdma_display_proto::TAG_INPUT {
            eprintln!("[server] input connection in testcard mode: dropped");
            continue;
        }
        eprintln!("client connected: {}", stream.peer_addr()?);
        // testcard has no console to inject into: drain input events
        let (input_tx, input_rx) = mpsc::channel();
        std::thread::spawn(move || {
            while input_rx.recv().is_ok() {}
        });
        // no input sink in testcard mode: nothing to relay rumble to
        let rumble_hub = rdma_display_proto::gamepad::RumbleHub::new();
        let mut link = rdma_display_rdma::ServerLink::accept(
            Conn::new(stream),
            transport,
            rdma,
            &meta,
            slots,
            rdma_display_codec::Codec::Raw,
            meta.frame_bytes as usize,
            0,
            input_tx,
            None,
            &rumble_hub,
            rdma_display_rdma::random_nonce(),
        )?;
        // TCP transport: the client opens its TAG_VIDEO socket right after the
        // codec handshake; grab it before the (blocking) testcard loop starts
        if link.transport() == rdma_display_rdma::Transport::Tcp {
            wait_for_video(&mut link, &listener)?;
        }
        eprintln!("session up: {slots} slots, {} frame bytes", link.frame_bytes());
        run_testcard(&mut link, fps)?;
    }
}

fn run_testcard(link: &mut rdma_display_rdma::ServerLink, fps: u64) -> Result<()> {
    let period = std::time::Duration::from_millis(1000 / fps.max(1));
    let mut frame: u64 = 0;
    let started = std::time::Instant::now();
    let mut sent: u64 = 0;
    let mut last_report = std::time::Instant::now();
    loop {
        let t0 = rdma_display_proto::unix_now_us();
        link.send_frame(|bytes| testcard_fill(bytes, frame), t0)?;
        frame += 1;
        sent += 1;
        if last_report.elapsed().as_secs() >= 10 {
            eprintln!("[stats] sent {} frames in {:?}", sent, started.elapsed());
            last_report = std::time::Instant::now();
        }
        std::thread::sleep(period);
    }
}

/// Simple moving gradient testcard in XRGB8888.
fn testcard_fill(bytes: &mut [u8], frame: u64) {
    let width = 1280usize;
    for (y, row) in bytes.chunks_exact_mut(width * 4).enumerate() {
        for (x, px) in row.chunks_exact_mut(4).enumerate() {
            let t = frame as u32;
            px[0] = ((x as u32 * 255 / width as u32) ^ ((y as u32) * 7) ^ t) as u8;
            px[1] = ((y as u32 * 255 / 800) + t / 4) as u8;
            px[2] = (x as u32 + t).wrapping_mul(7) as u8;
            px[3] = 0xff;
        }
    }
}

// ---------------------------------------------------------------------------
// capture mode
// ---------------------------------------------------------------------------

/// Per-session workers: the capture source and input sink, started on control
/// connect and stopped when the session ends. Audio is managed separately but
/// follows the same lifecycle.
struct SessionWorkers {
    /// None when a fallback eager source (preflight found no geometry) is
    /// already running and owned by the server.
    source: Option<rdma_display_capture::Running>,
    input: rdma_display_capture::Running,
    input_tx: mpsc::Sender<rdma_display_proto::InputEvent>,
}

impl SessionWorkers {
    fn stop(mut self) {
        if let Some(s) = self.source.take() {
            let mut s = s;
            s.stop();
        }
        self.input.stop();
    }
}

/// Best-effort audio capture start. Used per session (and by the startup
/// probe). Returns None when audio is disabled or unavailable.
fn start_audio(
    audio: bool,
    dev: Option<&str>,
) -> Option<rdma_display_audio::capture::AudioCapture> {
    if !audio {
        return None;
    }
    match rdma_display_audio::capture::start_capture(dev) {
        Ok(a) => Some(a),
        Err(e) => {
            eprintln!(
                "[audio] capture unavailable ({e:#}); streaming video without audio \
                 (pass --no-audio to silence this)"
            );
            None
        }
    }
}

/// `--input-test`: synthetic input injector. Runs for the life of the session
/// (it stops when the session's input channel closes).
fn spawn_input_test(tx: mpsc::Sender<rdma_display_proto::InputEvent>) -> Result<()> {
    std::thread::Builder::new()
        .name("input-test".into())
        .spawn(move || {
            eprintln!("[server] --input-test: injecting synthetic input");
            let mut n = 0u32;
            loop {
                // key 'a' (qnum 0x1e) press+release
                if tx
                    .send(rdma_display_proto::InputEvent::Key { code: 0x1e, down: true })
                    .is_err()
                {
                    return;
                }
                let _ = tx.send(rdma_display_proto::InputEvent::Key { code: 0x1e, down: false });
                // mouse: absolute sweep + click (console dims fallback 1080p)
                let (w, h) = (1920u32, 1080u32);
                let x = w / 2 + (n % 40) as u32 * 8;
                let y = h / 2 + (n % 20) as u32 * 10;
                let _ = tx.send(rdma_display_proto::InputEvent::Abs { x, y });
                if n % 2 == 0 {
                    let _ = tx.send(rdma_display_proto::InputEvent::Btn {
                        button: 0, // Left
                        down: true,
                    });
                    let _ = tx.send(rdma_display_proto::InputEvent::Btn {
                        button: 0,
                        down: false,
                    });
                }
                n += 1;
                eprintln!("[server] input-test #{n}: key 'a' + mouse abs ({x},{y})");
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        })
        .context("spawn input-test")?;
    Ok(())
}

/// Idle/session resource logger: RSS, CPU%, and thread count every `secs`.
/// Used to verify that an idle server holds little (Phase 3 measurement).
fn spawn_resource_metrics(
    secs: u64,
    session_active: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let _ = std::thread::Builder::new()
        .name("resource-metrics".into())
        .spawn(move || {
            let clk = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as u64;
            let mut last = read_cpu_ticks().unwrap_or((0, 0));
            let mut last_t = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(secs.max(1)));
                let now = std::time::Instant::now();
                let cpu = read_cpu_ticks().unwrap_or(last);
                let dt = now.duration_since(last_t).as_secs_f64();
                let dticks = (cpu.0 + cpu.1).saturating_sub(last.0 + last.1);
                let pct = if dt > 0.0 {
                    dticks as f64 / clk as f64 / dt * 100.0
                } else {
                    0.0
                };
                eprintln!(
                    "[metrics] {} rss={:.1} MB cpu={:.2}% threads={}",
                    if session_active.load(std::sync::atomic::Ordering::Relaxed) {
                        "session"
                    } else {
                        "idle   "
                    },
                    read_rss_bytes().unwrap_or(0) as f64 / 1e6,
                    pct,
                    read_threads().unwrap_or(0),
                );
                last = cpu;
                last_t = now;
            }
        });
}

/// Resident set size from /proc/self/statm (pages of 4 KiB). Field 0 is the
/// virtual size; field 1 is the resident set.
fn read_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

/// (utime, stime) in clock ticks from /proc/self/stat.
fn read_cpu_ticks() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // skip the comm field, which may itself contain spaces or parens
    let after = s.rsplit(')').next()?;
    let f: Vec<&str> = after.split_whitespace().collect();
    // utime/stime are overall fields 14/15 => indices 11/12 after the comm
    let utime: u64 = f.get(11)?.parse().ok()?;
    let stime: u64 = f.get(12)?.parse().ok()?;
    Some((utime, stime))
}

/// Thread count from /proc/self/status.
fn read_threads() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("Threads:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn run_capture(
    rdma: Option<(&ibverbs::Context, &ibverbs::ProtectionDomain)>,
    transport: rdma_display_rdma::TransportPref,
    source_name: &str,
    input_name: Option<&str>,
    dbus_socket: &Option<String>,
    console: u32,
    drm_card: &Option<String>,
    materialize: &str,
    rendernode: Option<&str>,
    listen: &str,
    slots: usize,
    fps: u64,
    input_test: bool,
    readback: &str,
    capture_sync: CaptureSync,
    audio: bool,
    audio_device: Option<&str>,
    codec: rdma_display_codec::Codec,
    lossy_budget: usize,
    lossy_chroma444: bool,
    idle_metrics: u64,
) -> Result<()> {
    if source_name == "drm" && materialize != "egl" && codec != rdma_display_codec::Codec::Lossy {
        bail!("--source drm requires --materialize egl (KMS capture is dmabuf-only)");
    }
    // idle/session resource logger (Phase 3 measurement)
    let session_active = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    if idle_metrics > 0 {
        spawn_resource_metrics(idle_metrics, session_active.clone());
    }
    // audio side-channel: per-session. At startup a one-shot probe (start +
    // first packet) validates the setup, then stops; capture is (re)started on
    // connect and stopped on disconnect, so nothing runs while idle.
    // Best-effort: a missing pipewire/pulse warns rather than blocking video.
    if let Some(mut a) = start_audio(audio, audio_device) {
        if a.mailbox.wait_first(std::time::Duration::from_secs(3)) {
            eprintln!("[preflight] audio capture ok");
        } else {
            eprintln!("[preflight] audio: no packet within 3 s (continuing)");
        }
        a.stop(); // preflight probe only
    }
    let mut audio_capture: Option<rdma_display_audio::capture::AudioCapture> = None;
    let audio_busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // One EGL context shared by the materializer and the lossless encoder:
    // both import the same scanout dmabufs and run on the server thread, so a
    // single GLES 3.x context serves both and saves a second context creation
    // (~49 ms on this stack) per session. Only built when something needs GL.
    let needs_egl = (codec != rdma_display_codec::Codec::Lossy && materialize == "egl")
        || codec == rdma_display_codec::Codec::Lossless;
    let gl: Option<Arc<rdma_display_egl::SharedContext>> = if needs_egl {
        Some(Arc::new(rdma_display_egl::SharedContext::new(rendernode)?))
    } else {
        None
    };

    // input sink: explicit --input wins, else paired default
    let input_name = input_name.unwrap_or(match source_name {
        "drm" => "uinput",
        _ => "qemu",
    });
    if input_name == "qemu" && source_name == "drm" {
        bail!("--input qemu injects into the QEMU guest, which --source drm does not capture (use --input uinput)");
    }
    if input_name == "uinput" && source_name == "qemu" {
        bail!("--input uinput injects into the host compositor, but --source qemu captures the guest (use --input qemu)");
    }

    // startup preflight: validate/probe each element once; the heavy pipeline
    // is (re)started on demand per client (Phase 2). The probed geometry lets
    // a client connect negotiate slots before the first captured frame.
    let geometry = preflight(source_name, input_name, drm_card, dbus_socket, console, codec, gl.as_ref());
    if let Some(m) = geometry.as_ref() {
        eprintln!(
            "[preflight] geometry cached: {}x{} stride {} (sizes slots on connect)",
            m.width, m.height, m.stride
        );
    }

    let mut mat: Box<dyn FrameMaterializer> = if codec == rdma_display_codec::Codec::Lossy {
        Box::new(lossy::LossyMaterializer::new(lossy_budget, lossy_chroma444)?)
    } else {
        match materialize {
            "egl" => Box::new(match gl.as_ref() {
                Some(gl) => rdma_display_capture::EglMaterializer::new_shared(gl)?,
                None => rdma_display_capture::EglMaterializer::new(rendernode)?,
            }),
            "plain" => Box::new(rdma_display_capture::materializer::PlainMaterializer::new()),
            other => bail!("materializer {other:?} not implemented (egl|plain)"),
        }
    };
    eprintln!("[server] capture materializer: {}", mat.name());

    // input: explicit --input wins, else paired default. The capture source
    // and input sink are started per session (on control connect) and stopped
    // on disconnect; the capture channel is long-lived so the server always
    // holds a sender and the receiver never disconnects between sessions.
    let screen = ScreenSize::new();
    // host-game rumble on the virtual pad rides this hub to the current
    // session's control stream (installed per-accept by the RDMA layer)
    let rumble_hub = rdma_display_proto::gamepad::RumbleHub::new();
    let (tx, rx) = mpsc::channel();
    // Fallback: if preflight could not read the geometry (rare), keep a source
    // running so the first frame can size the slots on connect; otherwise the
    // source is started lazily per session.
    let eager_source: Option<rdma_display_capture::Running> = if geometry.is_none() {
        eprintln!("[server] no preflight geometry; capturing eagerly to learn it");
        let source: Box<dyn CaptureSource> = match source_name {
            "qemu" => Box::new(QemuDbusSource::new(dbus_socket.clone(), console)),
            "drm" => Box::new(DrmKmsSource::with_sync(drm_card.clone(), capture_sync)),
            other => bail!("source {other:?} not implemented (qemu|drm)"),
        };
        Some(CaptureSource::spawn(source, tx.clone())?)
    } else {
        None
    };
    eprintln!("[server] listening on {listen} (pipeline starts on connect)");

    let listener = TcpListener::bind(listen)?;
    listener.set_nonblocking(true)?;

    let mut session: Option<rdma_display_rdma::ServerLink> = None;
    // when the current session's control connection was accepted, for
    // measuring connect -> first-frame latency
    let mut session_connected_at: Option<std::time::Instant> = None;
    // one-shot: log when the first captured scanout arrives after connect
    let mut first_scanout_logged = false;
    // diagnostic: count send-block entries to time the connect->first-send path
    let mut diag_send_entries: u32 = 0;
    // diagnostic: count accepted connections to time their arrival
    let mut diag_conns: u32 = 0;
    // per-session workers (capture source, input sink, audio); stopped when
    // the session ends so an idle server holds none of them
    let mut workers: Option<SessionWorkers> = None;
    let mut pending_meta: Option<DisplayMeta> = None;
    // fps cap: `--fps 0` = uncapped (consumer-paced via slot credits).
    // Otherwise it is only a safety ceiling — the real pacing is the client's
    // credit/compositor rate; the gate merely bounds the worst case.
    let cap_period: Option<std::time::Duration> = if fps == 0 {
        None
    } else {
        Some(std::time::Duration::from_millis(1000 / fps.max(1)))
    };
    let mut last_send: Option<std::time::Instant> = None;
    let mut frame_dirty = false;
    // t0 of the most recent frame event (pipeline stamp origin)
    let mut t0_unix_us: u64 = 0;
    // latest-wins capture mailbox: the newest scanout waiting to be
    // materialized. Older captures are dropped, so dropped frames cost no GPU
    // work; materialization happens only when the frame is actually sent.
    let mut pending_scanout: Option<rdma_display_capture::DmabufScanout> = None;
    // true once a scanout has been materialized at least once (guards a client
    // that connects before the first capture)
    let mut have_frame = false;
    // display meta currently negotiated with the live session (change detect)
    let mut session_meta: Option<DisplayMeta> = None;
    // input-connection session nonce (0 = no live session); set on control
    // accept and cleared when the session ends
    let session_nonce = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let started = std::time::Instant::now();

    // pipelined (PBO staging) state: frames issued to the materializer whose
    // fence hasn't been finished+posted yet (slot, t0 event, t0d issue time)
    let mut pipelined = false;
    // staging mode is a session property; the texture-direct lossless path
    // never pipelines (no readback), so `pipelined` is gated on it below
    let mut pipelined_capable = false;
    let mut issued: std::collections::VecDeque<(usize, u64, u64, u64, u64)> =
        std::collections::VecDeque::new();
    // pickup time of the frame the send step is currently working on (first
    // time it decided to send it); kept across slot-credit retries so the
    // credit wait lands in `slot`, not `queue`
    let mut send_pickup: Option<u64> = None;
    // prepare-done time of the same frame: first scanout import/materializer
    // prepare only, kept across slot-credit retries so a retry's idle time is
    // not billed to `prep` (mirrors send_pickup)
    let mut send_prep: Option<u64> = None;
    // frame codec state (raw = pass-through); configured per session
    let mut fc = FrameCodec::new();
    // lossy chroma advertised to the client: 0 = 4:2:0, 1 = 4:4:4
    let codec_chroma: u8 =
        if codec == rdma_display_codec::Codec::Lossy && lossy_chroma444 { 1 } else { 0 };

    loop {
        // 0. reap a hung-up client: the control-reader thread signals EOF
        // via the credit channel; without this, an idle capture loop (no
        // frames to send) would never notice, `session` would stay stale,
        // and a reconnecting client would wait forever on accept
        if session.is_some() {
            let gone = session.as_mut().unwrap().client_gone();
            if gone {
                eprintln!("[server] client disconnected");
                session = None;
            }
        }

        // a session that ended (hangup, send failure, Disabled) stops its
        // per-session workers and frees the geometry-sized GPU resources, so
        // an idle server holds none of the heavy pipeline
        if session.is_none() {
            if let Some(w) = workers.take() {
                w.stop();
                mat.release();
                fc.teardown();
                // audio is per-session too: stop it (and the helper) while idle
                if let Some(mut a) = audio_capture.take() {
                    a.stop();
                }
                session_nonce.store(0, std::sync::atomic::Ordering::SeqCst);
                pending_scanout = None;
                have_frame = false;
                frame_dirty = false;
                session_meta = None;
                pipelined = false;
                issued.clear();
                send_pickup = None;
                send_prep = None;
                session_connected_at = None;
                first_scanout_logged = false;
                diag_send_entries = 0;
                diag_conns = 0;
                eprintln!("[server] session ended; pipeline idle");
            }
        }
        session_active.store(session.is_some(), std::sync::atomic::Ordering::Relaxed);

        let gate_open =
            cap_period.map_or(true, |p| last_send.map_or(true, |t| t.elapsed() >= p));
        // 1. wait for something to happen: a capture event, the pacing
        // deadline, or (sessionless) the accept-poll tick. While a session
        // is up and the screen is idle, wait on a short tick so hangups and
        // (future) scanout changes are noticed promptly; credits/input
        // arrive on their own threads and wake nothing here.
        let wait: Option<std::time::Duration> = if session.is_none() {
            Some(std::time::Duration::from_millis(20))
        } else if frame_dirty {
            if pipelined && gate_open && !session.as_ref().unwrap().has_free_slot() {
                // frame ready but no staging slot free yet: wait for a client
                // credit on a fast tick (not a busy spin, not the cap period)
                Some(std::time::Duration::from_millis(2))
            } else {
                Some(
                    cap_period
                        .map(|p| p.saturating_sub(
                            last_send.map_or(p, |t| t.elapsed()),
                        ))
                        .unwrap_or_default(), // uncapped: frame is ready now
                )
            }
        } else if pipelined && !issued.is_empty() {
            Some(std::time::Duration::ZERO) // finish+post work pending
        } else {
            Some(std::time::Duration::from_millis(50))
        };
        match wait {
            None => match rx.recv() {
                Ok(ev) => {
                    handle_capture_event(
                        ev, &mut mat, &mut session, &mut pending_meta,
                        &mut t0_unix_us, &mut frame_dirty, &screen,
                        &mut pending_scanout,
                    )?;
                }
                Err(_) => bail!("capture source ended"),
            },
            Some(d) if d.is_zero() => {} // don't wait; send the ready frame
            Some(d) => match rx.recv_timeout(d) {
                Ok(ev) => {
                    handle_capture_event(
                        ev, &mut mat, &mut session, &mut pending_meta,
                        &mut t0_unix_us, &mut frame_dirty, &screen,
                        &mut pending_scanout,
                    )?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    bail!("capture source ended")
                }
            },
        }

        // 2. drain any backlog (each event latches state; newest t0 wins)
        loop {
            match rx.try_recv() {
                Ok(ev) => {
                    handle_capture_event(
                        ev, &mut mat, &mut session, &mut pending_meta,
                        &mut t0_unix_us, &mut frame_dirty, &screen,
                        &mut pending_scanout,
                    )?;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => bail!("capture source ended"),
            }
        }

        // diagnostic: when the first captured scanout lands after a connect,
        // attribute the connect -> first-frame latency
        if session.is_some() && !first_scanout_logged && pending_scanout.is_some() {
            first_scanout_logged = true;
            if let Some(t) = session_connected_at {
                eprintln!(
                    "[session] first scanout {} ms after connect",
                    t.elapsed().as_millis()
                );
            }
        }

        // 3. accept new connections and route by tag byte (audio uses a
        // SECOND connection to the same port; the control connection is
        // byte-compatible as before)
        if let Ok((mut stream, addr)) = listener.accept() {
            let _ = stream.set_nodelay(true);
            // explicit first-byte routing tag (control/input/audio); a short
            // timeout keeps a silent connection from stalling the capture loop
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
            let mut tag = [0u8; 1];
            let t_tag = std::time::Instant::now();
            let tag_ok = std::io::Read::read_exact(&mut stream, &mut tag).is_ok();
            if t_tag.elapsed().as_millis() > 50 {
                eprintln!(
                    "[diag] tag read from {addr} took {} ms (stalls the loop)",
                    t_tag.elapsed().as_millis()
                );
            }
            let _ = stream.set_read_timeout(None);
            if let Some(t) = session_connected_at {
                if diag_conns < 12 {
                    diag_conns += 1;
                    eprintln!(
                        "[diag] conn #{diag_conns} from {addr} tag={:#04x} at {} ms",
                        tag[0],
                        t.elapsed().as_millis()
                    );
                }
            }
            if !tag_ok {
                eprintln!("[server] connection with no routing tag dropped ({addr})");
            } else if tag[0] == rdma_display_audio::wire::TAG_AUDIO {
                match audio_capture.as_ref() {
                    Some(mbox) => {
                        if audio_busy
                            .compare_exchange(false, true, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
                            .is_ok()
                        {
                            eprintln!("[audio] client stream connected ({addr})");
                            let mbox = mbox.mailbox.clone();
                            let busy = audio_busy.clone();
                            std::thread::spawn(move || {
                                if let Err(e) =
                                    rdma_display_audio::capture::serve_connection(stream, &mbox)
                                {
                                    eprintln!("[audio] stream ended: {e:#}");
                                }
                                busy.store(false, std::sync::atomic::Ordering::SeqCst);
                            });
                        } else {
                            eprintln!("[audio] second audio connection refused ({addr})");
                        }
                    }
                    None => {
                        eprintln!(
                            "[server] audio connection but audio capture is unavailable: dropped ({addr})"
                        );
                    }
                }
            } else if tag[0] == rdma_display_proto::TAG_INPUT {
                // input has its own connection + thread, independent of the
                // video/control path; the nonce binds it to the live session
                let nonce_cell = session_nonce.clone();
                match workers.as_ref() {
                    Some(w) => {
                        let input_tx = w.input_tx.clone();
                        let _ = std::thread::Builder::new()
                            .name("input-reader".into())
                            .spawn(move || serve_input(stream, nonce_cell, input_tx, addr));
                    }
                    None => eprintln!(
                        "[input] input connection with no active session: dropped ({addr})"
                    ),
                }
            } else if tag[0] == rdma_display_proto::TAG_VIDEO {
                // client's TCP video socket (negotiated transport == TCP)
                match session.as_mut() {
                    Some(s)
                        if s.transport() == rdma_display_rdma::Transport::Tcp
                            && !s.has_video() =>
                    {
                        s.attach_video(stream);
                        if let Some(t) = session_connected_at {
                            eprintln!(
                                "[session] video socket attached {} ms after connect",
                                t.elapsed().as_millis()
                            );
                        }
                    }
                    _ => {
                        eprintln!("[server] unexpected TAG_VIDEO connection dropped ({addr})")
                    }
                }
            } else if session.is_none() {
                if let Some(meta) = geometry.clone().or_else(|| pending_meta.clone()) {
                    let t_connect = std::time::Instant::now();
                    eprintln!("client connected: {}", stream.peer_addr()?);
                    // start the per-session pipeline (source, input, audio)
                    let (input_tx, input_rx) = mpsc::channel::<rdma_display_proto::InputEvent>();
                    let sink: Box<dyn InputSink> = match input_name {
                        "qemu" => Box::new(QemuDbusInput::new(dbus_socket.clone(), console)),
                        "uinput" => Box::new(rdma_display_capture::UinputSink),
                        other => bail!("input sink {other:?} not implemented (qemu|uinput)"),
                    };
                    let input =
                        InputSink::spawn(sink, input_rx, screen.clone(), rumble_hub.clone())?;
                    eprintln!("[server] input sink: {input_name}");
                    let source = if eager_source.is_some() {
                        None
                    } else {
                        let source: Box<dyn CaptureSource> = match source_name {
                            "qemu" => Box::new(QemuDbusSource::new(dbus_socket.clone(), console)),
                            "drm" => {
                                Box::new(DrmKmsSource::with_sync(drm_card.clone(), capture_sync))
                            }
                            other => bail!("source {other:?} not implemented (qemu|drm)"),
                        };
                        Some(CaptureSource::spawn(source, tx.clone())?)
                    };
                    eprintln!("[server] capture running");
                    if input_test {
                        spawn_input_test(input_tx.clone())?;
                    }
                    if audio_capture.is_none() {
                        audio_capture = start_audio(audio, audio_device);
                    }
                    workers = Some(SessionWorkers {
                        source,
                        input,
                        input_tx: input_tx.clone(),
                    });
                    let nonce = rdma_display_rdma::random_nonce();
                    session_nonce.store(nonce, std::sync::atomic::Ordering::SeqCst);
                    let geom = rdma_display_codec::Geometry::new(
                        meta.width,
                        meta.height,
                        meta.stride,
                    );
                    let slot_len = match codec {
                        rdma_display_codec::Codec::Lossy => lossy_budget + (1 << 20),
                        _ => codec.slot_len(&geom).expect("implemented codec has a slot size"),
                    };
                    eprintln!(
                        "[server] codec {} → slot {} B (raw frame {} B)",
                        codec.name(),
                        slot_len,
                        meta.frame_bytes
                    );
                    // zero-copy staging (PBO readback): bind the materializer's
                    // persistent-mapped PBOs as the staging slots; None = fall
                    // back to verbs-allocated slots + direct readback
                    let external = if rdma.is_some()
                        && transport != rdma_display_rdma::TransportPref::Tcp
                        && codec != rdma_display_codec::Codec::Lossy
                        && readback == "pbo"
                        && materialize == "egl"
                    {
                        match mat.bind_staging(slots, slot_len) {
                            Some(v) => Some(v),
                            None => {
                                eprintln!("[server] PBO staging unavailable — direct readback slots");
                                None
                            }
                        }
                    } else {
                        None
                    };
                    // client input events join the same channel as the
                    // synthetic --input-test injector -> capture forwarder
                    match rdma_display_rdma::ServerLink::accept(
                        Conn::new(stream),
                        transport,
                        rdma,
                        &meta,
                        slots,
                        codec,
                        slot_len,
                        codec_chroma,
                        input_tx.clone(),
                        external.as_deref(),
                        &rumble_hub,
                        nonce,
                    ) {
                        Ok(link) => {
                            eprintln!("session up: {slots} slots, {} frame bytes", link.frame_bytes());
                            // the new client starts with zeroed slots: make
                            // the current scanout content sendable even
                            // though the guest hasn't repainted since the
                            // previous session's last frame
                            mat.mark_dirty();
                            // fresh codec reference: the new client's first
                            // frame must be full (no delta against a frame it
                            // never received)
                            fc.configure(
                                codec,
                                meta.width,
                                meta.height,
                                meta.stride,
                                meta.frame_bytes as usize,
                                gl.as_ref(),
                            )?;
                            // configure may have created/activated the shared
                            // GL context; restore the materializer's view
                            mat.activate()?;
                            frame_dirty = true;
                            pipelined_capable = external.is_some()
                                && link.transport() == rdma_display_rdma::Transport::Rdma;
                            pipelined = pipelined_capable;
                            session_meta = Some(meta.clone());
                            session = Some(link);
                            session_connected_at = Some(t_connect);
                            eprintln!(
                                "[session] setup {} ms (workers + link)",
                                t_connect.elapsed().as_millis()
                            );
                        }
                        Err(e) => {
                            eprintln!("[server] session setup failed: {e:#}");
                            session_nonce.store(0, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                }
            } else {
                eprintln!(
                    "[server] control connection while a session is active: dropped ({addr})"
                );
            }
        }

        // 4. send the newest frame if dirty and the (optional) cap allows.
        // Pipelined (PBO staging active): ISSUE the newest frame's readback
        // first (µs; the GPU DMA starts immediately), THEN finish+post the
        // oldest issued frame — its ~2.8 ms RDMA transfer overlaps the GPU
        // work of the frame just issued.
        if frame_dirty && session.is_some() && gate_open {
            let t_block = std::time::Instant::now();
            if diag_send_entries < 10 {
                if let Some(t) = session_connected_at {
                    diag_send_entries += 1;
                    eprintln!(
                        "[diag] send block #{} entered at {} ms after connect",
                        diag_send_entries,
                        t.elapsed().as_millis()
                    );
                }
            }
            // Pickup time: when the sender decided to send the frame. The
            // fps gate and event-queue wait before this point are `queue`
            // time; a wait for a staging slot after it is `slot` time.
            // Keeping the FIRST pickup across retries keeps a credit wait out
            // of `queue` (the serial path splits it the same way).
            let t0d =
                (*send_pickup.get_or_insert_with(rdma_display_proto::unix_now_us)).max(t0_unix_us);
            let link = session.as_mut().unwrap();
            let slot_bytes = link.frame_bytes();
            // Materialize the newest pending capture now — only now do we know
            // it will actually be sent. Everything superseded while a slot was
            // busy was already dropped by the latest-wins mailbox.
            match pending_scanout.take() {
                Some(d) => {
                    let fmt = d.format.clone();
                    let fb = fmt.frame_bytes() as usize;
                    // raw and lossless read the raw frame back into the slot,
                    // so it must be big enough to hold it. lossy encodes from
                    // the imported dmabuf into the slot (budget + 1 MiB), which
                    // is deliberately smaller than the raw frame, so the raw
                    // size is irrelevant there.
                    if codec != rdma_display_codec::Codec::Lossy && fb > slot_bytes {
                        eprintln!(
                            "[server] scanout {}x{} fourcc={:#010x} '{}' ({} B) exceeds negotiated slot ({} B); dropped",
                            fmt.width,
                            fmt.height,
                            fmt.fourcc,
                            rdma_display_capture::fourcc_name(fmt.fourcc),
                            fb,
                            slot_bytes
                        );
                        frame_dirty = false;
                    } else {
                        // a resized scanout invalidates the codec's tile
                        // geometry; rebuild it (resets the reference, so the
                        // next frame is full)
                        if fc.geometry_stale(fmt.width, fmt.height, fmt.stride) {
                            fc.configure(
                                codec,
                                fmt.width,
                                fmt.height,
                                fmt.stride,
                                fmt.frame_bytes() as usize,
                                gl.as_ref(),
                            )?;
                            mat.activate()?;
                        }
                        // Lossless on the GPU: import the scanout into the
                        // encoder and encode straight from GPU memory. When this
                        // succeeds we never touch the materializer — its
                        // per-frame dmabuf re-import would dominate the frame.
                        let mut texture_ok = false;
                        if codec == rdma_display_codec::Codec::Lossless && fc.has_gpu() {
                            match fc.set_scanout(&d) {
                                Ok(()) => {
                                    texture_ok = true;
                                    // texture-direct has no readback to pipeline
                                    pipelined = false;
                                }
                                Err(e) => {
                                    if !fc.scanout_warned {
                                        eprintln!(
                                            "[server] lossless texture-direct unavailable ({e:#}); using readback+encode"
                                        );
                                        fc.scanout_warned = true;
                                    }
                                    fc.clear_scanout();
                                }
                            }
                            // set_scanout made the codec's EGL context current
                            mat.activate()?;
                        }
                        if texture_ok {
                            have_frame = true;
                        } else {
                            // materializer path: raw/lossy, or the lossless
                            // fallback (readback + encode)
                            mat.activate()?;
                            match mat.prepare(&fmt, Some(&d)) {
                                Ok(()) => {
                                    have_frame = true;
                                    pipelined = pipelined_capable;
                                }
                                Err(e) => {
                                    // never tear down a session over a bad
                                    // capture: hold the last frame and resume
                                    // when the format returns
                                    eprintln!(
                                        "[server] prepare scanout failed: {e:#}; holding last frame"
                                    );
                                    frame_dirty = false;
                                }
                            }
                        }
                    }
                }
                // a client connected before the first capture: wait for one
                None if !have_frame => frame_dirty = false,
                None => {}
            }
            // The scanout import / materializer prepare is done. Stamped here
            // (not inside the link) so `queue = t0d - t0` stays the pacing wait
            // only: the import is `t0p - t0d`, the credit wait `t1s - t0p`.
            let t0p = *send_prep.get_or_insert_with(rdma_display_proto::unix_now_us);
            // Push a changed display meta (separate message, only when it
            // changes). Only same-size changes are safe without renegotiating
            // slots; a size change is remembered so we don't re-log per frame.
            if frame_dirty {
                if let Some(nm) = pending_meta.as_ref() {
                    if session_meta.as_ref() != Some(nm) {
                        if nm.frame_bytes as usize <= slot_bytes {
                            let mut m = nm.clone();
                            m.gen = session_meta.as_ref().map(|o| o.gen + 1).unwrap_or(1);
                            let _ = link.send_meta(&m);
                            eprintln!(
                                "[server] display meta gen {} ({}x{} stride {} fourcc {:#010x})",
                                m.gen, m.width, m.height, m.stride, m.fourcc
                            );
                            session_meta = Some(m);
                        } else {
                            session_meta = Some(nm.clone());
                        }
                    }
                }
            }
            if frame_dirty {
            let t_before_tx = std::time::Instant::now();
            if pipelined {
                if let Some(slot) = link.try_claim_slot() {
                    // slot claimed: pickup -> here is the slot-credit wait
                    let t1s = rdma_display_proto::unix_now_us();
                    let out = link.slot_bytes_mut(slot);
                    match mat.issue_frame(slot, out) {
                        Ok(true) => {
                            issued.push_back((slot, t0_unix_us, t0d, t0p, t1s));
                            frame_dirty = false;
                        }
                        Ok(false) => {
                            frame_dirty = false;
                            link.release_slot(slot);
                        }
                        Err(e) => {
                            eprintln!("[server] issue failed: {e:#}");
                            link.release_slot(slot);
                            session = None;
                            mat.drop_frames();
                            issued.clear();
                        }
                    }
                }
            } else {
                match link.send_frame_checked(
                    |out| {
                        if fc.uses_scanout() {
                            // lossless texture-direct: encode from the imported
                            // dmabuf, write the coded delta into the slot
                            let r = fc.encode_scanout(out)?;
                            // the codec's EGL context is current; restore ours
                            let t = std::time::Instant::now();
                            mat.activate()?;
                            fc.mat_activate_us = t.elapsed().as_micros() as u64;
                            return Ok(r);
                        }
                        let (changed, t) = mat.copy_frame(out)?;
                        if !changed {
                            return Ok((false, 0, t));
                        }
                        let n = match mat.coded_len() {
                            Some(n) => n,
                            None => {
                                let n = fc.encode_in_place(out)?;
                                // GPU encode leaves its own EGL context current
                                mat.activate()?;
                                n
                            }
                        };
                        Ok((true, n, t))
                    },
                    t0_unix_us,
                    t0d,
                    t0p,
                ) {
                    Ok(Some(seq)) => {
                        frame_dirty = false;
                        last_send = Some(std::time::Instant::now());
                        eprintln!("[server] frame {} sent", seq);
                        if seq == 0 {
                            eprintln!(
                                "[session] first frame block: materialize {} ms, encode+write {} ms",
                                t_before_tx.duration_since(t_block).as_millis(),
                                t_before_tx.elapsed().as_millis()
                            );
                            if let Some(t) = session_connected_at {
                                eprintln!(
                                    "[session] first frame {} ms after connect",
                                    t.elapsed().as_millis()
                                );
                            }
                        }
                        if seq % 60 == 0 {
                            eprintln!("[stats] frames={} uptime={:?}", seq, started.elapsed());
                        }
                    }
                    Ok(None) => {
                        if let Some(t) = session_connected_at {
                            if diag_send_entries <= 10 {
                                eprintln!(
                                    "[diag] send -> none (no video / unchanged) at {} ms",
                                    t.elapsed().as_millis()
                                );
                            }
                        }
                        frame_dirty = false;
                    }
                    Err(e) => {
                        eprintln!("[server] send failed: {e:#}");
                        session = None;
                    }
                }
            }
            } // if frame_dirty
        }
        // a resolved frame ends the pickup window (a still-dirty frame keeps
        // its pickup time so its slot-credit wait is not billed to `queue`)
        if !frame_dirty {
            send_pickup = None;
            send_prep = None;
        }
        // 5. pipelined: finish + post the oldest issued frame (FIFO preserves
        // frame order; the fence wait is mostly elapsed because the GPU DMA
        // overlapped the previous frame's transfer)
        if pipelined && session.is_some() {
            while let Some((slot, f_t0, f_t0d, f_t0p, f_t1s)) = issued.front().copied() {
                let link = session.as_mut().unwrap();
                let out = link.slot_bytes_mut(slot);
                let (changed, mat_times) = match mat.finish_frame(slot, out) {
                    Ok(x) => x,
                    Err(e) => {
                        eprintln!("[server] finish failed: {e:#}");
                        session = None;
                        mat.drop_frames();
                        issued.clear();
                        break;
                    }
                };
                issued.pop_front();
                if !changed {
                    link.release_slot(slot);
                    continue;
                }
                // compress the raw frame now that it is in the pinned slot
                let post_len = match fc.encode_in_place(out) {
                    Ok(n) => {
                        // GPU encode leaves its own EGL context current
                        mat.activate()?;
                        n
                    }
                    Err(e) => {
                        eprintln!("[server] encode failed: {e:#}");
                        session = None;
                        mat.drop_frames();
                        issued.clear();
                        break;
                    }
                };
                match link.post_filled(
                    slot,
                    f_t0,
                    f_t0d,
                    f_t0p,
                    f_t1s,
                    rdma_display_proto::unix_now_us(),
                    post_len,
                    mat_times,
                ) {
                    Ok(Some(seq)) => {
                        last_send = Some(std::time::Instant::now());
                        eprintln!("[server] frame {} sent", seq);
                        if seq == 0 {
                            if let Some(t) = session_connected_at {
                                eprintln!(
                                    "[session] first frame {} ms after connect",
                                    t.elapsed().as_millis()
                                );
                            }
                        }
                        if seq % 60 == 0 {
                            eprintln!("[stats] frames={} uptime={:?}", seq, started.elapsed());
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("[server] send failed: {e:#}");
                        session = None;
                        mat.drop_frames();
                        issued.clear();
                        break;
                    }
                }
            }
        }
    }
}

/// In-place frame coding for the RDMA staging slot. `raw` is a pass-through
/// (post the whole frame, no header). `lossless` runs the tile-delta encoder
/// on the GPU (`rdma-display-codec-gpu`, GLES 3.1 compute) against the frame
/// the materializer just read back; the CPU reference encoder is kept as a
/// fallback when the GPU codec can't be created. The encoder's reference is
/// reset in `configure` (once per session, and on a geometry change) so the
/// first frame a new client receives is always full.
///
/// The GPU encoder owns its own EGL context, so after it has run on the server
/// thread the caller must restore the materializer's context via
/// [`FrameMaterializer::activate`](rdma_display_capture::FrameMaterializer::activate).
struct FrameCodec {
    /// GPU lossless encoder (GLES compute); None = use the CPU encoder.
    gpu: Option<rdma_display_codec_gpu::GpuCodec>,
    /// CPU reference encoder: the fallback path.
    enc: Option<rdma_display_codec::Encoder>,
    buf: Vec<u8>,
    frame_bytes: usize,
    geom: Option<rdma_display_codec::Geometry>,
    /// the next encode must be a full frame (new session / new geometry)
    first: bool,
    /// lossless GPU: a scanout dmabuf is imported for texture-direct encode
    /// (no readback/re-upload). False = use the readback+encode path.
    scanout: bool,
    /// set once the texture-direct import failed (avoid per-frame log spam)
    scanout_warned: bool,
    /// periodic texture-direct breakdown: [activate, blit, encode, ref, copy,
    /// mat_restore] in µs for the last interval (reported as a p50)
    stat_buf: Vec<[u64; 6]>,
    /// time the caller spent restoring the materializer context last frame
    mat_activate_us: u64,
    /// import-cache hit/miss counters (cumulative)
    import_hits: u64,
    import_misses: u64,
    /// one-shot startup benchmark: iterations left (RDMA_LOSSLESS_BENCH=N)
    bench_left: u32,
    /// run the memprobe every N encodes (0 = off; RDMA_MEMPROBE=N)
    memprobe_every: u32,
    enc_frames: u64,
}

impl FrameCodec {
    fn new() -> Self {
        Self {
            gpu: None,
            enc: None,
            buf: Vec::new(),
            frame_bytes: 0,
            geom: None,
            first: true,
            scanout: false,
            scanout_warned: false,
            stat_buf: Vec::new(),
            mat_activate_us: 0,
            import_hits: 0,
            import_misses: 0,
            bench_left: std::env::var("RDMA_LOSSLESS_BENCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            memprobe_every: std::env::var("RDMA_MEMPROBE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            enc_frames: 0,
        }
    }

    /// Drop per-session encoder state so an idle server holds no codec
    /// buffers; the next session reconfigures.
    fn teardown(&mut self) {
        self.gpu = None;
        self.enc = None;
        self.geom = None;
        self.first = true;
        self.scanout = false;
        self.frame_bytes = 0;
        self.buf.clear();
    }

    fn configure(
        &mut self,
        codec: rdma_display_codec::Codec,
        width: u32,
        height: u32,
        stride: u32,
        frame_bytes: usize,
        gl: Option<&Arc<rdma_display_egl::SharedContext>>,
    ) -> Result<()> {
        self.frame_bytes = frame_bytes;
        self.geom = None;
        self.gpu = None;
        self.enc = None;
        self.first = true;
        self.scanout = false;
        match codec {
            rdma_display_codec::Codec::Raw | rdma_display_codec::Codec::Lossy => {}
            rdma_display_codec::Codec::Lossless => {
                let geom = rdma_display_codec::Geometry::new(width, height, stride);
                self.buf.resize(geom.max_coded_len(), 0);
                self.geom = Some(geom);
                let gpu = match gl {
                    Some(gl) => rdma_display_codec_gpu::GpuCodec::new_shared(geom, gl.clone()),
                    None => rdma_display_codec_gpu::GpuCodec::new(geom),
                };
                match gpu {
                    Ok(g) => {
                        eprintln!("[server] lossless encoder: GPU (GLES compute)");
                        self.gpu = Some(g);
                    }
                    Err(e) => {
                        eprintln!(
                            "[server] lossless encoder: GPU unavailable ({e:#}); falling back to CPU"
                        );
                        let mut e = rdma_display_codec::Encoder::new(geom)?;
                        // streaming path: the payload CRC costs tens of ms on a
                        // full frame and RC RDMA already guarantees delivery
                        e.set_crc(false);
                        self.enc = Some(e);
                    }
                }
            }
        }
        Ok(())
    }

    /// True when the active lossless encoder was built for a different
    /// geometry than `(width, height, stride)` and must be recreated (which
    /// resets its reference, so the next frame is full).
    fn geometry_stale(&self, width: u32, height: u32, stride: u32) -> bool {
        match self.geom {
            None => false,
            Some(g) => g.width != width || g.height != height || g.stride != stride,
        }
    }

    /// Encode `slot[..frame_bytes]` in place; returns the number of bytes to
    /// post (coded length, or `frame_bytes` for raw). When the GPU encoder
    /// ran, the materializer's GL context must be restored afterwards.
    fn encode_in_place(&mut self, slot: &mut [u8]) -> Result<usize> {
        if let Some(gpu) = self.gpu.as_mut() {
            let full = self.first;
            gpu.activate()?;
            gpu.upload(&slot[..self.frame_bytes])?;
            let n = gpu.encode(full)?;
            // advance the encoder's reference to the frame just coded, so the
            // next encode deltas against it (the decode pass writes the coded
            // tiles back into the reference buffer)
            gpu.update_reference()?;
            ensure!(
                n <= slot.len(),
                "coded frame {n} exceeds staging slot {}",
                slot.len()
            );
            let coded = gpu.coded(n);
            slot[..n].copy_from_slice(coded);
            self.first = false;
            return Ok(n);
        }
        let Some(enc) = self.enc.as_mut() else {
            return Ok(self.frame_bytes);
        };
        let n = enc.encode(&slot[..self.frame_bytes], &mut self.buf)?;
        ensure!(
            n <= slot.len(),
            "coded frame {n} exceeds staging slot {}",
            slot.len()
        );
        slot[..n].copy_from_slice(&self.buf[..n]);
        Ok(n)
    }

    fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Whether the texture-direct path is active for this frame.
    fn uses_scanout(&self) -> bool {
        self.scanout && self.gpu.is_some()
    }

    fn clear_scanout(&mut self) {
        self.scanout = false;
    }

    /// Import the scanout dmabuf into the GPU encoder so [`encode_scanout`]
    /// encodes straight from GPU memory — the readback to host and the
    /// re-upload are both skipped. Errors are returned so the caller keeps the
    /// readback+encode path (and only warns once).
    fn set_scanout(&mut self, d: &rdma_display_capture::DmabufScanout) -> Result<()> {
        let Some(gpu) = self.gpu.as_mut() else {
            anyhow::bail!("no GPU encoder");
        };
        if d.fds.len() != 1 {
            anyhow::bail!(
                "lossless texture-direct needs a single-plane dmabuf (got {})",
                d.fds.len()
            );
        }
        use std::os::fd::AsRawFd;
        gpu.set_scanout(&rdma_display_codec_gpu::DmabufImage {
            key: d.fb_id as u64,
            fd: d.fds[0].as_raw_fd(),
            offset: d.offsets.first().copied().unwrap_or(0),
            stride: d.strides.first().copied().unwrap_or(d.format.stride),
            fourcc: d.format.fourcc,
            modifier: d.format.modifier,
            width: d.format.width,
            height: d.format.height,
        })?;
        let (h, m) = gpu.cache_stats();
        self.import_hits = h;
        self.import_misses = m;
        if !self.scanout {
            eprintln!(
                "[server] lossless texture-direct: {} scanout (modifier={:#x}) — no readback/re-upload",
                if gpu.has_fetch_encode() {
                    "linear → texelFetch"
                } else {
                    "tiled → external sampler"
                },
                d.format.modifier,
            );
        }
        self.scanout = true;
        Ok(())
    }

    /// Texture-direct lossless encode: blit the imported scanout into the
    /// encoder's RGBA8 source image, encode against the reference, and write
    /// the coded delta into the staging slot. `changed` follows the encoder's
    /// changed-tile count (0 = idle delta, nothing to send). The codec's EGL
    /// context is left current; the caller restores the materializer's.
    fn encode_scanout(
        &mut self,
        out: &mut [u8],
    ) -> Result<(bool, usize, rdma_display_proto::MatTimes)> {
        let full = self.first;
        let Some(gpu) = self.gpu.as_mut() else {
            anyhow::bail!("no GPU encoder");
        };
        let t = std::time::Instant::now();
        gpu.activate()?;
        let act = t.elapsed();
        // source path: linear scanout -> texelFetch (fast); else the external
        // sampler (no blit either way); else blit + image encode
        let fetch = gpu.has_fetch_encode();
        let ext = gpu.has_ext_encode();
        let t = std::time::Instant::now();
        if !fetch && !ext {
            gpu.blit_source()?;
        }
        let blit = t.elapsed();
        let t = std::time::Instant::now();
        let n = if fetch {
            gpu.encode_fetch(full)?
        } else if ext {
            gpu.encode_ext(full)?
        } else {
            gpu.encode_tex(full)?
        };
        let enc = t.elapsed();
        let count = gpu.last_count();
        let changed = full || count > 0;
        let t = std::time::Instant::now();
        if changed {
            // The encode shader advances the reference in-pass (it writes each
            // coded tile's words back into `ref`), so there is no separate
            // decode-to-reference pass: the previous ref up-call was a second
            // pass over the coded payload (a PCIe read for every changed tile).
            self.first = false;
        }
        let rf = t.elapsed();
        anyhow::ensure!(
            n <= out.len(),
            "coded frame {n} exceeds staging slot {}",
            out.len()
        );
        let t = std::time::Instant::now();
        let coded = gpu.coded(n);
        out[..n].copy_from_slice(coded);
        let copy = t.elapsed();
        if self.bench_left > 0 {
            let n_iter = self.bench_left;
            self.bench_left = 0;
            self.bench_scanout_read(n_iter)?;
        }
        self.enc_frames += 1;
        if self.memprobe_every > 0 && self.enc_frames % self.memprobe_every as u64 == 0 {
            eprintln!(
                "[memprobe] --- encode #{} | gpu: {} ---",
                self.enc_frames,
                gpu_clocks()
            );
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.memprobe()?;
            }
        }
        self.stat_buf.push([
            act.as_micros() as u64,
            blit.as_micros() as u64,
            enc.as_micros() as u64,
            rf.as_micros() as u64,
            copy.as_micros() as u64,
            self.mat_activate_us,
        ]);
        if self.stat_buf.len() >= 120 {
            let mut med = [0u64; 6];
            let mut mx = [0u64; 6];
            for c in 0..6 {
                let mut v: Vec<u64> = self.stat_buf.iter().map(|r| r[c]).collect();
                v.sort_unstable();
                med[c] = v[v.len() / 2];
                mx[c] = v[v.len() - 1];
            }
            let sum: u64 = med.iter().sum();
            eprintln!(
                "[server] lossless texture p50/xmax us: activate={}/{} blit={}/{} encode={}/{} ref={}/{} copy={}/{} mat_restore={}/{} | p50 sum={} | import hit={} miss={} | tiles={} coded={} B",
                med[0], mx[0], med[1], mx[1], med[2], mx[2], med[3], mx[3],
                med[4], mx[4], med[5], mx[5], sum,
                self.import_hits, self.import_misses, count, n,
            );
            self.stat_buf.clear();
        }
        Ok((
            changed,
            n,
            rdma_display_proto::MatTimes::new(
                rdma_display_proto::MatKind::Lossless,
                [
                    act.as_micros() as u64,
                    blit.as_micros() as u64,
                    enc.as_micros() as u64,
                    rf.as_micros() as u64,
                    copy.as_micros() as u64,
                    // context restore is measured after this returns; report the
                    // running value (same staleness as the p50 line above)
                    self.mat_activate_us,
                ],
            ),
        ))
    }

    /// One-shot startup diagnostic (`RDMA_LOSSLESS_BENCH=N`): time N
    /// detection-only reads of the imported real scanout through the active
    /// path, after the first frame has seeded the reference. Compare with
    /// `codecbench --gbm` (a synthetic tiled dma-buf) to see whether the cost
    /// is the tiled layout or the compositor's buffer itself.
    fn bench_scanout_read(&mut self, n: u32) -> Result<()> {
        let Some(gpu) = self.gpu.as_mut() else {
            return Ok(());
        };
        gpu.activate()?;
        let path = if gpu.has_fetch_encode() {
            "fetch"
        } else if gpu.has_ext_encode() {
            "ext"
        } else {
            "blit+tex"
        };
        let mut us = Vec::with_capacity(n as usize);
        for _ in 0..n {
            gpu.sync();
            let t = std::time::Instant::now();
            match path {
                "fetch" => {
                    gpu.encode_fetch(false)?;
                }
                "ext" => {
                    gpu.encode_ext(false)?;
                }
                _ => {
                    gpu.blit_source()?;
                    gpu.encode_tex(false)?;
                }
            }
            us.push(t.elapsed().as_micros() as u64);
        }
        us.sort_unstable();
        eprintln!(
            "[server] real-scanout read bench (path={path}, n={n}): p50={}us min={}us max={}us",
            us[us.len() / 2],
            us[0],
            us[us.len() - 1],
        );
        Ok(())
    }
}

/// Input-connection worker: validate the session nonce, forward input events
/// to the injection sink, and release everything the client held when the
/// socket dies — a network blip must never leave a key stuck down.
/// Best-effort GPU clocks/power for the memprobe, to spot throttling.
fn gpu_clocks() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "n/a".to_string())
}

fn serve_input(
    mut stream: std::net::TcpStream,
    nonce_cell: std::sync::Arc<std::sync::atomic::AtomicU64>,
    input_tx: mpsc::Sender<rdma_display_proto::InputEvent>,
    addr: std::net::SocketAddr,
) {
    let mut nb = [0u8; 8];
    if std::io::Read::read_exact(&mut stream, &mut nb).is_err() {
        return;
    }
    let nonce = u64::from_le_bytes(nb);
    let expect = nonce_cell.load(std::sync::atomic::Ordering::SeqCst);
    if expect == 0 || nonce != expect {
        eprintln!("[input] rejected input connection with stale/invalid nonce ({addr})");
        return;
    }
    eprintln!("[input] client input stream connected ({addr})");
    let mut conn = Conn::new(stream);
    let mut keys: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut btns: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut gp_btns: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut gp_axes: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    loop {
        use rdma_display_proto::InputEvent as I;
        match conn.recv() {
            Ok(rdma_display_proto::Message::Input(ev)) => {
                match ev {
                    I::Key { code, down } => {
                        if down {
                            keys.insert(code);
                        } else {
                            keys.remove(&code);
                        }
                    }
                    I::Btn { button, down } => {
                        if down {
                            btns.insert(button);
                        } else {
                            btns.remove(&button);
                        }
                    }
                    I::GamepadBtn { slot, code, down } => {
                        if down {
                            gp_btns.insert((slot, code));
                        } else {
                            gp_btns.remove(&(slot, code));
                        }
                    }
                    I::GamepadAxis { slot, code, .. } => {
                        gp_axes.insert((slot, code));
                    }
                    I::Abs { .. } | I::Rel { .. } => {}
                }
                if input_tx.send(ev).is_err() {
                    break;
                }
            }
            Ok(rdma_display_proto::Message::Bye(_)) => break,
            Ok(other) => eprintln!("[input] unexpected message {other:?}"),
            Err(_) => break,
        }
    }
    use rdma_display_proto::InputEvent as I;
    for code in keys {
        let _ = input_tx.send(I::Key { code, down: false });
    }
    for button in btns {
        let _ = input_tx.send(I::Btn { button, down: false });
    }
    for (slot, code) in gp_btns {
        let _ = input_tx.send(I::GamepadBtn { slot, code, down: false });
    }
    for (slot, code) in gp_axes {
        let _ = input_tx.send(I::GamepadAxis { slot, code, value: 0 });
    }
    eprintln!("[input] client input stream ended ({addr}); released held input");
}

/// Latch one capture event into materializer/session state. Shared by the
/// blocking wait and the backlog drain (newest frame event wins for t0).
#[allow(clippy::too_many_arguments)]
fn handle_capture_event(
    ev: CaptureEvent,
    mat: &mut Box<dyn FrameMaterializer>,
    session: &mut Option<rdma_display_rdma::ServerLink>,
    pending_meta: &mut Option<DisplayMeta>,
    t0_unix_us: &mut u64,
    frame_dirty: &mut bool,
    screen: &Arc<ScreenSize>,
    pending_scanout: &mut Option<rdma_display_capture::DmabufScanout>,
) -> Result<()> {
    match ev {
        CaptureEvent::Scanout(dmabuf, t0) => {
            let fmt = dmabuf.format.clone();
            // latest-wins mailbox: keep only the newest capture. The GPU
            // import/blit/readback happens at send time (see the send step),
            // so a frame that gets superseded by a newer capture costs nothing.
            *pending_scanout = Some(dmabuf);
            screen.set(fmt.width, fmt.height);
            *pending_meta = Some(meta_linear(&fmt));
            // t0 comes from the source: for DRM it is the flip that put this
            // (one-behind) frame on screen, ~one frame before the poll noticed
            *t0_unix_us = t0;
            *frame_dirty = true;
        }
        CaptureEvent::PlainScanout(fmt, data, t0) => {
            *t0_unix_us = t0;
            mat.prepare(&fmt, None).context("prepare plain scanout")?;
            screen.set(fmt.width, fmt.height);
            mat.handle_event(CaptureEvent::PlainUpdate(
                rdma_display_capture::DirtyRect {
                    x: 0,
                    y: 0,
                    w: fmt.width as i32,
                    h: fmt.height as i32,
                },
                data,
                fmt.stride,
                fmt.fourcc,
                t0,
            ))?;
            *pending_meta = Some(meta_plain(&fmt));
            *frame_dirty = true;
        }
        update @ CaptureEvent::PlainUpdate(..) => {
            if let CaptureEvent::PlainUpdate(.., t0) = &update {
                *t0_unix_us = *t0;
            }
            if mat.handle_event(update)? {
                *frame_dirty = true;
            }
        }
        update @ CaptureEvent::DmabufFrame(..) => {
            if let CaptureEvent::DmabufFrame(.., t0) = &update {
                *t0_unix_us = *t0;
            }
            if mat.handle_event(update)? {
                *frame_dirty = true;
            }
        }
        CaptureEvent::Disabled => {
            eprintln!("[server] console scanout disabled");
            *session = None; // reconnect on next scanout
            *frame_dirty = false;
        }
    }
    Ok(())
}

/// DisplayMeta for materialized (linear) output: fourcc XR24, no modifier.
/// `gen` is assigned by the sender when a new generation is pushed.
fn meta_linear(fmt: &FrameFormat) -> DisplayMeta {
    DisplayMeta {
        width: fmt.width,
        height: fmt.height,
        stride: fmt.width * 4,
        fourcc: 0x3432_5258,
        modifier: 0,
        y0_top: false,
        frame_bytes: fmt.width * 4 * fmt.height,
        gen: 0,
    }
}

/// DisplayMeta passthrough for plain scanouts (already host-linear).
fn meta_plain(fmt: &FrameFormat) -> DisplayMeta {
    DisplayMeta {
        width: fmt.width,
        height: fmt.height,
        stride: fmt.stride,
        fourcc: fmt.fourcc,
        modifier: 0,
        y0_top: false,
        frame_bytes: fmt.frame_bytes(),
        gen: 0,
    }
}

/// Best-effort DisplayMeta from a probed size (QEMU console): XR24, linear.
fn meta_from_size(w: u32, h: u32) -> DisplayMeta {
    DisplayMeta {
        width: w,
        height: h,
        stride: w * 4,
        fourcc: 0x3432_5258,
        modifier: 0,
        y0_top: false,
        frame_bytes: w * 4 * h,
        gen: 0,
    }
}

/// Startup preflight: validate each element of the pipeline once and probe the
/// display geometry, without starting the (heavy, per-session) capture. Every
/// failure here is logged, not fatal — the pipeline still starts on connect and
/// re-learns what it can (except core pieces already opened before this).
/// Returns the probed geometry, if any, so a connect can size slots up front.
#[allow(clippy::too_many_arguments)]
fn preflight(
    source_name: &str,
    input_name: &str,
    drm_card: &Option<String>,
    dbus_socket: &Option<String>,
    console: u32,
    codec: rdma_display_codec::Codec,
    gl: Option<&Arc<rdma_display_egl::SharedContext>>,
) -> Option<DisplayMeta> {
    let mut geometry = None;

    // 1. display geometry (and proof the source can be opened at all)
    match source_name {
        "drm" => match rdma_display_capture::drm::probe_display(drm_card.as_deref()) {
            Ok(fmt) => {
                eprintln!(
                    "[preflight] drm scanout {}x{} '{}' modifier={:#x}",
                    fmt.width,
                    fmt.height,
                    rdma_display_capture::fourcc_name(fmt.fourcc),
                    fmt.modifier
                );
                geometry = Some(meta_linear(&fmt));
            }
            Err(e) => eprintln!("[preflight] drm probe failed: {e:#}"),
        },
        "qemu" => match rdma_display_capture::qemu::probe_console(dbus_socket, console) {
            Ok((w, h)) => {
                eprintln!("[preflight] qemu console {w}x{h}");
                geometry = Some(meta_from_size(w, h));
            }
            Err(e) => eprintln!("[preflight] qemu probe failed: {e:#}"),
        },
        _ => {}
    }

    // 2. lossless GPU encoder (shader compile + buffers) on the shared context
    if codec == rdma_display_codec::Codec::Lossless {
        if let Some(gl) = gl {
            let g = geometry
                .as_ref()
                .map(|m| rdma_display_codec::Geometry::new(m.width, m.height, m.stride))
                .unwrap_or_else(|| rdma_display_codec::Geometry::new(1920, 1080, 1920 * 4));
            match rdma_display_codec_gpu::GpuCodec::new_shared(g, gl.clone()) {
                Ok(c) => {
                    c.sync();
                    drop(c);
                    eprintln!("[preflight] lossless GPU encoder ok ({}x{})", g.width, g.height);
                }
                Err(e) => {
                    eprintln!("[preflight] lossless GPU encoder failed: {e:#} (will fall back to CPU)")
                }
            }
        }
    }

    // 3. input injection (uinput device creation; QEMU input shares the
    //    console probe above)
    if input_name == "uinput" {
        match rdma_display_capture::input::UinputMouse::open() {
            Ok(m) => {
                drop(m);
                eprintln!("[preflight] uinput ok");
            }
            Err(e) => eprintln!("[preflight] uinput failed: {e:#}"),
        }
    }

    geometry
}

#[cfg(test)]
mod metrics_tests {
    use super::{read_cpu_ticks, read_rss_bytes, read_threads};

    #[test]
    fn proc_readers_return_plausible_values() {
        let rss = read_rss_bytes().expect("rss");
        assert!(rss > 0 && rss < (1 << 40), "implausible rss {rss}");
        // ticks may legitimately be 0 on a virtualized /proc; just require a parse
        let _ = read_cpu_ticks().expect("cpu ticks");
        let t = read_threads().expect("threads");
        assert!(t >= 1, "no threads");
    }
}

#[cfg(test)]
mod codec_tests {
    use super::FrameCodec;
    use rdma_display_codec::{Codec, Decoder, Geometry};

    fn synthetic(w: u32, h: u32, seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for (i, b) in v.iter_mut().enumerate() {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            *b = (x >> 13) as u8;
        }
        v
    }

    /// The server's lossless encode path: `FrameCodec::encode_in_place` (GPU
    /// when available, CPU otherwise) must produce bytes the client's CPU
    /// decoder turns back into the exact frame, across a full frame followed
    /// by deltas — i.e. the reference is advanced correctly.
    #[test]
    fn lossless_encode_in_place_round_trips() {
        let (w, h) = (256u32, 256u32);
        let stride = w * 4;
        let geom = Geometry::new(w, h, stride);
        let fb = geom.frame_bytes();
        let mut fc = FrameCodec::new();
        fc.configure(Codec::Lossless, w, h, stride, fb, None).unwrap();
        let mut dec = Decoder::new(geom).unwrap();
        let mut slot = vec![0u8; geom.max_coded_len()];
        let base = synthetic(w, h, 1);
        for i in 0..4u32 {
            let mut orig = base.clone();
            let off = (i as usize * 4096) % orig.len();
            for k in 0..96 {
                let p = (off + k) % orig.len();
                orig[p] ^= 0x5a;
            }
            slot[..fb].copy_from_slice(&orig);
            let n = fc.encode_in_place(&mut slot).unwrap();
            assert!(n <= slot.len());
            let out = dec.decode(&slot[..n]).unwrap();
            assert_eq!(out, &orig[..], "frame {i} round trip");
        }
    }
}
