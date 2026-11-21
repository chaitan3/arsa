//! arsa server: captures a display source and streams frames to a
//! client over RDMA.
//!
//! Modes (defaults shown):
//!   --mode capture   stream a display source:
//!     --source drm (default)  KMS capture of the host screen (Sunshine-style):
//!             grabs the CRTC's current fb via GETFB2 as a dmabuf each flip;
//!             requires sudo (CAP_SYS_ADMIN) and kernel >= 5.8; Vulkan
//!             materialize only; --input uinput (default)
//!     --source qemu           QEMU dbus display capture (the guest):
//!             --materialize vk     Vulkan import + host staging (default)
//!             --materialize plain  D-Bus host-memory scanouts (gl=off)
//!             --input qemu         inject into the guest console (default)
//!     --codec lossless (default)|raw|lossy  frame coding (lossless =
//!             tile-delta against the previous frame; raw = unchanged wire;
//!             lossy = PyroWave; see docs/ARCHITECTURE.md). Client negotiates.
//!     --capture-sync auto (default) = fence, fallback settle:2ms
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

mod direct;
mod display;
mod gpu;
mod gpu_power;
mod lossy;
mod signals;

use gpu_power::GpuPower;

/// Default bind address/port when the user gives only a host (or nothing).
const DEFAULT_LISTEN: &str = "0.0.0.0:4791";
/// The signaling port (kept in sync with the client's `DEFAULT_PORT`).
const DEFAULT_PORT: u16 = 4791;

/// Append the default signaling port when the user gives only a host or IP
/// (`192.168.4.2`, or `--listen 192.168.4.2`). An explicit port is left alone,
/// including the bracketed IPv6 form `[::1]:4791`; a bare IPv6 literal is also
/// left alone (it needs brackets to carry a port).
fn with_default_port(s: &str) -> String {
    // Bracketed IPv6 carries a port only after "]:". Otherwise `host:port` has
    // exactly one colon; no colon means append the default, and two or more
    // means a bare IPv6 literal, which we must not rewrite.
    let has_port = if let Some(rest) = s.strip_prefix('[') {
        rest.contains("]:")
    } else {
        s.contains(':')
    };
    if s.is_empty() || has_port {
        s.to_string()
    } else {
        format!("{s}:{DEFAULT_PORT}")
    }
}

/// Initialize the logger. `-v`/`--verbose` enables debug for our crates; the
/// `RUST_LOG` environment variable overrides either default.
fn init_logging(verbose: bool) {
    use std::io::Write as _;
    let default = if verbose {
        "warn,rdma_display=debug,arsa_server=debug"
    } else {
        "warn,rdma_display=info,arsa_server=info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default))
        .format(|buf, record| writeln!(buf, "[{}] {}", record.level(), record.args()))
        .init();
}

fn main() -> Result<()> {
    let mut mode = "capture".to_string();
    let mut device: Option<String> = None;
    // the listen address: `--listen ADDR:PORT` or the first positional argument
    let mut listen: Option<String> = None;
    let mut slots: usize = 5;
    let mut fps: u64 = 0;
    let mut source = "drm".to_string();
    let mut input: Option<String> = None; // None = paired default for the source
    let mut dbus_socket: Option<String> = None;
    let mut console: u32 = 0;
    let mut materialize: String = "vk".to_string();
    let mut drm_card: Option<String> = None;
    let mut input_test = false;
    let mut audio = true;
    let mut audio_device: Option<String> = None;
    let mut audio_helper = false;
    // stream encryption: on by default; --no-encrypt leaves the TCP streams in
    // the clear (RDMA video is always unencrypted). The server's choice is sent
    // to the client inside the authenticated preamble.
    let mut no_encrypt = false;
    let mut capture_sync = CaptureSync::Auto;
    // frame-pacing policy: how the server sizes its flow-control window
    let mut pace = Pace::Auto;
    let mut codec = rdma_display_codec::Codec::Lossless;
    // `--codec auto` (default): honour the client's codec preference, falling
    // back to `codec` when the client has none. A specific `--codec` forces it.
    let mut codec_auto = true;
    // lossy bitstream budget in Mbit/s; the encoder only takes a per-frame byte
    // cap, so it is converted at a nominal 60 fps (see `lossy_budget_bytes`).
    // The client may override it via its Hello preference.
    let mut lossy_mbps: u16 = 480;
    let mut lossy_chroma444 = false;
    let mut transport = rdma_display_rdma::TransportPref::Auto;
    let mut gpu_power = GpuPower::Off;
    // compositor output control for the client's resolution request
    let mut display_control = "auto".to_string();
    let mut display_output: Option<String> = None;
    // log idle/session RSS+CPU every N seconds (0 = off)
    let mut idle_metrics: u64 = 0;
    // -v/--verbose: debug-level logging for our crates
    let mut verbose = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => mode = args.next().expect("--mode needs a value"),
            "--device" => device = Some(args.next().expect("--device needs a value")),
            "--listen" => listen = Some(args.next().expect("--listen needs a value")),
            // --slots kept as an accepted alias (older scripts/commands)
            "--frame-slots" | "--slots" => {
                slots = args
                    .next()
                    .expect("--frame-slots needs a value")
                    .parse()?
            }
            "--fps" => fps = args.next().expect("--fps needs a value").parse()?,

            "--source" => source = args.next().expect("--source needs a value"),
            "--input" => input = Some(args.next().expect("--input needs a value")),
            "--dbus-socket" => {
                dbus_socket = Some(args.next().expect("--dbus-socket needs a value"))
            }
            "--console" => console = args.next().expect("--console needs a value").parse()?,
            "--materialize" => materialize = args.next().expect("--materialize needs a value"),
            "--drm-card" => drm_card = Some(args.next().expect("--drm-card needs a value")),
            "--capture-sync" => {
                let v = args.next().expect("--capture-sync needs a value");
                capture_sync = CaptureSync::parse(&v).with_context(|| {
                    format!(
                        "--capture-sync must be auto|settle[:ms]|fence, got {v:?}"
                    )
                })?;
            }
            "--input-test" => input_test = true,
            "--pace" => {
                let v = args.next().expect("--pace needs a value");
                pace = Pace::parse(&v)
                    .with_context(|| format!("--pace must be auto|latency|smooth, got {v:?}"))?;
            }
            "--codec" => {
                let v = args.next().expect("--codec needs a value");
                if v == "auto" {
                    codec_auto = true;
                } else {
                    codec = rdma_display_codec::Codec::parse(&v).with_context(|| {
                        format!("--codec must be auto|raw|lossless|lossy, got {v:?}")
                    })?;
                    codec_auto = false;
                    if !codec.implemented() {
                        bail!(
                            "--codec {} is a reserved selector and not implemented yet",
                            codec.name()
                        );
                    }
                }
            }
            "--lossy-bitrate" => {
                lossy_mbps = args
                    .next()
                    .expect("--lossy-bitrate needs a value")
                    .parse()
                    .with_context(|| "--lossy-bitrate must be Mbit/s (e.g. 100)")?;
                if lossy_mbps == 0 {
                    bail!("--lossy-bitrate must be > 0 Mbit/s");
                }
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
            "--no-encrypt" => no_encrypt = true,
            "--display-control" => {
                display_control = args.next().expect("--display-control needs a value")
            }
            "--display-output" => {
                display_output = Some(args.next().expect("--display-output needs a value"))
            }
            "--gpu-power" => {
                let v = args.next().expect("--gpu-power needs a value");
                gpu_power = GpuPower::parse(&v)?;
            }
            "--idle-metrics" => {
                idle_metrics = args.next().expect("--idle-metrics needs a value").parse()?
            }
            "-v" | "--verbose" => verbose = true,
            "--help" | "-h" => {
                println!("usage: arsa-server [ADDR[:PORT]] [--mode capture|testcard] [--device mlx5_0] [--listen 0.0.0.0:4791] [--frame-slots 5] [--fps N]\n  \
  ADDR[:PORT]  listen address, the same as --listen (default 0.0.0.0:4791; the port defaults to 4791 when omitted)\n  \
  defaults: --mode capture --source drm --fps 0 --codec auto --audio on --capture-sync auto --pace auto (a bare command is a root KMS capture of the host)\n  \
capture: --source qemu|drm (qemu = QEMU dbus-display, the guest; drm = KMS capture, the host — requires sudo/CAP_SYS_ADMIN, kernel >= 5.8; portal = XDG Desktop Portal ScreenCast over PipeWire — needs a session bus and a portal backend, and needs no root)\n  \
  --input qemu|uinput (default auto: qemu↔qemu-dbus, drm↔uinput, portal↔uinput)\n  \
  qemu: [--dbus-socket /path/dbus.sock] [--console 0]   drm: [--drm-card /dev/dri/cardN] (default: first card0..card8 with an active scanout)\n  \
  [--materialize vk|plain] [--input-test]\n  \
  --materialize vk (default) imports the dmabuf into Vulkan; RDMA prefers dma-buf MRs (lossless coded GPU buffer, raw linear scanout) and falls back to host staging, where raw copies and lossless encodes. plain = D-Bus host-memory scanouts (qemu gl=off).\n  \
  --capture-sync auto|settle[:ms]|fence  (drm) when to capture, on top of the 1 ms poll that emits only when the plane's fb changes: auto (default) and fence read the current frame when the compositor's per-frame completion fence signals (tracefs; ~1 ms, tear-free, needs root), falling back to settle:2ms while the watch calibrates or if tracefs is unavailable; settle[:ms] = emit the current fb ms after its commit (default 2 ms; usually complete, but frames whose render outlasts the window tear)\n  \
  --pace auto|latency|smooth  frame pacing: the server's flow-control window — how many frames it keeps in flight ahead of the client's display. The client credits one frame per present, which paces both TCP and RDMA. auto (default) sizes the window from the measured link (ceil(turnaround / source frame period): keeps the pipe full without queueing); latency = 1 (present-paced, lowest latency, may stall if the link is slower than the display); smooth = the transport maximum (smoothest cadence, highest latency).\n  \
  --codec auto|raw|lossless|lossy  frame coding for the whole session (default auto = honour the client's Hello preference, else lossless). raw = unchanged wire; lossless = v2 tile-delta against the previous frame (desktop-friendly); lossy = PyroWave intra-only GPU codec (needs libpyrowave-shared.so). A non-auto value forces the codec. Build server and client together (lossless v2 is a wire-format version).\n  \
  --lossy-bitrate MBPS  lossy: bitstream budget in Mbit/s, converted to the encoder's per-frame byte cap at a nominal 60 fps (default 480 ~= the old 1 MB/frame; the slot size sent to the client is derived from it). The client can override it via --lossy-bitrate.\n  \
  --lossy-chroma 420|444  lossy: chroma subsampling (default 420).\n  \
  --gpu-power off|p5|p0|max|<sm>[,<mem>]  floor the NVIDIA GPU clocks with nvidia-smi, still allowing boost above the floor (needs root; warns and continues if unavailable). p5 = the measured fast floor (sm 435 / mem 810 MHz), p0/max = floor at the card's maximum (i.e. pinned), <sm>[,<mem>] = explicit MHz floors. Default off. GeForce cards reject the memory-clock floor.\n  \
  --transport auto|rdma|tcp  video transport (default auto: RDMA/RoCE when available, else TCP). Negotiated with the client; TCP sends length-prefixed coded frames on a TAG_VIDEO connection.\n  \
  --idle-metrics N  log RSS/CPU/threads every N seconds, labelled idle or session (default off).\n  \
  -v, --verbose  debug-level logging (per-frame/per-event detail; RUST_LOG overrides)\n  \
  --display-control auto|hyprctl|kscreen|none  apply the client's pre-session resolution request (--native / --set-mode) to this compositor's output before capture starts, so the scanout and the session slots match (--source drm; default auto = Hyprland via hyprctl, KDE via kscreen-doctor).\n  \
  --display-output NAME  DRM connector name (e.g. DP-1, HDMI-A-1, eDP-1) to capture and to apply the client's resolution request to (default: the compositor's primary output; with no compositor backend, the first active output).\n  \
  --fps N (capture): pacing cap; 0 = uncapped, consumer-paced by client credits (default 0).\n  \
  --frame-slots N  receive/staging slots, each holding one coded frame: sets the client's receive-buffer pool and the maximum flow-control window (default 5; --slots is accepted as an alias).\n  \
  --audio | --no-audio [--audio-device my-sink.monitor]  audio side-channel (default on; --no-audio disables): capture the named source's PCM through the pipewire-pulse\n  \
  --no-encrypt  leave the TCP streams in the clear (default: AES-256-GCM). RDMA video is always unencrypted.\n  \
  compat socket (XDG_RUNTIME_DIR must point at the user's session when running under sudo; '@DEFAULT_MONITOR@' = default sink).");
                return Ok(());
            }
            other if other.starts_with("--") => anyhow::bail!("unknown arg: {other}"),
            // first bare argument is the listen address (same format as
            // --listen); anything after that is not something we understand
            other => {
                if listen.is_none() {
                    listen = Some(other.to_string());
                } else {
                    anyhow::bail!("unexpected argument: {other}");
                }
            }
        }
    }
    let listen = with_default_port(listen.as_deref().unwrap_or(DEFAULT_LISTEN));
    init_logging(verbose);

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
            log::info!("[gpu-power] {e:#}");
            log::info!("[gpu-power] continuing with the driver's default power management");
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
                log::warn!("[server] no RDMA device ({e:#}); falling back to TCP video transport");
                None
            }
        },
    };
    let rdma_ref = rdma.as_ref().map(|(c, p)| (c, p));

    match mode.as_str() {
        "testcard" => run_testcard_server(rdma_ref, transport, &listen, slots, fps),
        "capture" => run_capture(
            rdma_ref, transport, &source, input.as_deref(), &dbus_socket, console, &drm_card,
            &materialize, &listen, slots, fps, input_test,
            capture_sync, audio, audio_device.as_deref(), codec, codec_auto, lossy_mbps,
            lossy_chroma444, &display_control, display_output.as_deref(), idle_metrics, pace,
            no_encrypt,
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
    log::info!("listening on {listen}, mode=testcard");
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
            log::warn!("[server] audio connection in testcard mode: dropped");
            continue;
        }
        if tag[0] == rdma_display_proto::TAG_INPUT {
            log::warn!("[server] input connection in testcard mode: dropped");
            continue;
        }
        let peer = stream.peer_addr()?;
        log::info!("client connected: {peer}");
        // testcard has no console to inject into: drain input events
        let (input_tx, input_rx) = mpsc::channel();
        std::thread::spawn(move || {
            while input_rx.recv().is_ok() {}
        });
        // no input sink in testcard mode: nothing to relay rumble to
        let rumble_hub = rdma_display_proto::gamepad::RumbleHub::new();
        // negotiate Hello + the display-mode request, then reply (testcard has
        // a fixed 1280x800 geometry, so a request is a no-op here)
        let (mut conn, neg, _mode, _codec_pref, _lossy_mbps, _client_caps) = rdma_display_rdma::handshake_server(
            stream,
            rdma_display_rdma::random_nonce(),
            transport,
            rdma.is_some(),
            None,
            None,
        )?;
        let _ = rdma_display_rdma::send_mode_reply(
            &mut conn,
            &rdma_display_proto::ModeReply::ok(
                meta.width,
                meta.height,
                0,
                "testcard: fixed 1280x800",
            ),
        );
        let mut link = rdma_display_rdma::ServerLink::accept_negotiated(
            conn,
            neg,
            rdma,
            &meta,
            slots,
            rdma_display_codec::Codec::Raw,
            meta.frame_bytes as usize,
            0,
            input_tx,
            None,
            &rumble_hub,
            None, // testcard: no auth, no encryption
            None,
            None,
        )?;
        // TCP transport: the client opens its TAG_VIDEO socket right after the
        // codec handshake; grab it before the (blocking) testcard loop starts
        if link.transport() == rdma_display_rdma::Transport::Tcp {
            wait_for_video(&mut link, &listener)?;
        }
        log::info!("session up: {slots} slots, {} frame bytes", link.frame_bytes());
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
            log::debug!("[stats] sent {} frames in {:?}", sent, started.elapsed());
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
            log::warn!(
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
            log::info!("[server] --input-test: injecting synthetic input");
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
                log::info!("[server] input-test #{n}: key 'a' + mouse abs ({x},{y})");
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

/// Frame-pacing policy: how the server sizes its flow-control window — the most
/// frames it lets be in flight ahead of the client's display. The client credits
/// one per frame it consumes, so the window is what ties the server's send rate
/// to the client's present rate (see `rdma_display_rdma::ServerLink`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pace {
    /// Present-paced: never more than one frame in flight — lowest latency, but
    /// can stall (drop the rate) when the link is slower than the display
    /// period.
    Latency,
    /// Never gate: buffer as far as the transport allows — smoothest cadence,
    /// highest latency.
    Smooth,
    /// Size the window from the measured link (default).
    Auto,
}

impl Pace {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "latency" => Pace::Latency,
            "smooth" => Pace::Smooth,
            "auto" => Pace::Auto,
            _ => return None,
        })
    }
}

/// Flow-control window for `pace`, derived from the measured link.
///
/// `turnaround` is the send -> client-credit time (it already includes the
/// client's decode and its wait for the next present). `source_period` is the
/// **source** frame period — the capture cadence, which is independent of the
/// window. Their ratio is the bandwidth-delay product counted in frames — how
/// many frames must be in flight for the pipeline not to stall — so the window
/// is `ceil(turnaround / source_period)`, floored at 1 (present-paced) and
/// capped at the transport maximum (the staging-slot count).
///
/// The denominator must be the *source* period, never the send interval: the
/// send interval is a consequence of the window (in steady state
/// `send_interval == turnaround / window`), so using it makes the formula
/// `ceil(turnaround / (turnaround/window)) == window` — a neutral fixed point
/// that can never climb out of a bad value (which is exactly how it collapsed
/// to 1 on a 30 ms link).
fn pace_window(pace: Pace, turnaround_us: u64, source_period_us: u64, max: usize) -> usize {
    let max = max.max(1);
    match pace {
        Pace::Latency => 1,
        Pace::Smooth => max,
        Pace::Auto => {
            if turnaround_us == 0 || source_period_us == 0 {
                // not measured yet: a small, safe, low-latency default
                return 2.min(max);
            }
            (turnaround_us.div_ceil(source_period_us) as usize).clamp(1, max)
        }
    }
}

/// The source frame period, measured from the capture timestamps (`t0`). Unlike
/// the send interval this is independent of the flow-control window, so it is
/// the correct denominator for the window.
///
/// The raw interval is noisy and bimodal: in fence mode `t0` is the trace
/// reader's own clock (scheduling jitter), and the compositor only flips on
/// damage, so a short idle gap shows up as a long interval. An EMA over the raw
/// intervals therefore wobbled between ~16 and ~155 ms and made the window
/// oscillate 1..3 (with the p95 latency spiking whenever it dipped to 1).
///
/// Use the *minimum* of a sliding window of intervals instead: it is the
/// display period (the fastest flip), which rejects the idle gaps, the slow
/// tail of the jitter, and — importantly — the pipeline's own achieved cadence,
/// so the window can't be throttled by the very rate it produced.
#[derive(Default)]
struct SourcePeriod {
    /// `t0` (unix µs) of the previous captured frame; 0 = none yet
    last_t0: u64,
    /// recent inter-frame intervals (µs), oldest first, bounded to `WINDOW`
    recent: std::collections::VecDeque<u64>,
    /// low percentile of `recent` (µs); 0 until measured
    estimate_us: u64,
    /// known output frame period (µs) from the display mode; 0 = unknown. The
    /// estimate is floored at this — an interval shorter than a vblank is a
    /// measurement artifact (two `t0` reads close together), not a real flip —
    /// and capped at twice it, so a sustained slow/idle source cannot halve the
    /// flow-control window.
    floor_us: u64,
}

impl SourcePeriod {
    /// Intervals kept for the estimate (~0.5 s at 60 Hz).
    const WINDOW: usize = 32;
    /// Percentile of the sorted intervals. The **minimum** (0) is the display
    /// period — the fastest the compositor flips. A higher percentile instead
    /// follows the *achieved* cadence, which is self-limiting: a throttled
    /// pipeline reports long intervals, those shrink the flow-control window,
    /// which throttles it further. The minimum floors the window at the display
    /// period and breaks that loop. It is then clamped into
    /// `[floor_us, 2 * floor_us]` so neither the artifact tail nor a slow
    /// stretch can move the period far off the display rate.
    const PCT: usize = 0;
    /// Upper clamp on the estimate, as a multiple of the output period.
    const MAX_FACTOR: u64 = 2;

    /// New estimator anchored on the output's frame period (0 = unknown).
    fn new(floor_us: u64) -> Self {
        SourcePeriod { floor_us, ..SourcePeriod::default() }
    }

    /// Feed one captured frame's `t0`.
    fn observe(&mut self, t0: u64) {
        if self.last_t0 != 0 && t0 > self.last_t0 {
            let dt = t0 - self.last_t0;
            // drop only absurd gaps (a paused/blanked screen), not the ordinary
            // damage-driven ones — the percentile handles those
            if dt < 1_000_000 {
                self.recent.push_back(dt);
                while self.recent.len() > Self::WINDOW {
                    self.recent.pop_front();
                }
                let p = Self::percentile(&self.recent, Self::PCT);
                self.estimate_us = if self.floor_us == 0 {
                    p
                } else {
                    p.clamp(self.floor_us, self.floor_us * Self::MAX_FACTOR)
                };
            }
        }
        self.last_t0 = t0;
    }

    /// The active source frame period (µs); 0 until measured.
    fn period_us(&self) -> u64 {
        self.estimate_us
    }

    /// `pct`-th percentile of `v` (0..=100); 0 for an empty window.
    fn percentile(v: &std::collections::VecDeque<u64>, pct: usize) -> u64 {
        if v.is_empty() {
            return 0;
        }
        let mut s: Vec<u64> = v.iter().copied().collect();
        s.sort_unstable();
        s[(s.len() * pct / 100).min(s.len() - 1)]
    }
}

/// p50/p95 of a bounded sample window, in KiB (judder instrumentation).
fn percentile_kib(v: &std::collections::VecDeque<usize>) -> (usize, usize) {
    let mut s: Vec<usize> = v.iter().copied().collect();
    s.sort_unstable();
    if s.is_empty() {
        return (0, 0);
    }
    (
        s[s.len() / 2] / 1024,
        s[(s.len() * 95 / 100).min(s.len() - 1)] / 1024,
    )
}

/// Push a sample into a bounded window, dropping the oldest (judder
/// instrumentation).
fn push_bounded(v: &mut std::collections::VecDeque<usize>, x: usize, cap: usize) {
    v.push_back(x);
    while v.len() > cap {
        v.pop_front();
    }
}

// --- capability drop for --source portal ------------------------------------

/// `_LINUX_CAPABILITY_VERSION_3`.
const LINUX_CAPABILITY_VERSION_3: u32 = 0x2008_0522;
/// `prctl(PR_SET_DUMPABLE)`.
const PR_SET_DUMPABLE: libc::c_long = 4;

/// `__user_cap_header_struct`.
#[repr(C)]
struct CapHeader {
    version: u32,
    pid: i32,
}

/// `__user_cap_data_struct`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CapData {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}

/// Make the process look like a plain user process for the portal.
///
/// `--source portal` is identified by `xdg-desktop-portal` through
/// `/proc/<pid>/root`. A binary carrying file capabilities (`install.sh`
/// setcaps `cap_sys_admin` for `--source drm`) executes with `suid_dumpable=0`
/// — i.e. **non-dumpable** — so the portal, a plain user process without
/// `CAP_SYS_PTRACE`, cannot read that path and refuses the request
/// ("Unable to open /proc/<pid>/root"). The portal path needs no capabilities,
/// so drop them (`capset` to empty) and restore dumpability.
fn drop_caps_for_portal() -> Result<()> {
    let hdr = CapHeader {
        version: LINUX_CAPABILITY_VERSION_3,
        pid: 0,
    };
    let data = [CapData::default(); 2];
    let rc = unsafe {
        libc::syscall(
            libc::SYS_capset,
            &hdr as *const CapHeader,
            data.as_ptr(),
        )
    };
    if rc != 0 {
        // nothing to drop when the binary carries no file capabilities
        log::debug!(
            "[server] capset(clear) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    if unsafe {
        libc::syscall(
            libc::SYS_prctl,
            PR_SET_DUMPABLE,
            1 as libc::c_long,
            0 as libc::c_long,
            0 as libc::c_long,
            0 as libc::c_long,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error())
            .context("prctl(PR_SET_DUMPABLE) for --source portal");
    }
    log::info!(
        "[server] --source portal: dropped capabilities and set dumpable \
         (so the portal can identify the caller)"
    );
    Ok(())
}

/// Build the configured capture source. `--source portal` (XDG Desktop Portal
/// screen capture over PipeWire) is behind the `portal` feature, so a default
/// build has no PipeWire dependency.
fn build_source(
    source_name: &str,
    dbus_socket: &Option<String>,
    console: u32,
    drm_card: &Option<String>,
    capture_connector: &Option<String>,
    capture_sync: CaptureSync,
) -> Result<Box<dyn CaptureSource>> {
    Ok(match source_name {
        "qemu" => Box::new(QemuDbusSource::new(dbus_socket.clone(), console)),
        "drm" => Box::new(DrmKmsSource::with_sync(
            drm_card.clone(),
            capture_connector.clone(),
            capture_sync,
        )),
        #[cfg(feature = "portal")]
        "portal" => Box::new(rdma_display_capture::PortalScreenCastSource::default()),
        other => bail!(
            "source {other:?} not implemented (qemu|drm{})",
            if cfg!(feature = "portal") {
                "|portal"
            } else {
                ""
            }
        ),
    })
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
    listen: &str,
    slots: usize,
    fps: u64,
    input_test: bool,
    capture_sync: CaptureSync,
    audio: bool,
    audio_device: Option<&str>,
    codec: rdma_display_codec::Codec,
    codec_auto: bool,
    lossy_mbps: u16,
    lossy_chroma444: bool,
    display_control: &str,
    display_output: Option<&str>,
    idle_metrics: u64,
    pace: Pace,
    no_encrypt: bool,
) -> Result<()> {
    if source_name == "drm" && materialize != "vk" && codec != rdma_display_codec::Codec::Lossy {
        bail!("--source drm requires --materialize vk (KMS capture is dmabuf-only)");
    }
    if source_name == "portal" {
        // the portal identifies the caller via /proc/<pid>/root; a setcap'd
        // binary is non-dumpable, so shed the caps we don't need here
        drop_caps_for_portal()?;
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
            log::debug!("[preflight] audio capture ok");
        } else {
            log::warn!("[preflight] audio: no packet within 3 s (continuing)");
        }
        a.stop(); // preflight probe only
    }
    let mut audio_capture: Option<rdma_display_audio::capture::AudioCapture> = None;
    let audio_busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // One Vulkan device shared by the raw materializer and the lossless/lossy
    // encoders: the scanout import is cached and reused.
    let gpu = gpu::Gpu::new().context("create the shared Vulkan GPU")?;
    // Host-visible staging shared by the raw materializer and the lossless
    // encoder (the transport's RDMA slots are registered from these buffers,
    // so coded output lands directly in the postable memory).
    let staging: gpu::SharedStaging = Arc::new(std::sync::Mutex::new(None));

    // input sink: explicit --input wins, else paired default
    let input_name = input_name.unwrap_or(match source_name {
        "drm" | "portal" => "uinput",
        _ => "qemu",
    });
    if input_name == "qemu" && source_name == "drm" {
        bail!("--input qemu injects into the QEMU guest, which --source drm does not capture (use --input uinput)");
    }
    if input_name == "uinput" && source_name == "qemu" {
        bail!("--input uinput injects into the host compositor, but --source qemu captures the guest (use --input qemu)");
    }

    // compositor output control for the client's resolution request: detect a
    // backend once; the client's request is applied on connect, before capture
    // starts, so the session is sized for the resulting scanout.
    let display_ctl: Option<Arc<dyn display::DisplayControl>> = if source_name == "drm" {
        match display::detect(display_control, display_output) {
            Ok(c) => c,
            Err(e) => {
                log::info!("[display] {e:#}");
                None
            }
        }
    } else {
        if display_control != "auto" && display_control != "none" {
            log::info!(
                "[display] --display-control applies to --source drm (host compositor); ignoring for {source_name}"
            );
        }
        None
    };
    // the mode the session changed away from, put back on disconnect (and on
    // SIGINT/SIGTERM via the shared signal watcher)
    let mut original_mode: Option<display::Mode> = None;
    // output frame period (µs) from the current mode: the floor for the
    // measured source period (0 = unknown, e.g. portal/guest sources)
    let mut mode_period_us: u64 = 0;
    if let Some(c) = display_ctl.as_deref() {
        match c.current() {
            Ok(m) => {
                mode_period_us = m.period_us();
                log::info!(
                    "[display] {} backend (output {}), current {}",
                    c.name(),
                    c.output().unwrap_or_else(|_| "?".into()),
                    m.label()
                )
            }
            Err(e) => log::warn!("[display] {} backend; probe failed: {e:#}", c.name()),
        }
        display::install_restore_watcher();
    }
    // the DRM connector to capture: --display-output forces it, else the
    // compositor's primary output — so capture and mode control agree on the
    // same output when there are several
    let capture_connector: Option<String> = display_output
        .map(str::to_string)
        .or_else(|| display_ctl.as_ref().and_then(|c| c.output().ok()));
    if let Some(name) = capture_connector.as_deref() {
        log::info!("[display] capture output: {name}");
    }

    // startup preflight: validate/probe each element once; the heavy pipeline
    // is (re)started on demand per client (Phase 2). The probed geometry lets
    // a client connect negotiate slots before the first captured frame.
    // server-default lossy budget (bytes/frame) from the Mbit/s target; a
    // client may override it per session via its Hello preference
    let lossy_budget = lossy_budget_bytes(lossy_mbps);
    let mut geometry = preflight(
        source_name,
        input_name,
        drm_card,
        dbus_socket,
        console,
        codec,
        codec_auto,
        lossy_budget,
        lossy_chroma444,
        &gpu,
        capture_connector.as_deref(),
    );
    if let Some(m) = geometry.as_ref() {
        log::debug!(
            "[preflight] geometry cached: {}x{} stride {} (sizes slots on connect)",
            m.width, m.height, m.stride
        );
    }

    let mut mat: Box<dyn FrameMaterializer> = build_materializer(
        codec,
        materialize,
        &gpu,
        &staging,
        lossy_budget,
        lossy_chroma444,
    )?;
    // whether the current materializer is the lossy one; raw/lossless share the
    // Vulkan/plain materializer, so the box is only rebuilt when the family changes
    let mut mat_is_lossy = codec == rdma_display_codec::Codec::Lossy;
    // the lossy budget baked into the current materializer (rebuild on change)
    let mut mat_lossy_budget = lossy_budget;
    log::info!("[server] capture materializer: {}", mat.name());

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
        log::info!("[server] no preflight geometry; capturing eagerly to learn it");
        let source = build_source(
            source_name,
            dbus_socket,
            console,
            drm_card,
            &capture_connector,
            capture_sync,
        )?;
        Some(CaptureSource::spawn(source, tx.clone())?)
    } else {
        None
    };
    log::info!("[server] listening on {listen} (pipeline starts on connect)");

    let listener = TcpListener::bind(listen)?;
    listener.set_nonblocking(true)?;

    let mut session: Option<rdma_display_rdma::ServerLink> = None;
    // per-session true-zero-copy staging (None = host Vulkan staging fallback)
    let mut gpu_direct = None;
    let mut raw_direct = None;
    // when the current session's control connection was accepted, for
    // measuring connect -> first-frame latency
    let mut session_connected_at: Option<std::time::Instant> = None;
    // one-shot: log when the first captured scanout arrives after connect
    let mut first_scanout_logged = false;
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
    // auth: this server's long-term key + the paired-client list (loaded once),
    // and the per-session secret the secondary sockets prove possession of.
    let host_identity = {
        let id = rdma_display_auth::Identity::load_or_generate(&rdma_display_auth::paths::host_key())?;
        log::info!(
            "[auth] required; server key {} (clients must pair, or already be in {})",
            id.fingerprint(),
            rdma_display_auth::paths::authorized_clients().display()
        );
        id
    };
    let mut authorized =
        rdma_display_auth::AuthorizedClients::load(&rdma_display_auth::paths::authorized_clients())?;
    let session_secret: std::sync::Arc<
        std::sync::Mutex<Option<[u8; rdma_display_auth::SECRET_LEN]>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(None));
    let started = std::time::Instant::now();
    // source frame period, measured from the capture t0s: the denominator for
    // the flow-control window (independent of the window, unlike the send
    // interval). Floored at the output's frame period so a sub-vblank artifact
    // cannot shrink it.
    let mut src_period = SourcePeriod::new(mode_period_us);
    // judder diagnostics: coded bytes of recently sent frames (bounded window)
    let mut coded_sizes: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    // whether the connected client asked for metrics (Hello::CAP_METRICS); when
    // false the server does no per-frame diagnostic work at all
    let mut client_metrics = false;
    // throttle for the periodic pacing log (the turnaround is otherwise only
    // logged when the window changes, so it was invisible after startup)
    let mut last_pace_log = std::time::Instant::now();

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
    // codec resolved for the active session (from --codec + the client's
    // preference); the startup `codec` is the fallback used for gl/mat/preflight
    let mut session_codec = codec;

    loop {
        // 0. reap a hung-up client: the control-reader thread signals EOF
        // via the credit channel; without this, an idle capture loop (no
        // frames to send) would never notice, `session` would stay stale,
        // and a reconnecting client would wait forever on accept
        if session.is_some() {
            let gone = session.as_mut().unwrap().client_gone();
            if gone {
                log::info!("[server] client disconnected");
                session = None;
            }
        }

        // a session that ended (hangup, send failure, Disabled) stops its
        // per-session workers and frees the geometry-sized GPU resources, so
        // an idle server holds none of the heavy pipeline
        if session.is_none() {
            if let Some(w) = workers.take() {
                w.stop();
                fc.clear_gpu_staging();
                gpu_direct = None;
                raw_direct = None;
                mat.release();
                // an idle server must not hold the per-session staging buffers
                // (lossy sessions don't bind them, so clear even then)
                *staging.lock().unwrap() = None;
                fc.teardown();
                // audio is per-session too: stop it (and the helper) while idle
                if let Some(mut a) = audio_capture.take() {
                    a.stop();
                }
                session_nonce.store(0, std::sync::atomic::Ordering::SeqCst);
                *session_secret.lock().unwrap() = None;
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
                coded_sizes.clear();
                client_metrics = false;
                log::info!("[server] session ended; pipeline idle");
            }
            // put the display mode back if a session changed it (also covers a
            // client that requested a change and then never established a
            // session)
            if let Some(orig) = original_mode.take() {
                if let Some(c) = display_ctl.as_deref() {
                    match c.set(orig) {
                        Ok(()) => log::info!("[display] restored {}", orig.label()),
                        Err(e) => log::warn!("[display] restore failed: {e:#}"),
                    }
                }
                display::disarm_restore();
            }
        }
        session_active.store(session.is_some(), std::sync::atomic::Ordering::Relaxed);

        // frame pacing: size the flow-control window from the measured link.
        // Cheap, so recompute every iteration; log on change *and* every 2 s so
        // the turnaround is visible after startup (it used to be logged only
        // when the window changed, which hid it).
        if let Some(link) = session.as_mut() {
            let (t, iv, max) = (link.turnaround_us(), link.send_interval_us(), link.max_window());
            let src = src_period.period_us();
            let target = pace_window(pace, t, src, max);
            let changed = target != link.window();
            if changed || last_pace_log.elapsed() >= std::time::Duration::from_secs(2) {
                last_pace_log = std::time::Instant::now();
                let applied = link.set_window(target);
                // judder diagnostics: in-flight count, coded frame size, and the
                // kernel send-queue depth on the TCP video socket (a growing
                // sockout ⇒ the link, not the pacing loop, is the bottleneck)
                if client_metrics {
                    let inflight = link.inflight();
                    let (c50, c95) = percentile_kib(&coded_sizes);
                    let sockout = match link.video_outq_bytes() {
                        Some(b) => format!("{} KiB", b / 1024),
                        None => "n/a".to_string(),
                    };
                    log::debug!(
                        "[server] pacing {pace:?}: window {applied} of max {max} \
                         (turnaround {t} us, source period {src} us, send interval {iv} us) \
                         | inflight {inflight} | coded p50 {c50} KiB p95 {c95} KiB | sockout {sockout}"
                    );
                } else {
                    log::debug!(
                        "[server] pacing {pace:?}: window {applied} of max {max} \
                         (turnaround {t} us, source period {src} us, send interval {iv} us)"
                    );
                }
            }
        }

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
                        &mut pending_scanout, &mut src_period,
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
                        &mut pending_scanout, &mut src_period,
                    )?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    bail!("capture source ended")
                }
            },
        }

        // flow control: if a frame is waiting and the client hasn't credited
        // room, block for a credit *before* draining, so the drain picks up the
        // freshest capture (latest-wins mailbox) and we never encode a frame we
        // cannot send. This is the only backpressure TCP gets (it has no
        // staging slots); RDMA would otherwise stall later, inside
        // `claim_free_slot_rdma`, after the frame had already been taken.
        // Skipped while pipelining: those frames must reach step 5 to be posted
        // (posting is what eventually frees them).
        if !pipelined && frame_dirty && session.is_some() {
            let hung = {
                let link = session.as_mut().unwrap();
                if link.window_open() {
                    false
                } else {
                    link.wait_for_window().is_err()
                }
            };
            if hung {
                log::info!("[server] client gone while waiting for a frame credit");
                session = None;
            }
        }

        // 2. drain any backlog (each event latches state; newest t0 wins)
        loop {
            match rx.try_recv() {
                Ok(ev) => {
                    handle_capture_event(
                        ev, &mut mat, &mut session, &mut pending_meta,
                        &mut t0_unix_us, &mut frame_dirty, &screen,
                        &mut pending_scanout, &mut src_period,
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
                log::info!(
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
            let tag_ok = std::io::Read::read_exact(&mut stream, &mut tag).is_ok();
            let _ = stream.set_read_timeout(None);
            if !tag_ok {
                log::warn!("[server] connection with no routing tag dropped ({addr})");
            } else if tag[0] == rdma_display_audio::wire::TAG_AUDIO {
                let authed = match *session_secret.lock().unwrap() {
                    Some(s) => {
                        rdma_display_auth::server_conn_verify(&mut stream, &s, tag[0]).is_ok()
                    }
                    None => false,
                };
                if !authed {
                    log::warn!("[audio] unauthenticated connection dropped ({addr})");
                    continue;
                }
                match audio_capture.as_ref() {
                    Some(mbox) => {
                        if audio_busy
                            .compare_exchange(false, true, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
                            .is_ok()
                        {
                            log::info!("[audio] client stream connected ({addr})");
                            let mbox = mbox.mailbox.clone();
                            let busy = audio_busy.clone();
                            let audio_seal = if no_encrypt {
                                None
                            } else {
                                (*session_secret.lock().unwrap()).map(|s| {
                                    let key = rdma_display_auth::derive_stream_key(
                                        &s,
                                        rdma_display_audio::wire::TAG_AUDIO,
                                        false,
                                    );
                                    rdma_display_auth::Seal::new(&key)
                                })
                            };
                            std::thread::spawn(move || {
                                if let Err(e) = rdma_display_audio::capture::serve_connection(
                                    stream, &mbox, audio_seal,
                                ) {
                                    log::info!("[audio] stream ended: {e:#}");
                                }
                                busy.store(false, std::sync::atomic::Ordering::SeqCst);
                            });
                        } else {
                            log::warn!("[audio] second audio connection refused ({addr})");
                        }
                    }
                    None => {
                        log::warn!(
                            "[server] audio connection but audio capture is unavailable: dropped ({addr})"
                        );
                    }
                }
            } else if tag[0] == rdma_display_proto::TAG_INPUT {
                // input has its own connection + thread, independent of the
                // video/control path; a proof of the session secret binds it to
                // the authenticated session
                let authed = match *session_secret.lock().unwrap() {
                    Some(s) => {
                        rdma_display_auth::server_conn_verify(&mut stream, &s, tag[0]).is_ok()
                    }
                    None => false,
                };
                if !authed {
                    log::warn!("[input] unauthenticated connection dropped ({addr})");
                    continue;
                }
                match workers.as_ref() {
                    Some(w) => {
                        let (enc, sec) = match *session_secret.lock().unwrap() {
                            Some(s) => (!no_encrypt, s),
                            None => (false, [0u8; rdma_display_auth::SECRET_LEN]),
                        };
                        let input_tx = w.input_tx.clone();
                        let _ = std::thread::Builder::new()
                            .name("input-reader".into())
                            .spawn(move || serve_input(stream, enc, sec, input_tx, addr));
                    }
                    None => log::warn!(
                        "[input] input connection with no active session: dropped ({addr})"
                    ),
                }
            } else if tag[0] == rdma_display_proto::TAG_VIDEO {
                // client's TCP video socket (negotiated transport == TCP); it
                // must prove possession of the session secret
                let authed = match *session_secret.lock().unwrap() {
                    Some(s) => {
                        rdma_display_auth::server_conn_verify(&mut stream, &s, tag[0]).is_ok()
                    }
                    None => false,
                };
                let mut video_attached = false;
                match session.as_mut() {
                    Some(s)
                        if authed
                            && s.transport() == rdma_display_rdma::Transport::Tcp
                            && !s.has_video() =>
                    {
                        s.attach_video(stream);
                        video_attached = true;
                        if let Some(t) = session_connected_at {
                            log::info!(
                                "[session] video socket attached {} ms after connect",
                                t.elapsed().as_millis()
                            );
                        }
                    }
                    _ => {
                        log::warn!("[server] unexpected TAG_VIDEO connection dropped ({addr})")
                    }
                }
                if video_attached {
                    // the client can receive now: re-arm so the CURRENT frame is
                    // sent immediately. The frame that was ready before the
                    // socket attached was dropped (send_frame_tcp returns None
                    // without a video socket), and on a static screen no new
                    // scanout may arrive for a long time.
                    mat.mark_dirty();
                    frame_dirty = true;
                }
            } else if session.is_none() {
                // Negotiate Hello + the display-mode request BEFORE anything
                // heavy: the requested mode must be in effect before the
                // capture source starts and the slots are sized, so there is
                // no mid-session resize and no reconnect.
                let peer = stream.peer_addr()?;
                // auth preamble: authenticate (and, if armed, pair) before any
                // session work, and record the session secret for the secondary
                // sockets.
                let stream = {
                    match rdma_display_auth::server_auth(
                        stream,
                        &host_identity,
                        &mut authorized,
                        "server",
                        !no_encrypt,
                        |client_name| {
                            let pin = rdma_display_auth::pair::random_pin(6)?;
                            eprintln!(
                                "\n  [auth] new client {client_name:?} wants to pair — enter PIN {pin} on the client\n"
                            );
                            Ok(pin)
                        },
                    ) {
                        Ok((stream, auth)) => {
                            if auth.newly_paired {
                                authorized.save(&rdma_display_auth::paths::authorized_clients())?;
                                log::info!(
                                    "[auth] paired new client {:?} ({})",
                                    auth.client_name,
                                    rdma_display_auth::fingerprint(&auth.client_public)
                                );
                            }
                            log::info!(
                                "[auth] client {:?} authenticated ({peer})",
                                auth.client_name
                            );
                            *session_secret.lock().unwrap() = Some(auth.secret);
                            stream
                        }
                        Err(e) => {
                            log::warn!("[auth] rejected {peer}: {e:#}");
                            continue;
                        }
                    }
                };
                let nonce = rdma_display_rdma::random_nonce();
                let (ctrl_seal, ctrl_open) = if no_encrypt {
                    (None, None)
                } else {
                    match *session_secret.lock().unwrap() {
                        Some(s) => {
                            let rd = rdma_display_auth::derive_stream_key(
                                &s,
                                rdma_display_proto::TAG_CONTROL,
                                true,
                            );
                            let wr = rdma_display_auth::derive_stream_key(
                                &s,
                                rdma_display_proto::TAG_CONTROL,
                                false,
                            );
                            (
                                Some(std::sync::Arc::new(std::sync::Mutex::new(
                                    rdma_display_auth::Seal::new(&wr),
                                ))),
                                Some(std::sync::Arc::new(std::sync::Mutex::new(
                                    rdma_display_auth::Open::new(&rd),
                                ))),
                            )
                        }
                        None => (None, None),
                    }
                };
                let (mut conn, neg, mode_req, client_codec_pref, client_lossy_mbps, client_caps) =
                    match rdma_display_rdma::handshake_server(
                        stream,
                        nonce,
                        transport,
                        rdma.is_some(),
                        ctrl_seal.clone(),
                        ctrl_open.clone(),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            log::warn!("[server] handshake failed ({peer}): {e:#}");
                            continue;
                        }
                    };
                let reply = apply_mode_request(&mode_req, display_ctl.as_ref(), &mut original_mode);
                log::info!(
                    "[display] mode request {:?} -> {} ({})",
                    mode_req,
                    reply.mode_label(),
                    reply.message
                );
                if let Err(e) = rdma_display_rdma::send_mode_reply(&mut conn, &reply) {
                    log::warn!("[server] mode reply failed ({peer}): {e:#}");
                    continue;
                }
                // the client's metrics flag (Hello::CAP_METRICS) gates the
                // server-side diagnostics, so a normal run does none of the
                // per-frame diagnostic work
                client_metrics = client_caps & rdma_display_proto::CAP_METRICS != 0;
                if client_metrics {
                    log::debug!("[server] client requested metrics (server diagnostics on)");
                }
                // re-probe the DRM scanout so the slots match the mode now in
                // effect (also catches an external change since preflight). The
                // compositor commits the modeset asynchronously, so wait for it
                // to land before sizing the session.
                if source_name == "drm" {
                    match reprobe_scanout(
                        drm_card.as_deref(),
                        capture_connector.as_deref(),
                        reply.width,
                        reply.height,
                    ) {
                        Ok(fmt) => geometry = Some(meta_linear(&fmt)),
                        Err(e) => log::warn!(
                            "[display] scanout re-probe failed: {e:#}; using cached geometry"
                        ),
                    }
                }
                // resolve the session codec from --codec (auto/forced) + the
                // client's preference, before sizing slots / building the encoder
                session_codec = resolve_codec(codec_auto, codec, client_codec_pref);
                // the materializer must match the session codec: raw/lossless
                // share the EGL/plain one, lossy needs its own (rebuild on change)
                let want_lossy = session_codec == rdma_display_codec::Codec::Lossy;
                // the client's Hello preference overrides the server default
                let eff_mbps = if client_lossy_mbps > 0 {
                    client_lossy_mbps
                } else {
                    lossy_mbps
                };
                let eff_budget = lossy_budget_bytes(eff_mbps);
                if want_lossy != mat_is_lossy || (want_lossy && eff_budget != mat_lossy_budget) {
                    mat = build_materializer(
                        session_codec,
                        materialize,
                        &gpu,
                        &staging,
                        eff_budget,
                        lossy_chroma444,
                    )?;
                    mat_is_lossy = want_lossy;
                    mat_lossy_budget = eff_budget;
                    log::info!(
                        "[server] materializer for this session: {} (lossy budget {} B/frame = {} Mbit/s @60fps{})",
                        mat.name(),
                        eff_budget,
                        eff_mbps,
                        if client_lossy_mbps > 0 {
                            ", client-requested"
                        } else {
                            ""
                        }
                    );
                }
                // lossy chroma advertised to the client: 0 = 4:2:0, 1 = 4:4:4
                let session_chroma: u8 = if want_lossy && lossy_chroma444 { 1 } else { 0 };
                // The session needs the scanout geometry. If the source has not
                // produced a frame yet — e.g. `--source portal`, where the picker
                // may still be open — wait for it (pumping capture events)
                // instead of dropping the client into a dead half-session.
                if geometry.is_none() && pending_meta.is_none() {
                    log::info!(
                        "[server] no geometry yet; waiting for the first captured frame \
                         before setting up the session (client {peer})"
                    );
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                    while pending_meta.is_none() && std::time::Instant::now() < deadline {
                        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(ev) => handle_capture_event(
                                ev, &mut mat, &mut session, &mut pending_meta,
                                &mut t0_unix_us, &mut frame_dirty, &screen,
                                &mut pending_scanout, &mut src_period,
                            )?,
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                bail!("capture source ended")
                            }
                        }
                    }
                    if pending_meta.is_none() {
                        log::warn!(
                            "[server] no captured frame within 30 s; dropping the client ({peer})"
                        );
                        continue;
                    }
                    log::info!("[server] geometry arrived; setting up the session");
                }
                if let Some(meta) = geometry.clone().or_else(|| pending_meta.clone()) {
                    let t_connect = std::time::Instant::now();
                    log::info!("client connected: {peer}");
                    // start the per-session pipeline (source, input, audio)
                    let (input_tx, input_rx) = mpsc::channel::<rdma_display_proto::InputEvent>();
                    let sink: Box<dyn InputSink> = match input_name {
                        "qemu" => Box::new(QemuDbusInput::new(dbus_socket.clone(), console)),
                        "uinput" => Box::new(rdma_display_capture::UinputSink),
                        other => bail!("input sink {other:?} not implemented (qemu|uinput)"),
                    };
                    let input =
                        InputSink::spawn(sink, input_rx, screen.clone(), rumble_hub.clone())?;
                    log::info!("[server] input sink: {input_name}");
                    let source = if eager_source.is_some() {
                        None
                    } else {
                        let source = build_source(
                            source_name,
                            dbus_socket,
                            console,
                            drm_card,
                            &capture_connector,
                            capture_sync,
                        )?;
                        Some(CaptureSource::spawn(source, tx.clone())?)
                    };
                    log::info!("[server] capture running");
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
                    session_nonce.store(nonce, std::sync::atomic::Ordering::SeqCst);
                    let geom = rdma_display_codec::Geometry::new(
                        meta.width,
                        meta.height,
                        meta.stride,
                    );
                    let slot_len = match session_codec {
                        rdma_display_codec::Codec::Lossy => eff_budget + (1 << 20),
                        _ => session_codec
                            .slot_len(&geom)
                            .expect("implemented codec has a slot size"),
                    };
                    log::info!(
                        "[server] codec {} → slot {} B (raw frame {} B)",
                        session_codec.name(),
                        slot_len,
                        meta.frame_bytes
                    );
                    // PBO staging: bind the materializer's persistent-mapped
                    // PBOs as the staging slots; None = fall back to
                    // verbs-allocated slots.
                    // RDMA registration domain (also used by true-zero-copy
                    // dma-buf staging below).
                    let pd = rdma.map(|(_, p)| p);
                    let external = if rdma.is_some()
                        && transport != rdma_display_rdma::TransportPref::Tcp
                        && session_codec != rdma_display_codec::Codec::Lossy
                        && materialize == "vk"
                    {
                        match mat.bind_staging(slots, slot_len) {
                            Some(v) => Some(v),
                            None => {
                                log::warn!("[server] PBO staging unavailable — verbs-allocated slots");
                                None
                            }
                        }
                    } else {
                        None
                    };
                    // client input events join the same channel as the
                    // synthetic --input-test injector -> capture forwarder
                    let video_key = if no_encrypt {
                        None
                    } else {
                        (*session_secret.lock().unwrap()).map(|s| {
                            rdma_display_auth::derive_stream_key(
                                &s,
                                rdma_display_proto::TAG_VIDEO,
                                false,
                            )
                        })
                    };
                    match rdma_display_rdma::ServerLink::accept_negotiated(
                        conn,
                        neg,
                        rdma,
                        &meta,
                        slots,
                        session_codec,
                        slot_len,
                        session_chroma,
                        input_tx.clone(),
                        external.as_deref(),
                        &rumble_hub,
                        video_key,
                        ctrl_seal,
                        ctrl_open,
                    ) {
                        Ok(mut link) => {
                            log::info!("session up: {slots} slots, {} frame bytes", link.frame_bytes());
                            // the new client starts with zeroed slots: make
                            // the current scanout content sendable even
                            // though the guest hasn't repainted since the
                            // previous session's last frame
                            mat.mark_dirty();
                            // fresh codec reference: the new client's first
                            // frame must be full (no delta against a frame it
                            // never received)
                            fc.configure(
                                &gpu,
                                &staging,
                                session_codec,
                                meta.width,
                                meta.height,
                                meta.stride,
                                meta.frame_bytes as usize,
                            )?;
                            frame_dirty = true;
                            // True RDMA zero-copy, preferred over the host
                            // staging buffers: lossless coded output in an
                            // exported GPU dma-buf; raw reads a linear XRGB
                            // scanout dma-buf directly. Both fall back to the
                            // host path on registration failure.
                            gpu_direct = None;
                            raw_direct = None;
                            fc.clear_gpu_staging();
                            if link.transport() == rdma_display_rdma::Transport::Rdma
                                && materialize == "vk"
                            {
                                if let Some(pd) = pd {
                                    match session_codec {
                                        rdma_display_codec::Codec::Lossless => {
                                            match direct::GpuDirect::new(
                                                gpu.core.clone(),
                                                pd,
                                                slots,
                                                slot_len,
                                            ) {
                                                Ok(g) => {
                                                    log::info!(
                                                        "[server] lossless dma-buf staging: {slots} coded buffers registered with ibv_reg_dmabuf_mr"
                                                    );
                                                    gpu_direct = Some(g);
                                                }
                                                Err(e) => log::warn!(
                                                    "[server] lossless dma-buf staging unavailable ({e:#}); using host Vulkan staging"
                                                ),
                                            }
                                        }
                                        rdma_display_codec::Codec::Raw => {
                                            log::info!(
                                                "[server] raw source: trying scanout dma-buf MRs (host copy fallback)"
                                            );
                                            raw_direct = Some(direct::RawDirect::new(pd));
                                        }
                                        rdma_display_codec::Codec::Lossy => {}
                                    }
                                }
                            }
                            if let Some(g) = gpu_direct.as_ref() {
                                fc.set_gpu_staging(&g.buffers(), slot_len);
                            }
                            // TCP with the Vulkan materializer: send GPU output
                            // straight from a host staging buffer (the same one
                            // the encoder writes), sealed in place.
                            if link.transport() == rdma_display_rdma::Transport::Tcp
                                && session_codec != rdma_display_codec::Codec::Lossy
                                && materialize == "vk"
                            {
                                if let Some(pts) = mat.bind_staging(1, slot_len) {
                                    if let Some((ptr, _)) = pts.first() {
                                        if link.set_tcp_buffer(*ptr, slot_len + 16) {
                                            log::debug!(
                                                "[server] TCP video buffer: Vulkan host staging (zero-copy output)"
                                            );
                                        }
                                    }
                                }
                            }
                            pipelined_capable = external.is_some()
                                && link.transport() == rdma_display_rdma::Transport::Rdma;
                            pipelined = pipelined_capable;
                            session_meta = Some(meta.clone());
                            session = Some(link);
                            session_connected_at = Some(t_connect);
                            log::info!(
                                "[session] setup {} ms (workers + link)",
                                t_connect.elapsed().as_millis()
                            );
                        }
                        Err(e) => {
                            log::warn!("[server] session setup failed: {e:#}");
                            gpu_direct = None;
                            raw_direct = None;
                            fc.clear_gpu_staging();
                            session_nonce.store(0, std::sync::atomic::Ordering::SeqCst);
                *session_secret.lock().unwrap() = None;
                        }
                    }
                }
            } else {
                log::warn!(
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
            // Pickup time: when the sender decided to send the frame. The
            // fps gate and event-queue wait before this point are `queue`
            // time; a wait for a staging slot after it is `slot` time.
            // Keeping the FIRST pickup across retries keeps a credit wait out
            // of `queue` (the serial path splits it the same way).
            let t0d =
                (*send_pickup.get_or_insert_with(rdma_display_proto::unix_now_us)).max(t0_unix_us);
            let link = session.as_mut().unwrap();
            let slot_bytes = link.frame_bytes();
            // Per-frame raw source MR (RDMA dma-buf direct), if available.
            let mut raw_src = None;
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
                    if session_codec != rdma_display_codec::Codec::Lossy && fb > slot_bytes {
                        log::warn!(
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
                                &gpu,
                                &staging,
                                session_codec,
                                fmt.width,
                                fmt.height,
                                fmt.stride,
                                fmt.frame_bytes() as usize,
                            )?;
                        }
                        // Lossless on the GPU: import the scanout into the
                        // encoder and encode straight from GPU memory. When this
                        // succeeds we never touch the materializer — its
                        // per-frame dmabuf re-import would dominate the frame.
                        let mut texture_ok = false;
                        if session_codec == rdma_display_codec::Codec::Lossless && fc.has_gpu() {
                            match fc.set_scanout(&d) {
                                Ok(()) => {
                                    texture_ok = true;
                                    // texture-direct has no readback to pipeline
                                    pipelined = false;
                                }
                                Err(e) => {
                                    if !fc.scanout_warned {
                                        log::warn!(
                                            "[server] lossless Vulkan scanout import failed ({e:#}); waiting for a usable scanout"
                                        );
                                        fc.scanout_warned = true;
                                    }
                                    fc.clear_scanout();
                                }
                            }
                        }
                        if texture_ok {
                            have_frame = true;
                        } else if session_codec == rdma_display_codec::Codec::Lossless {
                            // The texture-direct path is the only lossless path.
                            // Without it there is no readback/CPU fallback; wait
                            // for a scanout the Vulkan encoder can import.
                        } else {
                            // raw/lossy share the materializer
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
                                    log::warn!(
                                        "[server] prepare scanout failed: {e:#}; holding last frame"
                                    );
                                    frame_dirty = false;
                                }
                            }
                        }
                        // Raw RDMA: try to post the compositor's scanout
                        // dma-buf directly (linear XRGB only). If this returns
                        // None the host staging copy below is used.
                        if have_frame && session_codec == rdma_display_codec::Codec::Raw {
                            if let Some(cache) = raw_direct.as_mut() {
                                raw_src = cache.source(&d, &fmt, slot_bytes);
                            }
                            if raw_src.is_some() {
                                if issued.is_empty() {
                                    // no GPU work in flight: take the direct
                                    // source path for this and later frames
                                    pipelined = false;
                                } else {
                                    // cannot switch away from pipelining with
                                    // staged copies in flight; use the host path
                                    raw_src = None;
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
                            log::debug!(
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
                            log::warn!("[server] issue failed: {e:#}");
                            link.release_slot(slot);
                            session = None;
                            mat.drop_frames();
                            issued.clear();
                        }
                    }
                }
            } else {
                match link.send_frame_checked(
                    |slot, out| {
                        // Raw dma-buf direct: the scanout MR already holds the
                        // frame; post it without touching host memory.
                        if let Some(src) = raw_src.take() {
                            let n = src.len();
                            return Ok(rdma_display_rdma::FrameFill {
                                changed: true,
                                post_len: n,
                                mat_times: rdma_display_proto::MatTimes::ZERO,
                                source: Some(src),
                            });
                        }
                        if fc.uses_scanout() {
                            // lossless texture-direct: encode from the imported
                            // dmabuf. Prefer the dma-buf-registered GPU output;
                            // otherwise write into the host staging slot.
                            if gpu_direct.is_some() {
                                let (changed, n, t) = fc.encode_scanout_gpu(slot)?;
                                let source = if changed {
                                    Some(
                                        gpu_direct
                                            .as_ref()
                                            .unwrap()
                                            .source(slot, n)
                                            .context("lossless GPU staging source")?,
                                    )
                                } else {
                                    None
                                };
                                return Ok(rdma_display_rdma::FrameFill {
                                    changed,
                                    post_len: n,
                                    mat_times: t,
                                    source,
                                });
                            }
                            let (changed, n, t) = fc.encode_scanout(slot, out)?;
                            return Ok(rdma_display_rdma::FrameFill {
                                changed,
                                post_len: n,
                                mat_times: t,
                                source: None,
                            });
                        }
                        let (changed, t) = mat.copy_frame(out)?;
                        if !changed {
                            return Ok(rdma_display_rdma::FrameFill::host(false, 0, t));
                        }
                        let n = match mat.coded_len() {
                            Some(n) => n,
                            None => fc.encode_in_place(slot, out)?,
                        };
                        Ok(rdma_display_rdma::FrameFill::host(true, n, t))
                    },
                    t0_unix_us,
                    t0d,
                    t0p,
                ) {
                    Ok(Some(seq)) => {
                        frame_dirty = false;
                        last_send = Some(std::time::Instant::now());
                        if client_metrics {
                            push_bounded(&mut coded_sizes, link.last_sent_bytes(), 256);
                        }
                        log::debug!("[server] frame {} sent", seq);
                        if seq == 0 {
                            log::debug!(
                                "[session] first frame block: materialize {} ms, encode+write {} ms",
                                t_before_tx.duration_since(t_block).as_millis(),
                                t_before_tx.elapsed().as_millis()
                            );
                            if let Some(t) = session_connected_at {
                                log::info!(
                                    "[session] first frame {} ms after connect",
                                    t.elapsed().as_millis()
                                );
                            }
                        }
                        if seq % 60 == 0 {
                            log::debug!("[stats] frames={} uptime={:?}", seq, started.elapsed());
                        }
                    }
                    Ok(None) => frame_dirty = false,
                    Err(e) => {
                        log::warn!("[server] send failed: {e:#}");
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
                        log::warn!("[server] finish failed: {e:#}");
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
                let post_len = match fc.encode_in_place(slot, out) {
                    Ok(n) => n,
                    Err(e) => {
                        log::warn!("[server] encode failed: {e:#}");
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
                        if client_metrics {
                            push_bounded(&mut coded_sizes, link.last_sent_bytes(), 256);
                        }
                        log::debug!("[server] frame {} sent", seq);
                        if seq == 0 {
                            if let Some(t) = session_connected_at {
                                log::info!(
                                    "[session] first frame {} ms after connect",
                                    t.elapsed().as_millis()
                                );
                            }
                        }
                        if seq % 60 == 0 {
                            log::debug!("[stats] frames={} uptime={:?}", seq, started.elapsed());
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        log::warn!("[server] send failed: {e:#}");
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
/// on the GPU (Vulkan compute, `rdma-display-vkcodec`).
/// The encoder's reference is reset in `configure` (once per session, and on a
/// geometry change) so the first frame a new client receives is always full.
struct FrameCodec {
    /// Vulkan lossless encoder; None for raw.
    vk: Option<rdma_display_vkcodec::Encoder>,
    /// shared Vulkan device + scanout import cache
    gpu: Option<Arc<gpu::Gpu>>,
    /// currently sampled scanout (owned by the cache)
    last_image: Option<Arc<gpu::CachedImage>>,
    frame_bytes: usize,
    geom: Option<rdma_display_codec::Geometry>,
    /// the next encode must be a full frame (new session / new geometry)
    first: bool,
    /// a scanout dmabuf is imported for texture-direct encode
    scanout: bool,
    /// set once the texture-direct import failed (avoid per-frame log spam)
    scanout_warned: bool,
    /// shared per-slot Vulkan staging buffers (raw/lossless direct output)
    staging: gpu::SharedStaging,
    /// optional dma-buf-registered coded buffers (lossless RDMA zero-copy)
    gpu_buffers: Vec<rdma_display_vkcore::ash::vk::Buffer>,
    gpu_len: usize,
}

impl FrameCodec {
    fn new() -> Self {
        Self {
            vk: None,
            gpu: None,
            last_image: None,
            frame_bytes: 0,
            geom: None,
            first: true,
            scanout: false,
            scanout_warned: false,
            staging: Arc::new(std::sync::Mutex::new(None)),
            gpu_buffers: Vec::new(),
            gpu_len: 0,
        }
    }

    /// Drop per-session encoder state so an idle server holds no codec
    /// buffers; the next session reconfigures.
    fn teardown(&mut self) {
        self.vk = None;
        self.gpu = None;
        self.last_image = None;
        self.geom = None;
        self.first = true;
        self.scanout = false;
        self.frame_bytes = 0;
    }

    fn configure(
        &mut self,
        gpu: &Arc<gpu::Gpu>,
        staging: &gpu::SharedStaging,
        codec: rdma_display_codec::Codec,
        width: u32,
        height: u32,
        stride: u32,
        frame_bytes: usize,
    ) -> Result<()> {
        self.staging = staging.clone();
        self.frame_bytes = frame_bytes;
        self.geom = None;
        self.vk = None;
        self.gpu = Some(gpu.clone());
        self.last_image = None;
        self.first = true;
        self.scanout = false;
        match codec {
            rdma_display_codec::Codec::Raw | rdma_display_codec::Codec::Lossy => {}
            rdma_display_codec::Codec::Lossless => {
                let geom = rdma_display_codec::Geometry::new(width, height, stride);
                self.geom = Some(geom);
                self.vk = Some(
                    rdma_display_vkcodec::Encoder::with_core(gpu.core.clone(), geom)
                        .context("lossless Vulkan encoder (no CPU fallback)")?,
                );
                log::info!("[server] lossless encoder: GPU (Vulkan compute)");
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

    /// Encode `out[..frame_bytes]` in place; returns the number of bytes to
    /// post (coded length, or `frame_bytes` for raw). When `slot_idx` names a
    /// shared staging buffer the encoder writes the coded bytes straight into
    /// it (no CPU copy).
    fn encode_in_place(&mut self, slot_idx: usize, out: &mut [u8]) -> Result<usize> {
        let Some(vk) = self.vk.as_mut() else {
            // raw has no codec; lossless always installs the Vulkan encoder
            return Ok(self.frame_bytes);
        };
        let staging = self.staging.lock().unwrap().clone();
        let target = staging.as_ref().and_then(|s| {
            s.buffer_for(slot_idx, out.as_ptr() as *mut u8)
                .map(|b| (b, s.len + 16))
        });
        let direct = target.is_some();
        if let Some((buf, cap)) = target {
            vk.set_output(buf, cap)?;
        }
        let n = vk.encode_host(&out[..self.frame_bytes])?;
        ensure!(
            n <= out.len(),
            "coded frame {n} exceeds staging slot {}",
            out.len()
        );
        if !direct {
            out[..n].copy_from_slice(&vk.coded()[..n]);
        }
        self.first = false;
        Ok(n)
    }

    fn has_gpu(&self) -> bool {
        self.vk.is_some()
    }

    /// Whether the texture-direct path is active for this frame.
    fn uses_scanout(&self) -> bool {
        self.scanout && self.vk.is_some()
    }

    fn clear_scanout(&mut self) {
        self.scanout = false;
        self.last_image = None;
    }

    /// Import the scanout dmabuf into the Vulkan encoder so
    /// [`encode_scanout`](Self::encode_scanout) samples it directly.
    fn set_scanout(&mut self, d: &rdma_display_capture::DmabufScanout) -> Result<()> {
        let Some(gpu) = self.gpu.as_ref() else {
            anyhow::bail!("no Vulkan GPU");
        };
        let Some(vk) = self.vk.as_mut() else {
            anyhow::bail!("no Vulkan encoder");
        };
        use rdma_display_vkcore::ash::vk;
        let img = gpu.import(d)?;
        if !matches!(
            img.format,
            vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
        ) {
            anyhow::bail!(
                "scanout fourcc {:#010x} is not 8-bit BGRA/RGBA; the Vulkan lossless path cannot import it",
                d.format.fourcc
            );
        }
        if self
            .last_image
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, &img))
        {
            return Ok(());
        }
        img.ensure_layout(rdma_display_vkcore::ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        vk.set_scanout_image(img.view())?;
        self.last_image = Some(img);
        if !self.scanout {
            log::debug!(
                "[server] lossless Vulkan: scanout imported (modifier={:#x})",
                d.format.modifier
            );
        }
        self.scanout = true;
        Ok(())
    }

    /// Vulkan texture-direct encode. `changed` follows the changed-tile
    /// count; a zero-tile delta is not sent.
    fn encode_scanout(
        &mut self,
        slot_idx: usize,
        out: &mut [u8],
    ) -> Result<(bool, usize, rdma_display_proto::MatTimes)> {
        let staging = self.staging.lock().unwrap().clone();
        let target = staging.as_ref().and_then(|s| {
            s.buffer_for(slot_idx, out.as_ptr() as *mut u8)
                .map(|b| (b, s.len + 16))
        });
        self.encode_scanout_target(target, Some(out))
    }

    /// Lossless encode straight into the dma-buf-registered GPU coded buffer
    /// for `slot_idx`. No host bytes are produced or copied.
    fn encode_scanout_gpu(
        &mut self,
        slot_idx: usize,
    ) -> Result<(bool, usize, rdma_display_proto::MatTimes)> {
        let buffer = *self
            .gpu_buffers
            .get(slot_idx)
            .context("no dma-buf lossless staging slot")?;
        self.encode_scanout_target(Some((buffer, self.gpu_len)), None)
    }

    fn encode_scanout_target(
        &mut self,
        target: Option<(rdma_display_vkcore::ash::vk::Buffer, usize)>,
        out: Option<&mut [u8]>,
    ) -> Result<(bool, usize, rdma_display_proto::MatTimes)> {
        let full = self.first;
        let Some(vk) = self.vk.as_mut() else {
            anyhow::bail!("no Vulkan encoder");
        };
        if let Some((buf, cap)) = target {
            vk.set_output(buf, cap)?;
        }
        let t = std::time::Instant::now();
        let n = vk.encode_scanout()?;
        let enc = t.elapsed();
        let count = vk.last_count();
        let changed = full || count > 0;
        if changed {
            self.first = false;
        }
        if let Some((_, cap)) = target {
            ensure!(n <= cap, "coded frame {n} exceeds output buffer {cap}");
        }
        if let Some(out) = out.as_deref() {
            ensure!(
                n <= out.len(),
                "coded frame {n} exceeds staging slot {}",
                out.len()
            );
        }
        let t = std::time::Instant::now();
        if target.is_none() {
            let out = out.context("lossless output buffer")?;
            out[..n].copy_from_slice(&vk.coded()[..n]);
        }
        let copy = t.elapsed();
        Ok((
            changed,
            n,
            rdma_display_proto::MatTimes::new(
                rdma_display_proto::MatKind::Lossless,
                [0, 0, enc.as_micros() as u64, 0, copy.as_micros() as u64, 0],
            ),
        ))
    }

    /// Install the per-slot dma-buf coded buffers for this session.
    fn set_gpu_staging(&mut self, buffers: &[rdma_display_vkcore::ash::vk::Buffer], len: usize) {
        self.gpu_buffers.clear();
        self.gpu_buffers.extend_from_slice(buffers);
        self.gpu_len = len;
    }

    fn clear_gpu_staging(&mut self) {
        self.gpu_buffers.clear();
        self.gpu_len = 0;
    }
}

/// Input-connection worker: validate the session nonce, forward input events
/// to the injection sink, and release everything the client held when the
/// socket dies — a network blip must never leave a key stuck down.
fn serve_input(
    stream: std::net::TcpStream,
    encrypt: bool,
    secret: [u8; rdma_display_auth::SECRET_LEN],
    input_tx: mpsc::Sender<rdma_display_proto::InputEvent>,
    addr: std::net::SocketAddr,
) {
    log::info!("[input] client input stream connected ({addr})");
    let mut conn = if encrypt {
        let key =
            rdma_display_auth::derive_stream_key(&secret, rdma_display_proto::TAG_INPUT, true);
        Conn::encrypted(stream, None, Some(Box::new(rdma_display_auth::Open::new(&key))))
    } else {
        Conn::new(stream)
    };
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
            Ok(other) => log::warn!("[input] unexpected message {other:?}"),
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
    log::info!("[input] client input stream ended ({addr}); released held input");
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
    src_period: &mut SourcePeriod,
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
            src_period.observe(t0);
            *frame_dirty = true;
        }
        CaptureEvent::PlainScanout(fmt, data, t0) => {
            *t0_unix_us = t0;
            src_period.observe(t0);
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
                src_period.observe(*t0);
            }
            if mat.handle_event(update)? {
                *frame_dirty = true;
            }
        }
        CaptureEvent::Disabled => {
            log::warn!("[server] console scanout disabled");
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

/// Apply the client's pre-handshake display-mode request to the server's
/// compositor. On an actual change, remembers the previous mode in `original`
/// (and arms the signal-restore hook) so it can be put back on disconnect.
fn apply_mode_request(
    req: &rdma_display_proto::ModeRequest,
    ctl: Option<&Arc<dyn display::DisplayControl>>,
    original: &mut Option<display::Mode>,
) -> rdma_display_proto::ModeReply {
    use rdma_display_proto::{ModeReply, ModeRequest};

    let Some(ctl) = ctl else {
        return match req {
            ModeRequest::Keep => {
                ModeReply::ok(0, 0, 0, "no display-control backend; keeping current mode")
            }
            _ => ModeReply::err(0, 0, 0, "no display-control backend (--display-control)"),
        };
    };

    match req {
        ModeRequest::Keep => match ctl.current() {
            Ok(m) => ModeReply::ok(m.w, m.h, m.refresh_mhz, "keeping current mode"),
            Err(e) => ModeReply::err(0, 0, 0, format!("{e:#}")),
        },
        ModeRequest::Reset => match *original {
            Some(orig) => match ctl.set(orig) {
                Ok(()) => {
                    *original = None;
                    display::disarm_restore();
                    ModeReply::ok(
                        orig.w,
                        orig.h,
                        orig.refresh_mhz,
                        format!("restored {}", orig.label()),
                    )
                }
                Err(e) => ModeReply::err(0, 0, 0, format!("{e:#}")),
            },
            None => ModeReply::err(0, 0, 0, "no saved mode to restore"),
        },
        ModeRequest::Set {
            width,
            height,
            refresh_mhz,
        } => {
            let cur = match ctl.current() {
                Ok(m) => m,
                Err(e) => return ModeReply::err(0, 0, 0, format!("{e:#}")),
            };
            let modes = match ctl.modes() {
                Ok(m) => m,
                Err(e) => return ModeReply::err(cur.w, cur.h, cur.refresh_mhz, format!("{e:#}")),
            };
            let Some(chosen) = display::nearest(&modes, *width, *height, *refresh_mhz) else {
                return ModeReply::err(cur.w, cur.h, cur.refresh_mhz, "output advertises no modes");
            };
            if let Err(e) = ctl.set(chosen) {
                return ModeReply::err(cur.w, cur.h, cur.refresh_mhz, format!("{e:#}"));
            }
            if chosen != cur {
                if original.is_none() {
                    *original = Some(cur);
                    display::arm_restore(ctl.clone(), cur);
                }
                ModeReply::ok(
                    chosen.w,
                    chosen.h,
                    chosen.refresh_mhz,
                    format!("applied {}", chosen.label()),
                )
            } else {
                ModeReply::ok(
                    chosen.w,
                    chosen.h,
                    chosen.refresh_mhz,
                    format!("already at {}", chosen.label()),
                )
            }
        }
    }
}

/// Nominal frame rate used to convert a lossy bandwidth target into the
/// encoder's per-frame byte cap.
const LOSSY_NOMINAL_FPS: usize = 60;

/// The lossy encoder's per-frame bitstream cap (bytes) for a bandwidth target
/// in Mbit/s, assuming [`LOSSY_NOMINAL_FPS`]: `MBPS * 1e6 / 8 / 60`. PyroWave's
/// rate control only takes a size cap (there is no target-bitrate field), so
/// the achieved bitrate is `cap * fps` — at or below the target up to 60 fps.
fn lossy_budget_bytes(mbps: u16) -> usize {
    (mbps as usize) * 1_000_000 / 8 / LOSSY_NOMINAL_FPS
}

/// Build the frame materializer for `codec`: the lossy (PyroWave/Vulkan) one, or
/// the Vulkan/plain one shared by raw and lossless. Called at startup and again
/// when a session's resolved codec changes the materializer family.
fn build_materializer(
    codec: rdma_display_codec::Codec,
    materialize: &str,
    gpu: &Arc<gpu::Gpu>,
    staging: &gpu::SharedStaging,
    lossy_budget: usize,
    lossy_chroma444: bool,
) -> Result<Box<dyn FrameMaterializer>> {
    if codec == rdma_display_codec::Codec::Lossy {
        return Ok(Box::new(lossy::LossyMaterializer::new(
            gpu.clone(),
            lossy_budget,
            lossy_chroma444,
        )?));
    }
    Ok(match materialize {
        "vk" => Box::new(gpu::VkMaterializer::new(gpu.clone(), staging.clone())),
        "plain" => Box::new(rdma_display_capture::materializer::PlainMaterializer::new()),
        other => bail!("materializer {other:?} not implemented (vk|plain)"),
    })
}

/// Resolve the session codec from the server's `--codec` and the client's
/// `Hello` preference. With `--codec auto` the client may pick any codec it can
/// decode (raw/lossless/lossy); an unknown id falls back. A non-auto `--codec`
/// always wins (logged when it overrides the client's request).
fn resolve_codec(
    auto: bool,
    fallback: rdma_display_codec::Codec,
    client_pref: u8,
) -> rdma_display_codec::Codec {
    use rdma_display_codec::Codec;
    if client_pref == rdma_display_proto::CODEC_ANY {
        return fallback;
    }
    let Some(client) = Codec::from_id(client_pref) else {
        log::warn!(
            "[server] client requested unknown codec id {client_pref}; using {}",
            fallback.name()
        );
        return fallback;
    };
    if !auto {
        if client != fallback {
            log::info!(
                "[server] --codec {} overrides the client's {} request",
                fallback.name(),
                client.name()
            );
        }
        return fallback;
    }
    log::info!(
        "[server] codec {} (client preference; --codec auto)",
        client.name()
    );
    client
}

/// Re-probe the DRM scanout, waiting (bounded) for a requested mode to land.
///
/// `hyprctl`/`kscreen-doctor` return before the compositor's KMS modeset is
/// committed, so an immediate `probe_display` can still report the old size —
/// which sizes the session's slots for the wrong geometry and drops every
/// frame. `connector` is the captured output (None = first active); `want_w`/
/// `want_h` are the mode reply's dimensions; `0` means no mode was requested
/// and the first probe is used as-is.
fn reprobe_scanout(
    card: Option<&str>,
    connector: Option<&str>,
    want_w: u32,
    want_h: u32,
) -> Result<FrameFormat> {
    const SETTLE: std::time::Duration = std::time::Duration::from_millis(2000);
    const POLL: std::time::Duration = std::time::Duration::from_millis(20);
    let deadline = std::time::Instant::now() + SETTLE;
    let mut fmt = rdma_display_capture::drm::probe_display(card, connector)?;
    while want_w > 0 && want_h > 0 && (fmt.width, fmt.height) != (want_w, want_h) {
        if std::time::Instant::now() >= deadline {
            log::debug!(
                "[display] scanout still {}x{} after {} ms (wanted {}x{}); sizing from it",
                fmt.width,
                fmt.height,
                SETTLE.as_millis(),
                want_w,
                want_h
            );
            break;
        }
        std::thread::sleep(POLL);
        fmt = rdma_display_capture::drm::probe_display(card, connector)?;
    }
    Ok(fmt)
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
    codec_auto: bool,
    lossy_budget: usize,
    lossy_chroma444: bool,
    gpu: &Arc<gpu::Gpu>,
    drm_connector: Option<&str>,
) -> Option<DisplayMeta> {
    let mut geometry = None;

    // 1. display geometry (and proof the source can be opened at all)
    match source_name {
        "drm" => match rdma_display_capture::drm::probe_display(drm_card.as_deref(), drm_connector) {
            Ok(fmt) => {
                log::debug!(
                    "[preflight] drm scanout {}x{} '{}' modifier={:#x}",
                    fmt.width,
                    fmt.height,
                    rdma_display_capture::fourcc_name(fmt.fourcc),
                    fmt.modifier
                );
                geometry = Some(meta_linear(&fmt));
            }
            Err(e) => log::warn!("[preflight] drm probe failed: {e:#}"),
        },
        "qemu" => match rdma_display_capture::qemu::probe_console(dbus_socket, console) {
            Ok((w, h)) => {
                log::debug!("[preflight] qemu console {w}x{h}");
                geometry = Some(meta_from_size(w, h));
            }
            Err(e) => log::warn!("[preflight] qemu probe failed: {e:#}"),
        },
        _ => {}
    }

    // 2. lossless GPU encoder (Vulkan compute: SPIR-V shaders + buffers)
    if codec == rdma_display_codec::Codec::Lossless {
        let g = geometry
            .as_ref()
            .map(|m| rdma_display_codec::Geometry::new(m.width, m.height, m.stride))
            .unwrap_or_else(|| rdma_display_codec::Geometry::new(1920, 1080, 1920 * 4));
        match rdma_display_vkcodec::Encoder::with_core(gpu.core.clone(), g) {
            Ok(_) => log::debug!("[preflight] lossless encoder ok ({}x{})", g.width, g.height),
            Err(e) => {
                log::warn!("[preflight] lossless encoder failed: {e:#} (Vulkan/dma-buf required)")
            }
        }
    }

    // 2b. lossy encoder (PyroWave/Vulkan) — validate when lossy can be selected
    // (a client request under --codec auto, or a server-forced --codec lossy)
    if codec == rdma_display_codec::Codec::Lossy || codec_auto {
        match lossy::LossyMaterializer::new(gpu.clone(), lossy_budget, lossy_chroma444) {
            Ok(m) => {
                drop(m);
                log::debug!("[preflight] lossy encoder ok");
            }
            Err(e) => log::warn!(
                "[preflight] lossy encoder failed: {e:#} (client lossy requests will fail)"
            ),
        }
    }

    // 3. input injection (uinput device creation; QEMU input shares the
    //    console probe above)
    if input_name == "uinput" {
        match rdma_display_capture::input::UinputMouse::open() {
            Ok(m) => {
                drop(m);
                log::debug!("[preflight] uinput ok");
            }
            Err(e) => log::warn!("[preflight] uinput failed: {e:#}"),
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
        let gpu = crate::gpu::Gpu::new().unwrap();
        let staging: crate::gpu::SharedStaging =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut fc = FrameCodec::new();
        fc.configure(&gpu, &staging, Codec::Lossless, w, h, stride, fb).unwrap();
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
            let n = fc.encode_in_place(0, &mut slot).unwrap();
            assert!(n <= slot.len());
            let out = dec.decode(&slot[..n]).unwrap();
            assert_eq!(out, &orig[..], "frame {i} round trip");
        }
    }

    /// Direct-output variant: when the codec writes into the transport's
    /// staging `VkBuffer` (pointer matches), the coded bytes must land in that
    /// memory with no CPU copy and still round-trip.
    #[test]
    fn lossless_direct_staging_round_trips() {
        use rdma_display_capture::FrameMaterializer;
        let (w, h) = (256u32, 256u32);
        let stride = w * 4;
        let geom = Geometry::new(w, h, stride);
        let fb = geom.frame_bytes();
        let slot_len = geom.max_coded_len();
        let gpu = crate::gpu::Gpu::new().unwrap();
        let staging: crate::gpu::SharedStaging =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut mat = crate::gpu::VkMaterializer::new(gpu.clone(), staging.clone());
        assert!(mat.bind_staging(1, slot_len).is_some(), "bind staging");
        let ptr = staging.lock().unwrap().as_ref().unwrap().buffers[0].ptr;
        let slot = unsafe { std::slice::from_raw_parts_mut(ptr, slot_len) };
        let mut fc = FrameCodec::new();
        fc.configure(&gpu, &staging, Codec::Lossless, w, h, stride, fb).unwrap();
        let mut dec = Decoder::new(geom).unwrap();
        let base = synthetic(w, h, 7);
        for i in 0..3u32 {
            let mut orig = base.clone();
            let off = (i as usize * 8192) % orig.len();
            for k in 0..64 {
                let p = (off + k) % orig.len();
                orig[p] ^= 0x3c;
            }
            assert!(fb <= slot.len());
            slot[..fb].copy_from_slice(&orig);
            let n = fc.encode_in_place(0, slot).unwrap();
            assert!(n <= slot.len());
            let out = dec.decode(&slot[..n]).unwrap();
            assert_eq!(out, &orig[..], "direct frame {i} round trip");
        }
    }
}

#[cfg(test)]
mod display_apply_tests {
    use super::*;
    use std::sync::Mutex;

    /// A backend whose mode list is fixed and whose current mode is settable,
    /// so `apply_mode_request` can be exercised without a compositor.
    struct Mock {
        modes: Vec<display::Mode>,
        cur: Mutex<display::Mode>,
    }

    impl display::DisplayControl for Mock {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn output(&self) -> Result<String> {
            Ok("MOCK-1".into())
        }
        fn modes(&self) -> Result<Vec<display::Mode>> {
            Ok(self.modes.clone())
        }
        fn current(&self) -> Result<display::Mode> {
            Ok(*self.cur.lock().unwrap())
        }
        fn set(&self, m: display::Mode) -> Result<()> {
            *self.cur.lock().unwrap() = m;
            Ok(())
        }
    }

    fn m(w: u32, h: u32, r: u32) -> display::Mode {
        display::Mode {
            w,
            h,
            refresh_mhz: r,
        }
    }

    fn backend(cur: display::Mode, modes: Vec<display::Mode>) -> Arc<dyn display::DisplayControl> {
        Arc::new(Mock {
            modes,
            cur: Mutex::new(cur),
        })
    }

    #[test]
    fn keep_reports_current_without_changing() {
        let ctl = backend(m(1920, 1080, 60_000), vec![m(1920, 1080, 60_000)]);
        let mut orig = None;
        let r = apply_mode_request(
            &rdma_display_proto::ModeRequest::Keep,
            Some(&ctl),
            &mut orig,
        );
        assert!(r.ok);
        assert_eq!((r.width, r.height), (1920, 1080));
        assert!(orig.is_none());
    }

    #[test]
    fn set_picks_nearest_and_remembers_original() {
        let ctl = backend(
            m(1280, 720, 60_000),
            vec![m(1920, 1080, 60_000), m(2560, 1440, 144_000)],
        );
        let mut orig = None;
        let r = apply_mode_request(
            &rdma_display_proto::ModeRequest::Set {
                width: 2500,
                height: 1400,
                refresh_mhz: 143_000,
            },
            Some(&ctl),
            &mut orig,
        );
        assert!(r.ok, "{r:?}");
        assert_eq!((r.width, r.height), (2560, 1440));
        // the pre-change mode is remembered so it can be restored
        assert_eq!(orig, Some(m(1280, 720, 60_000)));
    }

    #[test]
    fn set_already_at_current_is_a_noop() {
        let cur = m(1920, 1080, 60_000);
        let ctl = backend(cur, vec![cur]);
        let mut orig = None;
        let r = apply_mode_request(
            &rdma_display_proto::ModeRequest::Set {
                width: 1920,
                height: 1080,
                refresh_mhz: 0,
            },
            Some(&ctl),
            &mut orig,
        );
        assert!(r.ok);
        assert!(orig.is_none(), "no change => nothing to restore");
    }

    #[test]
    fn reset_restores_the_remembered_mode() {
        let ctl = backend(m(2560, 1440, 144_000), vec![m(2560, 1440, 144_000)]);
        let mut orig = Some(m(1920, 1080, 60_000));
        let r = apply_mode_request(
            &rdma_display_proto::ModeRequest::Reset,
            Some(&ctl),
            &mut orig,
        );
        assert!(r.ok, "{r:?}");
        assert_eq!((r.width, r.height), (1920, 1080));
        assert!(orig.is_none(), "restored => nothing left to restore");
        // the backend really is back at the old mode
        assert_eq!(ctl.current().unwrap(), m(1920, 1080, 60_000));
    }

    #[test]
    fn no_backend_errors_for_set_but_not_for_keep() {
        let mut orig = None;
        let keep = apply_mode_request(
            &rdma_display_proto::ModeRequest::Keep,
            None,
            &mut orig,
        );
        assert!(keep.ok);
        let set = apply_mode_request(
            &rdma_display_proto::ModeRequest::Set {
                width: 1920,
                height: 1080,
                refresh_mhz: 0,
            },
            None,
            &mut orig,
        );
        assert!(!set.ok);
    }
}

#[cfg(test)]
mod listen_arg_tests {
    use super::*;

    #[test]
    fn appends_default_port_when_missing() {
        assert_eq!(
            with_default_port("192.168.4.2"),
            format!("192.168.4.2:{DEFAULT_PORT}")
        );
        assert_eq!(with_default_port("0.0.0.0"), format!("0.0.0.0:{DEFAULT_PORT}"));
    }

    #[test]
    fn preserves_explicit_port() {
        assert_eq!(with_default_port("192.168.4.2:5000"), "192.168.4.2:5000");
        assert_eq!(with_default_port("[::1]:4791"), "[::1]:4791");
    }

    #[test]
    fn leaves_bare_ipv6_and_empty_alone() {
        assert_eq!(with_default_port("::1"), "::1");
        assert_eq!(with_default_port(""), "");
    }
}

#[cfg(test)]
mod codec_resolve_tests {
    use super::resolve_codec;
    use rdma_display_codec::Codec;

    #[test]
    fn auto_honours_raw_and_lossless() {
        assert_eq!(resolve_codec(true, Codec::Lossless, Codec::Raw.id()), Codec::Raw);
        assert_eq!(
            resolve_codec(true, Codec::Raw, Codec::Lossless.id()),
            Codec::Lossless
        );
        // lossy is client-selectable under auto too
        assert_eq!(
            resolve_codec(true, Codec::Lossless, Codec::Lossy.id()),
            Codec::Lossy
        );
    }

    #[test]
    fn auto_falls_back_on_any_and_unknown() {
        assert_eq!(
            resolve_codec(true, Codec::Lossless, rdma_display_proto::CODEC_ANY),
            Codec::Lossless
        );
        assert_eq!(resolve_codec(true, Codec::Lossless, 0xEE), Codec::Lossless);
    }

    #[test]
    fn forced_codec_wins() {
        assert_eq!(
            resolve_codec(false, Codec::Lossless, Codec::Raw.id()),
            Codec::Lossless
        );
        assert_eq!(
            resolve_codec(false, Codec::Lossy, Codec::Raw.id()),
            Codec::Lossy
        );
        assert_eq!(
            resolve_codec(false, Codec::Lossy, rdma_display_proto::CODEC_ANY),
            Codec::Lossy
        );
    }
}

#[cfg(test)]
mod pace_tests {
    use super::{pace_window, Pace, SourcePeriod};

    /// A sub-vblank interval (two `t0` reads close together) must not shrink
    /// the source period below the output's frame period.
    #[test]
    fn source_period_is_floored_at_the_display_period() {
        let mut sp = SourcePeriod::new(16_667);
        let mut t0 = 1_000_000u64;
        sp.observe(t0);
        t0 += 16_667; // a real 60 Hz flip
        sp.observe(t0);
        t0 += 3_000; // a back-to-back artifact
        sp.observe(t0);
        t0 += 16_667;
        sp.observe(t0);
        assert_eq!(sp.period_us(), 16_667, "artifact must not shrink the period");

        // a sustained slow/idle source is capped at 2x the display period
        let mut sp = SourcePeriod::new(16_667);
        let mut t0 = 1_000_000u64;
        sp.observe(t0);
        for _ in 0..4 {
            t0 += 40_000;
            sp.observe(t0);
        }
        assert_eq!(sp.period_us(), 33_334, "slow source must not halve the window");

        // with no known mode period the raw minimum (the artifact) shows through
        let mut sp = SourcePeriod::new(0);
        let mut t0 = 1_000_000u64;
        sp.observe(t0);
        t0 += 16_667;
        sp.observe(t0);
        t0 += 3_000;
        sp.observe(t0);
        assert_eq!(sp.period_us(), 3_000);
    }

    #[test]
    fn latency_mode_is_present_paced() {
        // always one frame in flight, measured or not
        assert_eq!(pace_window(Pace::Latency, 10_000, 16_000, 3), 1);
        assert_eq!(pace_window(Pace::Latency, 0, 0, 3), 1);
    }

    #[test]
    fn smooth_mode_uses_the_transport_maximum() {
        assert_eq!(pace_window(Pace::Smooth, 10_000, 16_000, 3), 3);
        // max is always at least 1 (a zero max must not produce a zero window)
        assert_eq!(pace_window(Pace::Smooth, 0, 0, 0), 1);
    }

    #[test]
    fn auto_is_a_safe_default_then_the_frame_bdp() {
        // unmeasured (either measurement missing): small, safe, low-latency default
        assert_eq!(pace_window(Pace::Auto, 0, 0, 3), 2);
        assert_eq!(pace_window(Pace::Auto, 0, 16_000, 3), 2);
        // fast link: turnaround below one source frame period -> present-paced
        assert_eq!(pace_window(Pace::Auto, 8_000, 16_000, 3), 1);
        // turnaround spanning ~1.5 source periods -> two frames in flight
        assert_eq!(pace_window(Pace::Auto, 24_000, 16_000, 3), 2);
        // slow link: grows but is clamped to the staging-slot maximum
        assert_eq!(pace_window(Pace::Auto, 100_000, 16_000, 3), 3);
    }

    #[test]
    fn auto_uses_the_source_period_not_the_send_interval() {
        // The regression that collapsed the rate to ~1/turnaround: a 30 ms RTT
        // plus a 120 Hz client gives a ~40 ms loop, so with a 60 Hz source the
        // window must be ceil(40 / 16.7) = 3 (the max), not 1. The old code
        // divided by the *send* interval instead, making the formula
        // `ceil(turnaround / (turnaround/window)) == window` — a neutral fixed
        // point that just held whatever it first computed.
        assert_eq!(pace_window(Pace::Auto, 40_000, 16_700, 3), 3);
        // and it is stable: the source period does not move with the window, so
        // a correct window stays correct (unlike the send-interval form)
        assert_eq!(pace_window(Pace::Auto, 40_000, 16_700, 8), 3);
    }
}

#[cfg(test)]
mod source_period_tests {
    use super::SourcePeriod;

    #[test]
    fn empty_until_two_samples() {
        let mut s = SourcePeriod::default();
        assert_eq!(s.period_us(), 0);
        s.observe(1_000_000);
        assert_eq!(s.period_us(), 0); // one sample: no interval yet
        s.observe(1_016_667);
        assert_eq!(s.period_us(), 16_667);
    }

    #[test]
    fn converges_on_the_active_period_ignoring_idle_gaps() {
        // 60 Hz bursts with every 5th interval an idle gap — the case that made
        // the EMA wobble between ~16 and ~155 ms
        let mut s = SourcePeriod::default();
        let mut t = 1_000_000u64;
        s.observe(t);
        for i in 0..80u64 {
            t += if i % 5 == 4 { 150_000 } else { 16_667 };
            s.observe(t);
        }
        let p = s.period_us();
        assert!((15_000..=25_000).contains(&p), "period {p} us (want ~16667)");
    }
}
