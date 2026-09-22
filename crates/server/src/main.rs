//! rdma-display server: captures a display source and streams frames to a
//! client over RDMA.
//!
//! Modes:
//!   --mode testcard   synthetic frames (validates the transport end-to-end)
//!   --mode capture    --source qemu|drm:
//!     qemu  QEMU dbus display capture (the guest):
//!             --materialize egl    GPU blit+readback (default, gl=on)
//!             --materialize plain  D-Bus host-memory scanouts (gl=off)
//!             --readback pbo       zero-copy staging: GL readback DMA fills
//!                                  persistent-mapped PBOs registered as RDMA
//!                                  slots (default for egl); "direct" = legacy
//!                                  readPixels path
//!             --input qemu         inject into the guest console (default)
//!     drm   KMS capture of the host screen (Sunshine-style): grabs the
//!             CRTC's current fb via GETFB2 as a dmabuf each flip; requires
//!             sudo (CAP_SYS_ADMIN) and kernel >= 5.8; egl materialize only
//!             --input uinput        inject into the host compositor (default)

use std::net::TcpListener;
use std::sync::mpsc;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rdma_display_proto::{Conn, DisplayMeta};
use rdma_display_capture::{
    CaptureEvent, CaptureSource, DrmKmsSource, FrameFormat, FrameMaterializer, InputSink,
    QemuDbusInput, QemuDbusSource, ScreenSize,
};

fn main() -> Result<()> {
    let mut mode = "testcard".to_string();
    let mut device: Option<String> = None;
    let mut listen: String = "0.0.0.0:4791".to_string();
    let mut slots: usize = 3;
    let mut fps: u64 = 30;
    let mut source = "qemu".to_string();
    let mut input: Option<String> = None; // None = paired default for the source
    let mut dbus_socket: Option<String> = None;
    let mut console: u32 = 0;
    let mut materialize: String = "egl".to_string();
    let mut rendernode: Option<String> = None;
    let mut drm_card: Option<String> = None;
    let mut input_test = false;
    let mut readback: String = "pbo".to_string();
    let mut audio = false;
    let mut audio_device: Option<String> = None;
    let mut audio_helper = false;

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
            "--input-test" => input_test = true,
            "--audio" => audio = true,
            "--audio-device" => audio_device = Some(args.next().expect("--audio-device needs a value")),
            "--audio-helper" => audio_helper = true,
            "--help" | "-h" => {
                println!("usage: rdma-display-server [--mode testcard|capture] [--device mlx5_0] [--listen 0.0.0.0:4791] [--slots 3] [--fps 60]\n  \
capture: --source qemu|drm (qemu = QEMU dbus-display, the guest; drm = KMS capture, the host — requires sudo/CAP_SYS_ADMIN, kernel >= 5.8)\n  \
  --input qemu|uinput (default auto: qemu↔qemu-dbus, drm↔uinput)\n  \
  qemu: [--dbus-socket /path/dbus.sock] [--console 0]   drm: [--drm-card /dev/dri/card0]\n  \
  [--materialize egl|plain] [--rendernode /dev/dri/renderD128] [--readback pbo|direct] [--input-test]\n  \
  --fps N (capture): pacing cap; 0 = uncapped, consumer-paced by client credits (default 30). For lower latency use 60 or 0.\n  \
  --audio [--audio-device my-sink.monitor]  audio side-channel: capture the named source's PCM through the pipewire-pulse\n  \
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

    let ctx = rdma_display_rdma::open_device(device.as_deref())?;
    let pd = ctx.alloc_pd()?;

    match mode.as_str() {
        "testcard" => run_testcard_server(&ctx, &pd, &listen, slots, fps),
        "capture" => run_capture(
            &ctx, &pd, &source, input.as_deref(), &dbus_socket, console, &drm_card,
            &materialize, rendernode.as_deref(), &listen, slots, fps, input_test, &readback,
            audio, audio_device.as_deref(),
        ),
        other => bail!("mode {other:?} not implemented yet"),
    }
}

fn run_testcard_server(
    ctx: &ibverbs::Context,
    pd: &ibverbs::ProtectionDomain,
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
    for stream in listener.incoming() {
        let mut stream = stream?;
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
            ctx,
            pd,
            &meta,
            slots,
            input_tx,
            None,
            &rumble_hub,
            rdma_display_rdma::random_nonce(),
        )?;
        eprintln!("session up: {slots} slots, {} frame bytes", link.frame_bytes());
        run_testcard(&mut link, fps)?;
    }
    Ok(())
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

#[allow(clippy::too_many_arguments)]
fn run_capture(
    ctx: &ibverbs::Context,
    pd: &ibverbs::ProtectionDomain,
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
    audio: bool,
    audio_device: Option<&str>,
) -> Result<()> {
    if source_name == "drm" && materialize != "egl" {
        bail!("--source drm requires --materialize egl (KMS capture is dmabuf-only)");
    }
    // audio side-channel: capture starts with the server (video flows even
    // if no client ever connects the audio stream; the mailbox just
    // overwrites — realtime drop-oldest, zero backpressure)
    let audio_mbox = if audio {
        Some(rdma_display_audio::capture::start_capture(audio_device)?)
    } else {
        None
    };
    let audio_busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut mat: Box<dyn FrameMaterializer> = match materialize {
        "egl" => Box::new(rdma_display_capture::EglMaterializer::new(rendernode)?),
        "plain" => Box::new(rdma_display_capture::materializer::PlainMaterializer::new()),
        other => bail!("materializer {other:?} not implemented (egl|plain)"),
    };
    eprintln!("[server] capture materializer: {}", mat.name());

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
    let screen = ScreenSize::new();
    let (input_tx, input_rx) = mpsc::channel::<rdma_display_proto::InputEvent>();
    // host-game rumble on the virtual pad rides this hub to the current
    // session's control stream (installed per-accept by the RDMA layer)
    let rumble_hub = rdma_display_proto::gamepad::RumbleHub::new();
    let sink: Box<dyn InputSink> = match input_name {
        "qemu" => Box::new(QemuDbusInput::new(dbus_socket.clone(), console)),
        "uinput" => Box::new(rdma_display_capture::UinputSink),
        other => bail!("input sink {other:?} not implemented (qemu|uinput)"),
    };
    InputSink::spawn(sink, input_rx, screen.clone(), rumble_hub.clone())?;
    eprintln!("[server] input sink: {input_name}");

    let (tx, rx) = mpsc::channel();
    let source: Box<dyn CaptureSource> = match source_name {
        "qemu" => Box::new(QemuDbusSource::new(dbus_socket.clone(), console)),
        "drm" => Box::new(DrmKmsSource::new(drm_card.clone())),
        other => bail!("source {other:?} not implemented (qemu|drm)"),
    };
    CaptureSource::spawn(source, tx)?;
    eprintln!("[server] capture running; listening on {listen}");

    // --input-test: synthetic injector in the BACKGROUND so a client can
    // still connect and watch its own display while the guest types 'a's
    // and the synthetic mouse sweeps/clicks.
    if input_test {
        let injector_tx = input_tx.clone();
        std::thread::Builder::new()
            .name("input-test".into())
            .spawn(move || {
                eprintln!("[server] --input-test: injecting synthetic input");
                let mut n = 0u32;
                loop {
                    // key 'a' (qnum 0x1e) press+release
                    let _ = injector_tx.send(rdma_display_proto::InputEvent::Key {
                        code: 0x1e,
                        down: true,
                    });
                    let _ = injector_tx.send(rdma_display_proto::InputEvent::Key {
                        code: 0x1e,
                        down: false,
                    });
                    // mouse: absolute sweep + click (console dims fallback 1080p)
                    let (w, h) = (1920u32, 1080u32);
                    let x = w / 2 + (n % 40) as u32 * 8;
                    let y = h / 2 + (n % 20) as u32 * 10;
                    let _ = injector_tx.send(rdma_display_proto::InputEvent::Abs { x, y });
                    if n % 2 == 0 {
                        let _ = injector_tx.send(rdma_display_proto::InputEvent::Btn {
                            button: 0, // Left
                            down: true,
                        });
                        let _ = injector_tx.send(rdma_display_proto::InputEvent::Btn {
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
    }

    let listener = TcpListener::bind(listen)?;
    listener.set_nonblocking(true)?;

    let mut session: Option<rdma_display_rdma::ServerLink> = None;
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
    let mut issued: std::collections::VecDeque<(usize, u64, u64)> =
        std::collections::VecDeque::new();

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
                pipelined = false;
                issued.clear();
                mat.drop_frames();
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
                eprintln!("[server] connection with no routing tag dropped ({addr})");
            } else if tag[0] == rdma_display_audio::wire::TAG_AUDIO {
                match audio_mbox.as_ref() {
                    Some(mbox) => {
                        if audio_busy
                            .compare_exchange(false, true, std::sync::atomic::Ordering::SeqCst, std::sync::atomic::Ordering::SeqCst)
                            .is_ok()
                        {
                            eprintln!("[audio] client stream connected ({addr})");
                            let mbox = mbox.clone();
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
                        eprintln!("[server] audio connection but --audio is off: dropped ({addr})");
                    }
                }
            } else if tag[0] == rdma_display_proto::TAG_INPUT {
                // input has its own connection + thread, independent of the
                // video/control path; the nonce binds it to the live session
                let nonce_cell = session_nonce.clone();
                let input_tx = input_tx.clone();
                let _ = std::thread::Builder::new()
                    .name("input-reader".into())
                    .spawn(move || serve_input(stream, nonce_cell, input_tx, addr));
            } else if session.is_none() {
                if let Some(meta) = pending_meta.clone() {
                    eprintln!("client connected: {}", stream.peer_addr()?);
                    let nonce = rdma_display_rdma::random_nonce();
                    session_nonce.store(nonce, std::sync::atomic::Ordering::SeqCst);
                    // zero-copy staging (PBO readback): bind the materializer's
                    // persistent-mapped PBOs as the staging slots; None = fall
                    // back to verbs-allocated slots + direct readback
                    let external = if readback == "pbo" && materialize == "egl" {
                        match mat.bind_staging(slots, meta.frame_bytes as usize) {
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
                        ctx,
                        pd,
                        &meta,
                        slots,
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
                            frame_dirty = true;
                            pipelined = external.is_some();
                            session_meta = Some(meta.clone());
                            session = Some(link);
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
            let link = session.as_mut().unwrap();
            let slot_bytes = link.frame_bytes();
            // Materialize the newest pending capture now — only now do we know
            // it will actually be sent. Everything superseded while a slot was
            // busy was already dropped by the latest-wins mailbox.
            match pending_scanout.take() {
                Some(d) => {
                    let fmt = d.format.clone();
                    let fb = fmt.frame_bytes() as usize;
                    if fb > slot_bytes {
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
                    } else if let Err(e) = mat.prepare(&fmt, Some(&d)) {
                        // never tear down a session over a bad capture: hold
                        // the last frame and resume when the format returns
                        eprintln!("[server] prepare scanout failed: {e:#}; holding last frame");
                        frame_dirty = false;
                    } else {
                        have_frame = true;
                    }
                }
                // a client connected before the first capture: wait for one
                None if !have_frame => frame_dirty = false,
                None => {}
            }
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
            if pipelined {
                if let Some(slot) = link.try_claim_slot() {
                    let t0d = rdma_display_proto::unix_now_us();
                    let out = link.slot_bytes_mut(slot);
                    match mat.issue_frame(slot, out) {
                        Ok(true) => {
                            issued.push_back((slot, t0_unix_us, t0d));
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
                match link.send_frame_checked(|out| mat.copy_frame(out), t0_unix_us) {
                    Ok(Some(seq)) => {
                        frame_dirty = false;
                        last_send = Some(std::time::Instant::now());
                        eprintln!("[server] frame {} sent", seq);
                        if seq % 60 == 0 {
                            eprintln!("[stats] frames={} uptime={:?}", seq, started.elapsed());
                        }
                    }
                    Ok(None) => frame_dirty = false,
                    Err(e) => {
                        eprintln!("[server] send failed: {e:#}");
                        session = None;
                    }
                }
            }
            } // if frame_dirty
        }
        // 5. pipelined: finish + post the oldest issued frame (FIFO preserves
        // frame order; the fence wait is mostly elapsed because the GPU DMA
        // overlapped the previous frame's transfer)
        if pipelined && session.is_some() {
            while let Some((slot, f_t0, f_t0d)) = issued.front().copied() {
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
                match link.post_filled(slot, f_t0, f_t0d, rdma_display_proto::unix_now_us(), mat_times) {
                    Ok(Some(seq)) => {
                        last_send = Some(std::time::Instant::now());
                        eprintln!("[server] frame {} sent", seq);
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

/// Input-connection worker: validate the session nonce, forward input events
/// to the injection sink, and release everything the client held when the
/// socket dies — a network blip must never leave a key stuck down.
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
        CaptureEvent::Scanout(dmabuf) => {
            let fmt = dmabuf.format.clone();
            // latest-wins mailbox: keep only the newest capture. The GPU
            // import/blit/readback happens at send time (see the send step),
            // so a frame that gets superseded by a newer capture costs nothing.
            *pending_scanout = Some(dmabuf);
            screen.set(fmt.width, fmt.height);
            *pending_meta = Some(meta_linear(&fmt));
            *t0_unix_us = rdma_display_proto::unix_now_us();
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
