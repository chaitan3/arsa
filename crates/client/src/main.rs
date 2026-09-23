//! rdma-display client: connects to an rdma-display server over the control
//! channel, receives frames via RDMA WRITE into registered slots, and either
//! presents them on Wayland (`--output wayland`, C1: the slots themselves are
//! wl_buffers, credits pace on compositor release) or runs headless
//! (stats/checksums + optional PPM dump).

mod eglwin;
mod gamepad;
mod keymap;
mod metrics;
mod wayland;

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rdma_display_proto::Conn;

mod lossy;

fn main() -> Result<()> {
    let mut server: String = "127.0.0.1:4791".to_string();
    let mut device: Option<String> = None;
    let mut slots: usize = 3;
    let mut dump: Option<String> = None;
    let mut seconds: Option<u64> = None;
    let mut slot_mem: String = "host".to_string();
    let mut cuda_device: u32 = 0;
    let mut output: String = "none".to_string();
    let mut metrics_on = false;
    let mut fullscreen = false;
    let mut audio_on = false;
    let mut controller_on = true;
    let mut controller_grab = true;
    let mut fake_gamepad = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--server" => server = args.next().expect("--server needs a value"),
            "--device" => device = Some(args.next().expect("--device needs a value")),
            "--slots" => slots = args.next().expect("--slots needs a value").parse()?,
            "--dump" => dump = Some(args.next().expect("--dump needs a value")),
            "--seconds" => seconds = Some(args.next().expect("--seconds needs a value").parse()?),
            "--slot-mem" => slot_mem = args.next().expect("--slot-mem needs a value").to_string(),
            "--cuda-device" => cuda_device = args.next().expect("--cuda-device needs a value").parse()?,
            "--output" => output = args.next().expect("--output needs a value").to_string(),
            "--metrics" => metrics_on = true,
            "--fullscreen" => fullscreen = true,
            "--audio" => audio_on = true,
            "--no-controller" => controller_on = false,
            "--no-controller-grab" => controller_grab = false,
            "--fake-gamepad" => fake_gamepad = true,
            "--help" | "-h" => {
                println!("usage: rdma-display-client --server HOST:PORT [--device mlx5_0] [--slots 3] [--dump frame.ppm] [--seconds N]");
                println!("  --slot-mem host|cuda  receive slots: host = pinned RAM (default); cuda = dormant GPU path");
                println!("  --cuda-device N       CUDA device ordinal for --slot-mem cuda (default 0)");
                println!("  --output none|egl     egl = present in a Wayland window (needs --slot-mem host)");
                println!("  --fullscreen          fullscreen on the current output; frame letterboxed/scaled to the screen with aspect kept (compositor's own fullscreen toggle also works)");
                println!("  --audio               play the server's audio side-channel (default sink; server captures its own monitor source)");
                println!("  --metrics             latency/bandwidth overlay (top-left) + clock-sync pings");
                println!("  --no-controller       don't forward attached gamepads");
                println!("  --no-controller-grab  leave the gamepad exclusive-grabbable by the local compositor too (default: EVIOCGRAB while streaming)");
                println!("  --fake-gamepad        synthesize a controller (slot 0) for testing without hardware");
                println!("  Ctrl+Alt+M            (--output egl) grab keyboard+mouse for the guest: the local compositor's");
                println!("                        own shortcuts (alt-f4, alt-tab, alt-f1, ...) reach the guest too; Ctrl+Alt+M again");
                println!("                        is the ONLY way to give control back to the desktop");
                return Ok(());
            }
            other if other.starts_with("--") => anyhow::bail!("unknown arg: {other}"),
            _ => {}
        }
    }

    if !["host", "cuda"].contains(&slot_mem.as_str()) {
        anyhow::bail!("--slot-mem must be host|cuda, got {slot_mem:?}");
    }

    let addr = server
        .to_socket_addrs()?
        .next()
        .with_context(|| format!("bad --server {server:?}"))?;
    let stream = TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
    // control channel must not batch, and it must identify itself with the
    // routing tag before the RDMA handshake
    stream.set_nodelay(true)?;
    {
        use std::io::Write as _;
        (&stream).write_all(&[rdma_display_proto::TAG_CONTROL])?;
        (&stream).flush()?;
    }
    eprintln!("connected to {server}");

    let ctx = rdma_display_rdma::open_device(device.as_deref())?;
    let pd = ctx.alloc_pd()?;
    let mem = match slot_mem.as_str() {
        "host" => rdma_display_rdma::SlotMem::Host,
        "cuda" => rdma_display_rdma::SlotMem::Cuda { device: cuda_device },
        other => anyhow::bail!("--slot-mem must be host|cuda, got {other:?}"),
    };
    let mut link =
        rdma_display_rdma::ClientLink::connect_with(Conn::new(stream), &ctx, &pd, slots, mem)?;
    if link.slots_gpu_only() {
        eprintln!("[client] GPU slots without CPU mapping: content checks use the verification-only device->host copy (the render path itself stays zero-copy)");
    }
    eprintln!(
        "session up: {}x{} fourcc=0x{:08x} modifier=0x{:x} frame_bytes={} slots={}",
        link.meta.width,
        link.meta.height,
        link.meta.fourcc,
        link.meta.modifier,
        link.meta.frame_bytes,
        slots,
    );

    // Wayland presentation: the EGL upload backend — the mirror of the
    // server's copy path. Needs a CPU view of the slots (host memory):
    // CUDA slots are GPU-only and cannot be uploaded from.
    // user-input path: Wayland events -> this channel -> control TCP
    let (input_tx, input_rx) = std::sync::mpsc::channel();

    // controller: raw evdev capture (slot 0..n) + rumble player; rumble
    // events from the server arrive on the control channel below
    let rumble_tx = if controller_on || fake_gamepad {
        Some(gamepad::start(input_tx.clone(), controller_grab, fake_gamepad))
    } else {
        None
    };

    let mut presenter = match output.as_str() {
        "none" => None,
        "egl" => {
            if slot_mem != "host" {
                bail!("--output egl requires --slot-mem host (the upload reads the slot through a CPU pointer)");
            }
            let p = wayland::Presenter::new(link.meta.width, link.meta.height, input_tx, fullscreen)?;
            let win = eglwin::EglWindow::new(
                p.wl_display_ptr(),
                p.wl_surface_ptr(),
                link.meta.width,
                link.meta.height,
                metrics_on,
            )?;
            Some((p, win))
        }
        other => bail!("--output must be none|egl, got {other:?}"),
    };

    // input runs on its own thread + TCP connection (tagged `TAG_INPUT` plus
    // the session nonce), so a present or a control burst can never delay a
    // keystroke. Audio does the same on its own connection (tag 0x02).
    {
        let nonce = link.session_nonce;
        std::thread::Builder::new()
            .name("input".into())
            .spawn(move || input_thread(addr, nonce, input_rx))
            .context("spawn input thread")?;
    }

    // --metrics collector: consumes server Stamps + Pongs via the control
    // reader, samples t3/t4/t5 around the GL calls, pings every PING_SECS
    let mut metrics = metrics_on.then(|| {
        let mut m = metrics::Metrics::new();
        m.set_frame_bytes(link.meta.frame_bytes as usize);
        m
    });
    let mut last_overlay = Instant::now();

    // audio side-channel: second connection to the same port, tagged 0x02;
    // the thread owns its socket outright and retries on its own schedule —
    // a dead audio path never affects video/input
    if audio_on {
        std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || rdma_display_audio::playback::run(addr))
            .context("spawn audio thread")?;
        eprintln!("[audio] client stream connecting ({addr})");
    }

    let started = Instant::now();
    let deadline = seconds.map(|s| started + std::time::Duration::from_secs(s));
    let mut lossy_dec: Option<lossy::LossyDecoder> = None;
    let mut frames: u64 = 0;
    let mut last_seq = 0u64;
    let mut last_report = Instant::now();
    let mut dumped = false;

    loop {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        if let Some((p, win)) = presenter.as_mut() {
            // Dispatch Wayland events (input callbacks, configure) BEFORE the
            // blocking present, so a keystroke is handed to the input thread
            // without waiting for a swap; the input thread does the socket send.
            p.pump(1)?;
            // apply compositor-driven buffer resizes (fullscreen toggles,
            // fractional-scale changes) before the next frame is drawn
            if let Some((w, h)) = p.take_resize() {
                win.resize(w, h);
            }
            // presenter mode: drain RDMA completions, upload+present, drain
            // the compositor. Bytes are never CPU-touched: the NIC wrote the
            // slot, the GPU reads it through the upload pointer. Credits
            // fire at upload time (glTexImage2D consumed the pointer).
            // Drain every ready frame but present only the NEWEST: the display
            // should show the latest content, not replay a backlog (a FIFO
            // present adds one display interval of latency per queued frame).
            // Superseded frames are re-armed and credited immediately without
            // uploading them.
            let mut newest: Option<(u64, usize)> = None;
            while let Some((seq, slot)) = link.try_next_frame()? {
                link.rearm_slot(slot)?;
                // Lossless deltas reference the previous frame, so every
                // received frame must be decoded in order (skipping one
                // desyncs the chain). Lossy is intra-only: nothing to do.
                if link.codec != rdma_display_codec::Codec::Lossy {
                    link.frame(slot)?;
                }
                if let Some((_, old)) = newest.replace((seq, slot)) {
                    link.credit_slot(old)?; // superseded: never presented
                }
            }
            if let Some((seq, slot)) = newest {
                let t3 = rdma_display_proto::unix_now_us();
                // refresh the overlay raster at the metrics cadence only;
                // every presented frame still draws the last one (no flicker)
                let overlay = match metrics.as_mut() {
                    Some(m) => {
                        if last_overlay.elapsed().as_millis() as u64
                            >= metrics::REFRESH_MS
                        {
                            last_overlay = Instant::now();
                            m.refresh().map(|text| {
                                (
                                    metrics::rasterize(&text),
                                    metrics::OVERLAY_W,
                                    metrics::OVERLAY_H,
                                )
                            })
                        } else {
                            None
                        }
                    }
                    None => None,
                };
                let (t4, t5) = if link.codec == rdma_display_codec::Codec::Lossy {
                    if lossy_dec.is_none() {
                        lossy_dec = Some(lossy::LossyDecoder::new(
                            link.meta.width,
                            link.meta.height,
                            link.codec_chroma,
                        )?);
                    }
                    let coded = link.slot_bytes(slot);
                    let frame = lossy_dec.as_mut().unwrap().decode(coded)?;
                    win.present(
                        frame,
                        overlay.as_ref().map(|(r, w, h)| (&r[..], *w, *h)),
                    )?
                } else {
                    let frame = link.frame(slot)?;
                    win.present(
                        frame,
                        overlay.as_ref().map(|(r, w, h)| (&r[..], *w, *h)),
                    )?
                };
                link.credit_slot(slot)?; // upload consumed the pointer
                if let Some(m) = metrics.as_mut() {
                    m.record(seq, t3, t4, t5);
                }
                last_seq = seq;
                frames += 1;
                if seq % 60 == 0 || seq == 0 {
                    eprintln!("[client] presented seq={seq} slot={slot}");
                }
            }
            // metrics: drain control events (stamps/pongs) + keep the clock
            // sync warm
            if let Some(m) = metrics.as_mut() {
                while let Some(ev) = link.try_control() {
                    if let Some(meta) = route_control(ev, rumble_tx.as_ref(), Some(m)) {
                        apply_meta(&mut link, &meta);
                    }
                }
                if m.maybe_ping() {
                    link.send_ping()?;
                }
            } else {
                // without metrics there is no server→client traffic expected;
                // still drain so nothing accumulates
                while let Some(ev) = link.try_control() {
                    if let Some(meta) = route_control(ev, rumble_tx.as_ref(), None) {
                        apply_meta(&mut link, &meta);
                    }
                }
            }
            // forward queued user-input events to the server
            // (input has its own thread + connection now — see input_thread)
            if p.closed() {
                eprintln!("[client] window closed, ending");
                break;
            }
        } else {
            // headless mode: today's blocking stats/dump path
            let (seq, slot) = link.next_frame()?;
            let t3 = rdma_display_proto::unix_now_us();
            if let Some(m) = metrics.as_mut() {
                // no GL here: t4=t5=t3 (upload/present rows read 0.00)
                m.record(seq, t3, t3, t3);
            }
            let want_stats = seq % 60 == 0 || seq == 0;
            let want_dump = !dumped && dump.is_some();
            let mut frame_owned: Option<Vec<u8>> = None;
            if link.codec == rdma_display_codec::Codec::Lossy {
                if lossy_dec.is_none() {
                    lossy_dec = Some(lossy::LossyDecoder::new(
                        link.meta.width,
                        link.meta.height,
                        link.codec_chroma,
                    )?);
                }
                let coded = link.slot_bytes(slot);
                let dec = lossy_dec.as_mut().unwrap().decode(coded)?.to_vec();
                if want_stats {
                    let sum: u32 = dec.iter().map(|b| *b as u32).sum();
                    let nonzero = dec.iter().filter(|b| **b != 0).count();
                    eprintln!(
                        "[client-check] seq={} slot={} sum={} nonzero={}/{}",
                        seq, slot, sum, nonzero, dec.len()
                    );
                }
                if want_dump {
                    frame_owned = Some(dec);
                }
            } else if want_dump {
                frame_owned = Some(link.frame(slot)?.to_vec());
            } else if want_stats {
                let bytes = link.frame(slot)?;
                let sum: u32 = bytes.iter().map(|b| *b as u32).sum();
                let nonzero = bytes.iter().filter(|b| **b != 0).count();
                eprintln!(
                    "[client-check] seq={} slot={} sum={} nonzero={}/{}",
                    seq, slot, sum, nonzero, bytes.len()
                );
            } else {
                link.frame(slot)?;
            }
            if let Some(frame) = frame_owned {
                dumped = true;
                let path = dump.as_ref().expect("want_dump implies dump path");
                dump_ppm(path, &link.meta, &frame)?;
            }
            link.ack_frame(slot)?;
            frames += 1;
            last_seq = seq;
        }
        if last_report.elapsed() >= std::time::Duration::from_secs(5) {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "[stats] frames={} fps={:.1} seq={}",
                frames,
                frames as f64 / elapsed,
                last_seq
            );
            if let Some(m) = metrics.as_mut() {
                while let Some(ev) = link.try_control() {
                    if let Some(meta) = route_control(ev, rumble_tx.as_ref(), Some(m)) {
                        apply_meta(&mut link, &meta);
                    }
                }
                if m.maybe_ping() {
                    link.send_ping()?;
                }
                if presenter.is_none() {
                    eprintln!("[metrics]\n{}", metrics::stderr_line(m));
                } else {
                    eprintln!("[metrics]\n{}", metrics::snapshot(m));
                }
            }
            last_report = Instant::now();
        }
    }
    eprintln!("done: {} frames in {:?}", frames, started.elapsed());
    Ok(())
}

/// Input sender thread: owns its own control-port connection (routing tag
/// `TAG_INPUT` + session nonce) and drains the shared input channel into it.
/// This keeps user input off both the video present loop and the control
/// socket. Reconnects with backoff on transient failures.
fn input_thread(
    addr: std::net::SocketAddr,
    nonce: u64,
    rx: std::sync::mpsc::Receiver<rdma_display_proto::InputEvent>,
) {
    use std::io::Write as _;
    let mut backoff = std::time::Duration::from_millis(200);
    loop {
        let sock = match TcpStream::connect(addr) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[input] connect {addr} failed: {e}");
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(std::time::Duration::from_secs(2));
                continue;
            }
        };
        let _ = sock.set_nodelay(true);
        if (&sock).write_all(&[rdma_display_proto::TAG_INPUT]).is_err()
            || (&sock).write_all(&nonce.to_le_bytes()).is_err()
        {
            std::thread::sleep(backoff);
            backoff = (backoff * 2).min(std::time::Duration::from_secs(2));
            continue;
        }
        eprintln!("[input] connected ({addr})");
        backoff = std::time::Duration::from_millis(200);
        let mut conn = Conn::new(sock);
        loop {
            match rx.recv() {
                Ok(ev) => {
                    if conn.send(&rdma_display_proto::Message::Input(ev)).is_err() {
                        eprintln!("[input] send failed; reconnecting");
                        break;
                    }
                }
                Err(_) => return, // presenter gone: stop the thread
            }
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(std::time::Duration::from_secs(2));
    }
}

fn route_control(
    ev: rdma_display_rdma::ControlEvent,
    rumble_tx: Option<&std::sync::mpsc::Sender<rdma_display_proto::gamepad::Rumble>>,
    metrics: Option<&mut metrics::Metrics>,
) -> Option<rdma_display_proto::DisplayMeta> {
    match ev {
        rdma_display_rdma::ControlEvent::Rumble(r) => {
            if let Some(tx) = rumble_tx {
                let _ = tx.send(r);
            }
            None
        }
        // the presenter applies the new meta to the link/window
        rdma_display_rdma::ControlEvent::Meta(m) => Some(m),
        other => {
            if let Some(m) = metrics {
                m.on_control(other);
            }
            None
        }
    }
}

/// Apply a server-pushed display meta change. Only same-size changes are
/// applied (slots are sized at handshake); the wire format is always XR24, so
/// a fourcc/stride change at the same frame size does not affect the upload.
fn apply_meta(link: &mut rdma_display_rdma::ClientLink, meta: &rdma_display_proto::DisplayMeta) {
    eprintln!(
        "[client] display meta gen {} ({}x{} stride {} fourcc {:#010x})",
        meta.gen, meta.width, meta.height, meta.stride, meta.fourcc
    );
    if meta.frame_bytes != link.meta.frame_bytes {
        eprintln!(
            "[client] WARNING: frame size changed mid-session ({} -> {} bytes); reconnect to apply",
            link.meta.frame_bytes, meta.frame_bytes
        );
        return;
    }
    link.meta = meta.clone();
}

fn dump_ppm(path: &str, meta: &rdma_display_proto::DisplayMeta, bytes: &[u8]) -> Result<()> {
    // XRGB8888 -> binary PPM (P6), dropping the alpha/upper byte
    let width = meta.width as usize;
    let height = meta.height as usize;
    let stride = meta.stride as usize;
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            let o = y * stride + x * 4;
            // XRGB8888 little-endian: bytes are B, G, R, X
            let b = bytes[o];
            let g = bytes[o + 1];
            let r = bytes[o + 2];
            ppm.extend_from_slice(&[r, g, b]);
        }
    }
    std::fs::write(path, &ppm).with_context(|| format!("write {path}"))?;
    eprintln!("dumped {path}");
    Ok(())
}
