//! rdma-display client: connects to an rdma-display server over the control
//! channel, receives frames via RDMA WRITE into registered slots, and either
//! presents them on Wayland (`--output egl`, default: the slots themselves are
//! wl_buffers, credits pace on compositor release) or runs headless
//! (`--output none`: stats/checksums + optional PPM dump). `--audio` is on by
//! default; `--no-audio` disables it.

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

/// Default signaling port, used when `--server`/`-s` omits a port.
const DEFAULT_PORT: u16 = 4791;

fn main() -> Result<()> {
    // explicit server address (flag or first positional); None => 127.0.0.1
    let mut server: Option<String> = None;
    let mut device: Option<String> = None;
    let mut slots: usize = 3;
    let mut dump: Option<String> = None;
    let mut seconds: Option<u64> = None;
    let mut slot_mem: String = "host".to_string();
    let mut cuda_device: u32 = 0;
    let mut output: String = "egl".to_string();
    let mut metrics_on = false;
    let mut verbose = false;
    let mut fullscreen = false;
    let mut grab = false;
    let mut audio_on = true;
    let mut audio_buffer_ms: u32 = rdma_display_audio::wire::BUFFER_MS;
    let mut controller_on = true;
    let mut controller_grab = true;
    let mut fake_gamepad = false;
    let mut transport = rdma_display_rdma::TransportPref::Auto;
    let mut scale_filter: u32 = 0; // 0=auto 1=nearest 2=linear 3=cubic 4=area
    // display-mode request sent to the server before the session is set up:
    // --native asks for this client's own output mode; --set-mode names one
    let mut native = false;
    let mut set_mode: Option<String> = None;
    // codec preference sent to the server (CODEC_ANY = let the server pick)
    let mut codec_pref: u8 = rdma_display_proto::CODEC_ANY;
    // lossy bitstream budget to request from the server, in Mbit/s
    // (0 = no preference; use the server's default)
    let mut lossy_mbps: u16 = 0;
    // lossy: force the CPU download+upload path instead of the all-GPU
    // Vulkan->GL dma-buf interop (also honours RDMA_DISPLAY_LOSSY_CPU=1)
    let mut lossy_cpu = std::env::var_os("RDMA_DISPLAY_LOSSY_CPU").is_some();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-s" | "--server" => server = Some(args.next().expect("--server needs a value")),
            "--device" => device = Some(args.next().expect("--device needs a value")),
            // --slots kept as an accepted alias (older scripts/commands)
            "--frame-slots" | "--slots" => {
                slots = args
                    .next()
                    .expect("--frame-slots needs a value")
                    .parse()?
            }
            "--dump" => dump = Some(args.next().expect("--dump needs a value")),
            "--seconds" => seconds = Some(args.next().expect("--seconds needs a value").parse()?),
            "--slot-mem" => slot_mem = args.next().expect("--slot-mem needs a value").to_string(),
            "--cuda-device" => cuda_device = args.next().expect("--cuda-device needs a value").parse()?,
            "--output" => output = args.next().expect("--output needs a value").to_string(),
            "--metrics" => metrics_on = true,
            "-v" | "--verbose" => verbose = true,
            "-f" | "--fullscreen" => fullscreen = true,
            "-g" | "--grab" => grab = true,
            "--audio" => audio_on = true,
            "--no-audio" => audio_on = false,
            "--audio-buffer-ms" => {
                audio_buffer_ms = args.next().expect("--audio-buffer-ms needs a value").parse()?
            }
            "--no-controller" => controller_on = false,
            "--no-controller-grab" => controller_grab = false,
            "--fake-gamepad" => fake_gamepad = true,
            "--transport" => {
                let v = args.next().expect("--transport needs a value");
                transport = rdma_display_rdma::TransportPref::parse(&v)
                    .with_context(|| format!("--transport must be auto|rdma|tcp, got {v:?}"))?;
            }
            "--scale-filter" => {
                let v = args.next().expect("--scale-filter needs a value");
                scale_filter = match v.as_str() {
                    "auto" => 0,
                    "nearest" => 1,
                    "linear" => 2,
                    "cubic" => 3,
                    "area" => 4,
                    other => bail!(
                        "--scale-filter must be auto|nearest|linear|cubic|area, got {other:?}"
                    ),
                };
            }
            "--lossy-cpu" => lossy_cpu = true,
            "--lossy-bitrate" => {
                lossy_mbps = args
                    .next()
                    .expect("--lossy-bitrate needs a value")
                    .parse()
                    .with_context(|| "--lossy-bitrate must be Mbit/s (e.g. 100)")?
            }
            "--native" => native = true,
            "--set-mode" => set_mode = Some(args.next().expect("--set-mode needs a value")),
            "--codec" => {
                let v = args.next().expect("--codec needs a value");
                codec_pref = match v.as_str() {
                    "auto" | "any" => rdma_display_proto::CODEC_ANY,
                    _ => rdma_display_codec::Codec::parse(&v)
                        .with_context(|| format!("--codec must be auto|raw|lossless|lossy, got {v:?}"))?
                        .id(),
                };
            }
            "--help" | "-h" => {
                println!("usage: rdma-display-client [HOST[:PORT]] [--device mlx5_0] [--frame-slots 3] [--dump frame.ppm] [--seconds N]");
                println!("  HOST[:PORT]           server address, given as the first positional argument (default host 127.0.0.1; :{DEFAULT_PORT} assumed when the port is omitted)");
                println!("  -s, --server HOST[:PORT]  same as the positional HOST[:PORT] (either form may be used)");
                println!("  --frame-slots N       receive slots, each holding one coded frame: sets the receive-buffer pool and the maximum flow-control window (default 3; --slots is accepted as an alias)");
                println!("  --slot-mem host|cuda  receive slots: host = pinned RAM (default); cuda = dormant GPU path");
                println!("  --cuda-device N       CUDA device ordinal for --slot-mem cuda (default 0)");
                println!("  --output egl|none     egl = present in a Wayland window (default; needs --slot-mem host); none = headless (e.g. with --dump)");
                println!("  -f, --fullscreen      fullscreen on the current output; frame letterboxed/scaled to the screen with aspect kept (compositor's own fullscreen toggle also works)");
                println!("  -g, --grab            start with the input grab on (keyboard+mouse for the guest; same as Ctrl+Alt+G); default off");
                println!("  --audio | --no-audio  play the server's audio side-channel (default sink; server captures its own monitor source). Default on; --no-audio disables");
                println!("  --audio-buffer-ms N   client playout buffer depth / pulse tlength (default {}; lower = less latency, less jitter headroom)", rdma_display_audio::wire::BUFFER_MS);
                println!("  --metrics             per-stage latency overlay (capture, send, transfer, upload, present, on-screen) + clock-sync pings");
                println!("  -v, --verbose         debug-level logging (per-frame/per-event detail; RUST_LOG overrides)");
                println!("  --no-controller       don't forward attached gamepads");
                println!("  --no-controller-grab  leave the gamepad exclusive-grabbable by the local compositor too (default: EVIOCGRAB while streaming)");
                println!("  --fake-gamepad        synthesize a controller (slot 0) for testing without hardware");
                println!("  --transport auto|rdma|tcp  video transport (default auto: RDMA/RoCE when available, else TCP); negotiated with the server");
                println!("  --scale-filter auto|nearest|linear|cubic|area  sampler for the letterboxed frame (default auto: nearest at integer scale, area when downscaling, else linear)");
                println!("  --native              ask the server to set its output to this client's monitor mode (read from wl_output) before the session starts");
                println!("  --set-mode WxH[@R]    ask the server to set its output to this size/refresh before the session starts (R in Hz; omitted = any)");
                println!("  --codec auto|raw|lossless|lossy  codec to request from the server (honoured under the server's --codec auto). Default auto: the server picks");
                println!("  --lossy-cpu           lossy: decode + download + upload on the CPU instead of the all-GPU Vulkan->GL dma-buf interop (default all-GPU, with an automatic fallback when the interop is unavailable)");
                println!("  --lossy-bitrate MBPS  lossy: ask the server for this bitstream budget in Mbit/s (e.g. 100); 0 = use the server's default. The server converts it to a per-frame byte cap at a nominal 60 fps");
                println!("  RDMA_DISPLAY_LOSSY_DEBUG=1  lossy: verbose dma-buf import diagnostics (extension strings, importable formats/modifiers, every import attempt)");
                println!("  Ctrl+Alt+G            (--output egl) grab keyboard+mouse for the guest: the local compositor's");
                println!("                        own shortcuts (alt-f4, alt-tab, alt-f1, ...) reach the guest too; Ctrl+Alt+G again");
                println!("                        is the ONLY way to give control back to the desktop");
                return Ok(());
            }
            other if other.starts_with("--") => anyhow::bail!("unknown arg: {other}"),
            // first bare argument is the server address; anything after that is
            // not something we know how to interpret
            other => {
                if server.is_none() {
                    server = Some(other.to_string());
                } else {
                    anyhow::bail!("unexpected argument: {other}");
                }
            }
        }
    }

    if !["host", "cuda"].contains(&slot_mem.as_str()) {
        anyhow::bail!("--slot-mem must be host|cuda, got {slot_mem:?}");
    }
    init_logging(verbose);
    // `HOST` (positional or `-s`/`--server`) without a port uses the default port
    let server = with_default_port(server.as_deref().unwrap_or("127.0.0.1"));
    let addr = server
        .to_socket_addrs()?
        .next()
        .with_context(|| format!("bad --server {server:?}"))?;
    let t0 = Instant::now();
    let stream = TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
    // control channel must not batch, and it must identify itself with the
    // routing tag before the RDMA handshake
    stream.set_nodelay(true)?;
    {
        use std::io::Write as _;
        (&stream).write_all(&[rdma_display_proto::TAG_CONTROL])?;
        (&stream).flush()?;
    }
    log::info!("connected to {server} (t={} ms)", t0.elapsed().as_millis());

    // video transport: auto prefers RDMA and falls back to TCP (matching the
    // server); with TCP there is no RDMA device to open at all
    let rdma_ctx = match transport {
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
                log::warn!("[client] no RDMA device ({e:#}); using TCP video transport");
                None
            }
        },
    };
    let rdma_ref = rdma_ctx.as_ref().map(|(c, p)| (c, p));
    let mem = match slot_mem.as_str() {
        "host" => rdma_display_rdma::SlotMem::Host,
        "cuda" => rdma_display_rdma::SlotMem::Cuda { device: cuda_device },
        other => anyhow::bail!("--slot-mem must be host|cuda, got {other:?}"),
    };
    // display-mode request: resolved here (before the handshake) so the server
    // can reconfigure its output before it sizes the session
    let mode_req = resolve_mode_request(native, set_mode.as_deref(), &output)?;
    let mut link = rdma_display_rdma::ClientLink::connect_with(
        Conn::new(stream),
        transport,
        rdma_ref,
        Some(addr),
        slots,
        mem,
        mode_req,
        codec_pref,
        lossy_mbps,
        metrics_on,
    )?;
    if link.slots_gpu_only() {
        log::info!("[client] GPU slots without CPU mapping: content checks use the verification-only device->host copy (the render path itself stays zero-copy)");
    }
    log::info!(
        "session up: {}x{} fourcc=0x{:08x} modifier=0x{:x} frame_bytes={} slots={}",
        link.meta.width,
        link.meta.height,
        link.meta.fourcc,
        link.meta.modifier,
        link.meta.frame_bytes,
        slots,
    );
    log::info!("[client] handshake done t={} ms", t0.elapsed().as_millis());

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

    if grab && output != "egl" {
        log::warn!("[client] --grab needs --output egl (no window to grab with --output {output})");
    }
    let mut presenter = match output.as_str() {
        "none" => None,
        "egl" => {
            if slot_mem != "host" {
                bail!("--output egl requires --slot-mem host (the upload reads the slot through a CPU pointer)");
            }
            let p = wayland::Presenter::new(link.meta.width, link.meta.height, input_tx, fullscreen, grab)?;
            let mut win = eglwin::EglWindow::new(
                p.wl_display_ptr(),
                p.wl_surface_ptr(),
                link.meta.width,
                link.meta.height,
                metrics_on,
            )?;
            win.set_scale_filter(scale_filter);
            Some((p, win))
        }
        other => bail!("--output must be none|egl, got {other:?}"),
    };
    log::info!("[client] presenter ready t={} ms", t0.elapsed().as_millis());

    // GPU lossless decode: when presenting through eglwin's GL context and the
    // session uses the lossless codec, decode the tiles straight into a texture
    // in that context (the presenter draws it) instead of a full-frame CPU
    // decode followed by a full-frame upload.
    let mut gl_lossless: Option<(rdma_display_codec_gpu::GpuVideoDecoder, u32)> = None;
    if link.codec == rdma_display_codec::Codec::Lossless && presenter.is_some() {
        let geom = rdma_display_codec::Geometry::new(
            link.meta.width,
            link.meta.height,
            link.meta.stride,
        );
        match rdma_display_codec_gpu::GpuVideoDecoder::attach_current(geom) {
            Ok(dec) => match dec.create_texture(link.meta.width, link.meta.height) {
                Ok(tex) => {
                    log::info!("[client] lossless decode: GPU -> texture (no full-frame upload)");
                    gl_lossless = Some((dec, tex));
                }
                Err(e) => {
                    log::warn!("[client] lossless decode: GPU texture failed ({e:#}); using CPU")
                }
            },
            Err(e) => log::warn!("[client] lossless decode: GPU unavailable ({e:#}); using CPU"),
        }
    }

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

    // audio timing stats: produced by the playout thread, consumed by the
    // metrics overlay (audio e2e + A/V offset). Only allocated under
    // `--metrics`, so a normal run records nothing per packet.
    let audio_stats = (metrics_on && audio_on)
        .then(|| std::sync::Arc::new(rdma_display_audio::stats::AudioStats::new()));

    // --metrics collector: consumes server Stamps + Pongs via the control
    // reader, samples t3/t4/t5 around the GL calls, pings every PING_SECS
    let mut metrics = metrics_on.then(|| metrics::Metrics::new());
    // tell the overlay whether the compositor can report the true on-screen
    // time (wp_presentation), so `display` reads a number instead of n/a
    if let (Some((p, _)), Some(m)) = (presenter.as_ref(), metrics.as_mut()) {
        m.set_onscreen(p.has_presentation());
    }
    if let Some(m) = metrics.as_mut() {
        m.set_tcp(link.transport() == rdma_display_rdma::Transport::Tcp);
        if let Some(s) = &audio_stats {
            m.set_audio_stats(s.clone());
        }
    }
    let mut last_overlay = Instant::now();

    // audio side-channel: second connection to the same port, tagged 0x02;
    // the thread owns its socket outright and retries on its own schedule —
    // a dead audio path never affects video/input
    if audio_on {
        let stats = audio_stats.clone();
        std::thread::Builder::new()
            .name("audio".into())
            .spawn(move || rdma_display_audio::playback::run(addr, audio_buffer_ms, stats))
            .context("spawn audio thread")?;
        log::info!("[audio] client stream connecting ({addr})");
    }

    let started = Instant::now();
    let deadline = seconds.map(|s| started + std::time::Duration::from_secs(s));
    let mut lossy_dec: Option<lossy::LossyDecoder> = None;
    // whether lossy_dec feeds the GL presenter through imported dma-buf planes
    // (all-GPU) rather than the CPU download+upload path
    let mut lossy_gpu = false;
    let mut frames: u64 = 0;
    let mut first_frame_logged = false;
    let mut last_seq = 0u64;
    let mut last_report = Instant::now();
    let mut dumped = false;
    // judder diagnostics: how long after a frame's completion the client holds
    // it before crediting (credit is sent right after present), and the reader
    // frame counter for the receive-rate delta
    let mut credit_lags: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
    let mut last_reader_frames: u64 = 0;

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
            let mut newest: Option<(u64, usize, u64)> = None;
            while let Some((seq, slot)) = link.try_next_frame()? {
                // t3: the instant THIS completion was observed. Stamped before
                // the re-arm/credit below (those are TCP sends and would
                // otherwise be billed to the client's receive side).
                let t3 = rdma_display_proto::unix_now_us();
                // bandwidth: count every received frame's real wire bytes
                // (superseded frames included), not the uncompressed size
                if let Some(m) = metrics.as_mut() {
                    m.add_rx_bytes(wire_bytes(&link, slot));
                }
                link.rearm_slot(slot)?;
                // Lossless deltas reference the previous frame, so every
                // received frame must be decoded in order (skipping one
                // desyncs the chain): on the GPU into the shared texture when
                // available, otherwise the CPU decoder. Lossy is intra-only.
                if let Some((dec, tex)) = gl_lossless.as_ref() {
                    let g = dec.geometry();
                    if g.width == link.meta.width
                        && g.height == link.meta.height
                        && g.stride == link.meta.stride
                    {
                        let coded = link.slot_bytes(slot);
                        match rdma_display_codec::Header::parse(coded) {
                            Ok(h) => {
                                let n = rdma_display_codec::HEADER_LEN
                                    + g.tile_count() * rdma_display_codec::INDEX_WORDS * 4
                                    + h.payload_words as usize * 4;
                                if n <= coded.len() {
                                    dec.upload(&coded[..n])?;
                                    dec.decode_to_image(*tex, h.count as usize)?;
                                } else {
                                    log::debug!(
                                        "[client] lossless: coded {n} > slot {}",
                                        coded.len()
                                    );
                                }
                            }
                            Err(e) => log::debug!("[client] lossless: header parse failed: {e:#}"),
                        }
                    } else {
                        // geometry changed (Meta resize): the server resets the
                        // reference on the next frame (full), so the CPU
                        // decoder resyncs; stop using the stale GPU texture.
                        link.frame(slot)?;
                    }
                } else if link.codec != rdma_display_codec::Codec::Lossy {
                    link.frame(slot)?;
                }
                if let Some((_, old, _)) = newest.replace((seq, slot, t3)) {
                    link.credit_slot(old)?; // superseded: never presented
                }
            }
            if let Some((seq, slot, t3)) = newest {
                if !first_frame_logged {
                    first_frame_logged = true;
                    log::info!(
                        "[client] first frame t={} ms (seq {seq})",
                        t0.elapsed().as_millis()
                    );
                }
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
                // ask the compositor for the on-screen time of this commit
                // BEFORE the swap (feedback must be armed pre-commit); only
                // when the overlay is actually collecting metrics
                let fb = if metrics.is_some() {
                    p.request_present_feedback()
                } else {
                    None
                };
                let (t4, t5) = if let Some((_, tex)) = gl_lossless.as_ref() {
                    win.present_texture(
                        *tex,
                        overlay.as_ref().map(|(r, w, h)| (&r[..], *w, *h)),
                    )?
                } else if link.codec == rdma_display_codec::Codec::Lossy {
                    if lossy_dec.is_none() {
                        let (w, h, c) =
                            (link.meta.width, link.meta.height, link.codec_chroma);
                        // Prefer the all-GPU path: Vulkan decodes into exported
                        // dma-buf planes, GL samples them (no CPU work). If this
                        // client's Vulkan cannot export them or GL cannot import
                        // them, fall back to the download+upload path instead of
                        // failing — lossy still works, with a host round trip.
                        if lossy_cpu {
                            lossy_dec = Some(lossy::LossyDecoder::new(w, h, c, false)?);
                        } else {
                            match lossy::LossyDecoder::new(w, h, c, true).and_then(|dec| {
                                win.import_yuv(dec.plane_descs())?;
                                Ok(dec)
                            }) {
                                Ok(dec) => {
                                    lossy_gpu = true;
                                    lossy_dec = Some(dec);
                                }
                                Err(e) => {
                                    log::warn!(
                                        "[lossy] GPU YUV interop unavailable ({e:#}); \
                                         falling back to the CPU upload path"
                                    );
                                    lossy_dec = Some(lossy::LossyDecoder::new(w, h, c, false)?);
                                }
                            }
                        }
                    }
                    let d = lossy_dec.as_mut().unwrap();
                    let coded = link.slot_bytes(slot);
                    d.decode(coded)?;
                    let ov = overlay.as_ref().map(|(r, w, h)| (&r[..], *w, *h));
                    if lossy_gpu {
                        win.present_yuv(ov)?
                    } else {
                        win.present(d.read_rgba()?, ov)?
                    }
                } else if let Some(frame) = link.decoded() {
                    // The receive loop already decoded every frame in order to
                    // advance the delta chain, so the newest is the decoder's
                    // current frame. Decoding it again here would apply the
                    // same delta twice and corrupt the reference (a permanent
                    // trail on the CPU lossless path).
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
                if metrics_on {
                    let lag = rdma_display_proto::unix_now_us().saturating_sub(t3);
                    credit_lags.push_back(lag);
                    while credit_lags.len() > 128 {
                        credit_lags.pop_front();
                    }
                }
                if let Some(m) = metrics.as_mut() {
                    m.record(seq, t3, t4, t5, fb);
                }
                last_seq = seq;
                frames += 1;
                if seq % 60 == 0 || seq == 0 {
                    log::debug!("[client] presented seq={seq} slot={slot}");
                }
            }
            // metrics: drain control events (stamps/pongs) + keep the clock
            // sync warm; fold in any on-screen timestamps the compositor
            // reported (dispatched by the pump at the top of this iteration)
            if let Some(m) = metrics.as_mut() {
                for pr in p.take_presented() {
                    if pr.presented {
                        m.on_presented(pr.id, pr.unix_us, true, pr.flags, pr.refresh);
                    }
                }
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
                log::info!("[client] window closed, ending");
                break;
            }
        } else {
            // headless mode: today's blocking stats/dump path
            let (seq, slot) = link.next_frame()?;
            let t3 = rdma_display_proto::unix_now_us();
            if let Some(m) = metrics.as_mut() {
                // no GL and no compositor here: t4=t5=t3 and no presentation
                // feedback (upload/present/display rows read 0.00 / n/a)
                m.record(seq, t3, t3, t3, None);
            }
            let want_stats = seq % 60 == 0 || seq == 0;
            let want_dump = !dumped && dump.is_some();
            let mut frame_owned: Option<Vec<u8>> = None;
            if link.codec == rdma_display_codec::Codec::Lossy {
                if lossy_dec.is_none() {
                    // headless has no GL context: never export dma-bufs
                    lossy_dec = Some(lossy::LossyDecoder::new(
                        link.meta.width,
                        link.meta.height,
                        link.codec_chroma,
                        false,
                    )?);
                }
                let coded = link.slot_bytes(slot);
                let d = lossy_dec.as_mut().unwrap();
                d.decode(coded)?;
                if want_stats || want_dump {
                    // debug-only CPU readback (the streaming path stays on GPU)
                    let dec = d.read_rgba()?.to_vec();
                    if want_stats {
                        let sum: u32 = dec.iter().map(|b| *b as u32).sum();
                        let nonzero = dec.iter().filter(|b| **b != 0).count();
                        log::debug!(
                            "[client-check] seq={} slot={} sum={} nonzero={}/{}",
                            seq, slot, sum, nonzero, dec.len()
                        );
                    }
                    if want_dump {
                        frame_owned = Some(dec);
                    }
                }
            } else if want_dump {
                frame_owned = Some(link.frame(slot)?.to_vec());
            } else if want_stats {
                let bytes = link.frame(slot)?;
                let sum: u32 = bytes.iter().map(|b| *b as u32).sum();
                let nonzero = bytes.iter().filter(|b| **b != 0).count();
                log::debug!(
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
            if metrics_on {
                // judder diagnostics: the true delivery cadence at the reader
                // thread (independent of the flow-control math), how often the
                // reader had to wait for a recycled buffer, and how long the
                // client held a frame before crediting it. Only under
                // `--metrics`, so a normal run does none of this work.
                let rs = link.video_reader_stats();
                let recv_dt = last_report.elapsed().as_secs_f64().max(1e-6);
                let recv_fps = rs.frames.saturating_sub(last_reader_frames) as f64 / recv_dt;
                last_reader_frames = rs.frames;
                let credit_p50 = {
                    let mut v: Vec<u64> = credit_lags.iter().copied().collect();
                    v.sort_unstable();
                    if v.is_empty() {
                        0
                    } else {
                        v[v.len() / 2] / 1000
                    }
                };
                log::debug!(
                    "[stats] frames={} fps={:.1} seq={} | recv {:.1}/s gap p50 {:.1} p95 {:.1} max {:.1} ms stall {} | credit p50 {} ms",
                    frames,
                    frames as f64 / elapsed,
                    last_seq,
                    recv_fps,
                    rs.gap_p50_us as f64 / 1000.0,
                    rs.gap_p95_us as f64 / 1000.0,
                    rs.gap_max_us as f64 / 1000.0,
                    rs.stalls,
                    credit_p50,
                );
            } else {
                log::debug!(
                    "[stats] frames={} fps={:.1} seq={}",
                    frames,
                    frames as f64 / elapsed,
                    last_seq
                );
            }
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
    log::info!("done: {} frames in {:?}", frames, started.elapsed());
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
                log::warn!("[input] connect {addr} failed: {e}");
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
        log::info!("[input] connected ({addr})");
        backoff = std::time::Duration::from_millis(200);
        let mut conn = Conn::new(sock);
        loop {
            match rx.recv() {
                Ok(ev) => {
                    if conn.send(&rdma_display_proto::Message::Input(ev)).is_err() {
                        log::warn!("[input] send failed; reconnecting");
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
        rdma_display_rdma::ControlEvent::ModeReply(r) => {
            log::info!(
                "[client] display mode: {} ({}{})",
                r.mode_label(),
                if r.ok { "ok" } else { "failed" },
                if r.message.is_empty() {
                    String::new()
                } else {
                    format!("; {}", r.message)
                }
            );
            None
        }
        other => {
            if let Some(m) = metrics {
                m.on_control(other);
            }
            None
        }
    }
}

/// Bytes that crossed the wire for the frame currently in `slot`. Prefer the
/// transport's reported length (the RDMA WRITE length / TCP payload length);
/// fall back to the coded length the frame implies for transports/HCAs that
/// report 0 for a write-with-immediate receive.
fn wire_bytes(link: &rdma_display_rdma::ClientLink, slot: usize) -> usize {
    let n = link.last_rx_bytes();
    if n != 0 {
        return n;
    }
    match link.codec {
        rdma_display_codec::Codec::Raw => link.meta.frame_bytes as usize,
        rdma_display_codec::Codec::Lossless => {
            let g = rdma_display_codec::Geometry::new(
                link.meta.width,
                link.meta.height,
                link.meta.stride,
            );
            rdma_display_codec::Header::parse(link.slot_bytes(slot))
                .map(|h| {
                    rdma_display_codec::HEADER_LEN
                        + g.tile_count() * rdma_display_codec::INDEX_WORDS * 4
                        + h.payload_words as usize * 4
                })
                .unwrap_or(0)
        }
        // the lossy bitstream length isn't recoverable from the slot alone
        rdma_display_codec::Codec::Lossy => 0,
    }
}

/// Apply a server-pushed display meta change. Only same-size changes are
/// applied (slots are sized at handshake); the wire format is always XR24, so
/// a fourcc/stride change at the same frame size does not affect the upload.
fn apply_meta(link: &mut rdma_display_rdma::ClientLink, meta: &rdma_display_proto::DisplayMeta) {
    log::debug!(
        "[client] display meta gen {} ({}x{} stride {} fourcc {:#010x})",
        meta.gen, meta.width, meta.height, meta.stride, meta.fourcc
    );
    if meta.frame_bytes != link.meta.frame_bytes {
        log::warn!(
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
    log::info!("dumped {path}");
    Ok(())
}

/// Resolve the `--native` / `--set-mode` flags into a pre-handshake display
/// mode request. `--native` reads this client's own output mode (from
/// `wl_output`); a failure just means no change is requested.
fn resolve_mode_request(
    native: bool,
    set_mode: Option<&str>,
    output: &str,
) -> Result<rdma_display_proto::ModeRequest> {
    use rdma_display_proto::ModeRequest;
    if let Some(s) = set_mode {
        let (w, h, r) = parse_mode_spec(s).with_context(|| format!("--set-mode {s:?}"))?;
        return Ok(ModeRequest::Set {
            width: w,
            height: h,
            refresh_mhz: r,
        });
    }
    if native {
        if output == "none" {
            log::info!("[client] --native ignored with --output none (no output to query)");
            return Ok(ModeRequest::Keep);
        }
        match wayland::query_output_mode() {
            Ok((w, h, r)) => {
                log::info!("[client] --native: output mode {w}x{h}@{r} mHz");
                return Ok(ModeRequest::Set {
                    width: w,
                    height: h,
                    refresh_mhz: r,
                });
            }
            Err(e) => {
                log::warn!(
                    "[client] --native: could not read the output mode ({e:#}); \
                     not requesting a change"
                );
                return Ok(ModeRequest::Keep);
            }
        }
    }
    Ok(ModeRequest::Keep)
}

/// Parse `WxH` or `WxH@R` (R in Hz, fractional allowed; a trailing unit like
/// `Hz` is ignored) into `(width, height, refresh_mhz)`.
fn parse_mode_spec(s: &str) -> Result<(u32, u32, u32)> {
    let (wh, r) = match s.split_once('@') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let (w, h) = wh.split_once('x').context("expected WxH[@R]")?;
    let w: u32 = w.trim().parse().context("width")?;
    let h: u32 = h.trim().parse().context("height")?;
    let refresh_mhz = match r {
        Some(r) => {
            let rf: f64 = r
                .trim()
                .trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse()
                .context("refresh")?;
            (rf * 1000.0).round() as u32
        }
        None => 0,
    };
    Ok((w, h, refresh_mhz))
}

/// Initialize the logger. `-v`/`--verbose` enables debug for our crates; the
/// `RUST_LOG` environment variable overrides either default.
fn init_logging(verbose: bool) {
    use std::io::Write as _;
    let default = if verbose {
        "warn,rdma_display=debug"
    } else {
        "warn,rdma_display=info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default))
        .format(|buf, record| writeln!(buf, "[{}] {}", record.level(), record.args()))
        .init();
}

/// Append the default signaling port when the user gives only a host or IP
/// (`192.168.4.2`, or `-s 192.168.4.2`). An explicit port is left alone,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_default_port_appends_when_missing() {
        assert_eq!(with_default_port("192.168.4.2"), format!("192.168.4.2:{DEFAULT_PORT}"));
        assert_eq!(with_default_port("localhost"), format!("localhost:{DEFAULT_PORT}"));
    }

    #[test]
    fn with_default_port_preserves_explicit_port() {
        assert_eq!(with_default_port("192.168.4.2:5000"), "192.168.4.2:5000");
        assert_eq!(with_default_port("[::1]:4791"), "[::1]:4791");
    }

    #[test]
    fn with_default_port_leaves_bare_ipv6_and_empty_alone() {
        assert_eq!(with_default_port("::1"), "::1");
        assert_eq!(with_default_port(""), "");
    }
}
