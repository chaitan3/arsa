//! RDMA transport for rdma-display.
//!
//! Model: one RC queue pair per session. The *client* allocates N receive
//! slots (registered host memory), advertises them over the control channel,
//! and posts N zero-payload receives to catch WRITE-with-immediate
//! notifications. The *server* writes each frame directly into a client slot
//! and encodes `{frame_seq, slot}` in the immediate value. Credits flow back
//! over the control channel so the server only reuses a slot after the client
//! has consumed it.
//!
//! The data path uses polling completion loops; no async runtime. The control
//! channel is a plain blocking TCP connection. All verbs resources are
//! Arc-backed, so link structs can move to threads freely.

use std::collections::VecDeque;
use std::io::Write;
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use ibverbs::{
    AccessFlags, CompletionQueue, LocalMemorySlice, MemoryRegion, QueuePair,
    QueuePairEndpoint, Rc, RemoteMemorySlice,
};

use rdma_display_proto as proto;
use proto::{Conn, DisplayMeta, Message, SlotDesc};

/// Events that arrive on the client's control channel from the server
/// (server → client direction). Forwarded to the main loop by the
/// control-reader thread.
pub enum ControlEvent {
    Stamp(proto::FrameStamp),
    /// c0 = the client's clock when it sent the ping, s1 = server's reply
    /// time, c2 = client's clock when the pong arrived (reader thread).
    Pong { c0: u64, s1: u64, c2: u64 },
    /// server → client: force-feedback event for the physical controller
    Rumble(proto::gamepad::Rumble),
    /// server → client: the display meta changed (new generation)
    Meta(proto::DisplayMeta),
    /// server → client: the result of the pre-handshake display-mode request
    ModeReply(proto::ModeReply),
}

pub mod cuda;
pub mod slots;
use slots::{SlotAlloc, SlotBuf};

/// Where the client's receive slots live.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotMem {
    /// Registered host memory (today's copy path; the NIC lands frames in
    /// RAM and the consumer copies/uploads from there).
    Host,
    /// `cuMemAlloc` memory exported via `cuMemGetHandleForAddressRange` —
    /// NVIDIA's documented GPUDirect-over-dmabuf path (C0 route 2); refused
    /// CUDA_ERROR_NOT_SUPPORTED on GeForce (RM product gate). Dormant code
    /// for future cooperative hardware.
    Cuda { device: u32 },
}

/// Open an RDMA device by name (or the first available one).
pub fn open_device(name: Option<&str>) -> Result<ibverbs::Context> {
    let list = ibverbs::devices().context("ibv_get_device_list failed")?;
    if list.is_empty() {
        anyhow::bail!("no RDMA devices found");
    }
    for dev in &list {
        let dev_name = dev
            .name()
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.is_none() || name == Some(dev_name.as_str()) {
            return Ok(dev.open()?);
        }
    }
    anyhow::bail!("RDMA device {name:?} not found")
}

fn routable_gid_index(ctx: &ibverbs::Context) -> Result<u32> {
    if let Some(gid) = ctx.routable_gid(1).context("routable_gid failed")? {
        log::info!(
            "[rdma] sgid: index={} type={:?} gid={:?} ndev_ifindex={}",
            gid.gid_index, gid.gid_type, gid.gid, gid.ndev_ifindex
        );
        return Ok(gid.gid_index);
    }
    anyhow::bail!("no routable GID found on port 1 (check RoCE config)")
}

fn create_prepared_qp(
    ctx: &ibverbs::Context,
    pd: &ibverbs::ProtectionDomain,
) -> Result<(
    ibverbs::PreparedQueuePair<Rc>,
    CompletionQueue,
    CompletionQueue,
)> {
    let send_cq = ctx.create_cq(256).build().context("create send CQ")?;
    let recv_cq = ctx.create_cq(256).build().context("create recv CQ")?;
    let gid_index = routable_gid_index(ctx)?;
    let prepared = pd
        .create_qp::<Rc>(&send_cq, &recv_cq, 1)
        .context("create_qp")?
        .set_gid_index(gid_index)
        // both sides must accept inbound RDMA WRITEs (the client receives
        // frames; the server's QP is only the initiator today, but symmetric
        // permissions keep future two-way paths from tripping on this)
        .set_access(AccessFlags::PERMISSIVE)
        .build()
        .context("qp build")?;
    Ok((prepared, send_cq, recv_cq))
}

/// A per-session nonce (LAN-grade, not cryptographic): the server hands it to
/// the client in the Hello reply; the client presents it on its input
/// connection so a stale input socket can't feed a later session.
pub fn random_nonce() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    proto::unix_now_us().hash(&mut h);
    std::process::id().hash(&mut h);
    (&h as *const _ as usize).hash(&mut h);
    h.finish()
}

/// Video transport for a session. RDMA (ibverbs/RoCE) is preferred; TCP is
/// the fallback when no RDMA device (or RoCE GID) is available.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transport {
    Rdma,
    Tcp,
}

/// Caller's preference for [`Transport`]; `Auto` prefers RDMA.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransportPref {
    Auto,
    Rdma,
    Tcp,
}

impl TransportPref {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "rdma" | "roce" | "ib" => Some(Self::Rdma),
            "tcp" => Some(Self::Tcp),
            _ => None,
        }
    }
}

/// Client side of the handshake: advertise our transport capability, learn
/// the server's chosen transport (and the session nonce).
fn hello_client<T: std::io::Read + std::io::Write>(
    conn: &mut Conn<T>,
    caps: u8,
    codec_pref: u8,
    lossy_mbps: u16,
) -> Result<(u64, u8)> {
    conn.send(&Message::Hello(proto::Hello {
        magic: proto::MAGIC,
        version: proto::VERSION,
        nonce: 0,
        caps,
        codec_pref,
        lossy_mbps,
    }))?;
    match conn.recv()? {
        Message::Hello(h) if h.magic == proto::MAGIC => {
            if h.version != proto::VERSION {
                anyhow::bail!("peer protocol version {} != our {}", h.version, proto::VERSION);
            }
            Ok((h.nonce, h.caps))
        }
        other => Err(anyhow!("expected Hello, got {other:?}")),
    }
}

/// Server side of the handshake: read the client's capability, choose the
/// transport, and reply with the choice (and the session nonce).
fn hello_server<T: std::io::Read + std::io::Write>(
    conn: &mut Conn<T>,
    session_nonce: u64,
    pref: TransportPref,
    server_has_rdma: bool,
) -> Result<(Transport, u8, u16)> {
    let ch = match conn.recv()? {
        Message::Hello(h) if h.magic == proto::MAGIC => h,
        other => return Err(anyhow!("expected Hello, got {other:?}")),
    };
    if ch.version != proto::VERSION {
        anyhow::bail!("peer protocol version {} != our {}", ch.version, proto::VERSION);
    }
    let client_rdma = (ch.caps & proto::CAP_RDMA) != 0;
    let transport = match pref {
        TransportPref::Tcp => Transport::Tcp,
        TransportPref::Rdma => {
            if !server_has_rdma {
                anyhow::bail!("--transport rdma: no RDMA device on the server");
            }
            if !client_rdma {
                anyhow::bail!("--transport rdma: the client has no RDMA device");
            }
            Transport::Rdma
        }
        TransportPref::Auto => {
            if server_has_rdma && client_rdma {
                Transport::Rdma
            } else {
                Transport::Tcp
            }
        }
    };
    conn.send(&Message::Hello(proto::Hello {
        magic: proto::MAGIC,
        version: proto::VERSION,
        nonce: session_nonce,
        caps: if transport == Transport::Rdma { proto::CAP_RDMA } else { 0 },
        codec_pref: proto::CODEC_ANY, // ignored in the reply
        lossy_mbps: 0,                // ignored in the reply
    }))?;
    Ok((transport, ch.codec_pref, ch.lossy_mbps))
}

fn endpoint_exchange<T: std::io::Read + std::io::Write>(
    conn: &mut Conn<T>,
    ctx: &ibverbs::Context,
    pd: &ibverbs::ProtectionDomain,
) -> Result<(QueuePair<Rc>, CompletionQueue, CompletionQueue)> {
    let (prepared, send_cq, recv_cq) = create_prepared_qp(ctx, pd)?;
    let local = prepared.endpoint().context("qp endpoint")?;
    conn.send(&Message::QpEndpoint(local.to_bytes()))?;
    match conn.recv()? {
        Message::QpEndpoint(bytes) => {
            let remote = QueuePairEndpoint::from_bytes(&bytes).map_err(|e| anyhow!("{e}"))?;
            log::debug!(
                "[rdma] endpoints: local qpn={} psn={:#010x} gid={:?} | remote qpn={} psn={:#010x} gid={:?}",
                local.qp_num,
                local.psn,
                local.gid,
                remote.qp_num,
                remote.psn,
                remote.gid,
            );
            let qp = prepared.handshake(remote).context("qp handshake")?;
            Ok((qp, send_cq, recv_cq))
        }
        other => Err(anyhow!("expected QpEndpoint, got {other:?}")),
    }
}

/// Immediate-word layout: `frame_seq << 8 | slot_index`.
pub fn imm_encode(frame_seq: u64, slot: usize) -> u32 {
    (((frame_seq & 0x00ff_ffff) << 8) as u32) | ((slot as u32) & 0xff)
}

pub fn imm_decode(imm: u32) -> (u64, usize) {
    ((imm >> 8) as u64, (imm & 0xff) as usize)
}

// ---------------------------------------------------------------------------
// Server side
// ---------------------------------------------------------------------------

/// One server-side staging slot: either verbs-allocated heap (allocated
/// mode, testcard/plain) or externally provided memory (PBO readback mode —
/// the pages are the materializer's persistent-mapped PBOs, registered here
/// per connection with `register_from_raw` so the NIC reads exactly what the
/// GL readback DMA wrote: zero copy).
enum StagingSlot {
    Owned(MemoryRegion<Box<[u8]>>),
    Ext(MemoryRegion<()>),
}

impl StagingSlot {
    /// Postable slice covering the first `len` bytes.
    fn post_slice(&self, len: usize) -> LocalMemorySlice {
        match self {
            StagingSlot::Owned(m) => m.slice(..len),
            StagingSlot::Ext(m) => m.slice(..len),
        }
    }
    /// Registered byte length.
    fn len(&self) -> usize {
        match self {
            StagingSlot::Owned(m) => m.bytes().len(),
            StagingSlot::Ext(m) => m.remote().len,
        }
    }
    /// Mutable view of the registered memory. For `Ext` the memory is the
    /// GL-owned PBO map: `glReadPixels(GL_BGRA)` writes it; `fill` only
    /// recognizes the pointer and drives the GL side. Aliasing with the
    /// materializer's map is intentional and safe while the MR is alive
    /// (the backing memory is pinned and never moved).
    fn as_mut_bytes(&mut self, len: usize) -> &mut [u8] {
        match self {
            StagingSlot::Owned(m) => &mut m.bytes_mut()[..len],
            StagingSlot::Ext(m) => {
                let s = m.slice(..len);
                unsafe { core::slice::from_raw_parts_mut(s.addr() as *mut u8, len) }
            }
        }
    }
}

enum Tx {
    Rdma {
        qp: QueuePair<Rc>,
        send_cq: CompletionQueue,
        /// local staging slots (copy path / testcard writes here before WRITE)
        slots: Vec<StagingSlot>,
        remote: Vec<RemoteMemorySlice>,
        slot_free: Vec<bool>,
    },
    Tcp {
        slot_len: usize,
        buf: Vec<u8>,
        busy: bool,
    },
}

/// Server side of the pre-handshake: negotiate the video transport (the
/// `Hello` exchange) and read the client's display-mode request. Returns the
/// still-open control connection, the negotiated transport, the request, and
/// the client's codec preference ([`proto::CODEC_ANY`] = none).
///
/// The caller applies the mode (and re-probes the display) *before* calling
/// [`ServerLink::accept_negotiated`], so the session's slots are sized for the
/// final geometry — no mid-session resize and no reconnect. The client blocks
/// in its own handshake until it reads the [`proto::ModeReply`] the caller
/// sends with [`send_mode_reply`].
pub fn handshake_server(
    stream: TcpStream,
    session_nonce: u64,
    pref: TransportPref,
    server_has_rdma: bool,
) -> Result<(Conn<TcpStream>, Transport, proto::ModeRequest, u8, u16)> {
    let mut conn = Conn::new(stream);
    let (transport, codec_pref, lossy_mbps) =
        hello_server(&mut conn, session_nonce, pref, server_has_rdma)?;
    log::info!(
        "[rdma] video transport: {}{}",
        match transport {
            Transport::Rdma => "RDMA",
            Transport::Tcp => "TCP",
        },
        if transport == Transport::Tcp { " (RDMA unavailable)" } else { "" }
    );
    let req = match conn.recv()? {
        Message::SetMode(r) => r,
        other => anyhow::bail!("expected SetMode, got {other:?}"),
    };
    Ok((conn, transport, req, codec_pref, lossy_mbps))
}

/// Send the display-mode result to the client. Called after the mode has been
/// applied (or rejected), before [`ServerLink::accept_negotiated`].
pub fn send_mode_reply(conn: &mut Conn<TcpStream>, reply: &proto::ModeReply) -> Result<()> {
    conn.send(&Message::ModeReply(reply.clone()))?;
    Ok(())
}

/// Exponential moving average (1/8 step), seeded by the first sample. Used for
/// the link-measurement EMAs so one slow frame doesn't move the window.
fn ema(prev: u64, x: u64) -> u64 {
    if prev == 0 {
        x
    } else {
        (prev * 7 + x) / 8
    }
}

// --- SIOCOUTQ: bytes queued in a socket's kernel send buffer (Linux) ---------
/// `SIOCOUTQ`/`TIOCOUTQ` (identical value on Linux).
const SIOCOUTQ: u64 = 0x5411;

extern "C" {
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

pub struct ServerLink {
    /// credits arrive from the control-reader thread, each timestamped when the
    /// reader received it (so the measured turnaround excludes our own loop
    /// latency); None = client gone
    credit_rx: mpsc::Receiver<Option<(u32, u64)>>,
    tx: Tx,
    seq: u64,
    /// current `DisplayMeta` generation, stamped into every FrameStamp;
    /// `send_meta` bumps it and pushes the new meta to the control reader
    meta_gen: u32,
    /// cloned control stream: the send path writes per-frame `Stamp`s here
    /// while the control-reader thread owns the receive side
    stamp_w: Option<TcpStream>,
    transport: Transport,
    /// TCP video socket; None until the client's TAG_VIDEO connection arrives
    video: Option<TcpStream>,
    /// Flow-control window: the most frames that may be sent-but-uncredited.
    /// The client credits one per frame it consumes (presented or superseded),
    /// so this bounds how far the server may run ahead of the client's
    /// display. Both transports use it — on RDMA a credit also frees the
    /// staging slot. The caller sizes it from the measured link (see
    /// [`ServerLink::turnaround_us`] / [`ServerLink::send_interval_us`]).
    window: usize,
    /// Hard upper bound for `window`: RDMA cannot have more frames in flight
    /// than it has staging slots to hold them.
    max_window: usize,
    /// frames sent and not yet credited
    inflight: usize,
    /// send time (unix µs) of each un-credited frame, FIFO-paired with credits
    send_times: VecDeque<u64>,
    /// EMA of the send -> credit turnaround (µs): how long the client takes to
    /// consume a frame and release it (link + decode + present + link)
    turnaround_us: u64,
    /// EMA of the interval between sends (µs): the source frame period
    send_interval_us: u64,
    /// last send time (unix µs), for `send_interval_us`
    last_send_us: u64,
    /// coded bytes of the most recently sent frame (judder instrumentation)
    last_sent_bytes: usize,
}

impl ServerLink {
    /// Server half of a session whose `Hello` (and display-mode exchange) has
    /// already been handled by [`handshake_server`]. `transport` is the
    /// negotiated transport; `conn` is the still-open control connection.
    /// Everything from the RDMA endpoint exchange onward lives here.
    #[allow(clippy::too_many_arguments)]
    pub fn accept_negotiated(
        conn: Conn<TcpStream>,
        transport: Transport,
        rdma: Option<(&ibverbs::Context, &ibverbs::ProtectionDomain)>,
        meta: &DisplayMeta,
        slots: usize,
        codec: rdma_display_codec::Codec,
        slot_len: usize,
        chroma: u8,
        input_tx: mpsc::Sender<proto::InputEvent>,
        external: Option<&[(*mut u8, usize)]>,
        rumble: &proto::gamepad::RumbleHub,
    ) -> Result<ServerLink> {
        // split the stream: reader thread gets the receive half, the frame
        // send path keeps a write-only clone for per-frame Stamps
        let stream = conn.into_inner();
        let stamp_w = stream.try_clone().context("clone stream for stamps")?;
        let rumble_w = stream.try_clone().context("clone stream for rumble")?;
        let mut conn = Conn::new(stream);

        let tx = match transport {
            Transport::Rdma => {
                let (ctx, pd) = rdma.context("RDMA transport selected but no device")?;
                let (qp, send_cq, _recv_cq) = endpoint_exchange(&mut conn, ctx, pd)?;

                let mut stage: Vec<StagingSlot> = Vec::with_capacity(slots);
                match external {
                    Some(list) => {
                        anyhow::ensure!(
                            list.len() == slots,
                            "external staging slot count {} != {slots}",
                            list.len()
                        );
                        for (i, (ptr, len)) in list.iter().enumerate() {
                            anyhow::ensure!(
                                *len == slot_len,
                                "external slot {i} len {len} != expected {slot_len}"
                            );
                            // zero-copy staging: the NIC reads the same pages the GL
                            // readback DMA fills. LOCAL_WRITE|REMOTE_READ, matching
                            // the bench probe (deliberately not PERMISSIVE).
                            let mr = unsafe {
                                pd.register_from_raw(
                                    *ptr,
                                    *len,
                                    AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ,
                                )
                            }
                            .context("register external staging slot (PBO)")?;
                            stage.push(StagingSlot::Ext(mr));
                        }
                        log::info!(
                            "[rdma] external staging: {slots} slots from materializer PBOs (zero-copy)"
                        );
                    }
                    None => {
                        for _ in 0..slots {
                            let mr = pd
                                .allocate(slot_len, AccessFlags::PERMISSIVE)
                                .context("allocate staging slot")?;
                            stage.push(StagingSlot::Owned(mr));
                        }
                    }
                }

                conn.send(&Message::Meta(meta.clone()))?;
                // tell the client which codec the frames will use before it
                // allocates its receive slots (they must hold the worst case)
                conn.send(&Message::Codec(codec.id(), slot_len as u32, chroma))?;
                let remote: Vec<RemoteMemorySlice> = match conn.recv()? {
                    Message::Buffers(slots) => slots
                        .into_iter()
                        .map(|s| {
                            if s.len as usize != slot_len {
                                anyhow::bail!(
                                    "client slot len {} != expected slot {} for codec {}",
                                    s.len,
                                    slot_len,
                                    codec.name()
                                );
                            }
                            Ok(RemoteMemorySlice {
                                addr: s.addr,
                                len: s.len as usize,
                                rkey: s.rkey,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                    other => anyhow::bail!("expected Buffers, got {other:?}"),
                };
                conn.send(&Message::Ready)?;
                Tx::Rdma {
                    qp,
                    send_cq,
                    slots: stage,
                    remote,
                    slot_free: vec![true; slots],
                }
            }
            Transport::Tcp => {
                // no slots/buffers: Meta+Codec on control; frames ride the
                // TAG_VIDEO socket once the client opens it
                conn.send(&Message::Meta(meta.clone()))?;
                conn.send(&Message::Codec(codec.id(), slot_len as u32, chroma))?;
                Tx::Tcp {
                    slot_len,
                    buf: vec![0u8; slot_len],
                    busy: false,
                }
            }
        };

        // rumble: the pad thread's events ride this channel to a dedicated
        // forwarder on a private clone of the control stream; a live session
        // owns the hub's sender slot, and the control-reader clears it on
        // disconnect so pad threads never block on a dead session
        let (rumble_tx, rumble_rx) = mpsc::channel::<proto::gamepad::Rumble>();
        rumble.set(rumble_tx);
        std::thread::Builder::new()
            .name("rumble-fwd".into())
            .spawn(move || {
                let mut w = rumble_w;
                loop {
                    match rumble_rx.recv() {
                        Ok(r) => {
                            if proto::send_msg(&mut w, &proto::Message::Rumble(r)).is_err() {
                                break;
                            }
                        }
                        Err(_) => break, // all senders dropped (session gone)
                    }
                }
            })
            .context("spawn rumble forwarder")?;

        // Control channel reader: owns the TCP stream from here on. Credits
        // go to the sender loop via a channel; input events are forwarded to
        // the capture side (QEMU D-Bus input injection). Reading ONLY inside
        // wait_free_slot would starve input while slots are busy.
        let (credit_tx, credit_rx) = mpsc::channel::<Option<(u32, u64)>>();
        let reader_rumble = rumble.clone();
        let reader = std::thread::Builder::new()
            .name("control-reader".into())
            .spawn(move || {
                loop {
                    match conn.recv() {
                        Ok(proto::Message::Credit(slot)) => {
                            // timestamp on arrival: the sender's turnaround
                            // measurement must not absorb its own loop latency
                            if credit_tx.send(Some((slot, proto::unix_now_us()))).is_err() {
                                break; // sender gone (server shutting down)
                            }
                        }
                        Ok(proto::Message::Input(ev)) => {
                            if input_tx.send(ev).is_err() {
                                break;
                            }
                        }
                        Ok(proto::Message::Ping(c0)) => {
                            // clock-sync reply: echo the client's stamp + our time
                            if let Err(e) =
                                conn.send(&proto::Message::Pong(c0, proto::unix_now_us()))
                            {
                                log::debug!("[rdma] pong failed: {e}");
                            }
                        }
                        Ok(proto::Message::Bye(_)) => {
                            let _ = credit_tx.send(None);
                            break;
                        }
                        Ok(other) => {
                            log::debug!("[rdma] unexpected control message {other:?}");
                        }
                        Err(e) => {
                            if e.kind() != std::io::ErrorKind::UnexpectedEof
                                && e.kind() != std::io::ErrorKind::ConnectionReset
                            {
                                log::debug!("[rdma] control channel error: {e}");
                            }
                            let _ = credit_tx.send(None);
                            break;
                        }
                    }
                }
                // session is gone: drop the hub's sender so the pad thread's
                // rumbles become no-ops and the forwarder thread winds down
                reader_rumble.clear();
            })
            .context("spawn control reader")?;
        std::mem::forget(reader); // runs for the session lifetime

        Ok(ServerLink {
            credit_rx,
            tx,
            seq: 0,
            meta_gen: meta.gen,
            stamp_w: Some(stamp_w),
            transport,
            video: None,
            // start at the physical maximum (the staging-slot count); the
            // caller narrows it from the measured link via `set_window`
            window: slots.max(1),
            max_window: slots.max(1),
            inflight: 0,
            send_times: VecDeque::new(),
            turnaround_us: 0,
            send_interval_us: 0,
            last_send_us: 0,
            last_sent_bytes: 0,
        })
    }

    /// The negotiated video transport.
    pub fn transport(&self) -> Transport {
        self.transport
    }

    /// Attach the client's TAG_VIDEO socket (TCP transport). Until this is
    /// called, `send_frame*` returns `Ok(None)` (nothing is sent).
    pub fn attach_video(&mut self, stream: TcpStream) {
        let _ = stream.set_nodelay(true);
        log::debug!("[rdma] TCP video socket attached (fd {})", stream.as_raw_fd());
        self.video = Some(stream);
    }

    pub fn has_video(&self) -> bool {
        self.video.is_some()
    }

    /// Whether one more frame may be sent without exceeding the flow-control
    /// window (see [`ServerLink::set_window`]).
    pub fn window_open(&self) -> bool {
        self.inflight < self.window
    }

    /// Current flow-control window (max frames sent-but-uncredited).
    pub fn window(&self) -> usize {
        self.window
    }

    /// Upper bound for the window (the staging-slot count).
    pub fn max_window(&self) -> usize {
        self.max_window
    }

    /// Narrow (or widen) the flow-control window; clamped to `[1, max_window]`.
    /// The caller derives the value from [`ServerLink::turnaround_us`] and the
    /// source frame period it measures itself (from the capture cadence).
    pub fn set_window(&mut self, n: usize) -> usize {
        self.window = n.clamp(1, self.max_window.max(1));
        self.window
    }

    /// EMA of the send -> credit turnaround (µs); 0 until measured. That is how
    /// long the client takes to consume a frame and release it, so it already
    /// contains the client's decode and its wait for the next present.
    pub fn turnaround_us(&self) -> u64 {
        self.turnaround_us
    }

    /// EMA of the interval between sends (µs); 0 until measured. This is the
    /// *send* cadence, which is a consequence of the window — do NOT use it to
    /// size the window (that makes the formula a neutral fixed point). Kept for
    /// diagnostics only; the window uses the source frame period.
    pub fn send_interval_us(&self) -> u64 {
        self.send_interval_us
    }

    /// Frames sent and not yet credited (the in-flight count within the window).
    pub fn inflight(&self) -> usize {
        self.inflight
    }

    /// Coded bytes of the most recently sent frame (0 until the first send).
    pub fn last_sent_bytes(&self) -> usize {
        self.last_sent_bytes
    }

    /// Bytes still queued in the TCP video socket's kernel send buffer
    /// (`SIOCOUTQ`). Non-zero and growing ⇒ the link, not the pacing loop, is
    /// the bottleneck. `None` for RDMA or before the video socket is attached.
    pub fn video_outq_bytes(&self) -> Option<u32> {
        let s = self.video.as_ref()?;
        let mut q: i32 = 0;
        let rc = unsafe { ioctl(s.as_raw_fd(), SIOCOUTQ, &mut q) };
        if rc == 0 && q >= 0 {
            Some(q as u32)
        } else {
            None
        }
    }

    /// Block until the window has room (a client credit arrives) or the client
    /// goes away. The caller waits *before* taking/materializing a frame, so a
    /// frame that cannot be sent yet is never encoded and the latest-wins
    /// mailbox can collapse to the freshest capture while we wait.
    pub fn wait_for_window(&mut self) -> Result<()> {
        while !self.window_open() {
            match self.credit_rx.recv() {
                Ok(Some((slot, at))) => self.apply_credit(slot, at),
                Ok(None) => anyhow::bail!("client hung up"),
                Err(_) => anyhow::bail!("client disconnected"),
            }
        }
        Ok(())
    }

    /// Apply one client credit: free the staging slot (RDMA) and release one
    /// in-flight frame, folding its turnaround into the EMA. `at_us` is the
    /// unix-µs the control reader received the credit — using the *arrival*
    /// time (not `now`) keeps the sender's own loop latency out of the
    /// measurement.
    fn apply_credit(&mut self, slot: u32, at_us: u64) {
        if let Tx::Rdma { slot_free, .. } = &mut self.tx {
            if let Some(f) = slot_free.get_mut(slot as usize) {
                *f = true;
            }
        }
        self.inflight = self.inflight.saturating_sub(1);
        if let Some(t0) = self.send_times.pop_front() {
            self.turnaround_us = ema(self.turnaround_us, at_us.saturating_sub(t0));
        }
    }

    /// Record a sent frame: bump the in-flight count and stamp its send time.
    fn note_send(&mut self) {
        let now = proto::unix_now_us();
        if self.last_send_us != 0 {
            let dt = now.saturating_sub(self.last_send_us);
            if dt > 0 {
                self.send_interval_us = ema(self.send_interval_us, dt);
            }
        }
        self.last_send_us = now;
        self.send_times.push_back(now);
        if self.send_times.len() > 256 {
            self.send_times.pop_front();
        }
        self.inflight = self.inflight.saturating_add(1);
    }

    /// Undo [`note_send`](Self::note_send) for a claimed frame that turned out
    /// unchanged and is therefore not sent.
    fn note_unsend(&mut self) {
        self.inflight = self.inflight.saturating_sub(1);
        self.send_times.pop_back();
    }

    /// Non-blocking hangup/credit check: drains the credit channel (applying
    /// credits) and returns true when the control-reader thread signaled the
    /// client hung up (EOF/Bye). Call each loop iteration — an idle capture
    /// loop may send nothing for a long time and otherwise never notices.
    pub fn client_gone(&mut self) -> bool {
        loop {
            match self.credit_rx.try_recv() {
                Ok(Some((slot, at))) => self.apply_credit(slot, at),
                Ok(None) => return true, // reader thread saw EOF
                Err(mpsc::TryRecvError::Empty) => return false,
                Err(mpsc::TryRecvError::Disconnected) => return true,
            }
        }
    }

    /// Whether a staging slot is currently free *and* the window has room.
    pub fn has_free_slot(&self) -> bool {
        if !self.window_open() {
            return false;
        }
        match &self.tx {
            Tx::Rdma { slot_free, .. } => slot_free.iter().any(|f| *f),
            Tx::Tcp { busy, .. } => !*busy,
        }
    }

    /// Non-blocking claim of the first free slot (marks it in use).
    pub fn try_claim_slot(&mut self) -> Option<usize> {
        if !self.window_open() {
            return None;
        }
        let slot = match &mut self.tx {
            Tx::Rdma { slot_free, .. } => {
                let slot = slot_free.iter().position(|f| *f)?;
                slot_free[slot] = false;
                slot
            }
            Tx::Tcp { busy, .. } => {
                if *busy {
                    return None;
                }
                *busy = true;
                0
            }
        };
        self.note_send();
        Some(slot)
    }

    /// Return a claimed slot (used when a frame turns out unchanged).
    pub fn release_slot(&mut self, slot: usize) {
        match &mut self.tx {
            Tx::Rdma { slot_free, .. } => slot_free[slot] = true,
            Tx::Tcp { busy, .. } => *busy = false,
        }
        self.note_unsend();
    }

    /// The claimed slot's registered memory, for `issue_frame`.
    pub fn slot_bytes_mut(&mut self, slot: usize) -> &mut [u8] {
        match &mut self.tx {
            Tx::Rdma { slots, .. } => {
                let len = slots[slot].len();
                slots[slot].as_mut_bytes(len)
            }
            Tx::Tcp { buf, .. } => buf.as_mut_slice(),
        }
    }

    /// Blocks until an RDMA staging slot is free *and* the window has room,
    /// absorbing credit messages meanwhile. (Credits arrive via the
    /// control-reader thread's channel.)
    fn claim_free_slot_rdma(&mut self) -> Result<usize> {
        loop {
            let mut claimed = None;
            if self.window_open() {
                if let Tx::Rdma { slot_free, .. } = &mut self.tx {
                    if let Some(s) = slot_free.iter().position(|f| *f) {
                        slot_free[s] = false;
                        claimed = Some(s);
                    }
                }
            }
            if let Some(slot) = claimed {
                self.note_send();
                return Ok(slot);
            }
            match self.credit_rx.recv() {
                Ok(Some((slot, at))) => self.apply_credit(slot, at),
                Ok(None) => anyhow::bail!("client hung up"),
                Err(_) => anyhow::bail!("client disconnected"),
            }
        }
    }

    /// Send one frame. `fill` writes the frame bytes into the staging slot
    /// before it is posted to the client. `t0_unix_us` is the frame's
    /// pipeline t0 (D-Bus event arrival; for testcard just pass `now`).
    /// Returns the frame sequence number.
    pub fn send_frame(&mut self, fill: impl FnOnce(&mut [u8]), t0_unix_us: u64) -> Result<u64> {
        self.send_frame_checked(
            |out| {
                let n = out.len();
                fill(out);
                Ok((true, n, proto::MatTimes::ZERO))
            },
            t0_unix_us,
            t0_unix_us,
            t0_unix_us,
        )?
        .ok_or_else(|| anyhow!("frame unexpectedly unchanged"))
    }

    /// Like [`send_frame`](Self::send_frame), but `fill` decides whether the
    /// frame content changed and should be sent (`Ok(false)` = skip), returns
    /// the number of bytes to post (`post_len`, e.g. the codec's coded length),
    /// and returns the frame's GPU stage times for the Stamp.
    ///
    /// `t0d_unix_us` is the sender's pickup time (pacing/gate done, *before*
    /// the scanout import) and `t0p_unix_us` the prepare-done time, so
    /// `queue = t0d - t0` is the pacing wait only and `t1s - t0p` is the
    /// staging-slot credit wait; neither absorbs the per-frame prepare.
    pub fn send_frame_checked(
        &mut self,
        fill: impl FnOnce(&mut [u8]) -> anyhow::Result<(bool, usize, proto::MatTimes)>,
        t0_unix_us: u64,
        t0d_unix_us: u64,
        t0p_unix_us: u64,
    ) -> Result<Option<u64>> {
        match self.transport {
            Transport::Rdma => self.send_frame_rdma(fill, t0_unix_us, t0d_unix_us, t0p_unix_us),
            Transport::Tcp => self.send_frame_tcp(fill, t0_unix_us, t0d_unix_us, t0p_unix_us),
        }
    }

    fn send_frame_rdma(
        &mut self,
        fill: impl FnOnce(&mut [u8]) -> anyhow::Result<(bool, usize, proto::MatTimes)>,
        t0_unix_us: u64,
        t0d: u64,
        t0p: u64,
    ) -> Result<Option<u64>> {
        let slot = self.claim_free_slot_rdma()?;
        let t1s = proto::unix_now_us(); // staging slot available
        let slot_wait_us = t1s.saturating_sub(t0d);
        if slot_wait_us > 5_000 {
            // can fire every frame when the client credits lag: keep it behind -v
            log::debug!(
                "[rdma] slot wait {} ms (client credits lagging)",
                slot_wait_us / 1000
            );
        }
        let (changed, post_len, mat_times, frame_bytes) = {
            let Tx::Rdma { slots, .. } = &mut self.tx else {
                unreachable!()
            };
            let frame_bytes = slots[slot].len();
            let (changed, post_len, mat_times) = fill(slots[slot].as_mut_bytes(frame_bytes))?;
            #[cfg(feature = "frame-checksums")]
            {
                let out = slots[slot].as_mut_bytes(frame_bytes);
                let sum: u32 = out.iter().map(|b| *b as u32).sum();
                let nonzero = out.iter().filter(|b| **b != 0).count();
                log::info!(
                    "[rdma] staging slot {} checksum sum={} nonzero={}/{}",
                    slot, sum, nonzero, frame_bytes
                );
            }
            (changed, post_len, mat_times, frame_bytes)
        };
        if !changed {
            // unchanged frame: return the slot, don't send
            self.release_slot(slot);
            return Ok(None);
        }
        anyhow::ensure!(post_len <= frame_bytes, "post_len {post_len} > slot {frame_bytes}");
        let t1_unix_us = proto::unix_now_us(); // readback/materialize complete
        let seq = self.seq;
        let (t2_unix_us, t2c_unix_us) = {
            let Tx::Rdma {
                slots,
                remote,
                qp,
                send_cq,
                ..
            } = &mut self.tx
            else {
                unreachable!()
            };
            let remote = remote[slot];
            let imm = imm_encode(seq, slot);
            let mut batch = qp.start_send();
            batch
                .op()
                .signaled()
                .imm(imm)
                .write(seq, &[slots[slot].post_slice(post_len)], remote);
            unsafe { batch.submit() }.context("post RDMA WRITE")?;
            let t2 = proto::unix_now_us(); // WRITE posted to the NIC
            // drain completions so the signaled WRITE is confirmed before we
            // hand the staging slot back to the caller
            loop {
                let mut completions = send_cq.poll().context("poll send CQ")?;
                match completions.next() {
                    Some(wc) => {
                        wc.ok().context("RDMA WRITE failed")?;
                        break;
                    }
                    None => std::thread::sleep(Duration::from_micros(20)),
                }
            }
            // WRITE completion, same-host stamp: t2c - t2 is the true transfer
            // time and needs no clock sync at all (unlike the client's t3)
            (t2, proto::unix_now_us())
        };
        self.seq += 1;
        self.last_sent_bytes = post_len;
        self.send_stamp(
            seq, t0_unix_us, t0d, t0p, t1s, t1_unix_us, t2_unix_us, t2c_unix_us, mat_times,
        );
        Ok(Some(seq))
    }

    fn send_frame_tcp(
        &mut self,
        fill: impl FnOnce(&mut [u8]) -> anyhow::Result<(bool, usize, proto::MatTimes)>,
        t0_unix_us: u64,
        t0d: u64,
        t0p: u64,
    ) -> Result<Option<u64>> {
        if self.video.is_none() {
            return Ok(None); // client hasn't opened its TAG_VIDEO socket yet
        }
        // Flow control: the client credits one frame per present (see
        // `ClientLink::credit_slot`), so wait for room before encoding. This is
        // what paces TCP sends to the client's display — there are no staging
        // slots to gate on, unlike RDMA.
        self.wait_for_window()?;
        let t1s = t0d; // TCP has no staging-slot credit wait of its own
        let (changed, post_len, mat_times, capacity) = {
            let Tx::Tcp { buf, .. } = &mut self.tx else {
                unreachable!()
            };
            let capacity = buf.len();
            let (changed, post_len, mat_times) = fill(buf)?;
            (changed, post_len, mat_times, capacity)
        };
        if !changed {
            return Ok(None);
        }
        anyhow::ensure!(post_len <= capacity, "post_len {post_len} > slot {capacity}");
        let t1_unix_us = proto::unix_now_us();
        let seq = self.seq;
        let (t2_unix_us, t2c_unix_us) = {
            let Tx::Tcp { buf, .. } = &self.tx else {
                unreachable!()
            };
            let w = self.video.as_mut().expect("video attached");
            // per-frame fd sanity check: it costs a getpeername syscall, so run
            // it only when debug logging is enabled
            if log::log_enabled!(log::Level::Debug) {
                if let Err(e) = w.peer_addr() {
                    log::debug!(
                        "[rdma] video socket fd {} is not a socket before write ({e}); an fd was closed and reused",
                        w.as_raw_fd()
                    );
                }
            }
            w.write_all(&(post_len as u32).to_le_bytes())
                .context("tcp video frame length")?;
            w.write_all(&seq.to_le_bytes())
                .context("tcp video frame seq")?;
            w.write_all(&buf[..post_len]).context("tcp video frame")?;
            let t2 = proto::unix_now_us();
            // no remote-completion signal on TCP: t2c == t2 (the client's t3
            // read-observation is the meaningful receipt stamp)
            (t2, t2)
        };
        self.note_send();
        self.seq += 1;
        self.last_sent_bytes = post_len;
        self.send_stamp(
            seq, t0_unix_us, t0d, t0p, t1s, t1_unix_us, t2_unix_us, t2c_unix_us, mat_times,
        );
        Ok(Some(seq))
    }

    #[allow(clippy::too_many_arguments)]
    fn send_stamp(
        &mut self,
        seq: u64,
        t0_unix_us: u64,
        t0d: u64,
        t0p: u64,
        t1s: u64,
        t1: u64,
        t2: u64,
        t2c: u64,
        mat_times: proto::MatTimes,
    ) {
        let Some(w) = self.stamp_w.as_mut() else {
            return;
        };
        // Clamp: a frame that sat dirty >1 s (e.g. arrived before any client
        // connected, or a zero-initialized t0) would poison the latency stats
        // with a meaningless queue wait — collapse it to 0.
        let t0 = if t0_unix_us == 0 || t0d.saturating_sub(t0_unix_us) > 1_000_000 {
            t0d
        } else {
            t0_unix_us
        };
        let s = proto::FrameStamp {
            seq,
            t0,
            t0d,
            t0p,
            t1s,
            t1,
            t2,
            t2c,
            mat: mat_times,
            meta_gen: self.meta_gen,
        };
        if let Err(e) = proto::send_msg(w, &Message::Stamp(s)) {
            log::debug!("[rdma] stamp send failed: {e}");
            self.stamp_w = None;
        }
    }

    /// Pipelined post: WRITE a slot that `issue_frame` already filled (the
    /// materializer completed it via `finish_frame`). `t0d` is the pickup
    /// time (pacing/queue wait done), `t0p` the prepare (dma-buf import) done
    /// time, and `t1s` the slot-claim time, so the stamp keeps the same
    /// `queue`/`prepare`/`slot`/`material` split as the serial path; `t1` is
    /// the fence-completion time. `post_len` is the number of bytes to send
    /// (the codec's coded length, or the raw frame size).
    #[allow(clippy::too_many_arguments)]
    pub fn post_filled(
        &mut self,
        slot: usize,
        t0_unix_us: u64,
        t0d_unix_us: u64,
        t0p_unix_us: u64,
        t1s_unix_us: u64,
        t1_unix_us: u64,
        post_len: usize,
        mat_times: proto::MatTimes,
    ) -> Result<Option<u64>> {
        let seq = self.seq;
        self.last_sent_bytes = post_len;
        match self.transport {
            Transport::Rdma => {
                let (t2, t2c) = {
                    let Tx::Rdma {
                        slots,
                        remote,
                        qp,
                        send_cq,
                        ..
                    } = &mut self.tx
                    else {
                        unreachable!()
                    };
                    let frame_bytes = slots[slot].len();
                    anyhow::ensure!(
                        post_len <= frame_bytes,
                        "post_len {post_len} > slot {frame_bytes}"
                    );
                    let remote = remote[slot];
                    let imm = imm_encode(seq, slot);
                    let mut batch = qp.start_send();
                    batch
                        .op()
                        .signaled()
                        .imm(imm)
                        .write(seq, &[slots[slot].post_slice(post_len)], remote);
                    unsafe { batch.submit() }.context("post RDMA WRITE")?;
                    let t2 = proto::unix_now_us();
                    // drain completions so the signaled WRITE is confirmed;
                    // the slot stays busy until the client's credit arrives
                    loop {
                        let mut completions = send_cq.poll().context("poll send CQ")?;
                        match completions.next() {
                            Some(wc) => {
                                wc.ok().context("RDMA WRITE failed")?;
                                break;
                            }
                            None => std::thread::sleep(Duration::from_micros(20)),
                        }
                    }
                    (t2, proto::unix_now_us())
                };
                self.seq += 1;
                self.send_stamp(
                    seq, t0_unix_us, t0d_unix_us, t0p_unix_us, t1s_unix_us, t1_unix_us, t2, t2c,
                    mat_times,
                );
                Ok(Some(seq))
            }
            Transport::Tcp => {
                if self.video.is_none() {
                    return Ok(None);
                }
                let (t2, t2c) = {
                    let Tx::Tcp { buf, .. } = &self.tx else {
                        unreachable!()
                    };
                    anyhow::ensure!(
                        post_len <= buf.len(),
                        "post_len {post_len} > slot {}",
                        buf.len()
                    );
                    let w = self.video.as_mut().expect("video attached");
                    // per-frame fd sanity check: it costs a getpeername syscall,
                    // so run it only when debug logging is enabled
                    if log::log_enabled!(log::Level::Debug) {
                        if let Err(e) = w.peer_addr() {
                            log::debug!(
                                "[rdma] video socket fd {} is not a socket before write ({e}); an fd was closed and reused",
                                w.as_raw_fd()
                            );
                        }
                    }
                    w.write_all(&(post_len as u32).to_le_bytes())
                        .context("tcp video frame length")?;
                    w.write_all(&seq.to_le_bytes())
                        .context("tcp video frame seq")?;
                    w.write_all(&buf[..post_len]).context("tcp video frame")?;
                    let t2 = proto::unix_now_us();
                    (t2, t2)
                };
                self.seq += 1;
                self.send_stamp(
                    seq, t0_unix_us, t0d_unix_us, t0p_unix_us, t1s_unix_us, t1_unix_us, t2, t2c,
                    mat_times,
                );
                Ok(Some(seq))
            }
        }
    }

    /// Push a new `DisplayMeta` to the client (generation bumped by the
    /// caller) and use it for subsequent FrameStamps. Returns false when the
    /// control stream is gone.
    pub fn send_meta(&mut self, meta: &proto::DisplayMeta) -> bool {
        match self.stamp_w.as_mut() {
            Some(w) => {
                if let Err(e) = proto::send_msg(w, &Message::Meta(meta.clone())) {
                    log::debug!("[rdma] meta send failed: {e}");
                    self.stamp_w = None;
                    return false;
                }
                self.meta_gen = meta.gen;
                true
            }
            None => false,
        }
    }

    pub fn frame_bytes(&self) -> usize {
        match &self.tx {
            Tx::Rdma { slots, .. } => slots.first().map(|m| m.len()).unwrap_or(0),
            Tx::Tcp { slot_len, .. } => *slot_len,
        }
    }

    pub fn sequence(&self) -> u64 {
        self.seq
    }
}

// ---------------------------------------------------------------------------
// Client side
// ---------------------------------------------------------------------------

/// Spawn the client-side control-channel reader (server → client direction:
/// per-frame Stamps + clock-sync Pongs). Returns the receiver the main loop
/// drains. Ends silently when the link (and thus the channel) is dropped.
fn spawn_control_reader(stream: TcpStream) -> mpsc::Receiver<ControlEvent> {
    let (tx, rx) = mpsc::channel::<ControlEvent>();
    let _ = std::thread::Builder::new()
        .name("ctrl-reader".into())
        .spawn(move || {
            let mut conn = Conn::new(stream);
            loop {
                match conn.recv() {
                    Ok(proto::Message::Stamp(s)) => {
                        if tx.send(ControlEvent::Stamp(s)).is_err() {
                            break;
                        }
                    }
                    Ok(proto::Message::Pong(c0, s1)) => {
                        let ev = ControlEvent::Pong {
                            c0,
                            s1,
                            c2: proto::unix_now_us(),
                        };
                        if tx.send(ev).is_err() {
                            break;
                        }
                    }
                    Ok(proto::Message::Rumble(r)) => {
                        if tx.send(ControlEvent::Rumble(r)).is_err() {
                            break;
                        }
                    }
                    Ok(proto::Message::Meta(m)) => {
                        if tx.send(ControlEvent::Meta(m)).is_err() {
                            break;
                        }
                    }
                    Ok(proto::Message::ModeReply(r)) => {
                        if tx.send(ControlEvent::ModeReply(r)).is_err() {
                            break;
                        }
                    }
                    Ok(other) => {
                        log::warn!("[rdma] unexpected client control message {other:?}");
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::UnexpectedEof
                            && e.kind() != std::io::ErrorKind::ConnectionReset
                        {
                            log::debug!("[rdma] client control channel error: {e}");
                        }
                        break;
                    }
                }
            }
        });
    rx
}

/// One receive slot: either host memory or an exported GPU buffer.
/// For `Gpu`, field declaration order is load-bearing: `mr` drops (and
/// deregisters) before `gbm` unmaps/destroys the bo.
enum Slot {
    Host(MemoryRegion<Box<[u8]>>),
    Gpu {
        mr: MemoryRegion<()>,
        // declared after mr: deregisters before the buffer is freed
        gpu: Box<dyn SlotBuf>,
    },
}

impl Slot {
    fn desc(&self) -> SlotDesc {
        let r = match self {
            Slot::Host(mr) => mr.remote(),
            Slot::Gpu { mr, .. } => mr.remote(),
        };
        SlotDesc {
            addr: r.addr,
            len: r.len as u64,
            rkey: r.rkey,
        }
    }
}

enum Rx {
    Rdma {
        qp: QueuePair<Rc>,
        recv_cq: CompletionQueue,
        slots: Vec<Slot>,
    },
    Tcp {
        frame_rx: mpsc::Receiver<(u64, Vec<u8>)>,
        recycle_tx: mpsc::Sender<Vec<u8>>,
        slots: Vec<Vec<u8>>,
        slot_free: Vec<bool>,
        pending: Option<(u64, Vec<u8>)>,
        _reader: std::thread::JoinHandle<()>,
    },
}

pub struct ClientLink {
    pub conn: Conn<TcpStream>,
    pub meta: DisplayMeta,
    /// frame codec negotiated for this session (see rdma-display-codec)
    pub codec: rdma_display_codec::Codec,
    /// lossy chroma subsampling: 0 = 4:2:0, 1 = 4:4:4
    pub codec_chroma: u8,
    /// bytes per receive slot: the raw frame, the lossless worst case, or the
    /// lossy downlink budget
    slot_len: usize,
    /// decode state for codec != Raw (kept here so the presenter just calls
    /// [`ClientLink::frame`])
    decoder: Option<rdma_display_codec::Decoder>,
    /// session nonce the server assigned; presented on the input connection
    pub session_nonce: u64,
    /// server's answer to this session's pre-handshake display-mode request
    pub mode_reply: proto::ModeReply,
    rx: Rx,
    /// server→client control events (Stamps, Pongs) from the reader thread
    ctrl_rx: mpsc::Receiver<ControlEvent>,
    transport: Transport,
    /// bytes moved by the most recently returned frame (the RDMA WRITE / TCP
    /// payload length) — the actual wire transfer, for the bandwidth display
    last_rx_bytes: usize,
    /// diagnostics shared with the TCP video-reader thread (arrival cadence,
    /// buffer stalls); stays all-zero for RDMA
    video_reader: std::sync::Arc<VideoReaderStats>,
}

/// Diagnostics shared by the TCP video-reader thread and the presenter: the
/// frame arrival cadence and how often the bounded buffer pool starved.
/// (Temporary instrumentation for the judder investigation.)
#[derive(Default)]
pub struct VideoReaderStats {
    inner: std::sync::Mutex<ReaderInner>,
}

#[derive(Default)]
struct ReaderInner {
    last_arrival_us: u64,
    /// recent inter-arrival gaps (µs), oldest first, bounded
    gaps: VecDeque<u64>,
    frames: u64,
    /// times the reader had no free buffer and had to wait on `recycle_rx`
    stalls: u64,
}

/// A snapshot of [`VideoReaderStats`] for logging.
#[derive(Clone, Copy, Default)]
pub struct VideoReaderSnapshot {
    pub frames: u64,
    pub stalls: u64,
    pub gap_p50_us: u64,
    pub gap_p95_us: u64,
    pub gap_max_us: u64,
}

impl VideoReaderStats {
    /// Record one frame arrival (µs, reader clock) and whether the reader had
    /// to wait for a recycled buffer before it could read it.
    fn observe(&self, now_us: u64, stalled: bool) {
        let mut g = self.inner.lock().unwrap();
        if stalled {
            g.stalls += 1;
        }
        if g.last_arrival_us != 0 && now_us > g.last_arrival_us {
            let gap = now_us - g.last_arrival_us;
            g.gaps.push_back(gap);
            while g.gaps.len() > 256 {
                g.gaps.pop_front();
            }
        }
        g.last_arrival_us = now_us;
        g.frames += 1;
    }

    /// Percentiles over the recent gaps; zero until measured.
    pub fn snapshot(&self) -> VideoReaderSnapshot {
        let g = self.inner.lock().unwrap();
        let mut v: Vec<u64> = g.gaps.iter().copied().collect();
        v.sort_unstable();
        let pick = |p: usize| -> u64 {
            if v.is_empty() {
                0
            } else {
                v[(v.len() * p / 100).min(v.len() - 1)]
            }
        };
        VideoReaderSnapshot {
            frames: g.frames,
            stalls: g.stalls,
            gap_p50_us: pick(50),
            gap_p95_us: pick(95),
            gap_max_us: v.last().copied().unwrap_or(0),
        }
    }
}

/// Read length-prefixed `[u32 len][u64 seq][payload]` video frames off the
/// TAG_VIDEO socket into freshly recycled buffers, handing `(seq, buf)` to the
/// presenter. A bounded pool provides backpressure: with all buffers checked
/// out the reader blocks on `recycle_rx` rather than growing latency.
fn spawn_video_reader(
    mut stream: TcpStream,
    slot_len: usize,
    slots: usize,
    frame_tx: mpsc::Sender<(u64, Vec<u8>)>,
    recycle_rx: mpsc::Receiver<Vec<u8>>,
    stats: std::sync::Arc<VideoReaderStats>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("video-reader".into())
        .spawn(move || {
            let mut free: Vec<Vec<u8>> =
                (0..slots).map(|_| Vec::with_capacity(slot_len)).collect();
            loop {
                let stalled;
                let mut buf = match free.pop() {
                    Some(b) => {
                        stalled = false;
                        b
                    }
                    None => {
                        stalled = true;
                        match recycle_rx.recv() {
                            Ok(b) => b,
                            Err(_) => break, // presenter gone
                        }
                    }
                };
                let mut hdr = [0u8; 12];
                if std::io::Read::read_exact(&mut stream, &mut hdr).is_err() {
                    break;
                }
                let len = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
                let seq = u64::from_le_bytes(hdr[4..12].try_into().unwrap());
                if len > slot_len {
                    log::debug!("[tcp] video frame {len} > slot {slot_len}");
                    break;
                }
                buf.clear();
                buf.resize(len, 0);
                if std::io::Read::read_exact(&mut stream, &mut buf).is_err() {
                    break;
                }
                // arrival stamped after the payload lands in memory
                stats.observe(proto::unix_now_us(), stalled);
                if frame_tx.send((seq, buf)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn video-reader")
}

impl ClientLink {
    /// Client half of a session with host-memory slots.
    pub fn connect(
        conn: Conn<TcpStream>,
        pref: TransportPref,
        rdma: Option<(&ibverbs::Context, &ibverbs::ProtectionDomain)>,
        addr: Option<std::net::SocketAddr>,
        slots: usize,
    ) -> Result<ClientLink> {
        Self::connect_with(
            conn,
            pref,
            rdma,
            addr,
            slots,
            SlotMem::Host,
            proto::ModeRequest::Keep,
            proto::CODEC_ANY,
            0,
        )
    }

    /// Client half of a session; `mem` chooses host vs GPU (GBM) slots (GPU
    /// slots are RDMA-only; TCP always uses host buffers). `addr` is the
    /// server endpoint, used to open the TAG_VIDEO socket when the negotiated
    /// transport is TCP. `mode` is the pre-handshake display-mode request; the
    /// server applies it before `Meta`, so the session is sized for the result.
    /// `codec_pref` is the codec the client would like ([`proto::CODEC_ANY`] =
    /// no preference); the server decides and reports it in the `Codec` message.
    pub fn connect_with(
        conn: Conn<TcpStream>,
        pref: TransportPref,
        rdma: Option<(&ibverbs::Context, &ibverbs::ProtectionDomain)>,
        addr: Option<std::net::SocketAddr>,
        slots: usize,
        mem: SlotMem,
        mode: proto::ModeRequest,
        codec_pref: u8,
        lossy_mbps: u16,
    ) -> Result<ClientLink> {
        // split the stream: the frame/metrics reader gets the receive half as
        // its own clone; the link's conn keeps doing client→server sends
        let stream = conn.into_inner();
        let reader_stream = stream.try_clone().context("clone stream for control reader")?;
        let mut conn = Conn::new(stream);

        // handshake: advertise our RDMA capability, learn the server's choice
        let caps = if pref == TransportPref::Tcp || rdma.is_none() {
            0
        } else {
            proto::CAP_RDMA
        };
        let (session_nonce, server_caps) = hello_client(&mut conn, caps, codec_pref, lossy_mbps)?;
        let transport = if server_caps & proto::CAP_RDMA != 0 {
            Transport::Rdma
        } else {
            Transport::Tcp
        };
        log::info!(
            "[rdma] video transport: {}{}",
            match transport {
                Transport::Rdma => "RDMA",
                Transport::Tcp => "TCP",
            },
            if transport == Transport::Tcp { " (RDMA unavailable)" } else { "" }
        );

        // display-mode request: sent right after Hello and before the server's
        // Meta, so a changed mode is already in effect when slots are sized.
        conn.send(&Message::SetMode(mode))?;
        let mode_reply = match conn.recv()? {
            Message::ModeReply(r) => r,
            other => anyhow::bail!("expected ModeReply, got {other:?}"),
        };
        log::info!(
            "[rdma] display mode: {} ({}{})",
            mode_reply.mode_label(),
            if mode_reply.ok { "ok" } else { "failed" },
            if mode_reply.message.is_empty() {
                String::new()
            } else {
                format!("; {}", mode_reply.message)
            }
        );

        // RDMA endpoint exchange happens before Meta (server sends Meta after)
        let endpoint = if transport == Transport::Rdma {
            let (ctx, pd) = rdma.context("server chose RDMA but no device")?;
            Some(endpoint_exchange(&mut conn, ctx, pd)?)
        } else {
            None
        };
        let pd = match transport {
            Transport::Rdma => Some(
                rdma.context("server chose RDMA but no device")?
                    .1,
            ),
            Transport::Tcp => None,
        };

        let meta = match conn.recv()? {
            Message::Meta(m) => m,
            other => anyhow::bail!("expected Meta, got {other:?}"),
        };
        // codec + slot size + chroma follow Meta, before we size and advertise
        let (codec, slot_len, codec_chroma) = match conn.recv()? {
            Message::Codec(id, slot_len, chroma) => {
                let c = rdma_display_codec::Codec::from_id(id)
                    .ok_or_else(|| anyhow!("server chose unknown codec id {id}"))?;
                (c, slot_len as usize, chroma)
            }
            other => anyhow::bail!("expected Codec, got {other:?}"),
        };
        anyhow::ensure!(
            codec.implemented(),
            "server chose codec {} which this client cannot decode",
            codec.name()
        );
        // raw and lossless advertise a slot sized for the worst case (a full
        // frame), so a slot that cannot hold one is a protocol mismatch. lossy
        // instead advertises its downlink budget, deliberately smaller than the
        // raw frame; only reject a degenerate slot there.
        match codec {
            rdma_display_codec::Codec::Lossy => {
                anyhow::ensure!(slot_len > 0, "nonsensical slot size {slot_len}")
            }
            _ => anyhow::ensure!(
                slot_len >= meta.frame_bytes as usize,
                "nonsensical slot size {slot_len}"
            ),
        }
        let geom = rdma_display_codec::Geometry::new(meta.width, meta.height, meta.stride);
        let decoder = match codec {
            rdma_display_codec::Codec::Lossless => Some(rdma_display_codec::Decoder::new(geom)?),
            _ => None,
        };
        log::info!(
            "[rdma] codec={} slot_len={} (frame_bytes={})",
            codec.name(),
            slot_len,
            meta.frame_bytes
        );

        let video_reader = std::sync::Arc::new(VideoReaderStats::default());
        let rx = match transport {
            Transport::Rdma => {
                let (qp, _send_cq, recv_cq) = endpoint.expect("rdma endpoint");
                let pd = pd.expect("pd");
                let gpu_alloc: Option<Box<dyn SlotAlloc>> = match &mem {
                    SlotMem::Host => None,
                    SlotMem::Cuda { device } => Some(Box::new(cuda::CudaSlots::open(*device)?)),
                };
                let mut slots_v: Vec<Slot> = Vec::with_capacity(slots);
                for i in 0..slots {
                    let slot = match (&gpu_alloc, meta.width, meta.height) {
                        (None, _, _) => {
                            let mr = pd
                                .allocate(slot_len, AccessFlags::PERMISSIVE)
                                .context("allocate recv slot")?;
                            Slot::Host(mr)
                        }
                        (Some(dev), w, h) => {
                            let gpu = dev.alloc_slot(w, h, meta.stride)?;
                            // Deliberately NOT PERMISSIVE: PERMISSIVE includes
                            // RELAXED_ORDERING, which would let the NIC reorder
                            // posted writes across PCIe destinations (CQE vs GPU
                            // payload). Strict ordering is a zero-copy invariant.
                            let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE;
                            let mr = pd
                                .register_dmabuf(gpu.fd(), gpu.offset(), slot_len, gpu.iova(), flags)
                                .with_context(|| {
                                    format!("ibv_reg_dmabuf_mr slot {i} (C0 gate #3)")
                                })?;
                            log::info!(
                                "[slots] gpu slot {i}: iova={:#x} rkey={:#x}",
                                mr.remote().addr,
                                mr.remote().rkey
                            );
                            Slot::Gpu { mr, gpu }
                        }
                    };
                    slots_v.push(slot);
                }
                let descs: Vec<SlotDesc> = slots_v.iter().map(Slot::desc).collect();
                conn.send(&Message::Buffers(descs))?;
                match conn.recv()? {
                    Message::Ready => {}
                    other => anyhow::bail!("expected Ready, got {other:?}"),
                }
                let rx = Rx::Rdma {
                    qp,
                    recv_cq,
                    slots: slots_v,
                };
                let mut link = ClientLink {
                    conn,
                    meta,
                    codec,
                    codec_chroma,
                    slot_len,
                    decoder,
                    session_nonce,
                    mode_reply: mode_reply.clone(),
                    rx,
                    ctrl_rx: spawn_control_reader(reader_stream),
                    transport,
                    last_rx_bytes: 0,
                    video_reader: video_reader.clone(),
                };
                link.rearm_all()?;
                return Ok(link);
            }
            Transport::Tcp => {
                let addr = addr.context("TCP transport but no server address")?;
                let mut s = TcpStream::connect(addr).context("connect TAG_VIDEO socket")?;
                s.set_nodelay(true)?;
                std::io::Write::write_all(&mut s, &[proto::TAG_VIDEO])
                    .context("send TAG_VIDEO")?;
                let (frame_tx, frame_rx) = mpsc::channel();
                let (recycle_tx, recycle_rx) = mpsc::channel();
                let reader = spawn_video_reader(s, slot_len, slots, frame_tx, recycle_rx, video_reader.clone());
                Rx::Tcp {
                    frame_rx,
                    recycle_tx,
                    slots: vec![Vec::new(); slots],
                    slot_free: vec![true; slots],
                    pending: None,
                    _reader: reader,
                }
            }
        };

        Ok(ClientLink {
            conn,
            meta,
            codec,
            codec_chroma,
            slot_len,
            decoder,
            session_nonce,
            mode_reply,
            rx,
            ctrl_rx: spawn_control_reader(reader_stream),
            transport,
            last_rx_bytes: 0,
            video_reader,
        })
    }

    fn rearm_all(&mut self) -> Result<()> {
        let Rx::Rdma { qp, slots, .. } = &mut self.rx else {
            return Ok(()); // TCP: no recv re-arm
        };
        let n = slots.len();
        let reqs: Vec<ibverbs::RecvRequest> = (0..n)
            .map(|slot| ibverbs::RecvRequest::new(slot as u64, &[]))
            .collect();
        unsafe { qp.post_recv(reqs) }.context("post_recv")?;
        Ok(())
    }

    /// Blocking (poll with tiny sleeps): wait for the next frame notification.
    /// Returns `(frame_seq, slot)`. The slot's bytes stay valid until
    /// [`ack_frame`](Self::ack_frame) is called — the server respects credits,
    /// so nothing will overwrite the slot in between.
    pub fn next_frame(&mut self) -> Result<(u64, usize)> {
        loop {
            if let Some(f) = self.try_next_frame()? {
                return Ok(f);
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    /// Non-blocking variant: `Ok(None)` if no completion is ready. The
    /// presenter loop uses this to interleave RDMA with Wayland events.
    pub fn try_next_frame(&mut self) -> Result<Option<(u64, usize)>> {
        let (frame_seq, slot, nbytes) = match &mut self.rx {
            Rx::Rdma { recv_cq, .. } => {
                let mut completions = recv_cq.poll().context("poll recv CQ")?;
                let Some(wc) = completions.next() else {
                    return Ok(None);
                };
                wc.ok().context("recv work request failed")?;
                let imm = wc.imm_data().context("expected WRITE immediate")?;
                let (frame_seq, slot) = imm_decode(imm);
                // the RDMA WRITE length: the actual bytes that crossed the wire
                (frame_seq, slot, wc.len())
            }
            Rx::Tcp {
                frame_rx,
                slots,
                slot_free,
                pending,
                ..
            } => {
                let item = match pending.take() {
                    Some(p) => Some(p),
                    None => frame_rx.try_recv().ok(),
                };
                let Some((seq, buf)) = item else {
                    return Ok(None);
                };
                let nbytes = buf.len();
                if let Some(slot) = slot_free.iter().position(|f| *f) {
                    slot_free[slot] = false;
                    slots[slot] = buf;
                    (seq, slot, nbytes)
                } else {
                    *pending = Some((seq, buf)); // no free slot: retry next call
                    return Ok(None);
                }
            }
        };
        self.last_rx_bytes = nbytes;
        Ok(Some((frame_seq, slot)))
    }

    /// Bytes moved by the frame most recently returned by
    /// [`try_next_frame`](Self::try_next_frame): the RDMA WRITE length (TCP:
    /// payload length). This is the real wire transfer, i.e. the coded frame
    /// size, not the uncompressed frame size — use it for bandwidth.
    pub fn last_rx_bytes(&self) -> usize {
        self.last_rx_bytes
    }

    /// Snapshot of the TCP video-reader diagnostics (frame arrival cadence and
    /// buffer stalls); all-zero for RDMA. Temporary judder instrumentation.
    pub fn video_reader_stats(&self) -> VideoReaderSnapshot {
        self.video_reader.snapshot()
    }

    /// The negotiated video transport.
    pub fn transport(&self) -> Transport {
        self.transport
    }

    /// Re-arm the slot's notification receipt. Must happen right after
    /// [`try_next_frame`](Self::try_next_frame) took the completion; the
    /// *credit* (server permission to overwrite again) is separate —
    /// presenters hold it until the compositor releases the buffer. No-op for
    /// TCP (the reader thread manages its own pool).
    pub fn rearm_slot(&mut self, slot: usize) -> Result<()> {
        if let Rx::Rdma { qp, .. } = &mut self.rx {
            unsafe { qp.post_recv([ibverbs::RecvRequest::new(slot as u64, &[])]) }
                .context("re-arm recv")?;
        }
        Ok(())
    }

    /// Return a slot to the pool and tell the server the frame is consumed.
    ///
    /// RDMA: credits the staging slot for reuse. TCP: recycles the buffer to
    /// the video reader. Either way a `Credit` goes to the server, which is
    /// what paces its sends to our present rate (the flow-control window — see
    /// `ServerLink::set_window`); on TCP there are no staging slots, so this
    /// message is the only backpressure the server gets.
    pub fn credit_slot(&mut self, slot: usize) -> Result<()> {
        match &mut self.rx {
            Rx::Rdma { .. } => {
                self.conn.send(&Message::Credit(slot as u32))?;
            }
            Rx::Tcp {
                recycle_tx,
                slots,
                slot_free,
                ..
            } => {
                if slot < slots.len() {
                    slot_free[slot] = true;
                    let v = std::mem::take(&mut slots[slot]);
                    let _ = recycle_tx.send(v);
                }
                self.conn.send(&Message::Credit(slot as u32))?;
            }
        }
        Ok(())
    }

    /// Forward one user-input event to the server (proto v2).
    pub fn send_input(&mut self, ev: proto::InputEvent) -> Result<()> {
        self.conn.send(&proto::Message::Input(ev))?;
        Ok(())
    }

    /// Clock-sync probe: server replies with its time; the collector pairs
    /// the pong with our send/recv times to estimate the clock offset.
    pub fn send_ping(&mut self) -> Result<()> {
        self.conn.send(&proto::Message::Ping(proto::unix_now_us()))?;
        Ok(())
    }

    /// Drain one control event from the reader thread, if any.
    pub fn try_control(&mut self) -> Option<ControlEvent> {
        self.ctrl_rx.try_recv().ok()
    }

    /// Tell the server the slot may be reused, and re-arm its notification.
    /// (Headless/stats mode: consumption == "bytes copied out".)
    pub fn ack_frame(&mut self, slot: usize) -> Result<()> {
        self.rearm_slot(slot)?;
        self.credit_slot(slot)
    }


    /// Read the bytes of slot `slot`. For GPU slots this is the best-effort
    /// CPU mapping of the dma-buf; an empty slice means "no CPU view" (the
    /// data is still valid in GPU memory — verify GPU-side). This is the raw
    /// slot contents (the coded frame when a codec is active); use
    /// [`frame`](Self::frame) for decoded pixels.
    pub fn slot_bytes(&self, slot: usize) -> &[u8] {
        match &self.rx {
            Rx::Rdma { slots, .. } => {
                let n = self.slot_len;
                match &slots[slot] {
                    Slot::Host(mr) => &mr.bytes()[..n],
                    Slot::Gpu { gpu, .. } => gpu.cpu_bytes(n).unwrap_or(&[]),
                }
            }
            Rx::Tcp { slots, .. } => &slots[slot][..],
        }
    }

    /// The full decoded frame for `slot` (stride × height pixels). `Raw` is a
    /// borrow of the slot; a coded frame is decoded into the link's internal
    /// buffer. Valid until the next `frame`/`decode` call (and, for `Raw`,
    /// until the slot is credited).
    pub fn frame(&mut self, slot: usize) -> Result<&[u8]> {
        if self.decoder.is_none() {
            return Ok(self.slot_bytes(slot));
        }
        // Inline the slot access so the borrow checker sees the slot storage
        // and `decoder` as disjoint fields.
        match &self.rx {
            Rx::Rdma { slots, .. } => {
                let n = self.slot_len;
                let coded: &[u8] = match &slots[slot] {
                    Slot::Host(mr) => &mr.bytes()[..n],
                    Slot::Gpu { gpu, .. } => gpu.cpu_bytes(n).unwrap_or(&[]),
                };
                let dec = self.decoder.as_mut().expect("decoder present");
                dec.decode(coded)
            }
            Rx::Tcp { slots, .. } => {
                let coded: &[u8] = &slots[slot][..];
                let dec = self.decoder.as_mut().expect("decoder present");
                dec.decode(coded)
            }
        }
    }

    /// The frame produced by the most recent [`frame`](Self::frame) call,
    /// without decoding again. `None` when the link has no CPU decoder (raw).
    /// The presenter uses this on the CPU lossless path, where the receive loop
    /// has already decoded the newest frame in order.
    pub fn decoded(&self) -> Option<&[u8]> {
        self.decoder.as_ref().map(|d| d.frame())
    }

    /// True when frames land in GPU memory with no CPU view.
    pub fn slots_gpu_only(&self) -> bool {
        match &self.rx {
            Rx::Rdma { slots, .. } => {
                matches!(slots.first(), Some(Slot::Gpu { gpu, .. }) if gpu.cpu_bytes(1).is_none())
            }
            Rx::Tcp { .. } => false,
        }
    }

    /// Whole-slot read for content verification: CPU map when present, else
    /// the (verification-only) device->host copy. Errors when neither exists.
    pub fn slot_read(&self, slot: usize) -> Result<Vec<u8>> {
        match &self.rx {
            Rx::Rdma { slots, .. } => {
                let n = self.slot_len;
                match &slots[slot] {
                    Slot::Host(mr) => Ok(mr.bytes()[..n].to_vec()),
                    Slot::Gpu { gpu, .. } => match gpu.cpu_bytes(n) {
                        Some(b) => Ok(b.to_vec()),
                        None => gpu.dto_h_vec(n),
                    },
                }
            }
            Rx::Tcp { slots, .. } => Ok(slots[slot].clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ema;

    #[test]
    fn ema_seeds_then_smooths() {
        // the first sample seeds the average
        assert_eq!(ema(0, 1000), 1000);
        // later samples move it by 1/8, so one outlier can't swing the window
        assert_eq!(ema(1000, 2000), (1000 * 7 + 2000) / 8);
        assert_eq!(ema(1000, 9000), (1000 * 7 + 9000) / 8);
    }
}
