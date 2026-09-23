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

use std::net::TcpStream;
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
        eprintln!(
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

fn hello_exchange<T: std::io::Read + std::io::Write>(conn: &mut Conn<T>, our_nonce: u64) -> Result<u64> {
    conn.send(&Message::Hello(proto::Hello {
        magic: proto::MAGIC,
        version: proto::VERSION,
        nonce: our_nonce,
    }))?;
    match conn.recv()? {
        Message::Hello(h) if h.magic == proto::MAGIC => {
            if h.version != proto::VERSION {
                anyhow::bail!(
                    "peer protocol version {} != our {}",
                    h.version,
                    proto::VERSION
                );
            }
            Ok(h.nonce)
        }
        other => Err(anyhow!("expected Hello, got {other:?}")),
    }
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
            eprintln!(
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

pub struct ServerLink {
    /// credits arrive from the control-reader thread; None = client gone
    credit_rx: mpsc::Receiver<Option<u32>>,
    qp: QueuePair<Rc>,
    send_cq: CompletionQueue,
    /// local staging slots (copy path / testcard writes here before the WRITE)
    slots: Vec<StagingSlot>,
    remote: Vec<RemoteMemorySlice>,
    slot_free: Vec<bool>,
    seq: u64,
    /// current `DisplayMeta` generation, stamped into every FrameStamp;
    /// `send_meta` bumps it and pushes the new meta to the control reader
    meta_gen: u32,
    /// cloned control stream: the send path writes per-frame `Stamp`s here
    /// while the control-reader thread owns the receive side
    stamp_w: Option<TcpStream>,
}

impl ServerLink {
    /// Server half of a session on an accepted TCP stream. `meta` describes
    /// the frames we will send (testcard or captured display format).
    #[allow(clippy::too_many_arguments)]
    pub fn accept(
        conn: Conn<TcpStream>,
        ctx: &ibverbs::Context,
        pd: &ibverbs::ProtectionDomain,
        meta: &DisplayMeta,
        slots: usize,
        codec: rdma_display_codec::Codec,
        slot_len: usize,
        chroma: u8,
        input_tx: mpsc::Sender<proto::InputEvent>,
        external: Option<&[(*mut u8, usize)]>,
        rumble: &proto::gamepad::RumbleHub,
        session_nonce: u64,
    ) -> Result<ServerLink> {
        // split the stream: reader thread gets the receive half, the frame
        // send path keeps a write-only clone for per-frame Stamps
        let stream = conn.into_inner();
        let stamp_w = stream.try_clone().context("clone stream for stamps")?;
        let rumble_w = stream.try_clone().context("clone stream for rumble")?;
        let mut conn = Conn::new(stream);
        let _ = hello_exchange(&mut conn, session_nonce)?;
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
                eprintln!(
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
        // tell the client which codec the frames will use before it allocates
        // its receive slots (they must hold the codec's worst case)
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
        let (credit_tx, credit_rx) = mpsc::channel::<Option<u32>>();
        let reader_rumble = rumble.clone();
        let reader = std::thread::Builder::new()
            .name("control-reader".into())
            .spawn(move || {
                loop {
                    match conn.recv() {
                        Ok(proto::Message::Credit(slot)) => {
                            if credit_tx.send(Some(slot)).is_err() {
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
                                eprintln!("[rdma] pong failed: {e}");
                            }
                        }
                        Ok(proto::Message::Bye(_)) => {
                            let _ = credit_tx.send(None);
                            break;
                        }
                        Ok(other) => {
                            eprintln!("[rdma] unexpected control message {other:?}");
                        }
                        Err(e) => {
                            if e.kind() != std::io::ErrorKind::UnexpectedEof
                                && e.kind() != std::io::ErrorKind::ConnectionReset
                            {
                                eprintln!("[rdma] control channel error: {e}");
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

        let n = remote.len();
        Ok(ServerLink {
            credit_rx,
            qp,
            send_cq,
            slots: stage,
            remote,
            slot_free: vec![true; n],
            seq: 0,
            meta_gen: meta.gen,
            stamp_w: Some(stamp_w),
        })
    }

    /// Non-blocking hangup/credit check: drains the credit channel (applying
    /// credits) and returns true when the control-reader thread signaled the
    /// client hung up (EOF/Bye). Call each loop iteration — an idle capture
    /// loop may send nothing for a long time and otherwise never notices.
    pub fn client_gone(&mut self) -> bool {
        loop {
            match self.credit_rx.try_recv() {
                Ok(Some(slot)) => {
                    if let Some(f) = self.slot_free.get_mut(slot as usize) {
                        *f = true;
                    }
                }
                Ok(None) => return true, // reader thread saw EOF
                Err(mpsc::TryRecvError::Empty) => return false,
                Err(mpsc::TryRecvError::Disconnected) => return true,
            }
        }
    }

    /// Whether a staging slot is currently free (client-credited).
    pub fn has_free_slot(&self) -> bool {
        self.slot_free.iter().any(|f| *f)
    }

    /// Non-blocking claim of the first free slot (marks it in use).
    pub fn try_claim_slot(&mut self) -> Option<usize> {
        let slot = self.slot_free.iter().position(|f| *f)?;
        self.slot_free[slot] = false;
        Some(slot)
    }

    /// Return a claimed slot (used when a frame turns out unchanged).
    pub fn release_slot(&mut self, slot: usize) {
        self.slot_free[slot] = true;
    }

    /// The claimed slot's registered memory, for `issue_frame`.
    pub fn slot_bytes_mut(&mut self, slot: usize) -> &mut [u8] {
        let len = self.slots[slot].len();
        self.slots[slot].as_mut_bytes(len)
    }

    /// Blocks until a slot is free, absorbing credit messages meanwhile.
    /// (Credits now arrive via the control-reader thread's channel.)
    fn wait_free_slot(&mut self) -> Result<usize> {
        loop {
            if let Some(slot) = self.slot_free.iter().position(|f| *f) {
                return Ok(slot);
            }
            match self.credit_rx.recv() {
                Ok(Some(slot)) => {
                    if let Some(f) = self.slot_free.get_mut(slot as usize) {
                        *f = true;
                    }
                }
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
        )?
        .ok_or_else(|| anyhow!("frame unexpectedly unchanged"))
    }

    /// Like [`send_frame`](Self::send_frame), but `fill` decides whether the
    /// frame content changed and should be sent (`Ok(false)` = skip), returns
    /// the number of bytes to post (`post_len`, e.g. the codec's coded length),
    /// and returns the frame's GPU stage times for the Stamp.
    pub fn send_frame_checked(
        &mut self,
        fill: impl FnOnce(&mut [u8]) -> anyhow::Result<(bool, usize, proto::MatTimes)>,
        t0_unix_us: u64,
    ) -> Result<Option<u64>> {
        let t0d = proto::unix_now_us(); // frame picked up for sending
        let slot = self.wait_free_slot()?;
        let t1s = proto::unix_now_us(); // staging slot available
        let slot_wait_us = t1s.saturating_sub(t0d);
        if slot_wait_us > 5_000 {
            eprintln!(
                "[rdma] slot wait {} ms (client credits lagging)",
                slot_wait_us / 1000
            );
        }
        self.slot_free[slot] = false;
        let frame_bytes = self.slots[slot].len();
        let (changed, post_len, mat_times) = fill(self.slots[slot].as_mut_bytes(frame_bytes))?;
        if !changed {
            // unchanged frame: return the slot, don't send
            self.slot_free[slot] = true;
            return Ok(None);
        }
        anyhow::ensure!(post_len <= frame_bytes, "post_len {post_len} > slot {frame_bytes}");
        let t1_unix_us = proto::unix_now_us(); // readback/materialize complete
        #[cfg(feature = "frame-checksums")]
        {
            let out = self.slots[slot].as_mut_bytes(frame_bytes);
            let sum: u32 = out.iter().map(|b| *b as u32).sum();
            let nonzero = out.iter().filter(|b| **b != 0).count();
            eprintln!(
                "[rdma] staging slot {} checksum sum={} nonzero={}/{}",
                slot, sum, nonzero, frame_bytes
            );
        }

        let remote = self.remote[slot];
        let seq = self.seq;
        let imm = imm_encode(seq, slot);
        let mut batch = self.qp.start_send();
        batch.op().signaled().imm(imm).write(
            seq,
            &[self.slots[slot].post_slice(post_len)],
            remote,
        );
        unsafe { batch.submit() }.context("post RDMA WRITE")?;
        let t2_unix_us = proto::unix_now_us(); // WRITE posted to the NIC

        // drain completions so the signaled WRITE is confirmed before we hand
        // the staging slot back to the caller
        loop {
            let mut completions = self.send_cq.poll().context("poll send CQ")?;
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
        let t2c_unix_us = proto::unix_now_us();
        self.seq += 1;
        // pipeline stamp for the client's metrics overlay; the TCP trip is
        // intentionally after the WRITE completes, so the client sees the
        // stamp ~immediately after the frame itself
        if let Some(w) = self.stamp_w.as_mut() {
            // Clamp: a frame that sat dirty >1 s (e.g. arrived before any
            // client connected, or a zero-initialized t0) would poison the
            // latency stats with a meaningless queue wait — collapse it to 0.
            let t0 = if t0_unix_us == 0 || t0d.saturating_sub(t0_unix_us) > 1_000_000 {
                t0d
            } else {
                t0_unix_us
            };
            let s = proto::FrameStamp {
                seq,
                t0,
                t0d,
                t1s,
                t1: t1_unix_us,
                t2: t2_unix_us,
                t2c: t2c_unix_us,
                blit_us: mat_times.blit_us,
                read_us: mat_times.read_us,
                fin_us: mat_times.fin_us,
                meta_gen: self.meta_gen,
            };
            if let Err(e) = proto::send_msg(w, &Message::Stamp(s)) {
                eprintln!("[rdma] stamp send failed: {e}");
                self.stamp_w = None;
            }
        }
        Ok(Some(seq))
    }

    /// Pipelined post: WRITE a slot that `issue_frame` already filled (the
    /// materializer completed it via `finish_frame`). `t0d` is the pickup
    /// time (pacing/queue wait done) and `t1s` the slot-claim time, so the
    /// stamp keeps the same `queue`/`slot`/`material` split as the serial
    /// path; `t1` is the fence-completion time. `post_len` is the number of
    /// bytes to send (the codec's coded length, or the raw frame size).
    #[allow(clippy::too_many_arguments)]
    pub fn post_filled(
        &mut self,
        slot: usize,
        t0_unix_us: u64,
        t0d_unix_us: u64,
        t1s_unix_us: u64,
        t1_unix_us: u64,
        post_len: usize,
        mat_times: proto::MatTimes,
    ) -> Result<Option<u64>> {
        let frame_bytes = self.slots[slot].len();
        anyhow::ensure!(post_len <= frame_bytes, "post_len {post_len} > slot {frame_bytes}");
        let remote = self.remote[slot];
        let seq = self.seq;
        let imm = imm_encode(seq, slot);
        let mut batch = self.qp.start_send();
        batch.op().signaled().imm(imm).write(
            seq,
            &[self.slots[slot].post_slice(post_len)],
            remote,
        );
        unsafe { batch.submit() }.context("post RDMA WRITE")?;
        let t2_unix_us = proto::unix_now_us(); // WRITE posted to the NIC

        // drain completions so the signaled WRITE is confirmed (t2c stamp);
        // the staging slot stays busy until the client's credit arrives
        loop {
            let mut completions = self.send_cq.poll().context("poll send CQ")?;
            match completions.next() {
                Some(wc) => {
                    wc.ok().context("RDMA WRITE failed")?;
                    break;
                }
                None => std::thread::sleep(Duration::from_micros(20)),
            }
        }
        let t2c_unix_us = proto::unix_now_us();
        self.seq += 1;
        if let Some(w) = self.stamp_w.as_mut() {
            // same >1 s stale-t0 clamp as the serial path
            let t0 = if t0_unix_us == 0 || t0d_unix_us.saturating_sub(t0_unix_us) > 1_000_000 {
                t0d_unix_us
            } else {
                t0_unix_us
            };
            let s = proto::FrameStamp {
                seq,
                t0,
                t0d: t0d_unix_us,
                t1s: t1s_unix_us,
                t1: t1_unix_us,
                t2: t2_unix_us,
                t2c: t2c_unix_us,
                blit_us: mat_times.blit_us,
                read_us: mat_times.read_us,
                fin_us: mat_times.fin_us,
                meta_gen: self.meta_gen,
            };
            if let Err(e) = proto::send_msg(w, &Message::Stamp(s)) {
                eprintln!("[rdma] stamp send failed: {e}");
                self.stamp_w = None;
            }
        }
        Ok(Some(seq))
    }

    /// Push a new `DisplayMeta` to the client (generation bumped by the
    /// caller) and use it for subsequent FrameStamps. Returns false when the
    /// control stream is gone.
    pub fn send_meta(&mut self, meta: &proto::DisplayMeta) -> bool {
        match self.stamp_w.as_mut() {
            Some(w) => {
                if let Err(e) = proto::send_msg(w, &Message::Meta(meta.clone())) {
                    eprintln!("[rdma] meta send failed: {e}");
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
        self.slots.first().map(|m| m.len()).unwrap_or(0)
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
                    Ok(other) => {
                        eprintln!("[rdma] unexpected client control message {other:?}");
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::UnexpectedEof
                            && e.kind() != std::io::ErrorKind::ConnectionReset
                        {
                            eprintln!("[rdma] client control channel error: {e}");
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

pub struct ClientLink {
    pub conn: Conn<TcpStream>,
    pub meta: DisplayMeta,
    /// frame codec negotiated for this session (see rdma-display-codec)
    pub codec: rdma_display_codec::Codec,
    /// lossy chroma subsampling: 0 = 4:2:0, 1 = 4:4:4
    pub codec_chroma: u8,
    /// bytes per receive slot (raw frame, or the codec's worst case)
    slot_len: usize,
    /// decode state for codec != Raw (kept here so the presenter just calls
    /// [`ClientLink::frame`])
    decoder: Option<rdma_display_codec::Decoder>,
    /// session nonce the server assigned; presented on the input connection
    pub session_nonce: u64,
    qp: QueuePair<Rc>,
    recv_cq: CompletionQueue,
    slots: Vec<Slot>,
    /// server→client control events (Stamps, Pongs) from the reader thread
    ctrl_rx: mpsc::Receiver<ControlEvent>,
}

impl ClientLink {
    /// Client half of a session with host-memory slots.
    pub fn connect(
        conn: Conn<TcpStream>,
        ctx: &ibverbs::Context,
        pd: &ibverbs::ProtectionDomain,
        slots: usize,
    ) -> Result<ClientLink> {
        Self::connect_with(conn, ctx, pd, slots, SlotMem::Host)
    }

    /// Client half of a session; `mem` chooses host vs GPU (GBM) slots.
    pub fn connect_with(
        conn: Conn<TcpStream>,
        ctx: &ibverbs::Context,
        pd: &ibverbs::ProtectionDomain,
        slots: usize,
        mem: SlotMem,
    ) -> Result<ClientLink> {
        // split the stream: the frame/metrics reader gets the receive half as
        // its own clone; the link's conn keeps doing client→server sends
        let stream = conn.into_inner();
        let reader_stream = stream.try_clone().context("clone stream for control reader")?;
        let mut conn = Conn::new(stream);
        let session_nonce = hello_exchange(&mut conn, 0)?;
        let (qp, _send_cq, recv_cq) = endpoint_exchange(&mut conn, ctx, pd)?;

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
        anyhow::ensure!(slot_len >= meta.frame_bytes as usize, "nonsensical slot size {slot_len}");
        let geom = rdma_display_codec::Geometry::new(meta.width, meta.height, meta.stride);
        let decoder = match codec {
            rdma_display_codec::Codec::Lossless => Some(rdma_display_codec::Decoder::new(geom)?),
            _ => None,
        };
        eprintln!(
            "[rdma] codec={} slot_len={} (frame_bytes={})",
            codec.name(),
            slot_len,
            meta.frame_bytes
        );

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
                        .with_context(|| format!("ibv_reg_dmabuf_mr slot {i} (C0 gate #3)"))?;
                    eprintln!(
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

        let mut link = ClientLink {
            conn,
            meta,
            codec,
            codec_chroma,
            slot_len,
            decoder,
            session_nonce,
            qp,
            recv_cq,
            slots: slots_v,
            ctrl_rx: spawn_control_reader(reader_stream),
        };
        link.rearm_all()?;
        Ok(link)
    }

    fn rearm_all(&mut self) -> Result<()> {
        let n = self.slots.len();
        let reqs: Vec<ibverbs::RecvRequest> = (0..n)
            .map(|slot| ibverbs::RecvRequest::new(slot as u64, &[]))
            .collect();
        unsafe { self.qp.post_recv(reqs) }.context("post_recv")?;
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
        let mut completions = self.recv_cq.poll().context("poll recv CQ")?;
        if let Some(wc) = completions.next() {
            wc.ok().context("recv work request failed")?;
            let imm = wc.imm_data().context("expected WRITE immediate")?;
            let (frame_seq, slot) = imm_decode(imm);
            return Ok(Some((frame_seq, slot)));
        }
        Ok(None)
    }

    /// Re-arm the slot's notification receipt. Must happen right after
    /// [`try_next_frame`](Self::try_next_frame) took the completion; the
    /// *credit* (server permission to overwrite again) is separate —
    /// presenters hold it until the compositor releases the buffer.
    pub fn rearm_slot(&mut self, slot: usize) -> Result<()> {
        unsafe {
            self.qp
                .post_recv([ibverbs::RecvRequest::new(slot as u64, &[])])
        }
        .context("re-arm recv")?;
        Ok(())
    }

    /// Return a slot to the server pool (credits it for reuse).
    pub fn credit_slot(&mut self, slot: usize) -> Result<()> {
        self.conn.send(&Message::Credit(slot as u32))?;
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
        let n = self.slot_len;
        match &self.slots[slot] {
            Slot::Host(mr) => &mr.bytes()[..n],
            Slot::Gpu { gpu, .. } => gpu.cpu_bytes(n).unwrap_or(&[]),
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
        // Inline the slot access so the borrow checker sees `slots` and
        // `decoder` as disjoint fields.
        let n = self.slot_len;
        let coded: &[u8] = match &self.slots[slot] {
            Slot::Host(mr) => &mr.bytes()[..n],
            Slot::Gpu { gpu, .. } => gpu.cpu_bytes(n).unwrap_or(&[]),
        };
        let dec = self.decoder.as_mut().expect("decoder present");
        dec.decode(coded)
    }

    /// True when frames land in GPU memory with no CPU view.
    pub fn slots_gpu_only(&self) -> bool {
        matches!(&self.slots.first(), Some(Slot::Gpu { gpu, .. }) if gpu.cpu_bytes(1).is_none())
    }

    /// Whole-slot read for content verification: CPU map when present, else
    /// the (verification-only) device->host copy. Errors when neither exists.
    pub fn slot_read(&self, slot: usize) -> Result<Vec<u8>> {
        let n = self.slot_len;
        match &self.slots[slot] {
            Slot::Host(mr) => Ok(mr.bytes()[..n].to_vec()),
            Slot::Gpu { gpu, .. } => match gpu.cpu_bytes(n) {
                Some(b) => Ok(b.to_vec()),
                None => gpu.dto_h_vec(n),
            },
        }
    }
}
