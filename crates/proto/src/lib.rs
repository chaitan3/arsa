//! Control-channel protocol for rdma-display: length-prefixed little-endian
//! messages over TCP (or Unix sockets for local testing).
//!
//! Framing: every message is `u32 payload_len | u32 msg_type | payload`.
//! RDMA-specific data (QPN/PSN/GID, rkeys) uses the `ibverbs` crate's stable
//! wire encodings embedded in payloads.

use std::io::{Read, Write};

pub mod gamepad;
pub mod keymap;

pub const MAGIC: u32 = 0x5244_4453; // "RDDS"
pub const VERSION: u32 = 11; // v11: per-codec MatTimes stage breakdown in the frame stamp

/// First byte of every connection, so the server routes by an explicit tag
/// rather than by guessing protocol bytes. Audio uses
/// `rdma_display_audio::wire::TAG_AUDIO` (0x02); control and input are here.
pub const TAG_CONTROL: u8 = 0x01;
pub const TAG_INPUT: u8 = 0x03;
/// Video over TCP (fallback when RDMA/RoCE is unavailable): a length-prefixed
/// frame stream, separate from control so a big frame can't block input/stamps.
pub const TAG_VIDEO: u8 = 0x04;

/// `Hello::caps` bit: this side can use an RDMA (ibverbs/RoCE) video transport.
pub const CAP_RDMA: u8 = 0x01;

/// Wall-clock microseconds since the unix epoch. Cross-host comparable
/// (that is the point): per-frame server stamps and ping-pong clock-sync
/// samples both use this clock. `Instant` stays per-host by design.
pub fn unix_now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

const MAX_PAYLOAD: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MsgType {
    Hello = 1,
    Meta = 2,
    QpEndpoint = 3,
    Buffers = 4,
    Ready = 5,
    Credit = 6,
    Bye = 7,
    Input = 8,
    /// server → client, one per frame: pipeline timestamps (unix us).
    Stamp = 9,
    /// clock sync: payload = sender's clock at send (unix us)
    Ping = 10,
    /// clock sync reply: payload = the ping's stamp + server's time at reply
    Pong = 11,
    /// server → client: force-feedback event for the client's physical pad
    Rumble = 12,
    /// server → client: frame codec for this session (see rdma-display-codec).
    /// Sent right after `Meta`, before the client allocates its slots, so the
    /// client can size them for the codec's worst case.
    Codec = 13,
}

/// Server-side pipeline timestamps for one frame (all `unix_now_us`).
///   t0  = capture event latched (QEMU D-Bus arrival / DRM scanout poll)
///   t0d = the sender picked the frame up (pacing + event-queue wait done)
///   t1s = a staging slot was available (after any credit wait)
///   t1  = the capture → staging-slot work finished (slot holds payload bytes)
///   t2  = RDMA WRITE posted
///   t2c = send CQ reported the WRITE fully written on the client —
///         the true transfer time is `t2c - t2`, same-host (no clock skew)
///
/// Every stage the client reports is a difference of stamps taken on ONE
/// host (server: t0..t2c, client: its own t3/t4/t5 + presentation time), so
/// no latency number depends on the two clocks agreeing. The absolute unix
/// stamps travel on the wire only so the client can choose which pairs to
/// subtract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameStamp {
    pub seq: u64,
    pub t0: u64,
    pub t0d: u64,
    pub t1s: u64,
    pub t1: u64,
    pub t2: u64,
    pub t2c: u64,
    /// breakdown of the `t1 - t1s` span (the capture → staging-slot work),
    /// tagged with the codec/materializer that produced it: see [`MatTimes`].
    pub mat: MatTimes,
    /// which `DisplayMeta` generation this frame was produced with (see
    /// `Message::Meta`). Mid-session meta changes are not yet negotiated, so
    /// this is 0 today; it makes the frame↔meta association explicit.
    pub meta_gen: u32,
}

/// Stages in a [`MatTimes`] breakdown. Six slots; the meaning of `us[i]` is
/// given by [`MatKind::labels`] (unused slots are zero).
pub const MAT_STAGES: usize = 6;

/// Which capture path produced a frame, so its stage labels are unambiguous.
/// Every codec has a *different* set of stages inside the `t1 - t1s` span;
/// reporting them under one fixed label was actively misleading — lossless's
/// whole GPU encode used to surface as "blit" with "readback 0.00".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MatKind {
    /// No breakdown (plain host-memory / QEMU materializer).
    None = 0,
    /// EGL dmabuf materializer (`raw`): shader blit → glReadPixels → sync.
    Raw = 1,
    /// GPU lossless (codec-gpu): texture-direct, or readback + encode.
    Lossless = 2,
    /// PyroWave lossy materializer.
    Lossy = 3,
}

impl MatKind {
    pub fn from_u8(v: u8) -> MatKind {
        match v {
            1 => MatKind::Raw,
            2 => MatKind::Lossless,
            3 => MatKind::Lossy,
            _ => MatKind::None,
        }
    }

    /// Short name for logs and the metrics overlay.
    pub fn name(self) -> &'static str {
        match self {
            MatKind::None => "plain",
            MatKind::Raw => "raw",
            MatKind::Lossless => "lossless",
            MatKind::Lossy => "lossy",
        }
    }

    /// Short human labels for `MatTimes::us`, in order; `""` = unused slot.
    pub fn labels(self) -> [&'static str; MAT_STAGES] {
        match self {
            MatKind::None => ["", "", "", "", "", ""],
            // blit = shader blit (detile + channel normalize); readback =
            // glReadPixels into the staging PBO; sync = EGL fence / glFinish.
            // NB with a single reused FBO the draw can block on the previous
            // frame's readback, in which case that backpressure lands here.
            MatKind::Raw => ["blit", "readback", "sync", "", "", ""],
            // texture-direct GPU lossless: activate = make the codec's EGL
            // context current; blit = optional source blit (0 on the
            // external-sampler path); encode = dispatch + glFinish (reads the
            // scanout, XORs tiles, codes changes, and advances the reference
            // in-pass); ref = legacy separate reference pass (now ~0); copy =
            // coded bytes into the slot; restore = make the materializer's
            // context current again.
            MatKind::Lossless => ["activate", "blit", "encode", "ref", "copy", "restore"],
            // encode = PyroWave GPU encode; packetize = packet framing + copy
            // into the slot. (The Vulkan dmabuf import happens in prepare,
            // before the t1s stamp, so it is not part of this span.)
            MatKind::Lossy => ["encode", "packetize", "", "", "", ""],
        }
    }
}

/// Per-codec stage breakdown of the `t1 - t1s` span (µs, all server-local).
/// `kind` selects the meaning of `us`; see [`MatKind::labels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatTimes {
    pub kind: MatKind,
    pub us: [u64; MAT_STAGES],
}

impl MatTimes {
    pub const ZERO: MatTimes = MatTimes {
        kind: MatKind::None,
        us: [0; MAT_STAGES],
    };

    pub fn new(kind: MatKind, us: [u64; MAT_STAGES]) -> Self {
        MatTimes { kind, us }
    }

    /// Sum of the populated stages (the whole `t1 - t1s` span).
    pub fn total(&self) -> u64 {
        self.us.iter().sum()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Hello {
    pub magic: u32,
    pub version: u32,
    /// Session nonce. The client sends 0; the server replies with a random
    /// value that the client must present on its input connection so a stale
    /// input socket can't feed a later session.
    pub nonce: u64,
    /// Transport capabilities / choice (see `CAP_RDMA`): the client sets
    /// `CAP_RDMA` if it has an RDMA device; the server's reply sets it if the
    /// chosen video transport is RDMA (clear = TCP).
    pub caps: u8,
}

/// Display format metadata, sent by the server for the current scanout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayMeta {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub y0_top: bool,
    /// total bytes of one full frame (stride * height for single-plane formats)
    pub frame_bytes: u32,
    /// generation: bumped whenever the format changes, so a frame's
    /// [`FrameStamp::meta_gen`] identifies which meta produced it.
    pub gen: u32,
}

/// One client receive slot the server may RDMA-WRITE into.
#[derive(Debug, Clone, Copy)]
pub struct SlotDesc {
    pub addr: u64,
    pub len: u64,
    pub rkey: u32,
}

/// One user-input event, client → server → guest.
/// `code` semantics (per org.qemu.Display1.* on the host):
/// - Key: QEMU "key number" (qnum) — see client/src/keymap.rs
/// - Btn: 0=Left, 1=Middle, 2=Right, 3=WheelUp, 4=WheelDown, 5=Side, 6=Extra
/// - GamepadBtn/GamepadAxis (proto v7): kernel-standard gamepad codes —
///   `code` is a BTN_* (see proto::gamepad) / ABS_* axis, value already
///   rescaled into the virtual pad's ranges by the client. `slot` is the
///   controller index (v7 wires slot 0 only; the field is here from day one).
#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Key { code: u32, down: bool },
    Btn { button: u32, down: bool },
    Abs { x: u32, y: u32 },
    Rel { dx: i32, dy: i32 },
    GamepadBtn { slot: u32, code: u32, down: bool },
    GamepadAxis { slot: u32, code: u32, value: i32 },
}

impl InputEvent {
    const KIND_KEY: u32 = 0;
    const KIND_BTN: u32 = 1;
    const KIND_ABS: u32 = 2;
    const KIND_REL: u32 = 3;
    const KIND_GAMEPAD_BTN: u32 = 4;
    const KIND_GAMEPAD_AXIS: u32 = 5;

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            InputEvent::Key { code, down } => {
                out.extend_from_slice(&Self::KIND_KEY.to_le_bytes());
                out.extend_from_slice(&code.to_le_bytes());
                out.push(*down as u8);
            }
            InputEvent::Btn { button, down } => {
                out.extend_from_slice(&Self::KIND_BTN.to_le_bytes());
                out.extend_from_slice(&button.to_le_bytes());
                out.push(*down as u8);
            }
            InputEvent::Abs { x, y } => {
                out.extend_from_slice(&Self::KIND_ABS.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
                out.extend_from_slice(&y.to_le_bytes());
            }
            InputEvent::Rel { dx, dy } => {
                out.extend_from_slice(&Self::KIND_REL.to_le_bytes());
                out.extend_from_slice(&dx.to_le_bytes());
                out.extend_from_slice(&dy.to_le_bytes());
            }
            InputEvent::GamepadBtn { slot, code, down } => {
                out.extend_from_slice(&Self::KIND_GAMEPAD_BTN.to_le_bytes());
                out.extend_from_slice(&slot.to_le_bytes());
                out.extend_from_slice(&code.to_le_bytes());
                out.push(*down as u8);
            }
            InputEvent::GamepadAxis { slot, code, value } => {
                out.extend_from_slice(&Self::KIND_GAMEPAD_AXIS.to_le_bytes());
                out.extend_from_slice(&slot.to_le_bytes());
                out.extend_from_slice(&code.to_le_bytes());
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        if payload.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "short input payload",
            ));
        }
        let kind = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let u32_at = |o: usize| -> std::io::Result<u32> {
            payload
                .get(4 + o..4 + o + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "short input payload")
                })
        };
        let down = |o: usize| payload.get(4 + o).map(|&b| b != 0).unwrap_or(false);
        Ok(match kind {
            Self::KIND_KEY => InputEvent::Key {
                code: u32_at(0)?,
                down: down(4),
            },
            Self::KIND_BTN => InputEvent::Btn {
                button: u32_at(0)?,
                down: down(4),
            },
            Self::KIND_ABS => InputEvent::Abs {
                x: u32_at(0)?,
                y: u32_at(4)?,
            },
            Self::KIND_REL => InputEvent::Rel {
                dx: u32_at(0)? as i32,
                dy: u32_at(4)? as i32,
            },
            Self::KIND_GAMEPAD_BTN => InputEvent::GamepadBtn {
                slot: u32_at(0)?,
                code: u32_at(4)?,
                down: down(8),
            },
            Self::KIND_GAMEPAD_AXIS => InputEvent::GamepadAxis {
                slot: u32_at(0)?,
                code: u32_at(4)?,
                value: u32_at(8)? as i32,
            },
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown input event kind {other}"),
                ))
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    Hello(Hello),
    Meta(DisplayMeta),
    /// raw `ibverbs::QueuePairEndpoint::to_bytes` encoding
    QpEndpoint([u8; 27]),
    Buffers(Vec<SlotDesc>),
    Ready,
    Credit(u32),
    Bye(u32),
    Input(InputEvent),
    Stamp(FrameStamp),
    /// client → server: client's send time; server echoes it back
    Ping(u64),
    /// server → client: (ping's stamp, server's reply time)
    Pong(u64, u64),
    /// server → client: force-feedback event from the host game, to be
    /// replayed on the client's physical controller (see proto::gamepad)
    Rumble(gamepad::Rumble),
    /// server → client: frame codec id, the slot byte size, and (lossy) the
    /// chroma subsampling (0 = 4:2:0, 1 = 4:4:4).
    Codec(u8, u32, u8),
}

/// Encode + frame one message onto any writer. `Conn::send` is a wrapper;
/// free-standing so the RDMA layer can push stamps over a cloned stream
/// without a `Read` half.
pub fn send_msg<T: Write>(io: &mut T, msg: &Message) -> std::io::Result<()> {
    let (ty, payload): (u32, Vec<u8>) = match msg {
        Message::Hello(h) => (
            MsgType::Hello as u32,
            {
                let mut v = Vec::with_capacity(17);
                v.extend_from_slice(&h.magic.to_le_bytes());
                v.extend_from_slice(&h.version.to_le_bytes());
                v.extend_from_slice(&h.nonce.to_le_bytes());
                v.push(h.caps);
                v
            },
        ),
        Message::Meta(m) => (
            MsgType::Meta as u32,
            {
                let mut v = Vec::with_capacity(33);
                v.extend_from_slice(&m.width.to_le_bytes());
                v.extend_from_slice(&m.height.to_le_bytes());
                v.extend_from_slice(&m.stride.to_le_bytes());
                v.extend_from_slice(&m.fourcc.to_le_bytes());
                v.extend_from_slice(&m.modifier.to_le_bytes());
                v.push(m.y0_top as u8);
                v.extend_from_slice(&m.frame_bytes.to_le_bytes());
                v.extend_from_slice(&m.gen.to_le_bytes());
                v
            },
        ),
        Message::QpEndpoint(ep) => (MsgType::QpEndpoint as u32, ep.to_vec()),
        Message::Buffers(slots) => (
            MsgType::Buffers as u32,
            {
                let mut v = Vec::with_capacity(4 + slots.len() * 20);
                v.extend_from_slice(&(slots.len() as u32).to_le_bytes());
                for s in slots {
                    v.extend_from_slice(&s.addr.to_le_bytes());
                    v.extend_from_slice(&s.len.to_le_bytes());
                    v.extend_from_slice(&s.rkey.to_le_bytes());
                }
                v
            },
        ),
        Message::Ready => (MsgType::Ready as u32, Vec::new()),
        Message::Credit(slot) => (MsgType::Credit as u32, slot.to_le_bytes().to_vec()),
        Message::Bye(code) => (MsgType::Bye as u32, code.to_le_bytes().to_vec()),
        Message::Input(ev) => (MsgType::Input as u32, ev.encode()),
        Message::Stamp(s) => (
            MsgType::Stamp as u32,
            {
                let mut v = Vec::with_capacity(112);
                v.extend_from_slice(&s.seq.to_le_bytes());
                v.extend_from_slice(&s.t0.to_le_bytes());
                v.extend_from_slice(&s.t0d.to_le_bytes());
                v.extend_from_slice(&s.t1s.to_le_bytes());
                v.extend_from_slice(&s.t1.to_le_bytes());
                v.extend_from_slice(&s.t2.to_le_bytes());
                v.extend_from_slice(&s.t2c.to_le_bytes());
                v.push(s.mat.kind as u8);
                for u in s.mat.us {
                    v.extend_from_slice(&u.to_le_bytes());
                }
                v.extend_from_slice(&s.meta_gen.to_le_bytes());
                v
            },
        ),
        Message::Ping(c0) => (MsgType::Ping as u32, c0.to_le_bytes().to_vec()),
        Message::Pong(c0, s1) => (
            MsgType::Pong as u32,
            {
                let mut v = Vec::with_capacity(16);
                v.extend_from_slice(&c0.to_le_bytes());
                v.extend_from_slice(&s1.to_le_bytes());
                v
            },
        ),
        Message::Rumble(r) => (MsgType::Rumble as u32, r.encode()),
        Message::Codec(id, slot_len, chroma) => (
            MsgType::Codec as u32,
            {
                let mut v = Vec::with_capacity(6);
                v.push(*id);
                v.extend_from_slice(&slot_len.to_le_bytes());
                v.push(*chroma);
                v
            },
        ),
    };
    if payload.len() > MAX_PAYLOAD {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "payload too large",
        ));
    }
    let mut hdr = Vec::with_capacity(8);
    hdr.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    hdr.extend_from_slice(&ty.to_le_bytes());
    io.write_all(&hdr)?;
    io.write_all(&payload)?;
    io.flush()
}

pub struct Conn<T: Read + Write> {
    io: T,
}

impl<T: Read + Write> Conn<T> {
    pub fn new(io: T) -> Self {
        Self { io }
    }

    pub fn into_inner(self) -> T {
        self.io
    }

    pub fn send(&mut self, msg: &Message) -> std::io::Result<()> {
        send_msg(&mut self.io, msg)
    }

    pub fn recv(&mut self) -> std::io::Result<Message> {
        let mut hdr = [0u8; 8];
        self.io.read_exact(&mut hdr)?;
        let len = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let ty = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if len > MAX_PAYLOAD {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "payload too large",
            ));
        }
        let mut payload = vec![0u8; len];
        if len > 0 {
            self.io.read_exact(&mut payload)?;
        }

        fn u32_at(p: &[u8], o: usize) -> std::io::Result<u32> {
            let b = p
                .get(o..o + 4)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "short payload"))?;
            Ok(u32::from_le_bytes(b.try_into().unwrap()))
        }

        let msg = match ty {
            t if t == MsgType::Hello as u32 => Message::Hello(Hello {
                magic: u32_at(&payload, 0)?,
                version: u32_at(&payload, 4)?,
                nonce: u64::from_le_bytes(
                    payload
                        .get(8..16)
                        .ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "short hello payload",
                            )
                        })?
                        .try_into()
                        .unwrap(),
                ),
                caps: *payload.get(16).unwrap_or(&0),
            }),
            t if t == MsgType::Meta as u32 => {
                let width = u32_at(&payload, 0)?;
                let height = u32_at(&payload, 4)?;
                let stride = u32_at(&payload, 8)?;
                let fourcc = u32_at(&payload, 12)?;
                let modifier = u64::from_le_bytes(payload[16..24].try_into().unwrap());
                let y0_top = payload[24] != 0;
                let frame_bytes = u32_at(&payload, 25)?;
                let gen = u32_at(&payload, 29)?;
                Message::Meta(DisplayMeta {
                    width,
                    height,
                    stride,
                    fourcc,
                    modifier,
                    y0_top,
                    frame_bytes,
                    gen,
                })
            }
            t if t == MsgType::QpEndpoint as u32 => {
                if payload.len() != 27 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "bad endpoint payload",
                    ));
                }
                let mut b = [0u8; 27];
                b.copy_from_slice(&payload);
                Message::QpEndpoint(b)
            }
            t if t == MsgType::Buffers as u32 => {
                let n = u32_at(&payload, 0)? as usize;
                let mut slots = Vec::with_capacity(n);
                for i in 0..n {
                    let base = 4 + i * 20;
                    if payload.len() < base + 20 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "short buffers payload",
                        ));
                    }
                    slots.push(SlotDesc {
                        addr: u64::from_le_bytes(payload[base..base + 8].try_into().unwrap()),
                        len: u64::from_le_bytes(payload[base + 8..base + 16].try_into().unwrap()),
                        rkey: u32::from_le_bytes(payload[base + 16..base + 20].try_into().unwrap()),
                    });
                }
                Message::Buffers(slots)
            }
            t if t == MsgType::Ready as u32 => Message::Ready,
            t if t == MsgType::Credit as u32 => Message::Credit(u32_at(&payload, 0)?),
            t if t == MsgType::Bye as u32 => Message::Bye(u32_at(&payload, 0)?),
            t if t == MsgType::Input as u32 => Message::Input(InputEvent::decode(&payload)?),
            t if t == MsgType::Stamp as u32 => {
                // 56 bytes of stamps + kind + 6 stage u64s + meta_gen
                const STAMP_LEN: usize = 56 + 1 + MAT_STAGES * 8 + 4;
                if payload.len() < STAMP_LEN {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "short stamp payload",
                    ));
                }
                let mut us = [0u64; MAT_STAGES];
                for (i, u) in us.iter_mut().enumerate() {
                    let o = 57 + i * 8;
                    *u = u64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
                }
                Message::Stamp(FrameStamp {
                    seq: u64::from_le_bytes(payload[0..8].try_into().unwrap()),
                    t0: u64::from_le_bytes(payload[8..16].try_into().unwrap()),
                    t0d: u64::from_le_bytes(payload[16..24].try_into().unwrap()),
                    t1s: u64::from_le_bytes(payload[24..32].try_into().unwrap()),
                    t1: u64::from_le_bytes(payload[32..40].try_into().unwrap()),
                    t2: u64::from_le_bytes(payload[40..48].try_into().unwrap()),
                    t2c: u64::from_le_bytes(payload[48..56].try_into().unwrap()),
                    mat: MatTimes {
                        kind: MatKind::from_u8(payload[56]),
                        us,
                    },
                    meta_gen: u32::from_le_bytes(payload[105..109].try_into().unwrap()),
                })
            }
            t if t == MsgType::Ping as u32 => {
                if payload.len() < 8 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "short ping payload",
                    ));
                }
                Message::Ping(u64::from_le_bytes(payload[0..8].try_into().unwrap()))
            }
            t if t == MsgType::Pong as u32 => {
                if payload.len() < 16 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "short pong payload",
                    ));
                }
                Message::Pong(
                    u64::from_le_bytes(payload[0..8].try_into().unwrap()),
                    u64::from_le_bytes(payload[8..16].try_into().unwrap()),
                )
            }
            t if t == MsgType::Rumble as u32 => Message::Rumble(gamepad::Rumble::decode(&payload)?),
            t if t == MsgType::Codec as u32 => {
                let id = *payload
                    .first()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "short codec payload"))?;
                let slot_len = u32_at(&payload, 1)?;
                let chroma = *payload.get(5).unwrap_or(&0);
                Message::Codec(id, slot_len, chroma)
            }
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown message type {other}"),
                ))
            }
        };
        Ok(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn stamp_ping_pong_roundtrip() {
        let msgs = [
            Message::Meta(DisplayMeta {
                width: 3840,
                height: 2160,
                stride: 15360,
                fourcc: 0x3432_5258,
                modifier: 0x0300_0000_0060_6014,
                y0_top: false,
                frame_bytes: 3840 * 4 * 2160,
                gen: 2,
            }),
            Message::Stamp(FrameStamp {
                seq: 0x00ff_ffff,
                t0: 1_700_000_000_000_123,
                t0d: 1_700_000_000_001_000,
                t1s: 1_700_000_000_001_100,
                t1: 1_700_000_000_002_456,
                t2: 1_700_000_000_002_789,
                t2c: 1_700_000_000_004_000,
                mat: MatTimes::new(MatKind::Lossless, [11, 0, 2069, 77, 2, 23]),
                meta_gen: 2,
            }),
            Message::Ping(1_700_000_000_003_111),
            Message::Pong(1_700_000_000_003_111, 1_700_000_000_003_500),
            Message::Rumble(gamepad::Rumble {
                slot: 0,
                id: 7,
                strong: 0xffff,
                weak: 0x8000,
                duration_ms: 5000,
                action: gamepad::RUMBLE_UPLOAD,
            }),
        ];
        for m in msgs {
            let mut buf: Vec<u8> = Vec::new();
            {
                let mut c = Conn::new(Cursor::new(&mut buf));
                c.send(&m).expect("send");
            }
            let mut c = Conn::new(Cursor::new(&mut buf));
            let got = c.recv().expect("recv");
            assert_eq!(format!("{got:?}"), format!("{m:?}"));
        }
    }

    #[test]
    fn input_event_roundtrip() {
        let events = [
            InputEvent::Key { code: 0x1e, down: true },
            InputEvent::Key { code: 0x2a, down: false },
            InputEvent::Btn { button: 0, down: true },
            InputEvent::Btn { button: 4, down: false },
            InputEvent::Abs { x: 1919, y: 1079 },
            InputEvent::Abs { x: 0, y: 0 },
            InputEvent::Rel { dx: -17, dy: 42 },
            InputEvent::Rel { dx: 0, dy: 0 },
            InputEvent::GamepadBtn { slot: 0, code: 0x130, down: true },
            InputEvent::GamepadBtn { slot: 0, code: 0x220, down: false },
            InputEvent::GamepadAxis { slot: 0, code: 0, value: -32768 },
            InputEvent::GamepadAxis { slot: 0, code: 5, value: 255 },
        ];
        for ev in events {
            let enc = ev.encode();
            let dec = InputEvent::decode(&enc).expect("decode");
            assert_eq!(enc.len() >= 4, true);
            // compare via debug (no PartialEq needed field-wise)
            assert_eq!(format!("{dec:?}"), format!("{ev:?}"));
        }
    }
}
