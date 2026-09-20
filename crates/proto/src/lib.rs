//! Control-channel protocol for rdma-display: length-prefixed little-endian
//! messages over TCP (or Unix sockets for local testing).
//!
//! Framing: every message is `u32 payload_len | u32 msg_type | payload`.
//! RDMA-specific data (QPN/PSN/GID, rkeys) uses the `ibverbs` crate's stable
//! wire encodings embedded in payloads.

use std::io::{Read, Write};

pub mod keymap;

pub const MAGIC: u32 = 0x5244_4453; // "RDDS"
pub const VERSION: u32 = 5; // v5: Stamp carries the materializer's blit/read/finish times

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
}

/// Server-side pipeline timestamps for one frame (all `unix_now_us`).
///   t0  = QEMU D-Bus event arrived in the capture thread
///   t0d = the sender picked the frame up (pacing + event-queue wait done)
///   t1s = a staging slot was available (after any credit wait)
///   t1  = materializer readback finished (glFinish complete, slot populated)
///   t2  = RDMA WRITE posted
///   t2c = send CQ reported the WRITE fully written on the client —
///         the true transfer time is `t2c - t2`, same-host (no clock skew)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameStamp {
    pub seq: u64,
    pub t0: u64,
    pub t0d: u64,
    pub t1s: u64,
    pub t1: u64,
    pub t2: u64,
    pub t2c: u64,
    /// materializer stage times of this frame (µs, all server-local,
    /// wall-clock split): blit = draw-call latency (the draw is queued, not
    /// awaited), read = GPU round trip from draw-queued to pixels-in-host-RAM
    /// (blit execution + readback DMA + driver delivery), fin = residual
    /// glFinish
    pub blit_us: u64,
    pub read_us: u64,
    pub fin_us: u64,
}

/// Materializer stage times for one frame (µs, server-local wall clock,
/// coarse split — the draw is queued, not awaited, so these are CPU stamps):
///   blit = glDrawArrays return (CPU call latency)
///   read = glReadPixels return: GPU round trip from draw-queued to
///          pixels-in-host-RAM (draw execution + readback DMA + delivery)
///   fin  = glFinish residual drain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatTimes {
    pub blit_us: u64,
    pub read_us: u64,
    pub fin_us: u64,
}

impl MatTimes {
    pub const ZERO: MatTimes = MatTimes {
        blit_us: 0,
        read_us: 0,
        fin_us: 0,
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Hello {
    pub magic: u32,
    pub version: u32,
}

/// Display format metadata, sent by the server for the current scanout.
#[derive(Debug, Clone)]
pub struct DisplayMeta {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub y0_top: bool,
    /// total bytes of one full frame (stride * height for single-plane formats)
    pub frame_bytes: u32,
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
#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Key { code: u32, down: bool },
    Btn { button: u32, down: bool },
    Abs { x: u32, y: u32 },
    Rel { dx: i32, dy: i32 },
}

impl InputEvent {
    const KIND_KEY: u32 = 0;
    const KIND_BTN: u32 = 1;
    const KIND_ABS: u32 = 2;
    const KIND_REL: u32 = 3;

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
}

/// Encode + frame one message onto any writer. `Conn::send` is a wrapper;
/// free-standing so the RDMA layer can push stamps over a cloned stream
/// without a `Read` half.
pub fn send_msg<T: Write>(io: &mut T, msg: &Message) -> std::io::Result<()> {
    let (ty, payload): (u32, Vec<u8>) = match msg {
        Message::Hello(h) => (
            MsgType::Hello as u32,
            {
                let mut v = Vec::with_capacity(8);
                v.extend_from_slice(&h.magic.to_le_bytes());
                v.extend_from_slice(&h.version.to_le_bytes());
                v
            },
        ),
        Message::Meta(m) => (
            MsgType::Meta as u32,
            {
                let mut v = Vec::with_capacity(29);
                v.extend_from_slice(&m.width.to_le_bytes());
                v.extend_from_slice(&m.height.to_le_bytes());
                v.extend_from_slice(&m.stride.to_le_bytes());
                v.extend_from_slice(&m.fourcc.to_le_bytes());
                v.extend_from_slice(&m.modifier.to_le_bytes());
                v.push(m.y0_top as u8);
                v.extend_from_slice(&m.frame_bytes.to_le_bytes());
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
                let mut v = Vec::with_capacity(80);
                v.extend_from_slice(&s.seq.to_le_bytes());
                v.extend_from_slice(&s.t0.to_le_bytes());
                v.extend_from_slice(&s.t0d.to_le_bytes());
                v.extend_from_slice(&s.t1s.to_le_bytes());
                v.extend_from_slice(&s.t1.to_le_bytes());
                v.extend_from_slice(&s.t2.to_le_bytes());
                v.extend_from_slice(&s.t2c.to_le_bytes());
                v.extend_from_slice(&s.blit_us.to_le_bytes());
                v.extend_from_slice(&s.read_us.to_le_bytes());
                v.extend_from_slice(&s.fin_us.to_le_bytes());
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
            }),
            t if t == MsgType::Meta as u32 => {
                let width = u32_at(&payload, 0)?;
                let height = u32_at(&payload, 4)?;
                let stride = u32_at(&payload, 8)?;
                let fourcc = u32_at(&payload, 12)?;
                let modifier = u64::from_le_bytes(payload[16..24].try_into().unwrap());
                let y0_top = payload[24] != 0;
                let frame_bytes = u32_at(&payload, 25)?;
                Message::Meta(DisplayMeta {
                    width,
                    height,
                    stride,
                    fourcc,
                    modifier,
                    y0_top,
                    frame_bytes,
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
                if payload.len() < 80 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "short stamp payload",
                    ));
                }
                Message::Stamp(FrameStamp {
                    seq: u64::from_le_bytes(payload[0..8].try_into().unwrap()),
                    t0: u64::from_le_bytes(payload[8..16].try_into().unwrap()),
                    t0d: u64::from_le_bytes(payload[16..24].try_into().unwrap()),
                    t1s: u64::from_le_bytes(payload[24..32].try_into().unwrap()),
                    t1: u64::from_le_bytes(payload[32..40].try_into().unwrap()),
                    t2: u64::from_le_bytes(payload[40..48].try_into().unwrap()),
                    t2c: u64::from_le_bytes(payload[48..56].try_into().unwrap()),
                    blit_us: u64::from_le_bytes(payload[56..64].try_into().unwrap()),
                    read_us: u64::from_le_bytes(payload[64..72].try_into().unwrap()),
                    fin_us: u64::from_le_bytes(payload[72..80].try_into().unwrap()),
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
            Message::Stamp(FrameStamp {
                seq: 0x00ff_ffff,
                t0: 1_700_000_000_000_123,
                t0d: 1_700_000_000_001_000,
                t1s: 1_700_000_000_001_100,
                t1: 1_700_000_000_002_456,
                t2: 1_700_000_000_002_789,
                t2c: 1_700_000_000_004_000,
                blit_us: 980,
                read_us: 11_120,
                fin_us: 680,
            }),
            Message::Ping(1_700_000_000_003_111),
            Message::Pong(1_700_000_000_003_111, 1_700_000_000_003_500),
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
