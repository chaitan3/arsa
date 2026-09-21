//! Shared gamepad plumbing (proto v7): kernel-standard event codes, the
//! virtual pad's fixed axis ranges, real→virtual axis rescaling, and the
//! server→client rumble hub.
//!
//! The client reads the physical controller straight from /dev/input (raw
//! evdev — no Wayland involvement), normalizes axis values into the virtual
//! device's fixed ranges, and forwards `InputEvent::Gamepad*` events. The
//! server re-emits those codes on its uinput gamepad, so the game on the host
//! sees exactly the physical pad (after rescale — the virtual device's
//! absinfo defines axis semantics, so wide-range real devices must be mapped
//! into it). Both ends read the constants here; server+client deploy
//! together.
//!
//! Rumble is the opposite direction: the host game submits an FF effect to
//! the virtual pad, the kernel hands it to the uinput owner, and the server
//! forwards it here for replay on the physical controller. The `RumbleHub`
//! carries those events from the pad thread to the current session's control
//! stream.

use std::sync::{mpsc, Arc};

// linux/input-event-codes.h — the gamepad-standard codes we forward
pub const ABS_X: u32 = 0x00;
pub const ABS_Y: u32 = 0x01;
pub const ABS_Z: u32 = 0x02;
pub const ABS_RX: u32 = 0x03;
pub const ABS_RY: u32 = 0x04;
pub const ABS_RZ: u32 = 0x05;
pub const ABS_HAT0X: u32 = 0x10;
pub const ABS_HAT0Y: u32 = 0x11;

pub const BTN_SOUTH: u32 = 0x130;
pub const BTN_EAST: u32 = 0x131;
pub const BTN_C: u32 = 0x132;
pub const BTN_NORTH: u32 = 0x133;
pub const BTN_WEST: u32 = 0x134;
pub const BTN_Z: u32 = 0x135;
pub const BTN_TL: u32 = 0x136;
pub const BTN_TR: u32 = 0x137;
pub const BTN_TL2: u32 = 0x138;
pub const BTN_TR2: u32 = 0x139;
pub const BTN_SELECT: u32 = 0x13a;
pub const BTN_START: u32 = 0x13b;
pub const BTN_MODE: u32 = 0x13c;
pub const BTN_THUMBL: u32 = 0x13d;
pub const BTN_THUMBR: u32 = 0x13e;
pub const BTN_GAMEPAD: u32 = BTN_SOUTH;
pub const BTN_DPAD_UP: u32 = 0x220;
pub const BTN_DPAD_DOWN: u32 = 0x221;
pub const BTN_DPAD_LEFT: u32 = 0x222;
pub const BTN_DPAD_RIGHT: u32 = 0x223;

/// linux/input.h FF_RUMBLE (effect type)
pub const FF_RUMBLE: u16 = 0x50;
pub const FF_GAIN: u16 = 0x60;

/// Forwarded button codes: the kernel gamepad action set + triggers + dpad.
/// Registered on the virtual device bit-for-bit, so a client event maps to a
/// `UI_SET_KEYBIT` that always exists.
pub fn is_button(code: u32) -> bool {
    (BTN_SOUTH..=BTN_THUMBR).contains(&code) || (BTN_DPAD_UP..=BTN_DPAD_RIGHT).contains(&code)
}

/// Forwarded axis codes: sticks, triggers, hat.
pub fn is_axis(code: u32) -> bool {
    matches!(code, ABS_X | ABS_Y | ABS_Z | ABS_RX | ABS_RY | ABS_RZ | ABS_HAT0X | ABS_HAT0Y)
}

/// The virtual pad's axis ranges — inputtino's Xbox joypad profile
/// (src/uinput/joypad_xbox.cpp), so SDL games on the host treat our pad
/// exactly like a Sunshine pad: sticks ±32767 fuzz 16 flat 128, triggers
/// 0..255, hat -1..1.
pub fn virtual_range(code: u32) -> (i32, i32) {
    match code {
        ABS_X | ABS_Y | ABS_RX | ABS_RY => (-32768, 32767),
        ABS_Z | ABS_RZ => (0, 255),
        ABS_HAT0X | ABS_HAT0Y => (-1, 1),
        _ => (0, 0),
    }
}

/// Map a raw axis value from the real device's range into the virtual range.
/// Identity when the ranges already match (DualSense sticks are within 1 LSB
/// of the virtual range); saturating elsewhere (a DualSense 0..1023 trigger
/// compresses into the virtual Xbox profile's 0..255). Endpoints map to
/// endpoints.
pub fn rescale(value: i32, real: (i32, i32), virt: (i32, i32)) -> i32 {
    let (rmin, rmax) = real;
    let (vmin, vmax) = virt;
    if rmax <= rmin || vmax <= vmin {
        return value.clamp(vmin, vmax);
    }
    let clamped = (value - rmin).clamp(0, rmax - rmin) as f64;
    let out = vmin as f64 + clamped * (vmax - vmin) as f64 / (rmax - rmin) as f64;
    out.round() as i32
}

/// One server→client force-feedback event, replayed on the physical pad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rumble {
    pub slot: u32,
    /// kernel effect id on the server's virtual pad (pairs upload ↔ play/stop)
    pub id: u16,
    pub strong: u16,
    pub weak: u16,
    /// effect replay length in ms (0 = play until stopped)
    pub duration_ms: u16,
    /// `ACTION_*`
    pub action: u8,
}

pub const RUMBLE_UPLOAD: u8 = 0;
pub const RUMBLE_PLAY: u8 = 1;
pub const RUMBLE_STOP: u8 = 2;
pub const RUMBLE_ERASE: u8 = 3;

impl Rumble {
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(13);
        v.extend_from_slice(&self.slot.to_le_bytes());
        v.extend_from_slice(&self.id.to_le_bytes());
        v.extend_from_slice(&self.strong.to_le_bytes());
        v.extend_from_slice(&self.weak.to_le_bytes());
        v.extend_from_slice(&self.duration_ms.to_le_bytes());
        v.push(self.action);
        v
    }

    pub fn decode(p: &[u8]) -> std::io::Result<Self> {
        if p.len() < 13 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "short rumble payload",
            ));
        }
        Ok(Rumble {
            slot: u32::from_le_bytes(p[0..4].try_into().unwrap()),
            id: u16::from_le_bytes(p[4..6].try_into().unwrap()),
            strong: u16::from_le_bytes(p[6..8].try_into().unwrap()),
            weak: u16::from_le_bytes(p[8..10].try_into().unwrap()),
            duration_ms: u16::from_le_bytes(p[10..12].try_into().unwrap()),
            action: p[12],
        })
    }
}

/// The pad thread's door to the current session's control stream. The RDMA
/// layer installs a sender per session (`ServerLink::accept`) and clears it
/// on disconnect; sends when no session is live are silently dropped —
/// rumble is fire-and-forget, never backpressured onto the video path.
#[derive(Clone, Default)]
pub struct RumbleHub(Arc<std::sync::Mutex<Option<mpsc::Sender<Rumble>>>>);

impl RumbleHub {
    pub fn new() -> Self {
        Self(Arc::new(std::sync::Mutex::new(None)))
    }

    pub fn set(&self, tx: mpsc::Sender<Rumble>) {
        *self.0.lock().unwrap() = Some(tx);
    }

    pub fn clear(&self) {
        *self.0.lock().unwrap() = None;
    }

    pub fn send(&self, r: Rumble) {
        if let Some(tx) = self.0.lock().unwrap().as_ref() {
            let _ = tx.send(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescale_identity_on_matching_ranges() {
        // DualSense sticks (±32767) into the virtual ±32768..32767 range:
        // endpoints map to endpoints, center stays center
        let (real, virt) = ((-32767, 32767), (-32768, 32767));
        assert_eq!(rescale(-32767, real, virt), -32768);
        // center maps within 1 LSB (virtual range midpoint is -0.5)
        assert!(rescale(0, real, virt).abs() <= 1);
        assert_eq!(rescale(32767, real, virt), 32767);
        // DualSense triggers 0..1023 → virtual 0..255
        let (real, virt) = ((0, 1023), (0, 255));
        assert_eq!(rescale(0, real, virt), 0);
        assert_eq!(rescale(1023, real, virt), 255);
        assert_eq!(rescale(512, real, virt), 128);
        // Xbox triggers 0..255 → 0..255: identity
        for v in [0, 127, 255] {
            assert_eq!(rescale(v, (0, 255), (0, 255)), v);
        }
        // hat: identity
        for v in [-1, 0, 1] {
            assert_eq!(rescale(v, (-1, 1), (-1, 1)), v);
        }
    }

    #[test]
    fn rumble_roundtrip() {
        let r = Rumble { slot: 3, id: 42, strong: 0xbeef, weak: 0x1234, duration_ms: 5000, action: RUMBLE_PLAY };
        let dec = Rumble::decode(&r.encode()).expect("decode");
        assert_eq!(r, dec);
    }
}
