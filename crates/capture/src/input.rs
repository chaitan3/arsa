//! Input injection sinks, decoupled from capture sources.
//!
//! A capture source produces frames; an `InputSink` consumes client input
//! events. The server pairs them (`--source`/`--input`): QEMU D-Bus capture
//! injects into the guest console; DRM/KMS capture injects into the host
//! compositor via uinput — exactly the Sunshine split.

use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{mpsc, Arc};

use anyhow::{Context, Result};
use rdma_display_proto::InputEvent;

use crate::ScreenSize;

/// Consumes `InputEvent`s until the channel closes. `screen` carries the
/// current captured dimensions (updated by the server on every scanout);
/// absolute-position sinks need them for coordinate ranges.
pub trait InputSink: Send {
    fn name(&self) -> &'static str;
    fn spawn(
        self: Box<Self>,
        rx: mpsc::Receiver<InputEvent>,
        screen: Arc<ScreenSize>,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// QEMU D-Bus console injection (the pre-split behavior, moved from qemu.rs)
// ---------------------------------------------------------------------------

pub struct QemuDbusInput {
    /// `None` → session bus; `Some(path)` → private (libvirt) dbus socket.
    pub dbus_socket: Option<String>,
    pub console: u32,
}

impl QemuDbusInput {
    pub fn new(dbus_socket: Option<String>, console: u32) -> Self {
        Self { dbus_socket, console }
    }
}

impl InputSink for QemuDbusInput {
    fn name(&self) -> &'static str {
        "qemu-dbus"
    }

    fn spawn(
        self: Box<Self>,
        rx: mpsc::Receiver<InputEvent>,
        _screen: Arc<ScreenSize>,
    ) -> Result<()> {
        std::thread::Builder::new()
            .name("capture-input".into())
            .spawn(move || {
                let console = match async_std::task::block_on(async {
                    crate::qemu::open_console(&self.dbus_socket, self.console).await
                }) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("[input] console proxy failed: {e:#}");
                        return;
                    }
                };
                let kb = console.keyboard.clone();
                let mo = console.mouse.clone();
                // probe the guest's pointing-device mode once (IsAbsolute property)
                if let Ok(abs) = async_std::task::block_on(mo.is_absolute()) {
                    eprintln!("[input] guest mouse IsAbsolute={abs}");
                }
                while let Ok(ev) = rx.recv() {
                    let r = async_std::task::block_on(async {
                        match ev {
                            InputEvent::Key { code, down } => {
                                if down {
                                    kb.press(code).await
                                } else {
                                    kb.release(code).await
                                }
                            }
                            InputEvent::Btn { button, down } => {
                                // D-Bus button values: Left=0 Middle=1 Right=2
                                // WheelUp=3 WheelDown=4 Side=5 Extra=6
                                let btn = match button {
                                    0 => qemu_display::MouseButton::Left,
                                    1 => qemu_display::MouseButton::Middle,
                                    2 => qemu_display::MouseButton::Right,
                                    3 => qemu_display::MouseButton::WheelUp,
                                    4 => qemu_display::MouseButton::WheelDown,
                                    5 => qemu_display::MouseButton::Side,
                                    6 => qemu_display::MouseButton::Extra,
                                    other => {
                                        eprintln!("[input] unknown button {other}");
                                        return Ok(());
                                    }
                                };
                                if down {
                                    mo.press(btn).await
                                } else {
                                    mo.release(btn).await
                                }
                            }
                            InputEvent::Abs { x, y } => mo.set_abs_position(x, y).await,
                            InputEvent::Rel { dx, dy } => mo.rel_motion(dx, dy).await,
                        }
                    });
                    if let Err(e) = r {
                        eprintln!("[input] D-Bus call failed for {ev:?}: {e}");
                    }
                }
                eprintln!("[input] control channel closed; input forwarding stopped");
            })
            .context("spawn input forwarder")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// uinput injection into the host compositor (DRM capture path)
// ---------------------------------------------------------------------------

/// Injects into the host via /dev/uinput, device split matching Sunshine's
/// inputtino (upstream src/uinput/mouse.cpp, verified):
/// - a REL mouse: buttons + REL_X/Y/WHEEL only. libinput classifies it as a
///   plain mouse, so relative deltas reach pointer-locked games 1:1.
/// - an ABS pointer: INPUT_PROP_POINTER + buttons + wheel + ABS_X/Y over
///   giant fixed ranges; client pixel coords are scaled in, so the device
///   never needs recreation when the captured resolution changes.
/// A single combined ABS+REL device is NOT what Sunshine ships: rel events
/// on a device that also declares ABS axes move the cursor only minutely
/// (observed on hyprland/libinput). Buttons go to whichever device saw the
/// last motion, like inputtino's active_mouse_device.
pub struct UinputSink;

/// inputtino's giant ABS ranges: x scaled into 0..19200, y into 0..12000
const ABS_MAX_W: i32 = 19200;
const ABS_MAX_H: i32 = 12000;
/// linux/input.h INPUT_PROP_POINTER (prop bit 0)
const INPUT_PROP_POINTER: u32 = 0;
const REL_WHEEL_HI_RES: u16 = 0x0b;

// evdev constants (linux/input-event-codes.h)
const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_REL: u16 = 0x02;
const EV_ABS: u16 = 0x03;
const REL_X: u16 = 0x00;
const REL_Y: u16 = 0x01;
const REL_WHEEL: u16 = 0x08;
const ABS_X: u16 = 0x00;
const ABS_Y: u16 = 0x01;
const BTN_LEFT: u16 = 0x110;
const BTN_RIGHT: u16 = 0x111;
const BTN_MIDDLE: u16 = 0x112;
const BTN_SIDE: u16 = 0x113;
const BTN_EXTRA: u16 = 0x114;
const SYN_REPORT: u16 = 0x00;
const KEY_MAX: u32 = 0x2ff;
const BUS_VIRTUAL: u16 = 0x06;

// /dev/uinput ioctl numbers (linux/uinput.h), built by hand: _IOC
// dir(2)<<30 | size(14)<<16 | type(8)<<8 | nr(8); _IOC_WRITE=1, 'U'=0x55.
const U: u64 = 'U' as u64;
fn iow(nr: u16, size: u64) -> u64 {
    (1 << 30) | (size << 16) | (U << 8) | nr as u64
}
fn io(nr: u16) -> u64 {
    (U << 8) | nr as u64
}

const UI_DEV_SETUP: u16 = 3; // _IOW('U', 3, uinput_setup)
const UI_DEV_DESTROY: u16 = 2; // _IO('U', 2)
const UI_ABS_SETUP: u16 = 4; // _IOW('U', 4, uinput_abs_setup)
const UI_DEV_CREATE: u16 = 1; // _IO('U', 1)
const UI_SET_EVBIT: u16 = 100;
const UI_SET_KEYBIT: u16 = 101;
const UI_SET_RELBIT: u16 = 102;
const UI_SET_ABSBIT: u16 = 103;
const UI_SET_PROPBIT: u16 = 110; // _IOW('U', 110, unsigned int)

#[repr(C)]
struct UinputSetup {
    bustype: u16,
    vendor: u16,
    product: u16,
    version: u16,
    name: [u8; 80],
    ff_effects_max: u32,
}

#[repr(C)]
struct InputAbsinfo {
    value: i32,
    minimum: i32,
    maximum: i32,
    fuzz: i32,
    flat: i32,
    resolution: i32,
}

#[repr(C)]
struct UinputAbsSetup {
    code: u16,
    absinfo: InputAbsinfo, // 4-aligned: 2 pad bytes before it
}

/// One injected event, exactly `struct input_event` on 64-bit Linux.
#[repr(C)]
struct RawEvent {
    sec: i64,
    usec: i64,
    ty: u16,
    code: u16,
    value: i32,
}

struct UinputDev {
    fd: OwnedFd,
}

impl UinputDev {
    fn setup(self, name: &str) -> Result<Self> {
        let mut s = UinputSetup {
            bustype: BUS_VIRTUAL,
            vendor: 0x1ab0,
            product: 0x5250, // "RD"
            version: 1,
            name: [0; 80],
            ff_effects_max: 0,
        };
        let bytes = name.as_bytes();
        let n = bytes.len().min(79);
        s.name[..n].copy_from_slice(&bytes[..n]);
        unsafe {
            ioctl_u64(
                self.fd.as_raw_fd(),
                iow(UI_DEV_SETUP, std::mem::size_of::<UinputSetup>() as u64),
                &s as *const _ as usize,
            )
            .map(|_| ())
            .context("UI_DEV_SETUP")?;
        }
        Ok(self)
    }

    fn set_evbit(&self, v: u16) -> Result<()> {
        unsafe { ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_EVBIT, 4), v as u32) }
            .map(|_| ())
            .context("UI_SET_EVBIT")
    }
    fn set_keybit(&self, v: u32) -> Result<()> {
        unsafe { ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_KEYBIT, 4), v) }
            .map(|_| ())
            .context("UI_SET_KEYBIT")
    }
    fn set_relbit(&self, v: u16) -> Result<()> {
        unsafe { ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_RELBIT, 4), v as u32) }
            .map(|_| ())
            .context("UI_SET_RELBIT")
    }
    fn set_propbit(&self, bit: u32) -> Result<()> {
        unsafe { ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_PROPBIT, 4), bit) }
            .map(|_| ())
            .context("UI_SET_PROPBIT")
    }
    fn set_absbit(&self, v: u16, max: i32, fuzz: i32, resolution: i32) -> Result<()> {
        unsafe {
            ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_ABSBIT, 4), v as u32)
                .map(|_| ())
                .context("UI_SET_ABSBIT")?;
        }
        let s = UinputAbsSetup {
            code: v,
            absinfo: InputAbsinfo {
                value: 0,
                minimum: 0,
                maximum: max,
                fuzz,
                flat: 0,
                resolution,
            },
        };
        unsafe {
            ioctl_u64(
                self.fd.as_raw_fd(),
                iow(UI_ABS_SETUP, std::mem::size_of::<UinputAbsSetup>() as u64),
                &s as *const _ as usize,
            )
            .map(|_| ())
            .context("UI_ABS_SETUP")
        }
    }

    fn create(self) -> Result<Self> {
        unsafe {
            ioctl_u64(self.fd.as_raw_fd(), io(UI_DEV_CREATE), 0)
                .map(|_| ())
                .context("UI_DEV_CREATE")?;
        }
        Ok(self)
    }

    fn emit(&self, ty: u16, code: u16, value: i32) {
        let ev = RawEvent { sec: 0, usec: 0, ty, code, value };
        let _ = dev_write(self.fd.as_raw_fd(), &ev);
    }
    fn syn(&self) {
        self.emit(EV_SYN, SYN_REPORT, 0);
    }
}

impl Drop for UinputDev {
    fn drop(&mut self) {
        // UI_DEV_DESTROY so the compositor forgets the device when the server exits
        unsafe { libc::ioctl(self.fd.as_raw_fd(), io(UI_DEV_DESTROY), 0) };
    }
}

// thin ioctl/write wrappers over libc (no extra deps)
unsafe fn ioctl_u32(fd: i32, req: u64, arg: u32) -> Result<i32> {
    let r = libc::ioctl(fd, req as libc::c_ulong, arg);
    if r < 0 { Err(std::io::Error::last_os_error().into()) } else { Ok(r) }
}
unsafe fn ioctl_u64(fd: i32, req: u64, arg: usize) -> Result<i32> {
    let r = libc::ioctl(fd, req as libc::c_ulong, arg as *mut libc::c_void);
    if r < 0 { Err(std::io::Error::last_os_error().into()) } else { Ok(r) }
}
fn dev_write(fd: i32, ev: &RawEvent) -> bool {
    unsafe {
        let mut off = 0usize;
        let buf = std::slice::from_raw_parts(
            ev as *const RawEvent as *const u8,
            std::mem::size_of::<RawEvent>(),
        );
        while off < buf.len() {
            let n = libc::write(fd, buf[off..].as_ptr() as *const libc::c_void, buf.len() - off);
            if n <= 0 {
                return false;
            }
            off += n as usize;
        }
    }
    true
}

impl InputSink for UinputSink {
    fn name(&self) -> &'static str {
        "uinput"
    }

    fn spawn(
        self: Box<Self>,
        rx: mpsc::Receiver<InputEvent>,
        screen: Arc<ScreenSize>,
    ) -> Result<()> {
        std::thread::Builder::new()
            .name("input-uinput".into())
            .spawn(move || {
                if let Err(e) = uinput_loop(rx, screen) {
                    eprintln!("[input] uinput sink failed: {e:#}");
                }
            })
            .context("spawn uinput input thread")?;
        Ok(())
    }
}

/// Create a keyboard device registering every EV_KEY code so the compositor
/// sees the full keysym space (matching what xkb expects).
fn uinput_keyboard() -> Result<UinputDev> {
    let d = UinputDev { fd: open_uinput()? }.setup("rdma-display keyboard")?;
    d.set_evbit(EV_SYN)?;
    d.set_evbit(EV_KEY)?;
    for k in 1..=KEY_MAX {
        // ignore individual failures (some bits reject; harmless)
        let _ = d.set_keybit(k);
    }
    d.create()
}

/// REL mouse (inputtino's create_mouse): buttons + REL_X/Y + wheel. No ABS
/// axes — libinput must classify this as a plain mouse for 1:1 relative
/// motion into pointer-locked games.
fn uinput_pointer_rel() -> Result<UinputDev> {
    let d = UinputDev { fd: open_uinput()? }.setup("rdma-display mouse")?;
    d.set_evbit(EV_SYN)?;
    d.set_evbit(EV_KEY)?;
    d.set_evbit(EV_REL)?;
    for b in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_SIDE, BTN_EXTRA] {
        let _ = d.set_keybit(b as u32);
    }
    for r in [REL_X, REL_Y, REL_WHEEL] {
        d.set_relbit(r)?;
    }
    d.create()
}

/// Absolute pointer (inputtino's create_mouse_abs): INPUT_PROP_POINTER +
/// buttons + wheel + ABS X/Y over the giant fixed ranges. No REL_X/REL_Y.
fn uinput_pointer_abs() -> Result<UinputDev> {
    let d = UinputDev { fd: open_uinput()? }.setup("rdma-display mouse (absolute)")?;
    d.set_propbit(INPUT_PROP_POINTER)?;
    d.set_evbit(EV_SYN)?;
    d.set_evbit(EV_KEY)?;
    d.set_evbit(EV_REL)?;
    d.set_evbit(EV_ABS)?;
    for b in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_SIDE, BTN_EXTRA] {
        let _ = d.set_keybit(b as u32);
    }
    d.set_relbit(REL_WHEEL)?;
    // inputtino: fuzz=1, resolution=28
    d.set_absbit(ABS_X, ABS_MAX_W, 1, 28)?;
    d.set_absbit(ABS_Y, ABS_MAX_H, 1, 28)?;
    d.create()
}

fn open_uinput() -> Result<OwnedFd> {
    use std::os::fd::FromRawFd;
    // SAFETY: NUL-terminated static path; fd < 0 checked before wrapping
    let fd = unsafe {
        libc::open(
            c"/dev/uinput".as_ptr(),
            libc::O_WRONLY | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).context("open /dev/uinput (needs root)");
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn uinput_loop(rx: mpsc::Receiver<InputEvent>, screen: Arc<ScreenSize>) -> Result<()> {
    let kb = uinput_keyboard().context("create uinput keyboard")?;
    eprintln!("[input] uinput keyboard created");
    // both mice created up front: the ABS device uses inputtino's giant
    // fixed ranges, so nothing needs recreation when the dims change
    let rel_ptr = match uinput_pointer_rel() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[input] uinput rel mouse create failed: {e:#}");
            None
        }
    };
    let abs_ptr = match uinput_pointer_abs() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("[input] uinput abs mouse create failed: {e:#}");
            None
        }
    };
    // buttons route to whichever device saw the last motion (inputtino's
    // active_mouse_device); rel is the default
    let mut last_abs = false;
    // rel diagnostics: per-2s aggregate (event rate + total counts) to
    // locate motion attenuation between client and injection
    let mut agg_n: u64 = 0;
    let mut agg_sum: (i64, i64) = (0, 0);
    let mut agg_window = std::time::Instant::now();

    while let Ok(ev) = rx.recv() {
        match ev {
            InputEvent::Key { code, down } => {
                let Some(k) = rdma_display_proto::keymap::qnum_to_linux(code) else {
                    eprintln!("[input] no evdev mapping for qnum {code:#x}");
                    continue;
                };
                let d = &kb;
                d.emit(EV_KEY, k as u16, down as i32);
                d.syn();
            }
            InputEvent::Btn { button, down } => {
                // wheel lives on both mice; buttons go to the active one
                let Some(active) = (if last_abs {
                    abs_ptr.as_ref()
                } else {
                    rel_ptr.as_ref()
                }) else {
                    continue;
                };
                match button {
                    // Left=0 Middle=1 Right=2 WheelUp=3 WheelDown=4 Side=5 Extra=6
                    0 | 1 | 2 | 5 | 6 => {
                        let code = match button {
                            0 => BTN_LEFT,
                            1 => BTN_MIDDLE,
                            2 => BTN_RIGHT,
                            5 => BTN_SIDE,
                            _ => BTN_EXTRA,
                        };
                        active.emit(EV_KEY, code, down as i32);
                        active.syn();
                    }
                    3 => {
                        active.emit(EV_REL, REL_WHEEL, 1);
                        active.emit(EV_REL, REL_WHEEL_HI_RES, 120);
                        active.syn();
                    }
                    4 => {
                        active.emit(EV_REL, REL_WHEEL, -1);
                        active.emit(EV_REL, REL_WHEEL_HI_RES, -120);
                        active.syn();
                    }
                    other => eprintln!("[input] unknown button {other}"),
                }
            }
            InputEvent::Abs { x, y } => {
                let (w, h) = screen.get();
                if w == 0 || h == 0 {
                    continue; // no scanout yet
                }
                let Some(a) = abs_ptr.as_ref() else { continue };
                let x = x.min(w.saturating_sub(1));
                let y = y.min(h.saturating_sub(1));
                // client pixels -> inputtino's giant ABS ranges
                let ax = (ABS_MAX_W as f64 * x as f64 / w as f64).round() as i32;
                let ay = (ABS_MAX_H as f64 * y as f64 / h as f64).round() as i32;
                a.emit(EV_ABS, ABS_X, ax);
                a.emit(EV_ABS, ABS_Y, ay);
                a.syn();
                last_abs = true;
            }
            InputEvent::Rel { dx, dy } => {
                agg_n += 1;
                agg_sum.0 += dx as i64;
                agg_sum.1 += dy as i64;
                if agg_window.elapsed().as_secs_f64() >= 2.0 {
                    eprintln!(
                        "[input] rel agg: {agg_n} ev Σdx={:+} Σdy={:+} counts in 2s",
                        agg_sum.0, agg_sum.1
                    );
                    agg_n = 0;
                    agg_sum = (0, 0);
                    agg_window = std::time::Instant::now();
                }
                let Some(r) = rel_ptr.as_ref() else { continue };
                r.emit(EV_REL, REL_X, dx);
                r.emit(EV_REL, REL_Y, dy);
                r.syn();
                last_abs = false;
            }
        }
    }
    eprintln!("[input] control channel closed; uinput sink stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn uapi_struct_sizes() {
        // must match the kernel's (64-bit Linux) layouts exactly: a wrong
        // ioctl size silently truncates
        assert_eq!(size_of::<UinputSetup>(), 92);
        assert_eq!(size_of::<InputAbsinfo>(), 24);
        assert_eq!(size_of::<UinputAbsSetup>(), 28);
        assert_eq!(size_of::<RawEvent>(), 24); // struct input_event
        assert_eq!(iow(UI_SET_EVBIT, 4), 0x40045564);
        assert_eq!(iow(UI_SET_KEYBIT, 4), 0x40045565);
        assert_eq!(iow(UI_SET_RELBIT, 4), 0x40045566);
        assert_eq!(iow(UI_SET_ABSBIT, 4), 0x40045567);
        assert_eq!(iow(UI_SET_PROPBIT, 4), 0x4004556E);
        assert_eq!(iow(UI_DEV_SETUP, size_of::<UinputSetup>() as u64), 0x405C5503);
        assert_eq!(
            iow(UI_ABS_SETUP, size_of::<UinputAbsSetup>() as u64),
            0x401C5504
        );
        assert_eq!(io(UI_DEV_CREATE), 0x5501);
        assert_eq!(io(UI_DEV_DESTROY), 0x5502);
    }
}
