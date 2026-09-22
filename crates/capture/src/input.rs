//! Input injection sinks, decoupled from capture sources.
//!
//! A capture source produces frames; an `InputSink` consumes client input
//! events. The server pairs them (`--source`/`--input`): QEMU D-Bus capture
//! injects into the guest console; DRM/KMS capture injects into the host
//! compositor via uinput — exactly the Sunshine split.

use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{mpsc, Arc};

use anyhow::{Context, Result};
use rdma_display_proto::{self as proto, InputEvent};

use crate::ScreenSize;

/// Consumes `InputEvent`s until the channel closes. `screen` carries the
/// current captured dimensions (updated by the server on every scanout);
/// absolute-position sinks need them for coordinate ranges. `rumble` is the
/// door from uinput force-feedback back to the client's physical pad — only
/// the uinput sink uses it; every other sink drops it.
pub trait InputSink: Send {
    fn name(&self) -> &'static str;
    fn spawn(
        self: Box<Self>,
        rx: mpsc::Receiver<InputEvent>,
        screen: Arc<ScreenSize>,
        rumble: rdma_display_proto::gamepad::RumbleHub,
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
        _rumble: rdma_display_proto::gamepad::RumbleHub,
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
                            // the QEMU console has no gamepad channel; pads are
                            // a uinput-backend feature
                            InputEvent::GamepadBtn { .. } | InputEvent::GamepadAxis { .. } => {
                                Ok(())
                            }
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
fn iowr(nr: u16, size: u64) -> u64 {
    (3 << 30) | (size << 16) | (U << 8) | nr as u64
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
const UI_SET_FFBIT: u16 = 107; // _IOW('U', 107, unsigned int)
const UI_SET_PROPBIT: u16 = 110; // _IOW('U', 110, unsigned int)
// force-feedback handshake (linux/uinput.h); requests arrive as EV_UINPUT
// (0x0101) events on the uinput fd, `value` = request id
const EV_UINPUT: u16 = 0x0101;
const UI_FF_UPLOAD: u16 = 1;
const UI_FF_ERASE: u16 = 2;
const UI_BEGIN_FF_UPLOAD: u16 = 200; // _IOWR('U', 200, uinput_ff_upload)
const UI_END_FF_UPLOAD: u16 = 201; // _IOW('U', 201, uinput_ff_upload)
const UI_BEGIN_FF_ERASE: u16 = 202; // _IOWR('U', 202, uinput_ff_erase)
const UI_END_FF_ERASE: u16 = 203; // _IOW('U', 203, uinput_ff_erase)
const FF_RUMBLE: u16 = 0x50; // linux/input.h FF_RUMBLE (effect type)
const FF_GAIN: u16 = 0x60;
const EV_FF: u16 = 0x15;

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

/// `struct ff_effect` (linux/input.h, 64-bit): header + a 32-byte union
/// (the largest member is ff_periodic_effect; the pointer member gives the
/// whole struct 8-byte alignment, hence the explicit align).
#[repr(C, align(8))]
struct FfEffect {
    ty: u16,
    id: i16,
    direction: u16,
    trigger_button: u16,
    trigger_interval: u16,
    replay_length: u16, // ms
    replay_delay: u16,  // ms
    u: [u8; 32],        // union, opaque
}

/// `struct uinput_ff_upload`
#[repr(C)]
struct UinputFfUpload {
    request_id: u32,
    retval: i32,
    effect: FfEffect,
    old: FfEffect,
}

/// `struct uinput_ff_erase`
#[repr(C)]
struct UinputFfErase {
    request_id: u32,
    retval: i32,
    effect_id: u32,
}

struct UinputDev {
    fd: OwnedFd,
}

impl UinputDev {
    fn setup(self, name: &str, ff_effects_max: u32, bus: u16, vendor: u16, product: u16, version: u16) -> Result<Self> {
        let mut s = UinputSetup {
            bustype: bus,
            vendor,
            product,
            version,
            name: [0; 80],
            ff_effects_max,
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
    fn set_ffbit(&self, v: u32) -> Result<()> {
        unsafe { ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_FFBIT, 4), v) }
            .map(|_| ())
            .context("UI_SET_FFBIT")
    }
    fn set_absinfo(&self, code: u16, min: i32, max: i32, fuzz: i32, flat: i32, resolution: i32) -> Result<()> {
        unsafe {
            ioctl_u32(self.fd.as_raw_fd(), iow(UI_SET_ABSBIT, 4), code as u32)
                .map(|_| ())
                .context("UI_SET_ABSBIT")?;
        }
        let s = UinputAbsSetup {
            code,
            absinfo: InputAbsinfo {
                value: 0,
                minimum: min,
                maximum: max,
                fuzz,
                flat,
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
        rumble: rdma_display_proto::gamepad::RumbleHub,
    ) -> Result<()> {
        // gamepad routing: a dedicated FF-capable thread polls the virtual
        // pad's uinput fd (kernel rumble requests) alongside the eventfd
        // that wakes it when gamepad events are queued
        let (pad_tx, pad_rx) = mpsc::channel::<InputEvent>();
        let wake = eventfd_new().context("eventfd for gamepad wake")?;
        let wake_w = wake.try_clone().context("clone eventfd")?;
        std::thread::Builder::new()
            .name("input-gamepad".into())
            .spawn(move || gamepad_loop(pad_rx, rumble, wake))
            .context("spawn gamepad thread")?;
        std::thread::Builder::new()
            .name("input-uinput".into())
            .spawn(move || {
                if let Err(e) = uinput_loop(rx, screen, pad_tx, wake_w.as_raw_fd()) {
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
    let d = UinputDev { fd: open_uinput(false)? }
        .setup("rdma-display keyboard", 0, BUS_VIRTUAL, 0x1ab0, 0x5250, 1)?;
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
    let d = UinputDev { fd: open_uinput(false)? }
        .setup("rdma-display mouse", 0, BUS_VIRTUAL, 0x1ab0, 0x5250, 1)?;
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
    let d = UinputDev { fd: open_uinput(false)? }
        .setup("rdma-display mouse (absolute)", 0, BUS_VIRTUAL, 0x1ab0, 0x5250, 1)?;
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
    d.set_absinfo(ABS_X, 0, ABS_MAX_W, 1, 0, 28)?;
    d.set_absinfo(ABS_Y, 0, ABS_MAX_H, 1, 0, 28)?;
    d.create()
}

/// Virtual gamepad — deliberately indistinguishable from a REAL Xbox One pad
/// on Linux (xpad driver, Xbox One identity per inputtino's XboxOneJoypad
/// defaults): USB bus, VID 0x045E / PID 0x02EA / version 0x0408, the exact
/// 11-button keybit set (south/east/north/west/tl/tr/select/start/mode/
/// thumbl/thumbr), sticks ±32767 fuzz 16 flat 128, triggers 0..255, dpad via
/// ABS_HAT0X/Y. Faithfulness is load-bearing:
/// - the kernel orders evdev button indices by keybit code, so ANY extra bit
///   (BTN_C/BTN_Z/BTN_TL2/BTN_TR2/BTN_DPAD_*) shifts indices and desyncs
///   every game calibrated against real hardware — observed live: an
///   Xbox-"press X" prompt triggering on the top (north) button instead of
///   the left (west);
/// - SDL's GameControllerDB matches by (bus, vid, pid, version), so the Xbox
///   identity is what makes games load the proper Xbox mapping.
/// Controllers whose driver reports extra codes (e.g. the DualSense's BTN_TL2
/// for the L2 latch) simply have those events dropped by the kernel on the
/// virtual pad — the analog axes carry the same input. FF via the pad fd's
/// rumble handshake, relayed to the client's physical pad.
fn uinput_gamepad() -> Result<UinputDev> {
    const BUS_USB: u16 = 0x03;
    let d = UinputDev { fd: open_uinput(true)? }.setup(
        "rdma-display X-Box One (virtual) pad",
        4,
        BUS_USB,
        0x045e,
        0x02ea,
        0x0408,
    )?;
    d.set_evbit(EV_SYN)?;
    d.set_evbit(EV_KEY)?;
    d.set_evbit(EV_ABS)?;
    d.set_evbit(EV_FF)?;
    for b in [
        proto::gamepad::BTN_SOUTH,
        proto::gamepad::BTN_EAST,
        proto::gamepad::BTN_NORTH,
        proto::gamepad::BTN_WEST,
        proto::gamepad::BTN_TL,
        proto::gamepad::BTN_TR,
        proto::gamepad::BTN_SELECT,
        proto::gamepad::BTN_START,
        proto::gamepad::BTN_MODE,
        proto::gamepad::BTN_THUMBL,
        proto::gamepad::BTN_THUMBR,
    ] {
        // NOTE: a failed keybit makes the kernel silently drop that button's
        // events (input core filters EV_KEY by keybit) — so failures are
        // logged, not swallowed
        if let Err(e) = d.set_keybit(b) {
            eprintln!("[input] gamepad UI_SET_KEYBIT({b:#x}) failed: {e}");
        }
    }
    d.set_absinfo(proto::gamepad::ABS_X as u16, -32768, 32767, 16, 128, 0)?;
    d.set_absinfo(proto::gamepad::ABS_Y as u16, -32768, 32767, 16, 128, 0)?;
    d.set_absinfo(proto::gamepad::ABS_RX as u16, -32768, 32767, 16, 128, 0)?;
    d.set_absinfo(proto::gamepad::ABS_RY as u16, -32768, 32767, 16, 128, 0)?;
    d.set_absinfo(proto::gamepad::ABS_Z as u16, 0, 255, 0, 0, 0)?;
    d.set_absinfo(proto::gamepad::ABS_RZ as u16, 0, 255, 0, 0, 0)?;
    d.set_absinfo(proto::gamepad::ABS_HAT0X as u16, -1, 1, 0, 0, 0)?;
    d.set_absinfo(proto::gamepad::ABS_HAT0Y as u16, -1, 1, 0, 0, 0)?;
    d.set_ffbit(FF_RUMBLE as u32)?;
    d.set_ffbit(FF_GAIN as u32)?;
    d.create()
}

fn open_uinput(rw: bool) -> Result<OwnedFd> {
    use std::os::fd::FromRawFd;
    // SAFETY: NUL-terminated static path; fd < 0 checked before wrapping
    // O_RDWR|O_NONBLOCK for the gamepad: its fd is polled for kernel FF
    // requests (reads fail with EBADF on an O_WRONLY fd), and drained
    // nonblocking after each poll
    let flags = if rw {
        libc::O_RDWR | libc::O_CLOEXEC | libc::O_NONBLOCK
    } else {
        libc::O_WRONLY | libc::O_CLOEXEC
    };
    let fd = unsafe { libc::open(c"/dev/uinput".as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).context("open /dev/uinput (needs root)");
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// eventfd wake pair: the pad thread polls it alongside the pad fds so gamepad
/// events forwarded from `uinput_loop` wake its poll immediately (sticks at
/// per-event latency; no timeout polling).
fn eventfd_new() -> Result<OwnedFd> {
    use std::os::fd::FromRawFd;
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).context("eventfd");
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn eventfd_write(fd: i32) {
    let one: u64 = 1;
    unsafe {
        libc::write(fd, &one as *const u64 as *const libc::c_void, 8);
    }
}

fn eventfd_drain(fd: i32) {
    let mut buf = [0u8; 8];
    loop {
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, 8) };
        if n <= 0 {
            break; // EAGAIN when drained
        }
    }
}

/// The FF-capable gamepad thread: gamepad `InputEvent`s arrive via the
/// channel (eventfd wake), forwarded as uinput emits; the virtual pad's fd
/// (O_RDWR|O_NONBLOCK) is polled for kernel force-feedback requests —
/// upload/erase are completed via the UI_BEGIN/END handshake and relayed to
/// the client through the rumble hub. One virtual pad per slot, created
/// lazily on the slot's first event.
fn gamepad_loop(
    rx: mpsc::Receiver<InputEvent>,
    rumble: rdma_display_proto::gamepad::RumbleHub,
    wake: OwnedFd,
) {
    let mut pads: std::collections::HashMap<u32, UinputDev> = std::collections::HashMap::new();
    // slot 0 eager: the pad exists as soon as the server starts, so host-side
    // games and evtest see it before any client connects (a lazy pad also
    // only appears after the first event, which reads as "the gamepad
    // vanished" between sessions). Higher slots stay lazy.
    match uinput_gamepad() {
        Ok(d) => {
            eprintln!("[input] virtual gamepad slot 0 ready (eager)");
            pads.insert(0, d);
        }
        Err(e) => eprintln!("[input] eager uinput gamepad create failed (events will retry lazily): {e:#}"),
    }
    loop {
        // poll set: [0] = eventfd, [1..] = one entry per live pad
        let set: Vec<(u32, i32)> = pads
            .iter()
            .map(|(slot, d)| (*slot, d.fd.as_raw_fd()))
            .collect();
        let mut fds: Vec<libc::pollfd> = Vec::with_capacity(1 + set.len());
        fds.push(libc::pollfd { fd: wake.as_raw_fd(), events: libc::POLLIN, revents: 0 });
        for (_, fd) in &set {
            fds.push(libc::pollfd { fd: *fd, events: libc::POLLIN, revents: 0 });
        }
        let n = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) };
        if n <= 0 {
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if errno_is(&e, libc::EINTR) {
                    continue;
                }
                eprintln!("[input] gamepad poll failed: {e}");
            }
            continue;
        }
        if fds[0].revents & libc::POLLIN != 0 {
            eventfd_drain(wake.as_raw_fd());
            while let Ok(ev) = rx.try_recv() {
                let (slot, code, down, axis, value) = match ev {
                    InputEvent::GamepadBtn { slot, code, down } => (slot, code, down, false, 0),
                    InputEvent::GamepadAxis { slot, code, value } => (slot, code, false, true, value),
                    _ => continue,
                };
                let dev = match pads.get_mut(&slot) {
                    Some(d) => d,
                    None => match uinput_gamepad() {
                        Ok(d) => {
                            eprintln!("[input] virtual gamepad slot {slot} created");
                            pads.entry(slot).or_insert(d)
                        }
                        Err(e) => {
                            eprintln!("[input] uinput gamepad create failed: {e:#}");
                            continue;
                        }
                    },
                };
                let code = code as u16;
                if axis {
                    dev.emit(EV_ABS, code, value);
                } else {
                    dev.emit(EV_KEY, code, down as i32);
                }
                dev.syn();
            }
        }
        // kernel FF requests from the host game on each live pad
        for (i, (slot, fd)) in set.iter().enumerate() {
            if fds[i + 1].revents & libc::POLLIN != 0 {
                read_ff_events(*slot, *fd, &rumble);
            }
        }
    }
}

fn errno_is(e: &std::io::Error, raw: i32) -> bool {
    e.raw_os_error() == Some(raw)
}

/// Drain the pad fd (O_NONBLOCK: reads until EAGAIN) and complete any FF
/// requests. Playback notifications (`EV_FF <effect_id> 0/1`) and
/// upload/erase requests are relayed to the client; FF_GAIN is ignored (the
/// client's physical pad receives raw magnitudes).
fn read_ff_events(slot: u32, fd: i32, rumble: &rdma_display_proto::gamepad::RumbleHub) {
    let mut buf = [0u8; 24 * 16];
    loop {
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n <= 0 {
            return; // EAGAIN (drained) or error — back to poll
        }
        let count = n as usize / 24;
        for i in 0..count {
            let off = i * 24;
            // struct input_event: sec i64, usec i64, type u16, code u16, value i32
            let ty = u16::from_le_bytes(buf[off + 16..off + 18].try_into().unwrap());
            let code = u16::from_le_bytes(buf[off + 18..off + 20].try_into().unwrap());
            let value = i32::from_le_bytes(buf[off + 20..off + 24].try_into().unwrap());
            match (ty, code) {
                (EV_UINPUT, UI_FF_UPLOAD) => ff_upload(slot, fd, value as u32, rumble),
                (EV_UINPUT, UI_FF_ERASE) => ff_erase(slot, fd, value as u32, rumble),
                (EV_FF, FF_GAIN) => {} // per-controller gain scaling not forwarded
                (EV_FF, _) => {
                    rumble.send(rdma_display_proto::gamepad::Rumble {
                        slot,
                        id: code,
                        strong: 0,
                        weak: 0,
                        duration_ms: 0,
                        action: if value != 0 {
                            rdma_display_proto::gamepad::RUMBLE_PLAY
                        } else {
                            rdma_display_proto::gamepad::RUMBLE_STOP
                        },
                    });
                }
                _ => {}
            }
        }
    }
}

/// Complete one FF upload request: BEGIN fetches the effect, the rumble part
/// goes to the client, END (retval 0) lets the kernel finish the game's
/// EVIOCSFF call. A failed BEGIN still gets an END (retval -1) so the
/// requesting game never hangs.
fn ff_upload(
    slot: u32,
    fd: i32,
    request_id: u32,
    rumble: &rdma_display_proto::gamepad::RumbleHub,
) {
    let mut up = UinputFfUpload {
        request_id,
        retval: 0,
        effect: zeroed_ff(),
        old: zeroed_ff(),
    };
    let begin = unsafe {
        ioctl_u64(fd, iowr(UI_BEGIN_FF_UPLOAD, 104), &mut up as *mut _ as usize).is_ok()
    };
    if begin && up.effect.ty == FF_RUMBLE {
        let strong = u16::from_le_bytes(up.effect.u[0..2].try_into().unwrap());
        let weak = u16::from_le_bytes(up.effect.u[2..4].try_into().unwrap());
        rumble.send(rdma_display_proto::gamepad::Rumble {
            slot,
            id: up.effect.id as u16,
            strong,
            weak,
            duration_ms: up.effect.replay_length,
            action: rdma_display_proto::gamepad::RUMBLE_UPLOAD,
        });
    }
    up.retval = if begin { 0 } else { -1 };
    let _ = unsafe { ioctl_u64(fd, iow(UI_END_FF_UPLOAD, 104), &up as *const _ as usize) };
}

/// Same handshake for erase (the game releasing an effect id).
fn ff_erase(
    slot: u32,
    fd: i32,
    request_id: u32,
    rumble: &rdma_display_proto::gamepad::RumbleHub,
) {
    let mut er = UinputFfErase { request_id, retval: 0, effect_id: 0 };
    let begin =
        unsafe { ioctl_u64(fd, iowr(UI_BEGIN_FF_ERASE, 12), &mut er as *mut _ as usize).is_ok() };
    if begin {
        rumble.send(rdma_display_proto::gamepad::Rumble {
            slot,
            id: er.effect_id as u16,
            strong: 0,
            weak: 0,
            duration_ms: 0,
            action: rdma_display_proto::gamepad::RUMBLE_ERASE,
        });
    }
    er.retval = if begin { 0 } else { -1 };
    let _ = unsafe { ioctl_u64(fd, iow(UI_END_FF_ERASE, 12), &er as *const _ as usize) };
}

fn zeroed_ff() -> FfEffect {
    // SAFETY: all-zero bytes are a valid (zero) ff_effect
    unsafe { std::mem::zeroed() }
}

fn uinput_loop(
    rx: mpsc::Receiver<InputEvent>,
    screen: Arc<ScreenSize>,
    pad_tx: mpsc::Sender<InputEvent>,
    pad_wake_fd: i32,
) -> Result<()> {
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
                eprintln!("[input] key {code:#04x} down={down}");
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
            ev @ (InputEvent::GamepadBtn { .. } | InputEvent::GamepadAxis { .. }) => {
                if pad_tx.send(ev).is_ok() {
                    eventfd_write(pad_wake_fd);
                }
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
        // force-feedback handshake structs (verified against linux/input.h)
        assert_eq!(size_of::<FfEffect>(), 48); // struct ff_effect
        assert_eq!(size_of::<UinputFfUpload>(), 104); // uinput_ff_upload
        assert_eq!(size_of::<UinputFfErase>(), 12); // uinput_ff_erase
        assert_eq!(iow(UI_SET_EVBIT, 4), 0x40045564);
        assert_eq!(iow(UI_SET_KEYBIT, 4), 0x40045565);
        assert_eq!(iow(UI_SET_RELBIT, 4), 0x40045566);
        assert_eq!(iow(UI_SET_ABSBIT, 4), 0x40045567);
        assert_eq!(iow(UI_SET_FFBIT, 4), 0x4004556b);
        assert_eq!(iow(UI_SET_PROPBIT, 4), 0x4004556E);
        assert_eq!(iowr(UI_BEGIN_FF_UPLOAD, 104), 0xC06855C8);
        assert_eq!(iow(UI_END_FF_UPLOAD, 104), 0x406855C9);
        assert_eq!(iowr(UI_BEGIN_FF_ERASE, 12), 0xC00C55CA);
        assert_eq!(iow(UI_END_FF_ERASE, 12), 0x400C55CB);
        assert_eq!(iow(UI_DEV_SETUP, size_of::<UinputSetup>() as u64), 0x405C5503);
        assert_eq!(
            iow(UI_ABS_SETUP, size_of::<UinputAbsSetup>() as u64),
            0x401C5504
        );
        assert_eq!(io(UI_DEV_CREATE), 0x5501);
        assert_eq!(io(UI_DEV_DESTROY), 0x5502);
    }
}
