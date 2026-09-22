//! Client-side controller capture: raw evdev reads on /dev/input/event*,
//! forwarded as `InputEvent::Gamepad*`, plus rumble playback in the reverse
//! direction (host-game FF requests arrive on the control channel and are
//! replayed on the physical pad).
//!
//! Wayland has no gamepad protocol, so this bypasses the compositor entirely
//! — the pad is read straight from the kernel. The kernel gamepad standard
//! (Documentation/input/gamepad.rst) covers every mainstream driver (xpad,
//! hid-sony/hid-playstation, hid-nintendo): codes are forwarded verbatim, and
//! axis values are rescaled from the real device's EVIOCGABS ranges into the
//! virtual pad's fixed ranges (see proto::gamepad) so the host game sees sane
//! axis semantics regardless of the physical controller.
//!
//! The pad is grabbed exclusively (EVIOCGRAB) by default so the local
//! compositor doesn't double-consume it; the kernel releases the grab
//! automatically when the fd closes, so a crashed client can't strand it.
//! Needs the client user in the `input` group (read for capture, write for
//! rumble).

use std::collections::{HashMap, HashSet};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::{mpsc, Arc, Mutex};

use rdma_display_proto as proto;

// linux/input-event-codes.h
const EV_KEY: u32 = 0x01;
const EV_ABS: u32 = 0x03;
const EV_FF: u32 = 0x15;
const FF_RUMBLE: u16 = 0x50;
const KEY_MAX: u32 = 0x2ff;

/// `struct input_event` on 64-bit Linux
#[repr(C)]
#[derive(Clone, Copy)]
struct Ev {
    sec: i64,
    usec: i64,
    ty: u16,
    code: u16,
    value: i32,
}

/// `struct ff_effect` — layout must match linux/input.h (48 bytes; the
/// pointer member of ff_periodic_effect forces 8-byte alignment)
#[repr(C, align(8))]
struct FfEffect {
    ty: u16,
    id: i16,
    direction: u16,
    trigger_button: u16,
    trigger_interval: u16,
    replay_length: u16, // ms
    replay_delay: u16,
    u: [u8; 32],
}

/// `struct input_absinfo`
#[repr(C)]
#[derive(Clone, Copy)]
struct AbsInfo {
    value: i32,
    minimum: i32,
    maximum: i32,
    fuzz: i32,
    flat: i32,
    resolution: i32,
}

// 'E' ioctls (linux/input.h), built the same way as the uinput ones
const E: u64 = 'E' as u64;
fn ev_ior(nr: u16, size: u64) -> u64 {
    (2 << 30) | (size << 16) | (E << 8) | nr as u64
}
const EVIOCGRAB: u64 = 0x4004_4590; // _IOW('E', 0x90, int)
const EVIOCSFF: u64 = 0x4030_4580; // _IOW('E', 0x80, struct ff_effect)
const EVIOCRMFF: u64 = 0x4004_4581; // _IOW('E', 0x81, int)

fn eviocgabs(code: u32) -> u64 {
    ev_ior(0x40 + code as u16, 24)
}

// ---------------------------------------------------------------------------

/// Start the controller subsystem. Returns the sender the client's control
/// reader feeds `Rumble` events into. `grab` = EVIOCGRAB the pad exclusively;
/// `fake` additionally synthesizes a controller (slot 0) for testing without
/// hardware.
pub fn start(
    input_tx: mpsc::Sender<rdma_display_proto::InputEvent>,
    grab: bool,
    fake: bool,
) -> mpsc::Sender<proto::gamepad::Rumble> {
    let (rumble_tx, rumble_rx) = mpsc::channel::<proto::gamepad::Rumble>();
    // pad fds as seen by the rumble thread (only O_RDWR pads are registered —
    // read-only fallback pads can't receive FF effects)
    let pads: Arc<Mutex<HashMap<u32, OwnedFd>>> = Arc::new(Mutex::new(HashMap::new()));

    let pads_r = pads.clone();
    std::thread::Builder::new()
        .name("gamepad-rumble".into())
        .spawn(move || rumble_loop(rumble_rx, pads_r))
        .expect("spawn gamepad rumble thread");

    std::thread::Builder::new()
        .name("gamepad-scan".into())
        .spawn(move || {
            if fake {
                spawn_fake(input_tx.clone());
            }
            manager_loop(input_tx, grab, pads);
        })
        .expect("spawn gamepad manager");

    rumble_tx
}

/// Attach one evdev node if it is gamepad-capable (EV_KEY bitmap has
/// BTN_GAMEPAD — the kernel's "this is a gamepad" bit). Returns the slot.
/// `retries` > 0 gives udev time to apply the `input` group after a hotplug;
/// the initial scan passes 1 (fail fast — an unreadable /dev/input should
/// never stall client startup).
fn try_attach(
    path: &std::path::Path,
    grab: bool,
    retries: u32,
    input_tx: &mpsc::Sender<rdma_display_proto::InputEvent>,
    pads: &Arc<Mutex<HashMap<u32, OwnedFd>>>,
    next_slot: &mut u32,
) -> Option<u32> {
    // O_RDWR so rumble works; fall back to read-only capture
    static EACCES_HINT: std::sync::Once = std::sync::Once::new();
    let mut fd: Option<OwnedFd> = None;
    let mut can_rumble = false;
    for attempt in 0..=retries {
        if let Some(f) = open_evdev(path, true) {
            fd = Some(f);
            can_rumble = true;
            break;
        }
        let rw_err = std::io::Error::last_os_error();
        if let Some(f) = open_evdev(path, false) {
            fd = Some(f);
            break;
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EACCES) {
            EACCES_HINT.call_once(|| {
                eprintln!(
                    "[gamepad] cannot open {path:?} ({rw_err}) — the client user needs read+write access to /dev/input/event* (add to the `input` group); capture disabled for it",
                    rw_err = rw_err
                );
            });
        }
        if attempt == retries {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
    }
    let fd = fd?;

    // gamepad check: the EV_KEY bitmap must include BTN_GAMEPAD
    let mut keybits = vec![0u8; (KEY_MAX as usize + 8) / 8];
    let r = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            ev_ior(0x21, keybits.len() as u64) as libc::c_ulong,
            keybits.as_mut_ptr(),
        )
    };
    if r < 0 || !bit_test(&keybits, proto::gamepad::BTN_GAMEPAD) {
        return None;
    }

    let name = device_name(fd.as_raw_fd()).unwrap_or_else(|| "gamepad".to_string());
    if !can_rumble {
        eprintln!("[gamepad] {path:?} opened read-only (rumble unavailable; write access to /dev/input/event* needs the `input` group)");
    }

    if grab {
        let r = unsafe { libc::ioctl(fd.as_raw_fd(), EVIOCGRAB as libc::c_ulong, 1i32) };
        if r < 0 {
            eprintln!(
                "[gamepad] EVIOCGRAB failed on {path:?} ({}) — the local compositor will also see this pad",
                std::io::Error::last_os_error()
            );
        }
    }

    let slot = *next_slot;
    *next_slot += 1;
    if can_rumble {
        let rumble_fd = fd.try_clone().expect("clone evdev fd for rumble");
        pads.lock().unwrap().insert(slot, rumble_fd);
    }
    eprintln!("[gamepad] slot {slot}: {name:?} at {path:?}");

    let input_tx = input_tx.clone();
    let slot_name = name;
    std::thread::Builder::new()
        .name("gamepad-read".into())
        .spawn(move || reader_loop(fd, slot, slot_name, input_tx))
        .expect("spawn gamepad reader");
    Some(slot)
}

/// Initial scan + hotplug: watch /dev/input via inotify (IN_CREATE) and
/// attach every new gamepad-capable node. Detached pads die on their own
/// (the reader thread's read fails with ENODEV); their rumble entries just
/// stop being used.
fn manager_loop(
    input_tx: mpsc::Sender<rdma_display_proto::InputEvent>,
    grab: bool,
    pads: Arc<Mutex<HashMap<u32, OwnedFd>>>,
) {
    let mut next_slot: u32 = 0;
    let mut attached: HashSet<String> = HashSet::new();
    scan_once(&input_tx, grab, &pads, &mut next_slot, &mut attached);

    let watch = inotify_start();
    if watch.is_none() {
        // no inotify: fall back to a 1s rescan (cheap; ~10 open/close per s)
        eprintln!("[gamepad] inotify unavailable; scanning /dev/input every 1s");
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            scan_once(&input_tx, grab, &pads, &mut next_slot, &mut attached);
        }
    }
    let watch = watch.unwrap();
    let mut buf = [0u8; 4096];
    loop {
        let n = unsafe { libc::read(watch.as_raw_fd(), buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n <= 0 {
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
            continue;
        }
        // struct inotify_event: int wd, unsigned mask, unsigned cookie,
        // unsigned len, char name[len]
        let mut off = 0usize;
        while off + 16 <= n as usize {
            let mask = u32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap());
            let len = u32::from_le_bytes(buf[off + 12..off + 16].try_into().unwrap()) as usize;
            let name = String::from_utf8_lossy(&buf[off + 16..off + 16 + len])
                .trim_end_matches('\0')
                .to_string();
            off += 16 + len;
            if mask & (libc::IN_CREATE as u32) != 0 && name.starts_with("event") {
                let path = std::path::Path::new("/dev/input").join(&name);
                if attached.insert(path.display().to_string()) {
                    try_attach(&path, grab, 4, &input_tx, &pads, &mut next_slot);
                }
            }
        }
    }
}

fn scan_once(
    input_tx: &mpsc::Sender<rdma_display_proto::InputEvent>,
    grab: bool,
    pads: &Arc<Mutex<HashMap<u32, OwnedFd>>>,
    next_slot: &mut u32,
    attached: &mut HashSet<String>,
) {
    let mut nodes: Vec<String> = match std::fs::read_dir("/dev/input") {
        Ok(d) => d
            .filter_map(|e| e.ok())
            .map(|e| e.path().display().to_string())
            .filter(|p| {
                p.rsplit('/').next().map_or(false, |n| n.starts_with("event"))
            })
            .collect(),
        Err(_) => return,
    };
    nodes.sort();
    for p in nodes {
        if attached.insert(p.clone()) {
            try_attach(std::path::Path::new(&p), grab, 1, input_tx, pads, next_slot);
        }
    }
}

fn inotify_start() -> Option<OwnedFd> {
    let fd = unsafe { libc::inotify_init1(libc::IN_CLOEXEC) };
    if fd < 0 {
        return None;
    }
    let r = unsafe {
        libc::inotify_add_watch(
            fd,
            c"/dev/input".as_ptr(),
            (libc::IN_CREATE | libc::IN_DELETE) as u32,
        )
    };
    if r < 0 {
        unsafe { libc::close(fd) };
        None
    } else {
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

fn open_evdev(path: &std::path::Path, rw: bool) -> Option<OwnedFd> {
    use std::os::unix::ffi::OsStrExt;
    let mut cpath = path.as_os_str().as_bytes().to_vec();
    cpath.push(0);
    let flags = if rw {
        libc::O_RDWR | libc::O_CLOEXEC
    } else {
        libc::O_RDONLY | libc::O_CLOEXEC
    };
    let fd = unsafe { libc::open(cpath.as_ptr() as *const libc::c_char, flags) };
    if fd < 0 {
        None
    } else {
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

fn bit_test(bits: &[u8], code: u32) -> bool {
    let (byte, bit) = ((code / 8) as usize, (code % 8) as u8);
    bits.get(byte).map_or(false, |b| (b >> bit) & 1 != 0)
}

fn device_name(fd: i32) -> Option<String> {
    // EVIOCGNAME(0): variable-length _IOC, size 256
    let req = (2u64 << 30) | (256 << 16) | ('E' as u64) << 8 | 0x06;
    let mut buf = vec![0u8; 256];
    let r = unsafe { libc::ioctl(fd, req as libc::c_ulong, buf.as_mut_ptr()) };
    if r <= 0 {
        None
    } else {
        let n = r as usize;
        let s = String::from_utf8_lossy(&buf[..n.min(256)]);
        Some(s.trim_end_matches('\0').to_string())
    }
}

/// Per-pad reader: emit events into `input_tx`, only on actual changes
/// (evdev resends axis values; sticks jitter). Values are rescaled from the
/// real device's absinfo range into the virtual pad's range.
fn reader_loop(
    fd: OwnedFd,
    slot: u32,
    name: String,
    input_tx: mpsc::Sender<rdma_display_proto::InputEvent>,
) {
    // probe the real device's axis ranges once (missing axes stay unforwarded)
    let mut ranges: HashMap<u32, (i32, i32)> = HashMap::new();
    for code in [proto::gamepad::ABS_X, proto::gamepad::ABS_Y, proto::gamepad::ABS_RX, proto::gamepad::ABS_RY, proto::gamepad::ABS_Z, proto::gamepad::ABS_RZ, proto::gamepad::ABS_HAT0X, proto::gamepad::ABS_HAT0Y] {
        let mut ai = AbsInfo { value: 0, minimum: 0, maximum: 0, fuzz: 0, flat: 0, resolution: 0 };
        if unsafe { libc::ioctl(fd.as_raw_fd(), eviocgabs(code) as libc::c_ulong, &mut ai as *mut _ as *mut libc::c_void) } >= 0 {
            ranges.insert(code, (ai.minimum, ai.maximum));
        }
    }
    eprintln!("[gamepad] slot {slot}: forwarding ({name:?}, {} axes)", ranges.len());

    let mut btn_down: HashSet<u32> = HashSet::new();
    let mut axis_last: HashMap<u32, i32> = HashMap::new();
    let mut buf = vec![0u8; 24 * 16];

    loop {
        let n = unsafe { libc::read(fd.as_raw_fd(), buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n <= 0 {
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.raw_os_error() == Some(libc::EAGAIN) || e.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
            }
            eprintln!("[gamepad] slot {slot}: pad gone");
            return;
        }
        for ev in buf[..n as usize].chunks_exact(24) {
            let e = Ev {
                sec: i64::from_le_bytes(ev[0..8].try_into().unwrap()),
                usec: i64::from_le_bytes(ev[8..16].try_into().unwrap()),
                ty: u16::from_le_bytes(ev[16..18].try_into().unwrap()),
                code: u16::from_le_bytes(ev[18..20].try_into().unwrap()),
                value: i32::from_le_bytes(ev[20..24].try_into().unwrap()),
            };
            let (ty, code) = (e.ty as u32, e.code as u32);
            if ty == EV_KEY && proto::gamepad::is_button(code) {
                let down = e.value != 0;
                // forward only on state change (some drivers resend states).
                // NOTE: HashSet::insert returns true iff the value was NEWLY
                // present — the true branch is "newly pressed", not "already
                // down"; inverting it drops every first press (seen live:
                // DS5 ABXY dead while sticks flowed).
                if down {
                    if btn_down.insert(code) {
                        let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadBtn {
                            slot,
                            code,
                            down: true,
                        });
                    }
                    // already down: no change
                } else if btn_down.remove(&code) {
                    let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadBtn {
                        slot,
                        code,
                        down: false,
                    });
                }
            } else if ty == EV_ABS && proto::gamepad::is_axis(code) {
                let Some(real) = ranges.get(&code) else { continue };
                let virt = proto::gamepad::virtual_range(code);
                let v = proto::gamepad::rescale(e.value, *real, virt);
                if axis_last.get(&code) != Some(&v) {
                    axis_last.insert(code, v);
                    let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadAxis {
                        slot,
                        code,
                        value: v,
                    });
                }
            }
        }
    }
}

/// The client's rumble player: server-side FF events replay on the physical
/// pad. Effect ids are the SERVER's virtual-pad kernel ids; they map to local
/// effect ids uploaded here once per (slot, kernel id) and played/erased
/// after that.
fn rumble_loop(
    rx: mpsc::Receiver<proto::gamepad::Rumble>,
    pads: Arc<Mutex<HashMap<u32, OwnedFd>>>,
) {
    // (slot, kernel effect id) → local effect id
    let mut effects: HashMap<(u32, u16), i16> = HashMap::new();
    let mut warned: HashSet<u32> = HashSet::new();
    while let Ok(r) = rx.recv() {
        let fd = pads.lock().unwrap().get(&r.slot).map(|f| f.as_raw_fd());
        let Some(fd) = fd else {
            if warned.insert(r.slot) {
                eprintln!("[gamepad] rumble for slot {} dropped: pad has no write access", r.slot);
            }
            continue;
        };
        match r.action {
            proto::gamepad::RUMBLE_UPLOAD => {
                let mut eff: FfEffect = unsafe { std::mem::zeroed() };
                eff.ty = FF_RUMBLE;
                eff.id = -1; // kernel assigns
                eff.direction = 0;
                eff.replay_length = r.duration_ms;
                eff.u[0..2].copy_from_slice(&r.strong.to_le_bytes());
                eff.u[2..4].copy_from_slice(&r.weak.to_le_bytes());
                match unsafe {
                    libc::ioctl(
                        fd,
                        EVIOCSFF as libc::c_ulong,
                        &mut eff as *mut FfEffect as *mut libc::c_void,
                    )
                } {
                    0 => {
                        effects.insert((r.slot, r.id), eff.id);
                    }
                    _ => eprintln!("[gamepad] EVIOCSFF failed: {}", std::io::Error::last_os_error()),
                }
            }
            proto::gamepad::RUMBLE_PLAY | proto::gamepad::RUMBLE_STOP => {
                if let Some(local) = effects.get(&(r.slot, r.id)) {
                    let play_ev = Ev { sec: 0, usec: 0, ty: EV_FF as u16, code: *local as u16, value: i32::from(r.action == proto::gamepad::RUMBLE_PLAY) };
                    write_ev(fd, &play_ev);
                }
            }
            proto::gamepad::RUMBLE_ERASE => {
                if let Some(local) = effects.remove(&(r.slot, r.id)) {
                    let id = local as i32;
                    unsafe { libc::ioctl(fd, EVIOCRMFF as libc::c_ulong, id) };
                }
            }
            _ => {}
        }
    }
}

fn write_ev(fd: i32, e: &Ev) {
    let buf = unsafe { std::slice::from_raw_parts(e as *const Ev as *const u8, 24) };
    let _ = unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len()) };
}

/// Test mode without hardware: slot 0 traces a stick circle, taps A, and
/// sweeps the left trigger, so the full path (evdev-free) can be verified
/// with `evtest` on the host watching the virtual pad.
fn spawn_fake(input_tx: mpsc::Sender<rdma_display_proto::InputEvent>) {
    eprintln!("[gamepad] --fake-gamepad: synthesizing a controller on slot 0");
    std::thread::Builder::new()
        .name("gamepad-fake".into())
        .spawn(move || fake_loop(input_tx))
        .expect("spawn fake gamepad");
}

fn fake_loop(input_tx: mpsc::Sender<rdma_display_proto::InputEvent>) {
    let slot = 0u32;
    // DualSense-style ranges (what the reader would rescale into the
    // virtual pad): sticks ±32767, triggers 0..1023
    let mut t = 0.0f64;
    let mut btn_state = false;
    let hat_seq: [(i32, i32); 5] = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)];
    let mut sent_axis: HashMap<u32, i32> = HashMap::new();
    let mut sent_hat = (0i32, 0i32);
    loop {
        // left stick: circle at 70% radius, right stick stationary
        let x = (t * 0.8).sin() * 0.7 * 32767.0;
        let y = (t * 0.8).cos() * 0.7 * 32767.0;
        for (code, v) in [(proto::gamepad::ABS_X, x), (proto::gamepad::ABS_Y, y)] {
            let v = v.round() as i32;
            if sent_axis.get(&code) != Some(&v) {
                sent_axis.insert(code, v);
                let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadAxis {
                    slot,
                    code,
                    value: v,
                });
            }
        }
        // trigger: half-press pulsing
        let z = (((t * 1.6).sin() + 1.0) / 2.0 * 1023.0).round() as i32;
        if sent_axis.get(&proto::gamepad::ABS_Z) != Some(&z) {
            sent_axis.insert(proto::gamepad::ABS_Z, z);
            let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadAxis {
                slot,
                code: proto::gamepad::ABS_Z,
                value: z,
            });
        }
        // hat: step the compass every second
        let hat = hat_seq[(t as usize) % hat_seq.len()];
        if (hat.0, hat.1) != sent_hat {
            sent_hat = hat;
            let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadAxis {
                slot,
                code: proto::gamepad::ABS_HAT0X,
                value: hat.0,
            });
            let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadAxis {
                slot,
                code: proto::gamepad::ABS_HAT0Y,
                value: hat.1,
            });
        }
        // A button tap every 2 cycles
        let _ = input_tx.send(rdma_display_proto::InputEvent::GamepadBtn {
            slot,
            code: proto::gamepad::BTN_SOUTH,
            down: btn_state,
        });
        btn_state = !btn_state;
        std::thread::sleep(std::time::Duration::from_millis(500));
        t += 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers() {
        // verified against linux/input.h on the build box
        assert_eq!(EVIOCGRAB, 0x40044590);
        assert_eq!(EVIOCSFF, 0x40304580);
        assert_eq!(EVIOCRMFF, 0x40044581);
        assert_eq!(eviocgabs(proto::gamepad::ABS_X), 0x80184540);
        assert_eq!(size_of::<FfEffect>(), 48);
        assert_eq!(size_of::<AbsInfo>(), 24);
    }
}
