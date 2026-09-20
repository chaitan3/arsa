//! C1 Wayland window plumbing for the EGL-upload presenter: an xdg-toplevel
//! that `eglwin::EglWindow` paints into via `eglSwapBuffers`.
//!
//! This module deliberately contains NO buffer management: with the
//! dmabuf-attach presenter removed (KWin+NVIDIA refuses foreign system
//! dmabuf imports — see docs/client-display.md), the compositor only ever
//! sees EGL's own driver-native swapchain buffers. Credits are paced by the
//! upload path in `main.rs`, so there is no release-to-credit machinery
//! here either.
//!
//! One non-obvious behavior encoded below: KWin 6.7.5 emits the initial
//! xdg configure only after the surface's first commit. A client that
//! waits for configure before committing deadlocks forever — hence the
//! early empty commit, mirroring `crates/client/examples/simple_window.rs`.
//!
//! Plain-thread integration without an event-loop framework: create the
//! `ReadEventsGuard` BEFORE `poll()`ing its fd (per wayland-backend docs),
//! then `read()` — which is non-blocking after the poll — and dispatch.

use std::os::fd::AsRawFd;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use wayland_client::globals::GlobalListContents;
use wayland_client::protocol::wl_compositor::WlCompositor;
use wayland_client::protocol::wl_display::WlDisplay;
use wayland_client::protocol::wl_keyboard::{self, KeyState, WlKeyboard};
use wayland_client::protocol::wl_pointer::{self, Axis, ButtonState, WlPointer};
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_seat::{self, Capability, WlSeat};
use wayland_client::protocol::wl_surface::{self, WlSurface};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use std::sync::mpsc::Sender;
use rdma_display_proto::InputEvent;

use crate::keymap;
use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_device_v1::{Shape, WpCursorShapeDeviceV1};
use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_manager_v1::WpCursorShapeManagerV1;
use wayland_protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1::{self, Mode, ZxdgToplevelDecorationV1};
use wayland_protocols::xdg::decoration::zv1::client::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1;
use wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_v1::{self, WpFractionalScaleV1};
use wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;
use wayland_protocols::wp::viewporter::client::wp_viewporter::WpViewporter;
use wayland_protocols::xdg::shell::client::xdg_surface::{self, XdgSurface};
use wayland_protocols::xdg::shell::client::xdg_toplevel::{self, XdgToplevel};
use wayland_protocols::xdg::shell::client::xdg_wm_base::{self, XdgWmBase};

pub struct Presenter {
    conn: Connection,
    queue: EventQueue<St>,
    state: St,
}

struct St {
    // compositor/wmbase are used; the others are keep-alive handles (their
    // proxies must outlive the window; events keep flowing to the Dispatch
    // impls regardless)
    #[allow(dead_code)]
    compositor: WlCompositor,
    wmbase: XdgWmBase,
    /// keep-alive: seat owns the keyboard/pointer object lifetimes
    #[allow(dead_code)]
    seat: WlSeat,
    /// cursor-shape protocol: without it the compositor shows its default
    /// (KWin's fallback for an unnamed surface is the text beam)
    #[allow(dead_code)]
    cursor_mgr: Option<WpCursorShapeManagerV1>,
    /// server-side decorations (titlebar/close drawn by KWin)
    #[allow(dead_code)]
    decoration_mgr: Option<ZxdgDecorationManagerV1>,
    #[allow(dead_code)]
    decoration: Option<ZxdgToplevelDecorationV1>,
    /// shape device for our wl_pointer (created with the pointer)
    #[allow(dead_code)]
    cursor_device: Option<WpCursorShapeDeviceV1>,
    /// wl_pointer enter serial: set_shape must use the latest one
    pointer_serial: u32,
    /// output scale (120ths; 120 = 1.0) — the compositor's render scale.
    /// The buffer is always the NATIVE guest frame (no resampling, sharp
    /// pixels); the logical window size is native/scale via wp_viewporter.
    scale120: u32,
    logical_w: u32,
    logical_h: u32,
    /// scale plumbing keep-alives
    #[allow(dead_code)]
    viewporter: WpViewporter,
    #[allow(dead_code)]
    fs_mgr: Option<WpFractionalScaleManagerV1>,
    #[allow(dead_code)]
    fs_obj: Option<WpFractionalScaleV1>,
    viewport: WpViewport,
    surface: WlSurface,
    #[allow(dead_code)]
    xdg_surface: XdgSurface,
    #[allow(dead_code)]
    toplevel: XdgToplevel,
    configured: bool,
    closed: bool,
    /// user-input forwarding: gated on Wayland focus (keyboard enter/leave)
    input_tx: Sender<InputEvent>,
    kb_active: bool,
    pointer_active: bool,
    last_abs: Option<(u32, u32)>,
    width: u32,
    height: u32,
    /// scroll: axis direction pending its axis_value120 event
    pending_axis: Option<Axis>,
}

extern "C" {
    fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
}

#[repr(C)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

const POLLIN: i16 = 0x0001;
const POLLHUP: i16 = 0x0010;
const POLLERR: i16 = 0x0008;
const LIBC_EINTR: i32 = 4;

impl Presenter {
    /// Connect to the compositor, build the window, map it with an early
    /// empty commit, and block until the initial xdg configure arrives.
    /// `input_tx` receives user-input events (keyboard/pointer) for the
    /// session's input path.
    pub fn new(
        width: u32,
        height: u32,
        input_tx: Sender<InputEvent>,
    ) -> Result<Self> {
        let conn = Connection::connect_to_env().context(
            "wayland connect: needs WAYLAND_DISPLAY + XDG_RUNTIME_DIR in env \
             (sudo: use `sudo -E` or pass them through)",
        )?;
        let (globals, queue) =
            wayland_client::globals::registry_queue_init::<St>(&conn).context("registry")?;
        let qh = queue.handle();

        let compositor: WlCompositor = globals.bind(&qh, 1..=6, ())?;
        let wmbase: XdgWmBase = globals.bind(&qh, 1..=6, ())?;
        let seat: WlSeat = globals.bind(&qh, 1..=10, ())
            .context("bind wl_seat (keyboard/pointer input)")?;
        // cursor-shape-v1 (staging, KWin 6): lets us name the cursor we want
        // (arrow) instead of hand-rolling a wl_shm cursor theme; absence is
        // non-fatal — we'd just show the compositor's fallback again
        let cursor_mgr: Option<WpCursorShapeManagerV1> =
            globals.bind(&qh, 1..=1, ()).ok();
        // viewporter + fractional-scale: buffer stays at the NATIVE guest
        // frame size; the compositor maps it 1:1 onto physical pixels at
        // whatever output scale the desktop runs (no upscale/blur)
        let viewporter: WpViewporter = globals.bind(&qh, 1..=1, ())?;
        let fs_mgr: Option<WpFractionalScaleManagerV1> =
            globals.bind(&qh, 1..=1, ()).ok();
        // xdg-decoration: opt in to server-side decorations — without this
        // request KWin maps the toplevel with no titlebar at all
        let decoration_mgr: Option<ZxdgDecorationManagerV1> =
            globals.bind(&qh, 1..=1, ()).ok();
        eprintln!(
            "[wl] bound compositor v{} wm_base v{} seat v{} cursor-shape {} fractional-scale {}",
            compositor.version(),
            wmbase.version(),
            seat.version(),
            if cursor_mgr.is_some() { "v1" } else { "unavailable" },
            if fs_mgr.is_some() { "v1" } else { "unavailable" },
        );

        let surface = compositor.create_surface(&qh, ());
        let viewport = viewporter.get_viewport(&surface, &qh, ());
        let fs_obj: Option<WpFractionalScaleV1> =
            fs_mgr.as_ref().map(|m| m.get_fractional_scale(&surface, &qh, ()));
        let xdg_surface = wmbase.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title("rdma-display".into());
        toplevel.set_app_id("rdma-display-client".into());
        // 1:1 fixed-size window (resize/renegotiation is M7)
        xdg_surface.set_window_geometry(0, 0, width as i32, height as i32);
        toplevel.set_min_size(width as i32, height as i32);
        toplevel.set_max_size(width as i32, height as i32);
        // request the compositor-drawn titlebar before the map commit
        let decoration = decoration_mgr
            .as_ref()
            .map(|m| m.get_toplevel_decoration(&toplevel, &qh, ()));
        if let Some(d) = decoration.as_ref() {
            d.set_mode(Mode::ServerSide);
        }
        eprintln!(
            "[wl] decorations: {}",
            match decoration.as_ref() {
                Some(_) => "server-side requested",
                None => "unavailable (compositor will map bare)",
            }
        );

        let mut me = Presenter {
            conn,
            queue,
            state: St {
                compositor,
                wmbase,
                seat,
                cursor_mgr,
                cursor_device: None,
                pointer_serial: 0,
                decoration_mgr,
                decoration,
                scale120: 120,
                logical_w: width,
                logical_h: height,
                viewporter,
                fs_mgr,
                fs_obj,
                viewport,
                surface,
                xdg_surface,
                toplevel,
                configured: false,
                closed: false,
                input_tx,
                kb_active: false,
                pointer_active: false,
                last_abs: None,
                width,
                height,
                pending_axis: None,
            },
        };

        me.queue.flush()?;

        // Early empty commit, exactly like the crate's own canonical client
        // (crates/client/examples/simple_window.rs): KWin 6.7.5 emits the
        // initial xdg configure only once the surface has been committed at
        // least once. A client that waits for configure before its first
        // commit deadlocks forever — compositor processes every request and
        // emits nothing (verified by the roundtrip bisect).
        me.state.surface.commit();
        eprintln!("[wl] early empty commit sent (pre-configure map)");
        me.queue.flush()?;

        let start = Instant::now();
        while !me.state.configured && !me.state.closed {
            if start.elapsed().as_secs() > 5 {
                bail!("no xdg_toplevel configure within 5s (cached_protocol_error={:?})", me.conn.protocol_error());
            }
            me.pump(50)?;
        }
        if me.state.closed {
            bail!("window closed before first configure");
        }
        // configure received; frame presentation attaches + commits content
        me.state.apply_scale(); // apply scale if preferred_scale already arrived
        me.pump(0)?;
        Ok(me)
    }

    /// Non-blocking-ish event drain: poll the socket up to `timeout_ms`,
    /// read + dispatch whatever arrived. ALL errors propagate: a compositor
    /// that killed our connection (protocol error) must be heard, not
    /// swallowed (silent EPIPE == "no configure within 5s" symptom).
    pub fn pump(&mut self, timeout_ms: i32) -> Result<()> {
        if let Err(e) = self.queue.flush() {
            bail!("wayland flush failed (connection dead?): {e}");
        }
        // guard MUST be created before polling the fd (wayland-backend rule)
        if let Some(guard) = self.queue.prepare_read() {
            let fd = guard.connection_fd().as_raw_fd();
            let mut pfd = PollFd {
                fd,
                events: POLLIN,
                revents: 0,
            };
            let rc = unsafe { poll(&mut pfd, 1, timeout_ms) };
            if rc < 0 {
                let e = std::io::Error::last_os_error();
                if e.raw_os_error() != Some(LIBC_EINTR) {
                    bail!("wayland poll: {e}");
                }
            } else if rc > 0 && (pfd.revents & (POLLHUP | POLLERR)) != 0 {
                bail!(
                    "wayland socket hangup (revents {:#x}) — compositor closed the \
                     connection, most likely after a protocol error in our requests",
                    pfd.revents
                );
            }
            if let Err(e) = guard.read() {
                // poll-timeout => nothing to read => WouldBlock is the normal
                // idle outcome, not a failure
                let would_block = match &e {
                    wayland_client::backend::WaylandError::Io(io) => matches!(
                        io.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ),
                    _ => false,
                };
                if !would_block {
                    bail!("wayland read failed: {e}");
                }
            }
        }
        self.queue
            .dispatch_pending(&mut self.state)
            .context("wayland dispatch")?;
        Ok(())
    }

    /// Raw libwayland `struct wl_display *` for EGL interop
    /// (`eglGetDisplay`).
    pub fn wl_display_ptr(&self) -> *mut std::ffi::c_void {
        self.conn.display().id().as_ptr() as *mut std::ffi::c_void
    }

    /// Raw libwayland `struct wl_surface *` for `wl_egl_window_create`.
    pub fn wl_surface_ptr(&self) -> *mut std::ffi::c_void {
        self.state.surface.id().as_ptr() as *mut std::ffi::c_void
    }

    pub fn closed(&self) -> bool {
        self.state.closed
    }
}

impl St {
    /// Recompute the logical window size for the compositor's output scale:
    /// buffer stays native (W×H, the guest frame verbatim), the viewport
    /// destination becomes native/scale — so the compositor maps the buffer
    /// 1:1 onto physical pixels. Input coords are converted the same way.
    fn apply_scale(&mut self) {
        let lw = (self.width as u64 * 120 / self.scale120.max(1) as u64) as u32;
        let lh = (self.height as u64 * 120 / self.scale120.max(1) as u64) as u32;
        if (lw, lh) == (self.logical_w, self.logical_h) {
            return;
        }
        self.logical_w = lw;
        self.logical_h = lh;
        self.viewport.set_destination(lw as i32, lh as i32);
        self.xdg_surface.set_window_geometry(0, 0, lw as i32, lh as i32);
        self.toplevel.set_min_size(lw as i32, lh as i32);
        self.toplevel.set_max_size(lw as i32, lh as i32);
        eprintln!(
            "[wl] scale {}/120: window {}x{} logical, buffer {}x{} native (1:1 pixels)",
            self.scale120, lw, lh, self.width, self.height
        );
        self.surface.commit();
    }
}

// --- Dispatch implementations (wayland-client 0.31 requires them per object
//     data type; interfaces with no events get the quiet blanket) -----------

/// no events to handle on this object
macro_rules! quiet {
    ($($iface:ty => $data:ty),* $(,)?) => {
        $(
            impl Dispatch<$iface, $data> for St {
                fn event(
                    _state: &mut St,
                    _object: &$iface,
                    _event: <$iface as Proxy>::Event,
                    _data: &$data,
                    _conn: &Connection,
                    _qhandle: &QueueHandle<St>,
                ) {
                }
            }
        )*
    };
}

// objects created via the registry/global machinery carry GlobalListContents
impl Dispatch<WlDisplay, GlobalListContents> for St {
    fn event(
        _: &mut St,
        _: &WlDisplay,
        _: <WlDisplay as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
    }
}
impl Dispatch<WlRegistry, GlobalListContents> for St {
    fn event(
        _: &mut St,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
    }
}

quiet!(WlCompositor => ());
quiet!(WpCursorShapeManagerV1 => ());
quiet!(WpCursorShapeDeviceV1 => ());
quiet!(WpViewporter => ());
quiet!(WpViewport => ());
quiet!(WpFractionalScaleManagerV1 => ());
quiet!(ZxdgDecorationManagerV1 => ());

/// Decoration negotiate: we want the titlebar drawn by the compositor; if
/// KWin ever configures client-side, re-assert our request.
impl Dispatch<ZxdgToplevelDecorationV1, ()> for St {
    fn event(
        state: &mut St,
        obj: &ZxdgToplevelDecorationV1,
        ev: <ZxdgToplevelDecorationV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        if let zxdg_toplevel_decoration_v1::Event::Configure { mode } = ev {
            match mode {
                WEnum::Value(Mode::ServerSide) => {
                    eprintln!("[wl] decoration mode: server-side (titlebar active)")
                }
                other => {
                    eprintln!("[wl] decoration configure said {other:?}; re-asserting server-side");
                    obj.set_mode(Mode::ServerSide);
                    state.surface.commit();
                }
            }
        }
    }
}

/// Fractional scale: the compositor tells us the output scale (120ths);
/// the logical window is resized to native/scale so the native-size buffer
/// lands 1:1 on the physical display.
impl Dispatch<WpFractionalScaleV1, ()> for St {
    fn event(
        state: &mut St,
        _: &WpFractionalScaleV1,
        ev: <WpFractionalScaleV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale, .. } = ev {
            if state.scale120 != scale {
                state.scale120 = scale;
                state.apply_scale();
            }
        }
    }
}

/// Seat: create keyboard/pointer objects per reported capabilities.
impl Dispatch<WlSeat, ()> for St {
    fn event(
        state: &mut St,
        seat: &WlSeat,
        ev: <WlSeat as Proxy>::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<St>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(caps),
        } = ev
        {
            if caps.contains(Capability::Pointer) {
                let pointer = seat.get_pointer(qh, ());
                // cursor shape: a device per pointer; set on pointer-enter
                if let Some(mgr) = state.cursor_mgr.as_ref() {
                    state.cursor_device = Some(mgr.get_pointer(&pointer, qh, ()));
                }
            }
            if caps.contains(Capability::Keyboard) {
                seat.get_keyboard(qh, ());
            }
        }
    }
}

/// Keyboard: forward focused keys as QEMU key numbers. Wayland delivers
/// `key` only while this client's surface has keyboard focus, and we also
/// gate on enter/leave for belt-and-braces no-leak behavior.
impl Dispatch<WlKeyboard, ()> for St {
    fn event(
        state: &mut St,
        _: &WlKeyboard,
        ev: <WlKeyboard as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        match ev {
            wl_keyboard::Event::Keymap { .. } => {
                // we forward raw evdev codes; the xkb keymap is not needed
            }
            wl_keyboard::Event::Enter { .. } => state.kb_active = true,
            wl_keyboard::Event::Leave { .. } => state.kb_active = false,
            wl_keyboard::Event::Key { key, state: st, .. } => {
                if !state.kb_active {
                    return;
                }
                // `key` is the Linux evdev keycode (no xkb +8 offset needed
                // because we never interpret the keymap)
                let Some(code) = keymap::linux_to_qnum(key) else {
                    return; // unmapped key
                };
                let down = matches!(st, WEnum::Value(KeyState::Pressed));
                let _ = state.input_tx.send(InputEvent::Key { code, down });
            }
            _ => {} // modifiers / repeat_info
        }
    }
}

/// Pointer: absolute positioning (guest is a tablet), buttons, wheel.
impl Dispatch<WlPointer, ()> for St {
    fn event(
        state: &mut St,
        _: &WlPointer,
        ev: <WlPointer as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        match ev {
            wl_pointer::Event::Enter { serial, .. } => {
                state.pointer_active = true;
                state.pointer_serial = serial;
                // name our cursor: the plain arrow (compositors pick their
                // own fallback for silent clients; KWin's is the text beam)
                if let Some(dev) = state.cursor_device.as_ref() {
                    dev.set_shape(serial, Shape::Default);
                }
            }
            wl_pointer::Event::Leave { .. } => {
                state.pointer_active = false;
                state.pending_axis = None;
            }
            wl_pointer::Event::Motion { surface_x, surface_y, .. } => {
                if !state.pointer_active {
                    return;
                }
                // surface-local coords are LOGICAL pixels (scaled by the
                // compositor's output scale); convert to guest pixels and
                // clamp to the console's last row/column (SetAbsPosition
                // requires x < width, y < height strictly)
                let lw = state.logical_w.max(1) as f64;
                let lh = state.logical_h.max(1) as f64;
                let x = ((surface_x as f64 * state.width as f64 / lw) as u32)
                    .min(state.width.saturating_sub(1));
                let y = ((surface_y as f64 * state.height as f64 / lh) as u32)
                    .min(state.height.saturating_sub(1));
                if state.last_abs != Some((x, y)) {
                    state.last_abs = Some((x, y));
                    let _ = state.input_tx.send(InputEvent::Abs { x, y });
                }
            }
            wl_pointer::Event::Button { button, state: st, .. } => {
                if !state.pointer_active {
                    return;
                }
                // evdev button codes -> D-Bus button values
                let dbtn = match button {
                    0x110 => 0, // Left
                    0x112 => 1, // Middle
                    0x111 => 2, // Right
                    0x113 => 5, // Side
                    0x114 => 6, // Extra
                    other => {
                        eprintln!("[wl] unmapped pointer button {other:#x}");
                        return;
                    }
                };
                let down = matches!(st, WEnum::Value(ButtonState::Pressed));
                let _ = state.input_tx.send(InputEvent::Btn { button: dbtn, down });
            }
            wl_pointer::Event::AxisValue120 { axis, value120 } => {
                // value120 is in 120ths of a wheel notch; positive vertical
                // axis = scrolling down (WheelDown), negative = up.
                // Empirical — swap if the guest wheel appears inverted.
                let vertical = matches!(axis, WEnum::Value(Axis::VerticalScroll));
                if vertical && value120 != 0 {
                    let notches = (value120 / 120).unsigned_abs();
                    let btn = if value120 > 0 { 4 } else { 3 }; // 4=WheelDown 3=WheelUp
                    for _ in 0..notches {
                        let _ = state.input_tx.send(InputEvent::Btn { button: btn, down: true });
                        let _ = state.input_tx.send(InputEvent::Btn { button: btn, down: false });
                    }
                }
            }
            _ => {} // legacy axis events, axis_stop, frame
        }
    }
}

impl Dispatch<XdgWmBase, ()> for St {
    fn event(
        state: &mut St,
        _: &XdgWmBase,
        ev: <XdgWmBase as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = ev {
            state.wmbase.pong(serial);
        }
    }
}

impl Dispatch<XdgSurface, ()> for St {
    fn event(
        state: &mut St,
        obj: &XdgSurface,
        ev: <XdgSurface as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        if let xdg_surface::Event::Configure { serial } = ev {
            obj.ack_configure(serial);
            state.configured = true;
        }
    }
}

impl Dispatch<XdgToplevel, ()> for St {
    fn event(
        state: &mut St,
        _: &XdgToplevel,
        ev: <XdgToplevel as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        match ev {
            xdg_toplevel::Event::Close => state.closed = true,
            xdg_toplevel::Event::Configure { width, height, .. } => {
                // 0x0 = "no preference"; C1 keeps the native frame size
                let _ = (width, height);
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSurface, ()> for St {
    fn event(
        _: &mut St,
        _: &WlSurface,
        ev: <WlSurface as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        match ev {
            wl_surface::Event::Enter { .. } | wl_surface::Event::Leave { .. } => {}
            _ => {}
        }
    }
}
