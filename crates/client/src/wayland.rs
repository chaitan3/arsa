//! Wayland window plumbing for the Vulkan presenter: an xdg-toplevel
//! that `vkpresent::Renderer` uses as a Vulkan Wayland surface.
//!
//! This module deliberately contains NO buffer management: Vulkan WSI owns the
//! swapchain wl_buffers (driver-native), and decoding/upload in the renderer
//! paces credits. There is no release-to-credit machinery here.
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
use wayland_client::protocol::wl_output::{self, WlOutput};
use wayland_client::protocol::wl_keyboard::{self, KeyState, WlKeyboard};
use wayland_client::protocol::wl_pointer::{self, Axis, ButtonState, WlPointer};
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_region::WlRegion;
use wayland_client::protocol::wl_seat::{self, Capability, WlSeat};
use wayland_client::protocol::wl_surface::{self, WlSurface};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle, WEnum};
use std::sync::mpsc::Sender;
use rdma_display_proto::InputEvent;

use crate::keymap;
use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_device_v1::{Shape, WpCursorShapeDeviceV1};
use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::zwp_keyboard_shortcuts_inhibit_manager_v1::ZwpKeyboardShortcutsInhibitManagerV1;
use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::zwp_keyboard_shortcuts_inhibitor_v1::{self, ZwpKeyboardShortcutsInhibitorV1};
use wayland_protocols::wp::pointer_constraints::zv1::client::zwp_locked_pointer_v1::{self, ZwpLockedPointerV1};
use wayland_protocols::wp::pointer_constraints::zv1::client::zwp_pointer_constraints_v1::{Lifetime, ZwpPointerConstraintsV1};
use wayland_protocols::wp::relative_pointer::zv1::client::zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1;
use wayland_protocols::wp::relative_pointer::zv1::client::zwp_relative_pointer_v1::{self, ZwpRelativePointerV1};
use wayland_protocols::wp::cursor_shape::v1::client::wp_cursor_shape_manager_v1::WpCursorShapeManagerV1;
use wayland_protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1::{self, Mode, ZxdgToplevelDecorationV1};
use wayland_protocols::xdg::decoration::zv1::client::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1;
use wayland_protocols::wp::fractional_scale::v1::client::wp_fractional_scale_v1::{self, WpFractionalScaleV1};
use wayland_protocols::wp::presentation_time::client::wp_presentation::{self, WpPresentation};
use wayland_protocols::wp::presentation_time::client::wp_presentation_feedback::{
    self, WpPresentationFeedback,
};
use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_feedback_v1::{
    self, ZwpLinuxDmabufFeedbackV1,
};
use wayland_protocols::wp::linux_dmabuf::zv1::client::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
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

/// One in-flight wp_presentation feedback: `id` is our correlation handle,
/// `proxy` is kept alive until the compositor answers presented/discarded.
struct Fb {
    id: u64,
    proxy: WpPresentationFeedback,
}

/// A completed presentation feedback, folded into metrics by the main loop.
#[derive(Clone, Copy)]
pub struct Presented {
    pub id: u64,
    pub unix_us: u64,
    pub presented: bool,
    /// wp_presentation_feedback flags (vsync/hw_clock/hw_completion/zero_copy)
    pub flags: u32,
    /// refresh interval the compositor used, ns (0 = unknown). Varies under
    /// VRR, which is how the client can tell VRR is actually engaged.
    pub refresh: u32,
}

/// Accumulator for `zwp_linux_dmabuf_feedback_v1`: the compositor's advertised
/// format/modifier pairs, split into preference tranches. The tranche with the
/// `scanout` flag is what it can scan out directly, so this is the authoritative
/// answer to "which buffer modifier would direct-scanout".
#[derive(Default)]
struct DmabufFeedback {
    table: Vec<(u32, u64)>, // (fourcc, modifier)
    main_device: Vec<u8>,
    cur: Tranche,
    tranches: Vec<Tranche>,
    last: String,
}

#[derive(Default, Clone)]
struct Tranche {
    target: Vec<u8>,
    flags: u32,
    formats: Vec<u16>,
}

impl DmabufFeedback {
    fn dev_str(d: &[u8]) -> String {
        if d.len() >= 8 {
            let v = u64::from_ne_bytes([
                d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
            ]);
            format!("{v:#x}")
        } else {
            format!("{d:02x?}")
        }
    }

    fn log(&mut self) {
        if self.tranches.is_empty() {
            return;
        }
        let mut s = format!(
            "[wl] dmabuf feedback: main_device={} tranches={}",
            Self::dev_str(&self.main_device),
            self.tranches.len()
        );
        for (i, t) in self.tranches.iter().enumerate() {
            let mut fmts: Vec<String> = t
                .formats
                .iter()
                .filter_map(|&idx| {
                    self.table
                        .get(idx as usize)
                        .map(|(f, m)| format!("{f:#010x}/mod{m:#x}"))
                })
                .collect();
            if fmts.len() > 8 {
                fmts.truncate(8);
                fmts.push("...".into());
            }
            s.push_str(&format!(
                "\n[wl]   tranche {i}: target={} scanout={} sampling={} formats=[{}]",
                Self::dev_str(&t.target),
                t.flags & 1 != 0,
                t.flags & 2 != 0,
                fmts.join(", ")
            ));
        }
        // KWin re-sends feedback as the surface's output/state changes (e.g.
        // after fullscreen); log only when the set actually changes, and only
        // under -v (it is diagnostic only; the Vulkan presenter owns its swapchain).
        if s != self.last {
            log::debug!("{s}");
            self.last = s;
        }
    }
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
    /// keep-alive: our wl_pointer (needed to lock it / create the relative
    /// pointer for grab mode)
    #[allow(dead_code)]
    pointer: Option<WlPointer>,
    /// keep-alive: our wl_keyboard (focus comes from it; the grab's
    /// shortcut inhibitor binds to the seat, not to this object)
    #[allow(dead_code)]
    keyboard: Option<WlKeyboard>,
    /// shape device for our wl_pointer (created with the pointer)
    #[allow(dead_code)]
    cursor_device: Option<WpCursorShapeDeviceV1>,
    /// wl_pointer enter serial: set_shape must use the latest one
    pointer_serial: u32,
    /// zwp_relative_pointer_manager_v1 + zwp_pointer_constraints_v1 for grab
    /// mode (None when the compositor lacks either global)
    rel_mgr: Option<ZwpRelativePointerManagerV1>,
    constraints: Option<ZwpPointerConstraintsV1>,
    /// keyboard-shortcuts-inhibit: while grabbed, the local compositor must
    /// hand us EVERY key — its own desktop shortcuts (alt-f4, alt-tab,
    /// alt-f1, ...) included — instead of acting on them. None when the
    /// compositor lacks the global (then grab only takes the mouse).
    inhibit_mgr: Option<ZwpKeyboardShortcutsInhibitManagerV1>,
    /// the live inhibitor for our surface (Some ⇔ keyboard grabbed); kept
    /// alive across focus shuffles so it re-engages on re-focus
    inhibitor: Option<ZwpKeyboardShortcutsInhibitorV1>,
    /// wp_presentation: the compositor's true on-screen timestamp for the
    /// commit that carried a frame — the `display`/end-to-end metrics stage
    presentation: Option<WpPresentation>,
    /// linux-dmabuf feedback: keep-alive + the parsed tranches (scanout set)
    _dmabuf: Option<ZwpLinuxDmabufV1>,
    _dmabuf_fb: Option<ZwpLinuxDmabufFeedbackV1>,
    dmabuf_feedback: DmabufFeedback,
    /// clock domain wp_presentation timestamps use (KWin: CLOCK_MONOTONIC)
    pres_clock: i32,
    /// feedback objects still awaiting presented/discarded
    fbs: Vec<Fb>,
    /// completed feedbacks, arrival order: (id, client-unix-us, presented)
    fb_done: std::collections::VecDeque<Presented>,
    /// id to hand to the next feedback request
    fb_next: u64,
    /// grab-mode keep-alives (dropping them releases the constraint)
    rel_obj: Option<ZwpRelativePointerV1>,
    locked_obj: Option<ZwpLockedPointerV1>,
    /// input grab (Ctrl+Alt+G): pointer locked locally (deltas forwarded as
    /// InputEvent::Rel, local cursor hidden) AND the keyboard grabbed (the
    /// compositor's own shortcuts inhibited, so alt-f4/alt-tab/alt-f1 reach
    /// the guest). Ctrl+Alt+G is the only way out — see `set_grab`.
    grabbed: bool,
    /// fractional accumulator for relative deltas (logical pixels)
    grab_acc: (f64, f64),
    /// grab diagnostics: per-2s aggregate (event count, Σdelta) to spot
    /// attenuation between compositor delivery and host injection
    dbg_n: u64,
    dbg_sum: (f64, f64),
    dbg_window: std::time::Instant,
    /// Ctrl+Alt both held (from wl_keyboard.modifiers) — grab hotkey chord
    mods_ctrl_alt: bool,
    /// the G key currently down was swallowed by the grab chord (never
    /// forwarded), so its release must be swallowed too. Tracked separately
    /// from `held_keys` because the release arrives after Ctrl/Alt are
    /// usually already up — it cannot be recognised from the chord state, and
    /// it must swallow ONLY G, never whatever key is released next.
    hotkey_g_down: bool,
    /// output scale (120ths; 120 = 1.0) — the compositor's render scale.
    /// The buffer is always the NATIVE guest frame (no resampling, sharp
    /// pixels); the logical window size is native/scale via wp_viewporter.
    scale120: u32,
    logical_w: u32,
    logical_h: u32,
    /// buffer size in device pixels (logical × scale) — what the Vulkan
    /// swapchain must render at; letterboxed when aspect ≠ frame's
    dev_w: u32,
    dev_h: u32,
    /// compositor-driven fullscreen state (xdg_toplevel.configure states
    /// array): our own --fullscreen request AND the compositor's own
    /// toggle shortcuts both arrive through configure, so this is the
    /// single source of truth
    fullscreened: bool,
    /// toplevel configure that xdg_surface.ack_configure must precede:
    /// target logical size to apply at ack time
    pending_mode: Option<(u32, u32)>,
    /// buffer size change waiting to be applied by the Vulkan presenter
    /// (main loop calls Presenter::take_resize → Renderer::resize)
    pending_resize: Option<(u32, u32)>,
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
    /// forwarded evdev codes currently believed held (for repeat handling
    /// and focus-loss flushes); KWin repeats keys with state=2
    /// (WL_KEYBOARD_KEY_STATE_REPEATED) which surfaces as WEnum::Raw
    held_keys: std::collections::HashSet<u32>,
    /// forwarded pointer buttons currently believed held (dbtn values);
    /// flushed on pointer leave so a constraint break / focus churn while
    /// dragging cannot leave a button stuck down in the guest
    held_buttons: std::collections::HashSet<u32>,
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

// --- presentation clock conversion -----------------------------------------
// wp_presentation timestamps are expressed in the compositor's clock domain
// (KWin: CLOCK_MONOTONIC). Sampling both clocks once gives the constant
// realtime-monotonic offset; both tick at the same rate, so the conversion is
// exact to clock-read granularity.
#[allow(clippy::useless_conversion)] // time_t/c_long: i64 here, i32 elsewhere
fn clock_us(clk: libc::clockid_t) -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(clk, &mut ts) };
    i64::from(ts.tv_sec) * 1_000_000 + i64::from(ts.tv_nsec) / 1000
}

/// `wp_presentation` timestamp (in clock `clk`) -> unix-epoch microseconds.
fn presentation_to_unix_us(clk: libc::clockid_t, secs: u64, nsec: u32) -> u64 {
    let delta = clock_us(libc::CLOCK_REALTIME) - clock_us(clk);
    let ts = secs as i64 * 1_000_000 + nsec as i64 / 1000;
    (ts + delta).max(0) as u64
}

/// One-shot query of this client's current output mode, for `--native`: connect
/// to the compositor, bind the first `wl_output`, and read its `CURRENT` mode.
/// Returns `(width, height, refresh_mhz)`.
pub fn query_output_mode() -> Result<(u32, u32, u32)> {
    #[derive(Default)]
    struct OutputQuery {
        mode: Option<(u32, u32, u32)>,
    }
    impl Dispatch<WlRegistry, GlobalListContents> for OutputQuery {
        fn event(
            _: &mut Self,
            _: &WlRegistry,
            _: <WlRegistry as Proxy>::Event,
            _: &GlobalListContents,
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }
    impl Dispatch<WlOutput, ()> for OutputQuery {
        fn event(
            state: &mut Self,
            _: &WlOutput,
            ev: <WlOutput as Proxy>::Event,
            _: &(),
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
            if let wl_output::Event::Mode {
                flags,
                width,
                height,
                refresh,
            } = ev
            {
                let current = match flags {
                    WEnum::Value(m) => m.contains(wl_output::Mode::Current),
                    WEnum::Unknown(v) => v & 1 != 0,
                };
                if current {
                    state.mode = Some((
                        width.max(0) as u32,
                        height.max(0) as u32,
                        refresh.max(0) as u32,
                    ));
                }
            }
        }
    }

    let conn = Connection::connect_to_env()
        .context("wayland connect (for --native): needs WAYLAND_DISPLAY + XDG_RUNTIME_DIR")?;
    let (globals, mut queue) =
        wayland_client::globals::registry_queue_init::<OutputQuery>(&conn).context("registry")?;
    let qh = queue.handle();
    let _output: WlOutput = globals
        .bind(&qh, 1..=4, ())
        .context("bind wl_output (no output?)")?;
    let mut state = OutputQuery::default();
    // the geometry/mode/scale/done burst follows the bind; one roundtrip is
    // normally enough, a second is harmless
    for _ in 0..2 {
        queue.roundtrip(&mut state).context("wl_output roundtrip")?;
        if state.mode.is_some() {
            break;
        }
    }
    state
        .mode
        .context("no CURRENT wl_output mode reported")
}

impl Presenter {
    /// Connect to the compositor, build the window, map it with an early
    /// empty commit, and block until the initial xdg configure arrives.
    /// `input_tx` receives user-input events (keyboard/pointer) for the
    /// session's input path. `fullscreen` requests a fullscreen toplevel
    /// on the current output before mapping (the compositor answers with
    /// a configure carrying the real output size + the fullscreen state).
    /// `grab` engages the input grab (Ctrl+Alt+G's ON state) right after the
    /// window is mapped, instead of waiting for the chord.
    pub fn new(
        width: u32,
        height: u32,
        input_tx: Sender<InputEvent>,
        fullscreen: bool,
        grab: bool,
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
        // relative-pointer + pointer-constraints: grab mode (games) — lock the
        // local pointer and forward raw relative deltas. Absence just means
        // the toggle prints a hint instead of grabbing.
        let rel_mgr: Option<ZwpRelativePointerManagerV1> =
            globals.bind(&qh, 1..=1, ()).ok();
        let constraints: Option<ZwpPointerConstraintsV1> =
            globals.bind(&qh, 1..=1, ()).ok();
        // keyboard-shortcuts-inhibit: the keyboard half of the grab — the
        // compositor must route its OWN shortcuts (alt-f4, alt-tab, alt-f1,
        // ...) to us instead of acting on them. Absence is non-fatal: grab
        // then only takes the mouse (and says so).
        let inhibit_mgr: Option<ZwpKeyboardShortcutsInhibitManagerV1> =
            globals.bind(&qh, 1..=1, ()).ok();
        // wp_presentation: the compositor reports the true on-screen time of
        // each committed frame (KWin: the DRM page-flip timestamp), which is
        // the only honest end of the video-latency chain. Absence only means
        // the metrics `display` stage reads n/a.
        let presentation: Option<WpPresentation> = globals.bind(&qh, 1..=2, ()).ok();
        // linux-dmabuf feedback: which format+modifier pairs the compositor can
        // scan out directly (the `scanout` tranche). Diagnostic only — the
        // presenter uses Vulkan WSI's own swapchain buffers.
        let dmabuf: Option<ZwpLinuxDmabufV1> = globals.bind(&qh, 1..=5, ()).ok();
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
        log::debug!(
            "[wl] bound compositor v{} wm_base v{} seat v{} cursor-shape {} fractional-scale {} shortcuts-inhibit {} presentation {}",
            compositor.version(),
            wmbase.version(),
            seat.version(),
            if cursor_mgr.is_some() { "v1" } else { "unavailable" },
            if fs_mgr.is_some() { "v1" } else { "unavailable" },
            if inhibit_mgr.is_some() { "v1" } else { "unavailable" },
            match presentation.as_ref().map(|p| p.version()) {
                Some(v) => format!("v{v}"),
                None => "unavailable".to_string(),
            },
        );
        // Direct-scanout capabilities: KWin decides based on the buffer's
        // dmabuf format/modifier. Which buffer-allocation protocols the
        // compositor advertises (and their versions) tells us whether the
        // driver can use linux-dmabuf feedback (scanout-capable modifiers)
        // or the legacy wl_drm path (fixed format set).
        {
            let mut caps: Vec<String> = globals
                .contents()
                .clone_list()
                .iter()
                .filter(|g| {
                    g.interface.contains("dmabuf")
                        || g.interface.contains("drm")
                        || g.interface.contains("syncobj")
                        || g.interface.contains("linux")
                })
                .map(|g| format!("{} v{}", g.interface, g.version))
                .collect();
            caps.sort();
            log::debug!("[wl] scanout-related globals: {}", caps.join(", "));
        }

        let surface = compositor.create_surface(&qh, ());
        let viewport = viewporter.get_viewport(&surface, &qh, ());
        // surface feedback (v4+): KWin sends the format/modifier tranches for
        // this surface, including the direct-scanout set
        let dmabuf_fb = dmabuf.as_ref().and_then(|d| {
            (d.version() >= 4).then(|| d.get_surface_feedback(&surface, &qh, ()))
        });
        let fs_obj: Option<WpFractionalScaleV1> =
            fs_mgr.as_ref().map(|m| m.get_fractional_scale(&surface, &qh, ()));
        let xdg_surface = wmbase.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title("arsa".into());
        toplevel.set_app_id("arsa-client".into());
        // 1:1 fixed-size window (resize/renegotiation is M7); fullscreen
        // hands sizing to the compositor instead (configure-driven)
        xdg_surface.set_window_geometry(0, 0, width as i32, height as i32);
        if fullscreen {
            toplevel.set_fullscreen(None);
            log::debug!("[wl] fullscreen requested (current output)");
        } else {
            toplevel.set_min_size(width as i32, height as i32);
            toplevel.set_max_size(width as i32, height as i32);
        }
        // request the compositor-drawn titlebar before the map commit
        let decoration = decoration_mgr
            .as_ref()
            .map(|m| m.get_toplevel_decoration(&toplevel, &qh, ()));
        if let Some(d) = decoration.as_ref() {
            d.set_mode(Mode::ServerSide);
        }
        log::debug!(
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
                pointer: None,
                keyboard: None,
                cursor_device: None,
                pointer_serial: 0,
                rel_mgr,
                constraints,
                inhibit_mgr,
                inhibitor: None,
                presentation,
                _dmabuf: dmabuf,
                _dmabuf_fb: dmabuf_fb,
                dmabuf_feedback: DmabufFeedback::default(),
                pres_clock: libc::CLOCK_MONOTONIC,
                fbs: Vec::new(),
                fb_done: std::collections::VecDeque::new(),
                fb_next: 0,
                rel_obj: None,
                locked_obj: None,
                grabbed: false,
                grab_acc: (0.0, 0.0),
                dbg_n: 0,
                dbg_sum: (0.0, 0.0),
                dbg_window: Instant::now(),
                mods_ctrl_alt: false,
                hotkey_g_down: false,
                decoration_mgr,
                decoration,
                scale120: 120,
                logical_w: width,
                logical_h: height,
                dev_w: width,
                dev_h: height,
                fullscreened: false,
                pending_mode: None,
                pending_resize: None,
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
                held_keys: std::collections::HashSet::new(),
                held_buttons: std::collections::HashSet::new(),
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
        log::debug!("[wl] early empty commit sent (pre-configure map)");
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
        me.state.apply_scale(&qh); // apply scale if preferred_scale already arrived
        me.pump(0)?;
        if grab {
            // --grab: start in the grab (Ctrl+Alt+G's ON state). Best-effort:
            // the inhibitor/lock engage now and self-heal on focus/enter.
            me.state.set_grab(true, &qh);
            me.queue.flush()?;
        }
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

    /// Raw libwayland `struct wl_display *` for Vulkan WSI
    /// (`VkWaylandSurfaceCreateInfoKHR.display`).
    pub fn wl_display_ptr(&self) -> *mut std::ffi::c_void {
        self.conn.display().id().as_ptr() as *mut std::ffi::c_void
    }

    /// Raw libwayland `struct wl_surface *` for Vulkan WSI.
    pub fn wl_surface_ptr(&self) -> *mut std::ffi::c_void {
        self.state.surface.id().as_ptr() as *mut std::ffi::c_void
    }

    pub fn closed(&self) -> bool {
        self.state.closed
    }

    /// Current device-pixel buffer size, including fullscreen/scale changes
    /// already applied by the configure path. Used to create the initial
    /// Vulkan swapchain at the right size.
    pub fn buffer_size(&self) -> (u32, u32) {
        (self.state.dev_w.max(1), self.state.dev_h.max(1))
    }

    /// Pop a queued buffer resize (device pixels) produced by xdg configure
    /// (fullscreen/restore) or a fractional-scale change. The caller routes it
    /// to `Renderer::resize`.
    pub fn take_resize(&mut self) -> Option<(u32, u32)> {
        self.state.pending_resize.take()
    }

    /// Whether the compositor exposes wp_presentation (metrics `display`).
    pub fn has_presentation(&self) -> bool {
        self.state.presentation.is_some()
    }

    /// Request a presentation feedback for the NEXT commit of our surface.
    /// MUST be called immediately before the Vulkan present that attaches the
    /// frame. Returns the id to hand to `Metrics::record`; `None`
    /// when wp_presentation is unavailable.
    pub fn request_present_feedback(&mut self) -> Option<u64> {
        let pres = self.state.presentation.clone()?;
        let qh = self.queue.handle();
        let id = self.state.fb_next;
        self.state.fb_next += 1;
        let proxy = pres.feedback(&self.state.surface, &qh, ());
        self.state.fbs.push(Fb { id, proxy });
        Some(id)
    }

    /// Drain completed feedbacks in commit order. The caller folds each into
    /// the matching metrics sample.
    pub fn take_presented(&mut self) -> Vec<Presented> {
        self.state.fb_done.drain(..).collect()
    }
}

impl St {
    /// Enter/leave input grab (Ctrl+Alt+G). ON: the local compositor keeps
    /// its hands off the KEYBOARD (shortcuts inhibited — every key, alt-f4,
    /// alt-tab, alt-f1 and any other desktop binding included, arrives as a
    /// plain key event and is forwarded to the guest) and off the MOUSE
    /// (pointer locked in relative mode, local cursor hidden). OFF: the
    /// compositor gets its shortcuts and the absolute cursor back.
    ///
    /// Ctrl+Alt+G is the ONLY way out. A focus shuffle deliberately does not
    /// release the grab (that was the hole through which alt-tab and friends
    /// dumped the user out of the game mid-play), so the pieces self-heal
    /// instead — see the Enter (re-lock on pointer re-entry) and Unlocked
    /// (re-lock after the compositor breaks the constraint) handlers. The
    /// grab is also kept across keyboard focus loss: if the compositor yanks
    /// focus (lock screen, popup), the grab re-engages when the window is
    /// focused again and Ctrl+Alt+G works there.
    fn set_grab(&mut self, on: bool, qh: &QueueHandle<St>) {
        if on == self.grabbed {
            return;
        }
        if on {
            let kb_ok = self.grab_keys(qh);
            let ptr_ok = self.grab_pointer(qh);
            if !kb_ok && !ptr_ok {
                log::warn!(
                    "[wl] grab unavailable: compositor supports neither \
                     keyboard-shortcuts-inhibit nor relative-pointer/pointer-constraints"
                );
                return;
            }
            self.grabbed = true;
            self.dbg_n = 0;
            self.dbg_sum = (0.0, 0.0);
            self.dbg_window = Instant::now();
            log::debug!("[wl] input grabbed (Ctrl+Alt+G to release)");
        } else {
            self.release_keys();
            self.release_pointer();
            self.grabbed = false;
            self.grab_acc = (0.0, 0.0);
            log::debug!(
                "[wl] grab session: {} relative events, Σx={:+.1} Σy={:+.1}",
                self.dbg_n, self.dbg_sum.0, self.dbg_sum.1
            );
            log::debug!("[wl] input released (compositor back in control)");
        }
    }

    /// Keyboard grab: `keyboard-shortcuts-inhibit` makes the compositor pass
    /// its OWN shortcuts to us as plain keys (alt-f4/alt-tab/alt-f1, ...),
    /// which we forward to the guest. No client-side trick can take the
    /// desktop's shortcuts away — without the global there is nothing to
    /// grab, and we say so instead of pretending.
    fn grab_keys(&mut self, qh: &QueueHandle<St>) -> bool {
        if self.inhibitor.is_some() {
            return true;
        }
        let Some(mgr) = self.inhibit_mgr.clone() else {
            log::warn!(
                "[wl] keyboard grab UNAVAILABLE: no keyboard-shortcuts-inhibit \
                 global — desktop shortcuts (alt-f4/alt-tab/...) still hit the \
                 local desktop"
            );
            return false;
        };
        self.inhibitor = Some(mgr.inhibit_shortcuts(&self.surface, &self.seat, qh, ()));
        log::debug!("[wl] keyboard grabbed (compositor shortcuts inhibited)");
        true
    }

    /// give the compositor its shortcuts back. The destructor request is
    /// sent EXPLICITLY: dropping the proxy alone never sends it (same trap
    /// as the pointer constraint below) and the desktop would stay mute.
    fn release_keys(&mut self) {
        if let Some(inh) = self.inhibitor.take() {
            inh.destroy();
            log::debug!("[wl] keyboard released (compositor shortcuts restored)");
        }
    }

    /// Mouse grab: lock the local pointer (compositor freezes the cursor) and
    /// subscribe to relative motion; hide the local cursor since the guest
    /// draws its own (FPS games especially). Idempotent and safe to call
    /// without hover: a lock requested for a surface without pointer focus
    /// stays dormant until the pointer enters (pointer-constraints spec) —
    /// which is also how a broken lock gets re-acquired.
    fn grab_pointer(&mut self, qh: &QueueHandle<St>) -> bool {
        if self.locked_obj.is_some() {
            return true;
        }
        let (Some(rel_mgr), Some(con), Some(pointer)) = (
            self.rel_mgr.clone(),
            self.constraints.clone(),
            self.pointer.clone(),
        ) else {
            log::warn!(
                "[wl] mouse grab UNAVAILABLE: compositor lacks \
                 relative-pointer/pointer-constraints"
            );
            return false;
        };
        let rel = rel_mgr.get_relative_pointer(&pointer, qh, ());
        let locked =
            con.lock_pointer(&self.surface, &pointer, None, Lifetime::Persistent, qh, ());
        self.rel_obj = Some(rel);
        self.locked_obj = Some(locked);
        self.grab_acc = (0.0, 0.0);
        if self.pointer_active {
            // hide the local cursor (guest games draw their own; desktop
            // users get the guest cursor back on release)
            pointer.set_cursor(self.pointer_serial, None, 0, 0);
            log::debug!("[wl] mouse grabbed (relative mode)");
        } else {
            log::debug!("[wl] mouse not over the window: it locks when it enters");
        }
        true
    }

    /// tear down the constraint objects only (no cursor/UI side effects) —
    /// shared by the release path and the re-lock-after-break path
    fn drop_pointer_constraint(&mut self) {
        // destroy EXPLICITLY: both interfaces have a destructor request, and
        // dropping the proxy alone never sends it — the compositor kept the
        // constraint (and the frozen cursor) alive until the next focus
        // change forced a constraint break (observed on KWin)
        if let Some(p) = self.rel_obj.take() {
            p.destroy();
        }
        if let Some(p) = self.locked_obj.take() {
            p.destroy();
        }
    }

    fn release_pointer(&mut self) {
        let had_lock = self.locked_obj.is_some() || self.rel_obj.is_some();
        self.drop_pointer_constraint();
        self.grab_acc = (0.0, 0.0);
        if self.pointer_active {
            if let Some(dev) = self.cursor_device.as_ref() {
                dev.set_shape(self.pointer_serial, Shape::Default);
            }
        }
        if had_lock {
            log::debug!("[wl] mouse released (absolute mode)");
        }
    }

    /// Windowed mode: recompute the logical window size for the
    /// compositor's output scale — buffer stays native (W×H, the guest
    /// frame verbatim), the viewport destination becomes native/scale so
    /// the compositor maps the buffer 1:1 onto physical pixels.
    fn apply_scale(&mut self, qh: &QueueHandle<St>) {
        if self.fullscreened {
            // fullscreen: the logical size is configure-driven (output
            // size); a scale change only shifts the device-pixel buffer
            // size (e.g. output moved to a monitor with another scale)
            self.queue_swapchain_resize();
            self.set_opaque(qh, self.logical_w, self.logical_h);
            return;
        }
        let lw = (self.width as u64 * 120 / self.scale120.max(1) as u64) as u32;
        let lh = (self.height as u64 * 120 / self.scale120.max(1) as u64) as u32;
        if (lw, lh) == (self.logical_w, self.logical_h) {
            self.set_opaque(qh, lw, lh);
            return;
        }
        self.apply_mode(qh, lw, lh);
    }

    /// Mark the whole surface opaque for the current logical size. KWin will
    /// only direct-scanout (skip compositing) a fullscreen surface it knows is
    /// fully opaque; without this the buffer is treated as possibly
    /// translucent and always composited. Re-set on every size change; the
    /// compositor copies the region when it processes the request, so the
    /// wl_region is destroyed immediately.
    fn set_opaque(&self, qh: &QueueHandle<St>, w: u32, h: u32) {
        let region = self.compositor.create_region(qh, ());
        region.add(0, 0, w.max(1) as i32, h.max(1) as i32);
        self.surface.set_opaque_region(Some(&region));
        region.destroy();
        log::debug!(
            "[wl] opaque region {w}x{h} (direct scanout eligible when fullscreen)"
        );
    }

    /// Apply a target logical size: viewport destination, window geometry,
    /// min/max (windowed only), Vulkan swapchain resize, commit. Called from
    /// the xdg_surface.ack_configure path so the ack precedes our commit.
    fn apply_mode(&mut self, qh: &QueueHandle<St>, lw: u32, lh: u32) {
        self.logical_w = lw;
        self.logical_h = lh;
        // At scale 1.0 the buffer and the logical size are identical, so drop
        // the viewport scaling entirely: some compositors refuse direct scanout
        // when a wp_viewport destination is set, even when it is a no-op.
        if self.scale120 == 120 {
            self.viewport.set_destination(-1, -1);
        } else {
            self.viewport.set_destination(lw as i32, lh as i32);
        }
        self.xdg_surface.set_window_geometry(0, 0, lw as i32, lh as i32);
        self.set_opaque(qh, lw, lh);
        if !self.fullscreened {
            self.toplevel.set_min_size(lw as i32, lh as i32);
            self.toplevel.set_max_size(lw as i32, lh as i32);
        }
        self.queue_swapchain_resize();
        log::debug!(
            "[wl] mode: {}scale {}/120, window {}x{} logical, buffer {}x{} device (frame {}x{})",
            if self.fullscreened { "fullscreen, " } else { "" },
            self.scale120,
            lw,
            lh,
            self.dev_w,
            self.dev_h,
            self.width,
            self.height
        );
        log::debug!(
            "[wl] scanout-check: fullscreen={} buffer={}x{} scale={}/120 viewport_dest={}x{} opaque={}x{} frame={}x{}",
            self.fullscreened,
            self.dev_w,
            self.dev_h,
            self.scale120,
            lw,
            lh,
            lw,
            lh,
            self.width,
            self.height
        );
        self.surface.commit();
    }

    /// Recompute the device-pixel buffer size (logical × fractional scale)
    /// and queue a Vulkan resize if it changed. No commit: the viewport
    /// destination (logical) is unchanged, so compositor mapping is intact.
    fn queue_swapchain_resize(&mut self) {
        let dw = ((self.logical_w as u64 * self.scale120 as u64 / 120) as u32).max(1);
        let dh = ((self.logical_h as u64 * self.scale120 as u64 / 120) as u32).max(1);
        if (dw, dh) != (self.dev_w, self.dev_h) {
            self.dev_w = dw;
            self.dev_h = dh;
            self.pending_resize = Some((dw, dh));
        }
    }

    /// Queue a presentation result for the main loop (bounded).
    fn push_presented(&mut self, p: Presented) {
        if self.fb_done.len() >= 256 {
            self.fb_done.pop_front();
        }
        self.fb_done.push_back(p);
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
quiet!(WlRegion => ());
quiet!(WpCursorShapeManagerV1 => ());
quiet!(WpCursorShapeDeviceV1 => ());
quiet!(WpViewporter => ());
quiet!(WpViewport => ());
quiet!(ZwpLinuxDmabufV1 => ());
quiet!(WpFractionalScaleManagerV1 => ());
quiet!(ZxdgDecorationManagerV1 => ());
quiet!(ZwpRelativePointerManagerV1 => ());
quiet!(ZwpPointerConstraintsV1 => ());
quiet!(ZwpKeyboardShortcutsInhibitManagerV1 => ());

/// Pointer-constraint lifecycle: the compositor may break the lock at any
/// moment (focus change, its own UI, output churn). Ctrl+Alt+G is the only
/// way OUT of the grab, so a break must not end it either — re-lock instead.
impl Dispatch<ZwpLockedPointerV1, ()> for St {
    fn event(
        state: &mut St,
        obj: &ZwpLockedPointerV1,
        ev: <ZwpLockedPointerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<St>,
    ) {
        match ev {
            zwp_locked_pointer_v1::Event::Locked => {
                log::debug!("[wl] pointer lock active (relative mode)");
            }
            zwp_locked_pointer_v1::Event::Unlocked => {
                if !state.grabbed {
                    return;
                }
                // stale event from a constraint object we already replaced
                if state.locked_obj.as_ref().map(|p| p.id()) != Some(obj.id()) {
                    return;
                }
                log::debug!(
                    "[wl] grab: compositor broke the pointer lock — re-locking \
                     (Ctrl+Alt+G releases)"
                );
                state.drop_pointer_constraint();
                state.grab_pointer(qh);
            }
            _ => {}
        }
    }
}

/// Shortcut-inhibitor lifecycle: `active` = the compositor's own shortcuts
/// are off and alt-f4/alt-tab/alt-f1 reach us like any other key. `inactive`
/// means the compositor took them back (the protocol lets it keep an escape
/// combo of its own); there is no client-side way to force re-activation, so
/// this only warns — the grab itself stays until Ctrl+Alt+G.
impl Dispatch<ZwpKeyboardShortcutsInhibitorV1, ()> for St {
    fn event(
        _state: &mut St,
        _: &ZwpKeyboardShortcutsInhibitorV1,
        ev: <ZwpKeyboardShortcutsInhibitorV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        match ev {
            zwp_keyboard_shortcuts_inhibitor_v1::Event::Active => {
                log::debug!("[wl] keyboard grab active (compositor shortcuts inhibited)");
            }
            zwp_keyboard_shortcuts_inhibitor_v1::Event::Inactive => {
                log::warn!(
                    "[wl] keyboard grab INACTIVE: the compositor took its shortcuts \
                     back (reserved escape combo?) — desktop keys may hit the local \
                     desktop again"
                );
            }
            _ => {}
        }
    }
}

/// wp_presentation: the compositor announces which clock its feedback
/// timestamps use. KWin reports CLOCK_MONOTONIC; we honor whatever arrives so
/// the unix conversion stays correct on compositors that use CLOCK_REALTIME.
impl Dispatch<WpPresentation, ()> for St {
    fn event(
        state: &mut St,
        _: &WpPresentation,
        ev: <WpPresentation as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        if let wp_presentation::Event::ClockId { clk_id } = ev {
            state.pres_clock = clk_id as i32;
        }
    }
}

/// One feedback per presented commit. `presented` carries the on-screen time
/// (converted from the presentation clock to unix-epoch micros); `discarded`
/// means the compositor never showed that commit. Either event finishes the
/// object, so the kept-alive proxy is dropped here.
impl Dispatch<WpPresentationFeedback, ()> for St {
    fn event(
        state: &mut St,
        obj: &WpPresentationFeedback,
        ev: <WpPresentationFeedback as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        let presented = match ev {
            wp_presentation_feedback::Event::Presented {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
                refresh,
                flags,
                ..
            } => {
                let secs = ((tv_sec_hi as u64) << 32) | tv_sec_lo as u64;
                let flags = match flags {
                    wayland_client::WEnum::Value(k) => u32::from(k),
                    wayland_client::WEnum::Unknown(v) => v,
                };
                Presented {
                    id: 0, // filled from the matching Fb below
                    unix_us: presentation_to_unix_us(state.pres_clock, secs, tv_nsec),
                    presented: true,
                    flags,
                    refresh,
                }
            }
            wp_presentation_feedback::Event::Discarded => Presented {
                id: 0,
                unix_us: 0,
                presented: false,
                flags: 0,
                refresh: 0,
            },
            _ => return, // sync_output: which output; not needed here
        };
        if let Some(i) = state.fbs.iter().position(|f| f.proxy.id() == obj.id()) {
            let f = state.fbs.remove(i);
            state.push_presented(Presented { id: f.id, ..presented });
        }
    }
}

impl Dispatch<ZwpLinuxDmabufFeedbackV1, ()> for St {
    fn event(
        state: &mut St,
        _: &ZwpLinuxDmabufFeedbackV1,
        ev: zwp_linux_dmabuf_feedback_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        use zwp_linux_dmabuf_feedback_v1::Event as E;
        match ev {
            E::FormatTable { fd, size } => {
                let size = size as usize;
                if size >= 16 {
                    let n = size / 16;
                    let mut table = Vec::with_capacity(n);
                    unsafe {
                        let p = libc::mmap(
                            std::ptr::null_mut(),
                            size,
                            libc::PROT_READ,
                            libc::MAP_PRIVATE,
                            fd.as_raw_fd(),
                            0,
                        );
                        if p != libc::MAP_FAILED {
                            let bytes = std::slice::from_raw_parts(p as *const u8, size);
                            for i in 0..n {
                                let e = &bytes[i * 16..i * 16 + 16];
                                let format =
                                    u32::from_ne_bytes([e[0], e[1], e[2], e[3]]);
                                let modifier = u64::from_ne_bytes([
                                    e[8], e[9], e[10], e[11], e[12], e[13], e[14], e[15],
                                ]);
                                table.push((format, modifier));
                            }
                            libc::munmap(p, size);
                        }
                    }
                    state.dmabuf_feedback.table = table;
                    // a new format_table begins a new feedback set
                    state.dmabuf_feedback.tranches.clear();
                    state.dmabuf_feedback.cur = Tranche::default();
                }
            }
            E::MainDevice { device } => state.dmabuf_feedback.main_device = device,
            E::TrancheTargetDevice { device } => state.dmabuf_feedback.cur.target = device,
            E::TrancheFlags { flags } => {
                state.dmabuf_feedback.cur.flags = match flags {
                    wayland_client::WEnum::Value(k) => u32::from(k),
                    wayland_client::WEnum::Unknown(v) => v,
                };
            }
            E::TrancheFormats { indices } => {
                state.dmabuf_feedback.cur.formats = indices
                    .chunks_exact(2)
                    .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                    .collect();
            }
            E::TrancheDone => {
                let cur = std::mem::take(&mut state.dmabuf_feedback.cur);
                state.dmabuf_feedback.tranches.push(cur);
            }
            E::Done => state.dmabuf_feedback.log(),
            _ => {}
        }
    }
}

/// Relative motion while grabbed: forward the UNACCELERATED delta (games want
/// raw motion, not the compositor's acceleration curve) as InputEvent::Rel,
/// accumulating the fractional part between events.
///
/// wayland-rs decodes the protocol's wl_fixed (24.8) arguments itself: the
/// RelativeMotion fields arrive as `f64` already divided by 256
/// (wayland-scanner emits `(#arg as f64) / 256.`), i.e. the plain
/// logical-pixel delta. Do NOT divide again — doing so was a 256x attenuation
/// bug whose "raw counts (KWin-style)" diagnosis (commit 894de95) mistook the
/// already-decoded f64 for the raw wire integer. The wire value really is
/// count*256 on spec-compliant compositors, including KWin, whose
/// relativepointer_v1.cpp encodes with wl_fixed_from_double().
impl Dispatch<ZwpRelativePointerV1, ()> for St {
    fn event(
        state: &mut St,
        _: &ZwpRelativePointerV1,
        ev: <ZwpRelativePointerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<St>,
    ) {
        if let zwp_relative_pointer_v1::Event::RelativeMotion {
            dx_unaccel, dy_unaccel, ..
        } = ev
        {
            if !state.grabbed {
                return;
            }
            // dx_unaccel/dy_unaccel are already logical pixels (see above)
            state.grab_acc.0 += dx_unaccel;
            state.grab_acc.1 += dy_unaccel;
            state.dbg_n += 1;
            state.dbg_sum.0 += dx_unaccel;
            state.dbg_sum.1 += dy_unaccel;
            if state.dbg_window.elapsed().as_secs_f64() >= 2.0 {
                log::debug!(
                    "[wl] rel agg: {} ev Σx={:+.1} Σy={:+.1} in 2s",
                    state.dbg_n, state.dbg_sum.0, state.dbg_sum.1
                );
                state.dbg_window = Instant::now();
            }
            let dx = state.grab_acc.0.trunc() as i32;
            let dy = state.grab_acc.1.trunc() as i32;
            state.grab_acc.0 -= dx as f64;
            state.grab_acc.1 -= dy as f64;
            if dx != 0 || dy != 0 {
                let _ = state.input_tx.send(InputEvent::Rel { dx, dy });
            }
        }
    }
}

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
                    log::debug!("[wl] decoration mode: server-side (titlebar active)")
                }
                other => {
                    log::debug!("[wl] decoration configure said {other:?}; re-asserting server-side");
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
        qh: &QueueHandle<St>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale, .. } = ev {
            if state.scale120 != scale {
                state.scale120 = scale;
                state.apply_scale(qh);
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
                state.pointer = Some(pointer);
            }
            if caps.contains(Capability::Keyboard) && state.keyboard.is_none() {
                state.keyboard = Some(seat.get_keyboard(qh, ()));
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
        qh: &QueueHandle<St>,
    ) {
        match ev {
            wl_keyboard::Event::Keymap { .. } => {
                // we forward raw evdev codes; the xkb keymap is not needed
            }
            wl_keyboard::Event::Enter { .. } => {
                state.kb_active = true;
                state.held_keys.clear();
                state.hotkey_g_down = false;
                // modifier state is per-focus: a chord left half-pressed in
                // the previous focus must not make the next plain M toggle
                // the grab (the fresh Modifiers event will re-set it)
                state.mods_ctrl_alt = false;
                if state.grabbed {
                    // the inhibitor re-engages by itself on keyboard focus
                    // (protocol: active ⇔ our surface has keyboard focus)
                    log::debug!("[wl] kb enter (grab still on: Ctrl+Alt+G releases)");
                } else {
                    log::debug!("[wl] kb enter (kb_active=true)");
                }
            }
            wl_keyboard::Event::Leave { .. } => {
                state.kb_active = false;
                // flush held keys: a Wayland focus break means the guest must
                // not keep any keys down (this is what made walking stop when
                // the compositor churned focus under us)
                for code in state.held_keys.drain() {
                    let _ = state.input_tx.send(InputEvent::Key { code, down: false });
                }
                state.hotkey_g_down = false;
                state.mods_ctrl_alt = false;
                if state.grabbed {
                    // NOT released here on purpose: Ctrl+Alt+G is the only
                    // exit. The compositor taking focus away (lock screen,
                    // popup) must not dump the user out of the grab — it
                    // re-engages on re-focus (see Enter above).
                    log::debug!(
                        "[wl] kb leave (grab stays on; Ctrl+Alt+G releases once \
                         the window is focused again)"
                    );
                } else {
                    log::debug!("[wl] kb leave (kb_active=false, held keys flushed)");
                }
            }
            wl_keyboard::Event::Modifiers { mods_depressed, .. } => {
                // depressed modifier mask: Control = bit2, Alt = bit3 (xkb)
                state.mods_ctrl_alt =
                    mods_depressed & (0x4 | 0x8) == (0x4 | 0x8);
            }
            wl_keyboard::Event::Key { key, state: st, .. } => {
                // KWin repeats held keys with state=2 (WL_KEYBOARD_KEY_STATE_
                // REPEATED, v10+) which surfaces as WEnum::Raw — anything but
                // an explicit Released means "down". Forward only true
                // press/release transitions: a spurious or repeated release
                // stops the guest game mid-walk (verified by the up-flood in
                // the grab logs), and repeat presses add nothing over uinput.
                let released = matches!(st, WEnum::Value(KeyState::Released));
                log::debug!(
                    "[wl] key {key:#04x} {} kb_active={}",
                    if released { "up  " } else { "down" },
                    state.kb_active
                );
                if !state.kb_active {
                    return;
                }
                // grab hotkey: Ctrl+Alt+G toggles the input grab (keyboard
                // shortcut inhibition + relative mouse) — client-local,
                // never forwarded to the guest, and the ONLY way in or out.
                // Both the press and the release of the chorded G are
                // swallowed; only the INITIAL press toggles, so a key repeat
                // (KWin sends state=2, surfaced as "down") must not re-toggle
                // the grab.
                if key == 0x22 {
                    if !released {
                        if state.mods_ctrl_alt {
                            if !state.hotkey_g_down {
                                state.hotkey_g_down = true;
                                state.set_grab(!state.grabbed, qh);
                            }
                            return; // chorded press/repeat: never forwarded
                        }
                        // plain G: fall through and forward normally
                    } else if state.hotkey_g_down {
                        // Release of a swallowed chord press. Ctrl/Alt are
                        // usually already up by now, so this is recognised
                        // from the flag alone — and it swallows only G, so
                        // the modifiers' own releases still reach the guest.
                        state.hotkey_g_down = false;
                        return;
                    }
                }
                // `key` is the Linux evdev keycode (no xkb +8 offset needed
                // because we never interpret the keymap)
                let Some(code) = keymap::linux_to_qnum(key) else {
                    if state.grabbed {
                        log::debug!(
                            "[wl] key {key:#04x} has no qnum mapping — dropped \
                             (while grabbed, neither the guest nor the desktop sees it)"
                        );
                    }
                    return; // unmapped key
                };
                if released {
                    if state.held_keys.remove(&key) || true {
                        let _ = state.input_tx.send(InputEvent::Key { code, down: false });
                    }
                    return;
                }
                // down (initial press OR repeat)
                if state.held_keys.contains(&key) {
                    return; // repeat: guest compositor repeats on its own
                }
                state.held_keys.insert(key);
                let _ = state.input_tx.send(InputEvent::Key { code, down: true });
            }
            _ => {} // latched/locked mods, repeat_info
        }
    }
}

/// Pointer: absolute positioning (default), grab mode (Ctrl+Alt+G) for
/// relative motion, buttons, wheel.
impl Dispatch<WlPointer, ()> for St {
    fn event(
        state: &mut St,
        pointer: &WlPointer,
        ev: <WlPointer as Proxy>::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<St>,
    ) {
        match ev {
            wl_pointer::Event::Enter { serial, .. } => {
                state.pointer_active = true;
                state.pointer_serial = serial;
                if state.grabbed {
                    // self-heal: the lock can be missing here (grab started
                    // without hover, or the compositor broke the constraint
                    // while the pointer was away) — re-acquire it rather
                    // than end the grab (Ctrl+Alt+G is the only exit)
                    state.grab_pointer(qh);
                    // keep the local cursor hidden while grabbed (re-enter)
                    pointer.set_cursor(serial, None, 0, 0);
                } else {
                    // name our cursor: the plain arrow (compositors pick their
                    // own fallback for silent clients; KWin's is the text beam)
                    if let Some(dev) = state.cursor_device.as_ref() {
                        dev.set_shape(serial, Shape::Default);
                    }
                }
            }
            wl_pointer::Event::Leave { .. } => {
                state.pointer_active = false;
                state.pending_axis = None;
                // same reasoning as the keyboard flush: if the pointer leaves
                // mid-press (the compositor can break the locked-pointer
                // constraint at any moment), the release may never arrive and
                // the guest would keep dragging a window with no button held
                for btn in state.held_buttons.drain() {
                    let _ = state.input_tx.send(InputEvent::Btn { button: btn, down: false });
                }
                if state.grabbed {
                    // NOT released here on purpose: Ctrl+Alt+G is the only
                    // exit (the old auto-release meant any focus churn — a
                    // desktop shortcut, an OSD, another output — dumped the
                    // user out of relative mode mid-play). The lock is
                    // re-acquired on re-enter / on the Unlocked event.
                    log::debug!(
                        "[wl] grab: pointer left the window — grab stays on \
                         (Ctrl+Alt+G releases)"
                    );
                }
            }
            wl_pointer::Event::Motion { surface_x, surface_y, .. } => {
                if !state.pointer_active || state.grabbed {
                    return;
                }
                // surface-local coords are LOGICAL pixels (scaled by the
                // compositor's output scale). Convert to device pixels,
                // then map through the letterboxed picture rect (bars
                // clamp to the guest edge): guest_px = (dev - origin)/fit.
                let sx = state.scale120 as f64 / 120.0;
                let dw = state.dev_w.max(1) as f64;
                let dh = state.dev_h.max(1) as f64;
                let px = surface_x as f64 * sx;
                let py = surface_y as f64 * sx;
                let fit = (dw / state.width as f64).min(dh / state.height.max(1) as f64);
                let ox = (dw - state.width as f64 * fit) * 0.5;
                let oy = (dh - state.height as f64 * fit) * 0.5;
                let x = (((px - ox) / fit).floor().clamp(
                    0.0,
                    (state.width - 1) as f64,
                )) as u32;
                let y = (((py - oy) / fit).floor().clamp(
                    0.0,
                    (state.height - 1) as f64,
                )) as u32;
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
                        log::debug!("[wl] unmapped pointer button {other:#x}");
                        return;
                    }
                };
                let down = matches!(st, WEnum::Value(ButtonState::Pressed));
                if down {
                    state.held_buttons.insert(dbtn);
                } else {
                    state.held_buttons.remove(&dbtn);
                }
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
        qh: &QueueHandle<St>,
    ) {
        if let xdg_surface::Event::Configure { serial } = ev {
            obj.ack_configure(serial);
            state.configured = true;
            // Apply a staged mode change (fullscreen/restore/resize) AFTER
            // the ack: xdg-shell requires the ack to precede any commit
            // that adapts to the configure.
            if let Some((lw, lh)) = state.pending_mode.take() {
                state.apply_mode(qh, lw, lh);
            }
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
            xdg_toplevel::Event::Configure { width, height, states } => {
                // The states array is uint32 values; Fullscreen (2) means
                // the compositor has us fullscreen — whether from our own
                // set_fullscreen request or its own toggle shortcut, the
                // reaction is the same: honor the configure's size.
                let fs = states.chunks_exact(4).any(|c| {
                    u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                        == xdg_toplevel::State::Fullscreen as u32
                });
                if fs != state.fullscreened {
                    state.fullscreened = fs;
                    log::debug!(
                        "[wl] toplevel {} (compositor configure)",
                        if fs { "FULLSCREEN" } else { "windowed" }
                    );
                }
                // Fullscreen configure always carries the output's logical
                // size; a windowed 0x0 means "no opinion" — restore the
                // fixed frame-size window.
                let target = if fs {
                    if width > 0 && height > 0 {
                        Some((width as u32, height as u32))
                    } else {
                        Some((state.logical_w, state.logical_h))
                    }
                } else {
                    let lw = (state.width as u64 * 120 / state.scale120.max(1) as u64) as u32;
                    let lh = (state.height as u64 * 120 / state.scale120.max(1) as u64) as u32;
                    if width > 0 && height > 0 && (width as u32, height as u32) != (lw, lh) {
                        // compositor suggests a floating size: honor it only
                        // if it matches the frame aspect (else keep fixed)
                        let fa = state.width as f64 / state.height.max(1) as f64;
                        let ca = width as f64 / height.max(1) as f64;
                        if (fa - ca).abs() < 0.01 {
                            Some((width as u32, height as u32))
                        } else {
                            Some((lw, lh))
                        }
                    } else {
                        Some((lw, lh))
                    }
                };
                // windowed no-op must still be staged: min/max may need
                // re-asserting after a fullscreen round trip
                state.pending_mode = target;
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
