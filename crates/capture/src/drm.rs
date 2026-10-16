//! DRM/KMS capture source (Sunshine-style): mirror the host's primary plane
//! by grabbing its current framebuffer as a dmabuf.
//!
//! No compositor cooperation needed: open a card node, find the bound
//! *primary plane* (atomic-only drivers like nvidia-drm never populate the
//! legacy `crtc.fb_id`, so the fb must be read from the plane state), and
//! poll `plane.fb_id`. The compositor flips the fb on every repaint — that
//! flip *is* our dirty signal (same semantics as QEMU's alternating
//! ScanoutDMABUF announcements, so the existing Scanout → EglMaterializer
//! path consumes it unchanged).
//!
//! Permissions: GETFB2 leaks GEM handles, so the kernel gates it on DRM
//! master *or* CAP_SYS_ADMIN. The compositor holds master; run the server
//! under sudo (as always) and we inherit CAP_SYS_ADMIN. We deliberately do
//! NOT call drmSetMaster (it would fight the running compositor).
//!
//! Requires kernel ≥ 5.8 (GETFB2); no libdrm dependency — hand-rolled ioctl
//! wrappers, same style as the verbs/EGL code.
//!
//! Polling: we re-read `plane.fb_id` every 1 ms and emit a scanout only when it
//! changes — a flip is noticed within ~1 ms and the capture stays silent while
//! the screen is idle (no fd churn, no re-encoding the same frame). The export
//! happens in the same iteration as the read, so the fb cannot vanish in
//! between. `--capture-sync` is accepted for compatibility but not used by the
//! DRM path: a scanout dmabuf on this stack carries no usable producer fence
//! (the `syncprobe` verdict: stub fences only), so there is nothing to wait on.
//!
//! Known gaps: no cursor-plane blending (hardware cursors are invisible;
//! software cursors are in the fb), no damage info (every flip is full-frame),
//! readback is unfenced (scanout buffers are stable while scanned out; the
//! compositor's next render into the same buffer is ~1-2 flips away).

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::mpsc::Sender;

use anyhow::{bail, Context, Result};

use crate::{
    CaptureEvent, CaptureSource, CaptureSync, DmabufScanout, FrameFormat, Running, Stop,
    DEFAULT_SETTLE_MS,
};

// --- drm uapi (numbers + layouts from /usr/include/drm/drm.h, drm_mode.h) ---

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeCardRes {
    fb_id_ptr: u64,
    crtc_id_ptr: u64,
    connector_id_ptr: u64,
    encoder_id_ptr: u64,
    count_fbs: u32,
    count_crtcs: u32,
    count_connectors: u32,
    count_encoders: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeModeinfo {
    clock: u32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    vrefresh: u32,
    flags: u32,
    ty: u32,
    name: [u8; 32],
}

/// Legacy CRTC readout: GETCRTC reads the atomic state's fb on modern kernels
/// (verified live on nvidia-drm: fb_id populated, planes unbound — so the fb
/// must be taken from the CRTC, not the plane, on this driver).
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeCrtc {
    set_connectors_ptr: u64,
    count_connectors: u32,
    crtc_id: u32,
    fb_id: u32,
    x: u32,
    y: u32,
    gamma_size: u32,
    mode_valid: u32,
    mode: DrmModeModeinfo,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeGetPlaneRes {
    plane_id_ptr: u64,
    count_planes: u32,
    // trailing padding to 8-align (implicit in Rust repr(C))
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeGetPlane {
    plane_id: u32,
    crtc_id: u32,
    fb_id: u32,
    possible_crtcs: u32,
    gamma_size: u32,
    count_format_types: u32,
    format_type_ptr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmSetClientCap {
    capability: u64,
    value: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeGetEncoder {
    encoder_id: u32,
    encoder_type: u32,
    crtc_id: u32,
    possible_crtcs: u32,
    possible_clones: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeGetConnector {
    encoders_ptr: u64,
    modes_ptr: u64,
    props_ptr: u64,
    prop_values_ptr: u64,
    count_modes: u32,
    count_props: u32,
    count_encoders: u32,
    encoder_id: u32,
    connector_id: u32,
    connector_type: u32,
    connector_type_id: u32,
    connection: u32,
    mm_width: u32,
    mm_height: u32,
    subpixel: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeObjGetProperties {
    props_ptr: u64,
    prop_values_ptr: u64,
    count_props: u32,
    obj_id: u32,
    obj_type: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeGetProperty {
    values_ptr: u64,
    enum_blob_ptr: u64,
    prop_id: u32,
    flags: u32,
    name: [u8; 32],
    count_values: u32,
    count_enum_blobs: u32,
}

#[repr(C)]
#[derive(Default)]
struct DrmModeFbCmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}

#[repr(C)]
#[derive(Default)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

#[repr(C)]
#[derive(Default)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

// ioctl numbers: _IOWR('d', nr, T) = (3<<30) | sizeof(T)<<16 | 'd'<<8 | nr.
// Direction bits: _IOC_NONE=0, _IOC_WRITE=1, _IOC_READ=2 — _IOWR = READ|WRITE
// = 3. (This bit is load-bearing: drm_ioctl skips copy_from_user unless the
// caller's cmd has IOC_IN set like the kernel's registered cmd.)
const D: u64 = 'd' as u64;
const fn iowr(nr: u16, size: usize) -> u64 {
    (3 << 30) | ((size as u64) << 16) | (D << 8) | nr as u64
}
fn ioctl(
    fd: i32,
    req: u64,
    arg: *mut core::ffi::c_void,
) -> Result<i32> {
    // SAFETY: caller passes the matching struct for each ioctl number
    let r = unsafe { libc::ioctl(fd, req as libc::c_ulong, arg) };
    if r < 0 {
        Err(std::io::Error::last_os_error().into())
    } else {
        Ok(r)
    }
}

const DRM_IOCTL_MODE_GETRESOURCES: u64 = iowr(0xA0, std::mem::size_of::<DrmModeCardRes>());
const DRM_IOCTL_MODE_GETPLANERESOURCES: u64 =
    iowr(0xB5, std::mem::size_of::<DrmModeGetPlaneRes>());
const DRM_IOCTL_MODE_GETCRTC: u64 = iowr(0xA1, std::mem::size_of::<DrmModeCrtc>());
const DRM_IOCTL_MODE_GETPLANE: u64 = iowr(0xB6, std::mem::size_of::<DrmModeGetPlane>());
const DRM_IOCTL_MODE_GETENCODER: u64 = iowr(0xA6, std::mem::size_of::<DrmModeGetEncoder>());
const DRM_IOCTL_MODE_GETCONNECTOR: u64 =
    iowr(0xA7, std::mem::size_of::<DrmModeGetConnector>());
const DRM_IOCTL_MODE_OBJ_GETPROPERTIES: u64 =
    iowr(0xB9, std::mem::size_of::<DrmModeObjGetProperties>());
const DRM_IOCTL_MODE_GETPROPERTY: u64 =
    iowr(0xAA, std::mem::size_of::<DrmModeGetProperty>());
const DRM_IOCTL_MODE_GETFB2: u64 = iowr(0xCE, std::mem::size_of::<DrmModeFbCmd2>());
const DRM_IOCTL_PRIME_HANDLE_TO_FD: u64 =
    iowr(0x2d, std::mem::size_of::<DrmPrimeHandle>());
const DRM_IOCTL_GEM_CLOSE: u64 = iowr_ioctl(0x09, std::mem::size_of::<DrmGemClose>());
const DRM_IOCTL_SET_CLIENT_CAP: u64 =
    iowr_ioctl(0x0D, std::mem::size_of::<DrmSetClientCap>());

// plane object type for OBJ_GETPROPERTIES (drm_mode.h)
const DRM_MODE_OBJECT_PLANE: u32 = 0xeeee_eeee;
// plane "type" property enum values (kernel drm_plane_type): 0=overlay
// 1=primary 2=cursor
const DRM_PLANE_TYPE_CURSOR: u64 = 2;
// drm_set_client_cap capabilities (drm.h)
const DRM_CLIENT_CAP_UNIVERSAL_PLANES: u64 = 2;
const DRM_CLIENT_CAP_ATOMIC: u64 = 3;

// GEM_CLOSE is _IOW (write dir, =1), not _IOWR
const fn iowr_ioctl(nr: u16, size: usize) -> u64 {
    (1 << 30) | ((size as u64) << 16) | (D << 8) | nr as u64
}

// --- dma-buf UAPI (include/uapi/linux/dma-buf.h) ----------------------------
// DMA_BUF_IOCTL_EXPORT_SYNC_FILE (kernel >= 5.20) exports the reservation's
// read-side fence as a sync_file, so a non-master observer can wait for the
// producer's GPU work before reading a buffer. This is the probe for whether
// KMS capture can read the CURRENT scanout fb with no tearing and no
// one-frame lag (i.e. drop the one-behind workaround).
const DMA_BUF_BASE: u64 = 'b' as u64;
const fn iowr_buf(nr: u16, size: usize) -> u64 {
    (3 << 30) | ((size as u64) << 16) | (DMA_BUF_BASE << 8) | nr as u64
}

#[repr(C)]
#[derive(Default)]
struct DmaBufExportSyncFile {
    flags: u32,
    fd: i32,
}

const DMA_BUF_SYNC_READ: u32 = 1 << 0;
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: u64 =
    iowr_buf(2, std::mem::size_of::<DmaBufExportSyncFile>());

// --- sync_file UAPI (include/uapi/linux/sync_file.h) ------------------------
// SYNC_IOC_FILE_INFO lets us inspect the exported sync_file: the number of
// fences and, per fence, the driver name and status. This is how we tell a
// *real* producer fence from the always-signaled **stub** fence the kernel
// substitutes when there is no fence on the reservation (driver name "stub").
#[repr(C)]
#[derive(Default)]
struct SyncFileInfo {
    name: [u8; 32],
    status: i32,
    flags: u32,
    num_fences: u32,
    pad: u32,
    sync_fence_info: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SyncFenceInfo {
    obj_name: [u8; 32],
    driver_name: [u8; 32],
    /// 0 = active/pending, 1 = signaled, <0 = error
    status: i32,
    flags: u32,
    timestamp_ns: u64,
}

const SYNC_MAGIC: u64 = '>' as u64;
const fn iowr_sync(nr: u16, size: usize) -> u64 {
    (3 << 30) | ((size as u64) << 16) | (SYNC_MAGIC << 8) | nr as u64
}
const SYNC_IOC_FILE_INFO: u64 = iowr_sync(4, std::mem::size_of::<SyncFileInfo>());

// DRM_FORMAT_MOD_INVALID = fourcc_mod_code(NONE, 0x00ffffff...); EGL refuses
// it, and it means "driver gave no real modifier" — treat as linear.
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

pub struct DrmKmsSource {
    /// Card node to capture on. None = probe /dev/dri/card0.. card8 for the
    /// first device with an active CRTC.
    pub card: Option<String>,
    /// DRM connector to capture (e.g. `DP-1`). None = the first active CRTC.
    pub connector: Option<String>,
    /// Which fb to capture relative to the compositor's flips.
    pub sync: CaptureSync,
}

impl DrmKmsSource {
    pub fn new(card: Option<String>, connector: Option<String>) -> Self {
        Self {
            card,
            connector,
            sync: CaptureSync::Auto,
        }
    }

    /// Pick the flip-sync policy explicitly (see [`CaptureSync`]).
    pub fn with_sync(card: Option<String>, connector: Option<String>, sync: CaptureSync) -> Self {
        Self {
            card,
            connector,
            sync,
        }
    }
}

/// Cheap preflight probe for the DRM source: open the card, find the scanout
/// (the named `connector`, or the first active one), and read the current
/// framebuffer's format — without starting the polling capture thread. Used at
/// server init to validate the setup and to cache the display geometry so a
/// client connect can negotiate slots before the first captured frame arrives.
/// The card is closed on return.
pub fn probe_display(card: Option<&str>, connector: Option<&str>) -> Result<FrameFormat> {
    let (fd, target) = open_card(card, connector)?;
    let fdi = fd.as_raw_fd();
    let fb = match target {
        Target::Crtc(id) => get_crtc(fdi, id)?.fb_id,
        Target::Plane(id) => get_plane(fdi, id)?.fb_id,
    };
    if fb == 0 {
        bail!("no bound framebuffer (display off?)");
    }
    let d = import_fb(&fd, fb)?;
    Ok(d.format.clone())
}

impl CaptureSource for DrmKmsSource {
    fn name(&self) -> &'static str {
        "drm-kms"
    }

    fn spawn(self: Box<Self>, tx: Sender<CaptureEvent>) -> Result<Running> {
        let sync = self.sync;
        let stop = Stop::new();
        let worker_stop = stop.clone();
        let join = std::thread::Builder::new()
            .name("capture-drm".into())
            .spawn(move || {
                if let Err(e) = run(self.card.as_deref(), self.connector.as_deref(), sync, tx, worker_stop) {
                    log::warn!("[capture] source failed: {e:#}");
                }
            })
            .context("spawn capture thread")?;
        Ok(Running::new(stop, vec![join]))
    }
}

/// Open a card node and return (fd, target) for the active display.
///
/// When `connector` is given (a DRM connector name like `DP-1`), it is resolved
/// to its current CRTC and used directly. The target is then a *bound plane* on
/// that CRTC when one exists (see [`find_plane_for_crtc`]), else the CRTC.
///
/// We enable `DRM_CLIENT_CAP_UNIVERSAL_PLANES` + `DRM_CLIENT_CAP_ATOMIC` before
/// probing (as Sunshine does). Without them the legacy getters report stale or
/// zero state on an atomic compositor, so a fullscreen game's direct scanout is
/// invisible and the capture sees a static placeholder.
fn open_card(preferred: Option<&str>, connector: Option<&str>) -> Result<(OwnedFd, Target)> {
    let candidates: Vec<String> = match preferred {
        Some(p) => vec![p.to_string()],
        None => (0..8).map(|i| format!("/dev/dri/card{i}")).collect(),
    };
    let mut last_err: Option<anyhow::Error> = None;
    for path in &candidates {
        let r = open_card_and_find_target(path, connector);
        match r {
            Ok(v) => return Ok(v),
            Err(e) => last_err = Some(e),
        }
    }
    let last = last_err.context("no /dev/dri/cardN with an active scanout")?;
    Err(last).context("drm capture needs root (CAP_SYS_ADMIN) and an active display")
}

/// Where the fb pointer lives on this card: the CRTC (legacy GETCRTC) or a
/// primary plane (GETPLANE). Poll the same object we selected on.
#[derive(Debug, Clone, Copy)]
enum Target {
    Crtc(u32),
    Plane(u32),
}

/// `drm_mode_connector_type` -> the name the kernel, sysfs and Wayland
/// compositors use. A connector's full name is `<type>-<type_id>`, e.g. `DP-1`,
/// `HDMI-A-1`, `eDP-1`.
fn connector_type_name(t: u32) -> &'static str {
    match t {
        1 => "VGA",
        2 => "DVI-I",
        3 => "DVI-D",
        4 => "DVI-A",
        5 => "Composite",
        6 => "SVIDEO",
        7 => "LVDS",
        8 => "Component",
        9 => "DIN",
        10 => "DP",
        11 => "HDMI-A",
        12 => "HDMI-B",
        13 => "TV",
        14 => "eDP",
        15 => "Virtual",
        16 => "DSI",
        17 => "DPI",
        18 => "Writeback",
        19 => "SPI",
        20 => "USB",
        _ => "Unknown",
    }
}

/// Resolve a connector name (`DP-1`, `HDMI-A-1`, `eDP-1` — the same string
/// sysfs and the compositors use) to the CRTC it is currently bound to.
fn resolve_connector_crtc(fd: i32, want: &str) -> Result<u32> {
    let mut res = DrmModeCardRes {
        count_connectors: 0,
        ..zeroed()
    };
    ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res as *mut _ as *mut _)
        .context("GETRESOURCES")?;
    let n = res.count_connectors as usize;
    if n == 0 {
        bail!("no connectors");
    }
    let mut conns: Vec<u32> = vec![0; n];
    let mut res = DrmModeCardRes {
        connector_id_ptr: conns.as_mut_ptr() as u64,
        count_connectors: n as u32,
        ..zeroed()
    };
    ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res as *mut _ as *mut _)
        .context("GETRESOURCES #2")?;

    let mut names: Vec<String> = Vec::new();
    for &id in conns.iter().take(res.count_connectors as usize) {
        let mut c: DrmModeGetConnector = zeroed();
        c.connector_id = id;
        ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut c as *mut _ as *mut _)
            .with_context(|| format!("GETCONNECTOR {id}"))?;
        let name = format!(
            "{}-{}",
            connector_type_name(c.connector_type),
            c.connector_type_id
        );
        if name.eq_ignore_ascii_case(want) {
            if c.encoder_id == 0 {
                bail!("connector {name} is not bound to an encoder (disabled?)");
            }
            let mut e: DrmModeGetEncoder = zeroed();
            e.encoder_id = c.encoder_id;
            ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &mut e as *mut _ as *mut _)
                .with_context(|| format!("GETENCODER {}", c.encoder_id))?;
            if e.crtc_id == 0 {
                bail!("connector {name} is not bound to a CRTC (disabled?)");
            }
            log::debug!("[capture-drm] connector {name} -> crtc {}", e.crtc_id);
            return Ok(e.crtc_id);
        }
        names.push(name);
    }
    bail!("no connector named {want:?} (have: {})", names.join(", "));
}

fn open_card_and_find_target(path: &str, connector: Option<&str>) -> Result<(OwnedFd, Target)> {
    use std::os::fd::FromRawFd;
    // SAFETY: NUL-terminated path, fd < 0 checked before wrapping
    let fd = unsafe {
        libc::open(
            format!("{path}\0").as_ptr() as *const libc::c_char,
            libc::O_RDWR | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        bail!("open {path}: {}", std::io::Error::last_os_error());
    }
    let card = unsafe { OwnedFd::from_raw_fd(fd) };
    let fdr = card.as_raw_fd();

    // Universal planes + atomic (what Sunshine sets): without them the legacy
    // getters report stale/zero state on an atomic compositor, so a fullscreen
    // game's direct scanout is invisible and we capture a static placeholder.
    for cap in [DRM_CLIENT_CAP_UNIVERSAL_PLANES, DRM_CLIENT_CAP_ATOMIC] {
        let mut c = DrmSetClientCap { capability: cap, value: 1 };
        match ioctl(fdr, DRM_IOCTL_SET_CLIENT_CAP, &mut c as *mut _ as *mut _) {
            Ok(_) => log::debug!("[capture-drm] client cap {cap} enabled"),
            Err(e) => log::warn!("[capture-drm] SET_CLIENT_CAP {cap}: {e:#}"),
        }
    }

    // Sunshine's plane selection: the first bound, non-cursor plane on the
    // card, in GETPLANERESOURCES order — it does NOT filter by the connector's
    // CRTC. A fullscreen game can direct-scan-out onto a plane while the CRTC
    // primary shows a stale/black placeholder, so the first *bound* plane is
    // the live scanout.
    if let Some(plane_id) = find_first_bound_plane(fdr) {
        return Ok((card, Target::Plane(plane_id)));
    }

    // Nothing bound (display off / no scanout). Sunshine bails here; we keep a
    // connector's CRTC as the target so `--display-output` mode control still
    // has an output, and the empty-plane timer re-picks once a plane appears.
    if let Some(name) = connector {
        let crtc_id = resolve_connector_crtc(fdr, name)?;
        log::debug!("[capture-drm] no bound plane; using connector {name:?} crtc {crtc_id}");
        return Ok((card, Target::Crtc(crtc_id)));
    }

    bail!("{path}: no bound plane (display off?)")
}

/// Sunshine's plane selection: the first plane in `GETPLANERESOURCES` order
/// that has a bound framebuffer and is not a cursor plane. `None` if nothing
/// is bound. (This is exactly `display_t::init`'s loop in Sunshine's
/// `kmsgrab.cpp`, for monitor index 0.)
fn find_first_bound_plane(fd: i32) -> Option<u32> {
    let mut pres = DrmModeGetPlaneRes::default();
    if ioctl(fd, DRM_IOCTL_MODE_GETPLANERESOURCES, &mut pres as *mut _ as *mut _).is_err() {
        return None;
    }
    let pn = pres.count_planes as usize;
    if pn == 0 {
        return None;
    }
    let mut planes: Vec<u32> = vec![0; pn];
    let mut pres = DrmModeGetPlaneRes {
        plane_id_ptr: planes.as_mut_ptr() as u64,
        count_planes: pn as u32,
    };
    if ioctl(fd, DRM_IOCTL_MODE_GETPLANERESOURCES, &mut pres as *mut _ as *mut _).is_err() {
        return None;
    }
    for &id in planes.iter().take(pres.count_planes as usize) {
        let Ok(p) = get_plane(fd, id) else { continue };
        if p.fb_id == 0 {
            continue;
        }
        if plane_type(fd, id).unwrap_or(u64::MAX) == DRM_PLANE_TYPE_CURSOR {
            continue;
        }
        log::debug!(
            "[capture-drm] plane {id}: crtc={} fb={} (Sunshine first bound non-cursor)",
            p.crtc_id,
            p.fb_id
        );
        return Some(id);
    }
    None
}

/// Legacy CRTC readout (scan + per-poll).
fn get_crtc(fd: i32, crtc_id: u32) -> Result<DrmModeCrtc> {
    let mut c: DrmModeCrtc = zeroed();
    c.crtc_id = crtc_id;
    ioctl(fd, DRM_IOCTL_MODE_GETCRTC, &mut c as *mut _ as *mut _)?;
    Ok(c)
}

/// GETPLANE with a pre-sized format list (drivers cap well below 64 formats,
/// so one call suffices — no two-call dance needed for the format array).
fn get_plane(fd: i32, plane_id: u32) -> Result<DrmModeGetPlane> {
    let mut formats = [0u32; 64];
    let mut p = DrmModeGetPlane {
        plane_id,
        count_format_types: 64,
        format_type_ptr: formats.as_mut_ptr() as u64,
        ..zeroed()
    };
    ioctl(fd, DRM_IOCTL_MODE_GETPLANE, &mut p as *mut _ as *mut _)?;
    Ok(p)
}

/// Plane "type" property value (1 = primary, 2 = cursor, 0 = overlay).
/// Two-call OBJ_GETPROPERTIES, then a GETPROPERTY per prop for the name.
fn plane_type(fd: i32, plane_id: u32) -> Result<u64> {
    let mut q = DrmModeObjGetProperties {
        obj_id: plane_id,
        obj_type: DRM_MODE_OBJECT_PLANE,
        count_props: 0,
        ..zeroed()
    };
    ioctl(fd, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &mut q as *mut _ as *mut _)
        .context("OBJ_GETPROPERTIES")?;
    let n = q.count_props as usize;
    if n == 0 {
        bail!("plane {plane_id}: no properties");
    }
    let mut props: Vec<u32> = vec![0; n];
    let mut vals: Vec<u64> = vec![0; n];
    q.props_ptr = props.as_mut_ptr() as u64;
    q.prop_values_ptr = vals.as_mut_ptr() as u64;
    q.count_props = n as u32;
    ioctl(fd, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &mut q as *mut _ as *mut _)
        .context("OBJ_GETPROPERTIES #2")?;
    for i in 0..n {
        if prop_name(fd, props[i]).ok().as_deref() == Some("type") {
            return Ok(vals[i]);
        }
    }
    bail!("plane {plane_id}: no \"type\" property")
}

fn prop_name(fd: i32, prop_id: u32) -> Result<String> {
    // single call with NULL arrays: the kernel fills name/flags/counts
    let mut gp = DrmModeGetProperty { prop_id, ..zeroed() };
    ioctl(fd, DRM_IOCTL_MODE_GETPROPERTY, &mut gp as *mut _ as *mut _)
        .context("GETPROPERTY")?;
    let end = gp.name.iter().position(|&c| c == 0).unwrap_or(gp.name.len());
    Ok(String::from_utf8_lossy(&gp.name[..end]).into_owned())
}

fn zeroed<T: Default>() -> T {
    T::default()
}

/// Stable identity of an exported dma-buf fd, for import caching.
///
/// The kernel keeps exactly one dma-buf per GEM object
/// (`drm_gem_prime_handle_to_dmabuf` returns the cached `obj->dma_buf`), and
/// every dma-buf is an anonymous inode with a globally unique `st_ino`
/// (`alloc_anon_inode` -> `get_next_ino`). So `(st_dev, st_ino)` names the
/// underlying buffer and is stable for its whole lifetime — unlike the DRM fb
/// id, which the kernel recycles (it hands out the lowest free id), so a
/// freed-and-recreated buffer can reuse an id and a stale import would be
/// served for the new buffer. Returns 0 if the fd can't be stat'd; callers
/// must not cache then.
fn dmabuf_identity(fd: i32) -> u64 {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return 0;
    }
    // st_ino is unique on its own; st_dev is folded in so the id stays unique
    // if the inode space were ever shared with another filesystem.
    ((st.st_dev as u64) << 32) | (st.st_ino as u64 & 0xffff_ffff)
}

fn import_fb(card: &OwnedFd, fb_id: u32) -> Result<DmabufScanout> {
    let mut fb: DrmModeFbCmd2 = zeroed();
    fb.fb_id = fb_id;
    ioctl(card.as_raw_fd(), DRM_IOCTL_MODE_GETFB2, &mut fb as *mut _ as *mut _)
        .context("GETFB2")?;
    if fb.handles[0] == 0 {
        bail!("GETFB2 returned no plane handle");
    }
    let num_planes = fb.handles.iter().filter(|&&h| h != 0).count();
    if num_planes != 1 {
        bail!("multi-plane fb not supported yet (planes={num_planes})");
    }

    // GEM handle -> dmabuf fd, then close the handle (kernel note: handles
    // from GETFB2 must be closed exactly once)
    let mut prime = DrmPrimeHandle { handle: fb.handles[0], flags: 0, fd: -1 };
    ioctl(
        card.as_raw_fd(),
        DRM_IOCTL_PRIME_HANDLE_TO_FD,
        &mut prime as *mut _ as *mut _,
    )
    .context("PRIME_HANDLE_TO_FD")?;
    let dmabuf = unsafe { OwnedFd::from_raw_fd(prime.fd) };
    // identity of the buffer itself (stable across the buffer's lifetime), as
    // opposed to `fb_id`, which the kernel recycles
    let buf_id = dmabuf_identity(dmabuf.as_raw_fd());

    let mut close = DrmGemClose { handle: fb.handles[0], pad: 0 };
    ioctl(
        card.as_raw_fd(),
        DRM_IOCTL_GEM_CLOSE,
        &mut close as *mut _ as *mut _,
    )
    .context("GEM_CLOSE (leaked GEM handle)")?;

    let mut modifier = fb.modifier[0];
    if modifier == DRM_FORMAT_MOD_INVALID {
        // driver gave no real modifier: EGL refuses INVALID, linear import works
        modifier = 0;
    }
    let format = FrameFormat {
        width: fb.width,
        height: fb.height,
        stride: fb.pitches[0],
        fourcc: fb.pixel_format,
        modifier,
        y0_top: true, // compositor fbs are top-first
        num_planes: 1,
    };
    Ok(DmabufScanout {
        fb_id,
        buf_id,
        fds: vec![dmabuf],
        offsets: vec![fb.offsets[0]],
        strides: vec![fb.pitches[0]],
        format,
    })
}

/// Export the dma-buf's read-side reservation fence as a sync_file fd. Waiting
/// on it (`poll`) before reading means the producer finished writing.
fn export_sync_file(dmabuf_fd: i32) -> Result<OwnedFd> {
    let mut a = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ,
        fd: -1,
    };
    ioctl(
        dmabuf_fd,
        DMA_BUF_IOCTL_EXPORT_SYNC_FILE,
        &mut a as *mut _ as *mut _,
    )
    .context("DMA_BUF_IOCTL_EXPORT_SYNC_FILE")?;
    if a.fd < 0 {
        bail!("DMA_BUF_IOCTL_EXPORT_SYNC_FILE returned fd {}", a.fd);
    }
    // SAFETY: the kernel returned a fresh fd we own
    Ok(unsafe { OwnedFd::from_raw_fd(a.fd) })
}

/// Inspect a sync_file: the first fence's `(driver_name, status)`.
/// `status`: 0 active/pending, 1 signaled, <0 error. Driver name `"stub"`
/// means the kernel substituted the always-signaled stub because the
/// reservation had no writer fence.
fn sync_file_first_fence(fd: i32) -> Option<(String, i32)> {
    // call 1: fence count
    let mut info = SyncFileInfo::default();
    ioctl(fd, SYNC_IOC_FILE_INFO, &mut info as *mut _ as *mut _).ok()?;
    let n = info.num_fences as usize;
    if n == 0 {
        return None;
    }
    // call 2: fence details
    let mut fences = vec![SyncFenceInfo::default(); n];
    info.sync_fence_info = fences.as_mut_ptr() as u64;
    info.num_fences = n as u32;
    ioctl(fd, SYNC_IOC_FILE_INFO, &mut info as *mut _ as *mut _).ok()?;
    let n = (info.num_fences as usize).min(fences.len());
    if n == 0 {
        return None;
    }
    let f = &fences[0];
    let end = f.driver_name.iter().position(|&c| c == 0).unwrap_or(32);
    Some((
        String::from_utf8_lossy(&f.driver_name[..end]).into_owned(),
        f.status,
    ))
}

#[derive(Default)]
struct PhaseStats {
    commits: u64,
    /// real producer fences (driver != "stub")
    real: u64,
    /// real fences still pending at observation
    pending: u64,
    /// real fences already signaled at observation
    signaled: u64,
    /// kernel stub fence => no reservation fence exists
    stub: u64,
    export_err: u64,
    waits_us: Vec<u64>,
}

fn probe_phase(
    card: &OwnedFd,
    target: Target,
    name: &str,
    dur: std::time::Duration,
) -> Result<PhaseStats> {
    let fdi = card.as_raw_fd();
    let mut last_fb = 0u32;
    let mut st = PhaseStats::default();
    let started = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();
    while started.elapsed() < dur {
        let fb = match target {
            Target::Crtc(id) => get_crtc(fdi, id)?.fb_id,
            Target::Plane(id) => get_plane(fdi, id)?.fb_id,
        };
        if fb == 0 || fb == last_fb {
            std::thread::sleep(std::time::Duration::from_micros(1000));
            continue;
        }
        last_fb = fb;
        let Ok(d) = import_fb(card, fb) else {
            std::thread::sleep(std::time::Duration::from_micros(1000));
            continue;
        };
        let Ok(sf) = export_sync_file(d.fds[0].as_raw_fd()) else {
            st.export_err += 1;
            continue;
        };
        st.commits += 1;
        match sync_file_first_fence(sf.as_raw_fd()) {
            Some((drv, status)) if drv != "stub" => {
                st.real += 1;
                if status == 0 {
                    st.pending += 1;
                    let t0 = std::time::Instant::now();
                    let mut p = libc::pollfd {
                        fd: sf.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    // bound the wait so a stuck fence can't hang the probe
                    unsafe { libc::poll(&mut p, 1, 50) };
                    st.waits_us.push(t0.elapsed().as_micros() as u64);
                } else {
                    st.signaled += 1;
                }
                if st.real <= 3 {
                    eprintln!(
                        "[syncprobe] {name}: fb {fb} REAL fence driver='{drv}' status={status} ({})",
                        if status == 0 { "pending" } else { "signaled" }
                    );
                }
            }
            Some((drv, _)) => {
                st.stub += 1;
                if st.stub <= 2 {
                    eprintln!("[syncprobe] {name}: fb {fb} fence driver='{drv}' (no producer fence)");
                }
            }
            None => {}
        }
        if last_report.elapsed() >= std::time::Duration::from_secs(5) {
            last_report = std::time::Instant::now();
            eprintln!(
                "[syncprobe] {name}: {} commits | real {} (pending {}, signaled {}) | stub {} | export_err {}",
                st.commits, st.real, st.pending, st.signaled, st.stub, st.export_err
            );
        }
        std::thread::sleep(std::time::Duration::from_micros(1000));
    }
    Ok(st)
}

/// Self-driving diagnostic. For `idle_secs` it watches the scanout with no
/// input; for `active_secs` it drives a synthetic uinput mouse in a bounded
/// circle (real cursor repaints, no physical input device) while sampling the
/// producer fences. It then prints a verdict: a real reservation fence means
/// the KMS capture can read the CURRENT fb with no tearing and no one-frame
/// lag (the one-behind workaround can go); a kernel stub fence means there is
/// nothing to wait for. Read-only apart from the virtual mouse; run under root.
pub fn sync_probe(
    card_pref: Option<&str>,
    idle_secs: u64,
    active_secs: u64,
    inject: bool,
) -> Result<()> {
    eprintln!(
        "[syncprobe] opening card {}",
        card_pref.unwrap_or("(probe /dev/dri/card0..8)")
    );
    let (card, target) = open_card(card_pref, None)?;
    eprintln!("[syncprobe] target: {target:?}");

    let move_on = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mover = if inject {
        let m = move_on.clone();
        let s = stop.clone();
        Some(
            std::thread::Builder::new()
                .name("syncprobe-mouse".into())
                .spawn(move || {
                    let mouse = match crate::input::UinputMouse::open() {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("[syncprobe] synthetic mouse unavailable: {e:#}");
                            return;
                        }
                    };
                    eprintln!("[syncprobe] synthetic uinput mouse up");
                    let mut theta = 0.0f64;
                    while !s.load(std::sync::atomic::Ordering::Relaxed) {
                        if m.load(std::sync::atomic::Ordering::Relaxed) {
                            // bounded lissajous: zero-mean velocity, the cursor
                            // traces a small circle and never runs off-screen
                            let dx = (16.0 * theta.cos()).round() as i32;
                            let dy = (16.0 * theta.sin()).round() as i32;
                            if dx != 0 || dy != 0 {
                                mouse.move_rel(dx, dy);
                            }
                            theta += 0.25;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(8));
                    }
                })
                .context("spawn synthetic mouse")?,
        )
    } else {
        None
    };

    eprintln!("[syncprobe] phase 1/2: {idle_secs}s idle baseline");
    let idle = probe_phase(&card, target, "idle", std::time::Duration::from_secs(idle_secs.max(1)))?;

    eprintln!("[syncprobe] phase 2/2: {active_secs}s synthetic mouse movement");
    move_on.store(true, std::sync::atomic::Ordering::Relaxed);
    let active =
        probe_phase(&card, target, "moving", std::time::Duration::from_secs(active_secs.max(1)))?;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(h) = mover {
        let _ = h.join();
    }

    let pct = |s: &PhaseStats| {
        if s.commits == 0 {
            0.0
        } else {
            100.0 * s.pending as f64 / s.commits as f64
        }
    };
    let rate = |s: &PhaseStats, secs: u64| s.commits as f64 / secs.max(1) as f64;
    let mut waits = active.waits_us.clone();
    waits.sort_unstable();
    let p50 = waits.get(waits.len() / 2).copied().unwrap_or(0);
    let pmax = waits.last().copied().unwrap_or(0);

    eprintln!("[syncprobe] ===== summary =====");
    eprintln!(
        "[syncprobe] idle:   {} commits ({:.1}/s), real {} (pending {}), stub {}, export_err {}",
        idle.commits, rate(&idle, idle_secs), idle.real, idle.pending, idle.stub, idle.export_err
    );
    eprintln!(
        "[syncprobe] moving: {} commits ({:.1}/s), real {} (pending {} = {:.1}%, signaled {}), stub {}, export_err {}",
        active.commits, rate(&active, active_secs), active.real, active.pending,
        pct(&active), active.signaled, active.stub, active.export_err
    );
    if !waits.is_empty() {
        eprintln!("[syncprobe] pending-fence wait: p50 {p50} us, max {pmax} us (n={})", waits.len());
    }

    if active.commits == 0 && idle.commits == 0 {
        eprintln!("[syncprobe] VERDICT: no primary-plane commits at all.");
        eprintln!("[syncprobe]   No active compositor scanout, or the screen is static and the cursor is a");
        eprintln!("[syncprobe]   hardware overlay plane (not part of the captured fb).");
    } else if active.real == 0 && active.stub > 0 {
        eprintln!("[syncprobe] VERDICT: NO producer fence on the reservation (kernel stub fence observed).");
        eprintln!("[syncprobe]   The compositor fences only the KMS commit, not the dma-buf, so there is nothing");
        eprintln!("[syncprobe]   to wait on. Keep the one-behind, or move to the compositor capture protocol");
        eprintln!("[syncprobe]   (ext-image-copy-capture), which is synchronized by construction.");
    } else if active.real > 0 {
        eprintln!(
            "[syncprobe] VERDICT: REAL producer fence present ({} of {} commits sampled; pending {:.1}%).",
            active.real, active.commits, pct(&active)
        );
        eprintln!("[syncprobe]   Implement a fence-checked read: export the reservation fence, poll it when pending,");
        eprintln!("[syncprobe]   then read the CURRENT fb. Tearing-free with NO one-frame lag — drop the one-behind.");
    } else {
        eprintln!("[syncprobe] VERDICT: inconclusive (no fences sampled).");
    }
    Ok(())
}

/// Read framebuffer `id` and emit a `Scanout`, exporting its dma-buf fd
/// immediately — in the same breath as the caller's `GETPLANE` (Sunshine's
/// `refresh()` does exactly this), so the compositor cannot destroy the fb in
/// between. Returns the scanout `(width, height)`; `Err` means the fb is gone
/// (`GETFB2` ENOENT) and the caller should retry / re-pick the plane.
fn emit_scanout(fd: &OwnedFd, id: u32, t0: u64, tx: &Sender<CaptureEvent>) -> Result<(u32, u32)> {
    let dmabuf = import_fb(fd, id)?;
    log::debug!(
        "[capture-drm] fb {id} buf {} {}x{} fourcc={:#010x} '{}' modifier={:#x} stride={}",
        dmabuf.buf_id,
        dmabuf.format.width,
        dmabuf.format.height,
        dmabuf.format.fourcc,
        crate::fourcc_name(dmabuf.format.fourcc),
        dmabuf.format.modifier,
        dmabuf.format.stride,
    );
    let size = (dmabuf.format.width, dmabuf.format.height);
    let _ = tx.send(CaptureEvent::Scanout(dmabuf, t0));
    Ok(size)
}

/// Calibration window for [`crate::fence::FenceWatch`] (the `fence` sync mode).
const FENCE_CALIB: std::time::Duration = std::time::Duration::from_millis(400);

/// Emit decision for one poll (flip-sync modes).
enum Step {
    None,
    Capture(u32, u64),
}

/// `settle` mode: emit the current fb `ms` after its commit was first seen.
///
/// The compositor commits with a non-blocking atomic commit and finishes the
/// frame's GPU work afterwards; on this stack the per-frame completion fence
/// (NVIDIA semsurf) signals ~0.75 ms (p50), ~8-16 ms (p99) after handover, so a
/// small settle captures a complete frame most of the time without waiting a
/// full refresh. Frames whose render outlasts the window are read mid-render
/// and can tear — an explicit latency/tearing tradeoff, not a guarantee. A small
/// backlog keeps a settle larger than the frame interval progressing in order
/// (bounded so we cannot fall arbitrarily behind).
#[derive(Default)]
struct SettleState {
    /// latest fb seen on the plane
    seen: u32,
    /// last fb handed to the importer
    last_captured: u32,
    /// commits awaiting their settle deadline: (fb, due_ms, t0_us)
    due: std::collections::VecDeque<(u32, u64, u64)>,
    /// settle window (ms)
    ms: u64,
}

impl SettleState {
    fn new(ms: u64) -> Self {
        Self {
            ms,
            ..Default::default()
        }
    }

    fn update(&mut self, now_ms: u64, now_us: u64, fb: u32) -> Step {
        if fb != self.seen {
            self.seen = fb;
            self.due
                .push_back((fb, now_ms.saturating_add(self.ms), now_us));
            while self.due.len() > 4 {
                self.due.pop_front();
            }
        }
        while let Some(&(id, due_ms, t0)) = self.due.front() {
            if now_ms < due_ms {
                break;
            }
            self.due.pop_front();
            if id != self.last_captured {
                self.last_captured = id;
                return Step::Capture(id, t0);
            }
        }
        Step::None
    }

    /// The import failed: forget the claim so a later poll can retry.
    fn failed(&mut self) {
        self.last_captured = 0;
    }
}

/// `fence` mode: emit the current fb when the compositor's per-frame completion
/// fence signals. The caller drains the (process-wide) signal stream and passes
/// the latest timestamp; we capture the current fb when it differs from the
/// last one captured. See [`crate::fence`].
#[derive(Default)]
struct FenceState {
    /// last fb handed to the importer
    last_captured: u32,
}

impl FenceState {
    /// `trigger` is the fence signal's timestamp, or `None` if no signal arrived
    /// since the last poll.
    fn update(&mut self, fb: u32, trigger: Option<u64>) -> Step {
        if let Some(t0) = trigger {
            if fb != self.last_captured {
                self.last_captured = fb;
                return Step::Capture(fb, t0);
            }
        }
        Step::None
    }

    /// The import failed: forget the claim so a later signal can retry.
    fn failed(&mut self) {
        self.last_captured = 0;
    }
}

fn run(
    card_pref: Option<&str>,
    connector: Option<&str>,
    sync: CaptureSync,
    tx: Sender<CaptureEvent>,
    stop: Stop,
) -> Result<()> {
    log::debug!(
        "[capture-drm] opening card {}",
        card_pref.unwrap_or("(probe /dev/dri/card0..8)")
    );
    // Flip-sync policy. `auto` (the default) and `fence` use the compositor's
    // per-frame completion fence when it is ready and fall back to a short
    // settle otherwise; `settle[:ms]` uses the settle window alone.
    let (use_fence, settle_ms) = match sync {
        CaptureSync::Settle(ms) => (false, ms),
        CaptureSync::Auto | CaptureSync::Fence => (true, DEFAULT_SETTLE_MS),
    };
    let mut settle = SettleState::new(settle_ms);
    let mut fence_state = FenceState::default();
    // process-wide completion-fence watch (started once, calibrates in the
    // background); `fence_seq` is the last signal consumed
    let mut fence: Option<&'static crate::fence::FenceWatch> = None;
    let mut fence_tried = false;
    let mut fence_seq: u64 = 0;
    let mut card: Option<(OwnedFd, Target)> = None;
    let mut disabled_reported = false;
    let started = std::time::Instant::now();
    // throttles card (re)opens to once a second after a failure
    let mut last_open_attempt: Option<std::time::Instant> = None;
    // Sunshine's `empty_plane_timer`: re-pick the plane once it has had no
    // framebuffer for this long.
    const EMPTY_MS: u64 = 250;
    let mut empty_since_ms: Option<u64> = None;
    // consecutive failed exports (GETFB2 ENOENT): re-pick the plane after a few
    const FAIL_RESCAN: u32 = 3;
    let mut fail_streak: u32 = 0;
    // size of the first captured scanout, to re-pick the plane on a mode change
    let mut captured_size: Option<(u32, u32)> = None;
    loop {
        if stop.stopped() {
            return Ok(());
        }
        if card.is_none() {
            if let Some(t) = last_open_attempt {
                if t.elapsed() < std::time::Duration::from_secs(1) {
                    if crate::sleep_or_stop(&stop, std::time::Duration::from_millis(100)) {
                        return Ok(());
                    }
                    continue;
                }
            }
            last_open_attempt = Some(std::time::Instant::now());
            match open_card(card_pref, connector) {
                Ok((fdr, target)) => {
                    log::debug!("[capture-drm] target: {target:?} (Sunshine plane selection)");
                    card = Some((fdr, target));
                    empty_since_ms = None;
                    fail_streak = 0;
                    captured_size = None;
                    disabled_reported = false;
                    // a (re)opened card means a fresh scanout: reset the sync
                    // state so the first frame is captured immediately
                    settle = SettleState::new(settle_ms);
                    fence_state = FenceState::default();
                }
                Err(e) => {
                    log::debug!("[capture-drm] {e:#}; retrying…");
                    continue;
                }
            }
        }
        let (fdr, target) = card.as_ref().unwrap();
        let fdi = fdr.as_raw_fd();
        // Sunshine's `refresh()`: read the plane's CURRENT framebuffer.
        let fb = match *target {
            Target::Crtc(crtc_id) => match get_crtc(fdi, crtc_id) {
                Ok(c) => c.fb_id,
                Err(e) => {
                    log::debug!("[capture-drm] GETCRTC: {e}; re-picking");
                    card = None;
                    last_open_attempt = Some(std::time::Instant::now());
                    if crate::sleep_or_stop(&stop, std::time::Duration::from_millis(500)) {
                        return Ok(());
                    }
                    continue;
                }
            },
            Target::Plane(plane_id) => match get_plane(fdi, plane_id) {
                Ok(p) => p.fb_id,
                Err(e) => {
                    log::debug!("[capture-drm] GETPLANE: {e}; re-picking");
                    card = None;
                    last_open_attempt = Some(std::time::Instant::now());
                    if crate::sleep_or_stop(&stop, std::time::Duration::from_millis(500)) {
                        return Ok(());
                    }
                    continue;
                }
            },
        };
        let now_ms = started.elapsed().as_millis() as u64;
        let now_us = rdma_display_proto::unix_now_us();
        if fb == 0 {
            // no framebuffer (DPMS off / no scanout). Tell the server once, and
            // re-pick the plane once the empty window elapses (Sunshine).
            let since = *empty_since_ms.get_or_insert(now_ms);
            if !disabled_reported {
                disabled_reported = true;
                let _ = tx.send(CaptureEvent::Disabled);
            }
            if now_ms.saturating_sub(since) >= EMPTY_MS {
                log::debug!(
                    "[capture-drm] target has had no framebuffer for {} ms; re-picking",
                    now_ms - since
                );
                card = None;
                last_open_attempt = Some(std::time::Instant::now());
                empty_since_ms = None;
                continue;
            }
            continue;
        }
        empty_since_ms = None;
        disabled_reported = false;
        // Decide whether to emit this poll, per the flip-sync policy. The export
        // happens in the same iteration as the GETPLANE read above, so the fb
        // cannot vanish in between — that is Sunshine's `refresh()`.
        let step = if use_fence {
            // start the process-wide fence watch once; it calibrates in the
            // background and reports `ready` once signals flow
            if fence.is_none() && !fence_tried {
                fence_tried = true;
                match crate::fence::start(FENCE_CALIB) {
                    Ok(w) => fence = Some(w),
                    Err(e) => log::warn!(
                        "[capture-drm] fence watch unavailable: {e:#}; using settle:{settle_ms}ms"
                    ),
                }
            }
            match fence.filter(|w| w.ready()) {
                // calibrating (an idle screen has no commits to calibrate
                // against), or the watch is unavailable: short settle
                None => settle.update(now_ms, now_us, fb),
                Some(w) => {
                    // newest completion-fence signal since we last looked
                    let trigger = w.signal_since(fence_seq).map(|(s, t)| {
                        fence_seq = s;
                        t
                    });
                    fence_state.update(fb, trigger)
                }
            }
        } else {
            settle.update(now_ms, now_us, fb)
        };
        if let Step::Capture(id, t0) = step {
            match emit_scanout(fdr, id, t0, &tx) {
                Ok(size) => {
                    fail_streak = 0;
                    match captured_size {
                        Some(prev) if prev != size => {
                            log::debug!(
                                "[capture-drm] scanout {}x{} -> {}x{}; re-picking the plane",
                                prev.0,
                                prev.1,
                                size.0,
                                size.1
                            );
                            card = None;
                            last_open_attempt = Some(std::time::Instant::now());
                            captured_size = None;
                            continue;
                        }
                        None => captured_size = Some(size),
                        _ => {}
                    }
                }
                Err(e) => {
                    // Sunshine returns `timeout` here and retries the same plane
                    // on the next iteration; only after a few do we re-pick.
                    settle.failed();
                    fence_state.failed();
                    fail_streak += 1;
                    if fail_streak >= FAIL_RESCAN {
                        log::debug!(
                            "[capture-drm] GETFB2 failed {}x in a row ({e:#}); re-picking the plane",
                            fail_streak
                        );
                        card = None;
                        last_open_attempt = Some(std::time::Instant::now());
                        fail_streak = 0;
                    }
                }
            }
        }
        if crate::sleep_or_stop(&stop, std::time::Duration::from_millis(1)) {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // ioctl size fields must match the kernel's struct layouts exactly
    #[test]
    fn uapi_struct_sizes() {
        assert_eq!(size_of::<DrmModeCrtc>(), 104);
        assert_eq!(size_of::<DrmModeGetPlaneRes>(), 16);
        assert_eq!(size_of::<DrmModeGetPlane>(), 32); // u64 format ptr pads
        assert_eq!(size_of::<DrmModeGetEncoder>(), 20);
        assert_eq!(size_of::<DrmModeGetConnector>(), 80);
        assert_eq!(size_of::<DrmModeObjGetProperties>(), 32);
        assert_eq!(size_of::<DrmModeGetProperty>(), 64);
        assert_eq!(size_of::<DrmModeFbCmd2>(), 104); // u64 modifier[] pads
        assert_eq!(size_of::<DrmPrimeHandle>(), 12);
        assert_eq!(size_of::<DrmGemClose>(), 8);
        // ioctl request numbers: _IOWR dir bits = 3 (READ|WRITE). Verified
        // against /usr/include/drm/drm.h macros via gcc (not self-computed!)
        assert_eq!(DRM_IOCTL_MODE_GETRESOURCES, 0xC04064A0);
        assert_eq!(DRM_IOCTL_MODE_GETCRTC, 0xC06864A1);
        assert_eq!(DRM_IOCTL_MODE_GETPLANERESOURCES, 0xC01064B5);
        assert_eq!(DRM_IOCTL_MODE_GETPLANE, 0xC02064B6);
        assert_eq!(DRM_IOCTL_MODE_GETENCODER, 0xC01464A6);
        assert_eq!(DRM_IOCTL_MODE_GETCONNECTOR, 0xC05064A7);
        assert_eq!(DRM_IOCTL_MODE_OBJ_GETPROPERTIES, 0xC02064B9);
        assert_eq!(DRM_IOCTL_MODE_GETPROPERTY, 0xC04064AA);
        assert_eq!(DRM_IOCTL_MODE_GETFB2, 0xC06864CE);
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD, 0xC00C642D);
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x40086409); // _IOW, dir=1
        // dma-buf fence export: _IOWR('b', 2, struct {u32; s32}) = dir3, size8
        assert_eq!(DMA_BUF_IOCTL_EXPORT_SYNC_FILE, 0xC0086202);
        assert_eq!(std::mem::size_of::<DmaBufExportSyncFile>(), 8);
        // sync_file info: _IOWR('>', 4, struct sync_file_info) = dir3, size56
        assert_eq!(SYNC_IOC_FILE_INFO, 0xC0383E04);
        assert_eq!(std::mem::size_of::<SyncFileInfo>(), 56);
        assert_eq!(std::mem::size_of::<SyncFenceInfo>(), 80);
    }

    #[test]
    fn connector_names_match_sysfs() {
        // the `<type>-<type_id>` rendering must equal sysfs / compositor names
        assert_eq!(format!("{}-{}", connector_type_name(10), 1), "DP-1");
        assert_eq!(format!("{}-{}", connector_type_name(11), 1), "HDMI-A-1");
        assert_eq!(format!("{}-{}", connector_type_name(14), 1), "eDP-1");
        assert_eq!(format!("{}-{}", connector_type_name(0), 1), "Unknown-1");
    }

    #[test]
    fn dmabuf_identity_is_stable_and_distinct() {
        use std::os::fd::AsRawFd;
        let a = std::fs::File::open("/dev/null").unwrap();
        let a2 = std::fs::File::open("/dev/null").unwrap();
        let b = std::fs::File::open("/dev/zero").unwrap();
        // the same underlying object -> the same identity (this is what makes
        // the import cache safe across the buffer's lifetime)
        assert_eq!(
            dmabuf_identity(a.as_raw_fd()),
            dmabuf_identity(a2.as_raw_fd())
        );
        // a different object -> a different identity (so a recycled fb id that
        // now names a new buffer can never hit a stale import)
        assert_ne!(
            dmabuf_identity(a.as_raw_fd()),
            dmabuf_identity(b.as_raw_fd())
        );
        // a real object is cacheable...
        assert_ne!(dmabuf_identity(a.as_raw_fd()), 0);
        // ...an unusable fd is "no stable identity" (must not cache)
        assert_eq!(dmabuf_identity(-1), 0);
    }
}
