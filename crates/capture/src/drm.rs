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
//! Flip sync: a scanout dmabuf on this stack carries no usable producer fence
//! (the `syncprobe` verdict: stub fences only), so we cannot wait on the
//! compositor's GPU work before reading a buffer. Two safe policies instead:
//!   * one-behind (default fallback): capture the fb that was just superseded
//!     — it scanned out for a full interval, so it is complete, at the cost of
//!     one compositor frame of lag;
//!   * vblank (`--capture-sync vblank|auto`): wait for the vblank that retires
//!     the *current* commit, then capture it — the kernel only latches a ready
//!     buffer, so it is complete and it is the frame actually on screen. A
//!     bounded wait falls back to one-behind per frame and degrades for the
//!     session if vblank never advances (VRR idle, broken accounting).
//!
//! There is also a deliberately unsafe `--capture-sync zero` (testing only):
//! grab the current fb the instant its commit is seen, with no wait at all.
//! It is the latency floor and a tear-behaviour test, not a mode to ship.
//!
//! Known gaps: no cursor-plane blending (hardware cursors are invisible;
//! software cursors are in the fb), no damage info (every flip is full-frame),
//! readback is unfenced (scanout buffers are stable while scanned out; the
//! compositor's next render into the same buffer is ~1-2 flips away).

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::mpsc::Sender;

use anyhow::{bail, Context, Result};

use crate::{CaptureEvent, CaptureSource, CaptureSync, DmabufScanout, FrameFormat};

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
const DRM_IOCTL_MODE_OBJ_GETPROPERTIES: u64 =
    iowr(0xB9, std::mem::size_of::<DrmModeObjGetProperties>());
const DRM_IOCTL_MODE_GETPROPERTY: u64 =
    iowr(0xAA, std::mem::size_of::<DrmModeGetProperty>());
const DRM_IOCTL_MODE_GETFB2: u64 = iowr(0xCE, std::mem::size_of::<DrmModeFbCmd2>());
const DRM_IOCTL_PRIME_HANDLE_TO_FD: u64 =
    iowr(0x2d, std::mem::size_of::<DrmPrimeHandle>());
const DRM_IOCTL_GEM_CLOSE: u64 = iowr_ioctl(0x09, std::mem::size_of::<DrmGemClose>());

// plane object type for OBJ_GETPROPERTIES (drm_mode.h)
const DRM_MODE_OBJECT_PLANE: u32 = 0xeeee_eeee;
// plane "type" property enum values (libdrm): 0=overlay 1=primary 2=cursor
const DRM_PLANE_TYPE_PRIMARY: u64 = 1;

// GEM_CLOSE is _IOW (write dir, =1), not _IOWR
const fn iowr_ioctl(nr: u16, size: usize) -> u64 {
    (1 << 30) | ((size as u64) << 16) | (D << 8) | nr as u64
}

// --- vblank UAPI (include/uapi/drm/drm.h) -----------------------------------
// GET_SEQUENCE reports the most recent vblank's count and its timestamp
// ("time of first pixel out", CLOCK_MONOTONIC ns). It is registered with no
// DRM_MASTER/DRM_AUTH flag, so a non-master observer (us) may call it, and it
// keeps the vblank counter enabled — which matters on VRR displays where the
// interrupt is otherwise off while idle.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmCrtcGetSequence {
    crtc_id: u32,
    active: u32,
    sequence: u64,
    sequence_ns: i64,
}
const DRM_IOCTL_CRTC_GET_SEQUENCE: u64 =
    iowr(0x3b, std::mem::size_of::<DrmCrtcGetSequence>());

/// Most recent vblank for `crtc_id` as `(sequence, ktime_ns)`.
fn get_sequence(fd: i32, crtc_id: u32) -> Result<(u64, i64)> {
    let mut s = DrmCrtcGetSequence {
        crtc_id,
        ..Default::default()
    };
    ioctl(
        fd,
        DRM_IOCTL_CRTC_GET_SEQUENCE,
        &mut s as *mut _ as *mut core::ffi::c_void,
    )?;
    Ok((s.sequence, s.sequence_ns))
}

#[allow(clippy::useless_conversion)] // time_t/c_long: i64 here, i32 elsewhere
fn clock_us(clk: libc::clockid_t) -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(clk, &mut ts) };
    i64::from(ts.tv_sec) * 1_000_000 + i64::from(ts.tv_nsec) / 1000
}

/// A CLOCK_MONOTONIC nanosecond timestamp (the kernel's vblank time) ->
/// unix-epoch microseconds. Same host, so the two clocks differ by a constant.
fn mono_ns_to_unix_us(ns: i64) -> u64 {
    if ns <= 0 {
        return rdma_display_proto::unix_now_us();
    }
    let delta = clock_us(libc::CLOCK_REALTIME) - clock_us(libc::CLOCK_MONOTONIC);
    ((ns / 1000) + delta).max(0) as u64
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
    /// Which fb to capture relative to the compositor's flips.
    pub sync: CaptureSync,
}

impl DrmKmsSource {
    pub fn new(card: Option<String>) -> Self {
        Self {
            card,
            sync: CaptureSync::Auto,
        }
    }

    /// Pick the flip-sync policy explicitly (see [`CaptureSync`]).
    pub fn with_sync(card: Option<String>, sync: CaptureSync) -> Self {
        Self { card, sync }
    }
}

impl CaptureSource for DrmKmsSource {
    fn name(&self) -> &'static str {
        "drm-kms"
    }

    fn spawn(self: Box<Self>, tx: Sender<CaptureEvent>) -> Result<()> {
        let sync = self.sync;
        std::thread::Builder::new()
            .name("capture-drm".into())
            .spawn(move || {
                if let Err(e) = run(self.card.as_deref(), sync, tx) {
                    eprintln!("[capture] source failed: {e:#}");
                }
            })
            .context("spawn capture thread")?;
        Ok(())
    }
}

/// Open a card node and return (fd, target) for the active display.
///
/// Probe order (what the C probe showed live on nvidia-drm):
///   1. CRTCs: GETCRTC's fb_id IS populated on nvidia-drm (the atomic state
///      is backfilled for this ioctl), while planes report crtc=0/fb=0 —
///      the legacy fb pointer lives on the CRTC here.
///   2. Planes (fallback): other atomic drivers populate plane.fb_id but not
///      crtc.fb_id; find a bound plane of type PRIMARY.
///
/// Every probe is logged — a silent skip cost us a whole round-trip.
fn open_card(preferred: Option<&str>) -> Result<(OwnedFd, Target)> {
    let candidates: Vec<String> = match preferred {
        Some(p) => vec![p.to_string()],
        None => (0..8).map(|i| format!("/dev/dri/card{i}")).collect(),
    };
    let mut last_err: Option<anyhow::Error> = None;
    for path in &candidates {
        let r = open_card_and_find_target(path);
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

fn open_card_and_find_target(path: &str) -> Result<(OwnedFd, Target)> {
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

    // 1. CRTC path: GETRESOURCES (two-call; zero counts for lists not provided)
    let mut res = DrmModeCardRes {
        count_crtcs: 0,
        ..zeroed()
    };
    ioctl(fdr, DRM_IOCTL_MODE_GETRESOURCES, &mut res as *mut _ as *mut _)
        .context("GETRESOURCES")?;
    let n = res.count_crtcs as usize;
    if n > 0 {
        let mut crtcs: Vec<u32> = vec![0; n];
        let mut res = DrmModeCardRes {
            crtc_id_ptr: crtcs.as_mut_ptr() as u64,
            count_crtcs: n as u32,
            ..zeroed()
        };
        ioctl(fdr, DRM_IOCTL_MODE_GETRESOURCES, &mut res as *mut _ as *mut _)
            .context("GETRESOURCES #2")?;
        for &crtc_id in crtcs.iter().take(res.count_crtcs as usize) {
            match get_crtc(fdr, crtc_id) {
                Ok(c) => {
                    eprintln!(
                        "[capture-drm] CRTC {}: fb_id={} active={} mode={}x{}",
                        crtc_id, c.fb_id, c.mode_valid, c.mode.hdisplay, c.mode.vdisplay
                    );
                    if c.fb_id != 0 {
                        return Ok((card, Target::Crtc(crtc_id)));
                    }
                }
                Err(e) => eprintln!("[capture-drm] CRTC {crtc_id}: GETCRTC: {e}"),
            }
        }
    }

    // 2. plane fallback: bound plane of type PRIMARY (other atomic drivers)
    let mut pres = DrmModeGetPlaneRes::default();
    ioctl(fdr, DRM_IOCTL_MODE_GETPLANERESOURCES, &mut pres as *mut _ as *mut _)
        .context("GETPLANERESOURCES")?;
    let pn = pres.count_planes as usize;
    if pn > 0 {
        let mut planes: Vec<u32> = vec![0; pn];
        let mut pres = DrmModeGetPlaneRes {
            plane_id_ptr: planes.as_mut_ptr() as u64,
            count_planes: pn as u32,
        };
        ioctl(fdr, DRM_IOCTL_MODE_GETPLANERESOURCES, &mut pres as *mut _ as *mut _)
            .context("GETPLANERESOURCES #2")?;
        let mut fallback: Option<u32> = None;
        for &plane_id in planes.iter().take(pres.count_planes as usize) {
            let p = match get_plane(fdr, plane_id) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[capture-drm] plane {plane_id}: GETPLANE: {e}");
                    continue;
                }
            };
            if p.fb_id == 0 || p.crtc_id == 0 {
                eprintln!(
                    "[capture-drm] plane {plane_id}: unbound (crtc={} fb={})",
                    p.crtc_id, p.fb_id
                );
                continue;
            }
            let ty = plane_type(fdr, plane_id).unwrap_or(u64::MAX);
            eprintln!("[capture-drm] plane {plane_id}: crtc={} fb={} type={ty}", p.crtc_id, p.fb_id);
            if ty == DRM_PLANE_TYPE_PRIMARY {
                return Ok((card, Target::Plane(plane_id)));
            }
            if fallback.is_none() {
                fallback = Some(plane_id);
            }
        }
        if let Some(p) = fallback {
            return Ok((card, Target::Plane(p)));
        }
    }

    bail!("{path}: no CRTC or plane with a bound framebuffer (display off?)")
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

fn import_fb(card: &OwnedFd, fb_id: u32) -> Result<DmabufScanout> {
    let mut fb: DrmModeFbCmd2 = zeroed();
    fb.fb_id = fb_id;
    ioctl(card.as_raw_fd(), DRM_IOCTL_MODE_GETFB2, &mut fb as *mut _ as *mut _)
        .context("GETFB2 (needs root/CAP_SYS_ADMIN; kernel >= 5.8)")?;
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
    let (card, target) = open_card(card_pref)?;
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

/// Settle window: once the current fb has been unchanged this long, flips have
/// stopped and it is certainly fully rendered, so it is safe to capture it.
const SETTLE_MS: u64 = 20;

/// One poll's decision.
#[derive(Debug, PartialEq, Eq)]
enum FbAction {
    /// Nothing new worth capturing.
    None,
    /// No scanout (DPMS off).
    Disabled,
    /// Capture this fb id — already known to be complete — plus the unix-us
    /// time it became visible on screen (the flip that latched it).
    Capture(u32, u64),
}

/// KMS flip tracker implementing the "one-behind" read.
///
/// `GETCRTC`/`GETPLANE` report the *pending* fb: the compositor may have
/// committed it with an in-fence that has not signaled yet, so reading it races
/// the GPU still rendering into it (blank/torn regions). The fb it just
/// replaced, however, was scanned out for a full interval and is therefore
/// complete. So on every change we capture the *previous* fb; when flips stop,
/// we capture the settled current fb once it has been stable for [`SETTLE_MS`].
/// Cost: one frame of capture latency. Benefit: never sample a buffer mid-render.
#[derive(Default)]
struct FbTracker {
    /// most recent fb id seen on the CRTC/plane (the pending one)
    seen: u32,
    /// last fb id handed to the importer
    last_captured: u32,
    /// when `seen` last changed (ms since capture start)
    changed_ms: u64,
    /// when `seen` was first observed, unix us: the moment that fb went on
    /// screen (within the poll granularity). This is the captured frame's t0.
    seen_since_us: u64,
}

impl FbTracker {
    fn update(&mut self, now_ms: u64, now_us: u64, fb: u32) -> FbAction {
        if fb == 0 {
            if self.seen == 0 {
                return FbAction::None;
            }
            self.seen = 0;
            self.last_captured = 0;
            return FbAction::Disabled;
        }
        if fb != self.seen {
            let prev = self.seen;
            // before overwriting: `seen_since_us` is when the PREVIOUS fb
            // (the one we are about to capture) first appeared
            let prev_since = self.seen_since_us;
            self.seen = fb;
            self.changed_ms = now_ms;
            self.seen_since_us = now_us;
            // The previous fb was on screen for a full interval -> complete. On
            // the very first observation there is no previous, so take the
            // current one immediately (a one-time possible tear) to get a
            // picture up; it is corrected on the next flip.
            let candidate = if prev != 0 { prev } else { fb };
            let t0 = if prev != 0 { prev_since } else { now_us };
            if candidate != self.last_captured {
                self.last_captured = candidate;
                return FbAction::Capture(candidate, t0);
            }
            return FbAction::None;
        }
        // Flips stopped: the current fb has been stable long enough to be
        // certainly rendered -> capture it so the settled frame is shown.
        if self.last_captured != self.seen
            && now_ms.saturating_sub(self.changed_ms) >= SETTLE_MS
        {
            self.last_captured = self.seen;
            return FbAction::Capture(self.seen, self.seen_since_us);
        }
        FbAction::None
    }

    /// The import failed: forget the claim so a later poll can retry.
    fn capture_failed(&mut self) {
        self.last_captured = 0;
    }
}

/// Zero-delay capture (TESTING ONLY): on every change, hand the *current* fb to
/// the importer immediately — no vblank wait, no one-behind. This is the
/// Sunshine-style "grab the pending fb". The buffer may still be mid-render (we
/// have no producer fence) or not yet latched, so the mirrored image can tear or
/// be a frame early; it exists to measure the latency floor and to observe how
/// this stack tears, not for regular use.
#[derive(Default)]
struct ZeroTracker {
    /// latest fb seen on the crtc/plane
    seen: u32,
    /// fb handed to the importer last
    last_captured: u32,
    /// when `seen` last changed (ms since capture start)
    changed_ms: u64,
    /// unix us when `seen` first appeared (its t0)
    seen_since_us: u64,
}

impl ZeroTracker {
    fn update(&mut self, now_ms: u64, now_us: u64, fb: u32) -> FbAction {
        if fb == 0 {
            if self.seen == 0 {
                return FbAction::None;
            }
            self.seen = 0;
            self.last_captured = 0;
            return FbAction::Disabled;
        }
        if fb != self.seen {
            self.seen = fb;
            self.changed_ms = now_ms;
            self.seen_since_us = now_us;
            self.last_captured = fb;
            return FbAction::Capture(fb, now_us);
        }
        // Flips stopped: if the immediate import failed, retry the settled
        // current fb so a picture still gets up.
        if self.last_captured != self.seen
            && now_ms.saturating_sub(self.changed_ms) >= SETTLE_MS
        {
            self.last_captured = self.seen;
            return FbAction::Capture(self.seen, self.seen_since_us);
        }
        FbAction::None
    }

    /// The import failed: forget the claim so a later poll can retry.
    fn capture_failed(&mut self) {
        self.last_captured = 0;
    }
}

/// Vblank-synced capture: on every flip to fb B, wait for the next vblank — the
/// moment the kernel latches B onto the display. B is complete then (the kernel
/// only latches a ready buffer) and stays on screen for at least one interval,
/// so reading it is tear-free without a producer fence (this box's scanout
/// dmabuf carries only stub fences). That removes the one-frame lag of
/// [`FbTracker`] while keeping the same safety argument. (Like the one-behind,
/// this relies on the read finishing before the compositor renders into B
/// again — two intervals away, so comfortable at 60 Hz and tight above
/// ~240 Hz.)
///
/// The wait is bounded: if no vblank advances within [`VBLANK_TIMEOUT_MS`] the
/// frame falls back to one-behind, and after [`VBLANK_FAILS_TO_GIVEUP`]
/// consecutive failures the caller degrades to one-behind for the session.
#[derive(Default)]
struct VblankTracker {
    /// latest fb seen on the crtc/plane
    seen: u32,
    /// fb handed to the importer last
    last_captured: u32,
    /// when `seen` last changed (ms since capture start)
    changed_ms: u64,
    /// unix us when `seen` first appeared (fallback t0)
    seen_since_us: u64,
    /// a commit waiting for the vblank that retires it
    wait: Option<VblankWait>,
    /// consecutive vblank timeouts
    failures: u32,
}

struct VblankWait {
    /// fb whose commit we are waiting to see retired
    fb: u32,
    target_seq: u64,
    since_ms: u64,
    /// the fb the wait replaced (one-behind fallback)
    prev: u32,
    prev_t0_us: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum VStep {
    None,
    Disabled,
    Capture(u32, u64),
}

const VBLANK_TIMEOUT_MS: u64 = 60;
const VBLANK_FAILS_TO_GIVEUP: u32 = 3;

impl VblankTracker {
    /// One poll. `seq`/`ns` come from `get_sequence` (0/0 when unavailable).
    fn update(&mut self, now_ms: u64, now_us: u64, fb: u32, seq: u64, ns: i64) -> VStep {
        if fb == 0 {
            if self.seen == 0 {
                return VStep::None;
            }
            self.seen = 0;
            self.last_captured = 0;
            self.wait = None;
            return VStep::Disabled;
        }
        if fb != self.seen {
            let prev = self.seen;
            let prev_t0 = self.seen_since_us;
            self.seen = fb;
            self.changed_ms = now_ms;
            self.seen_since_us = now_us;
            // the next vblank retires this commit; a newer commit before then
            // just replaces the wait (latest-wins, like the send mailbox)
            self.wait = Some(VblankWait {
                fb,
                target_seq: seq.wrapping_add(1),
                since_ms: now_ms,
                prev,
                prev_t0_us: prev_t0,
            });
            return VStep::None;
        }
        if let Some(w) = self.wait.as_ref() {
            if seq >= w.target_seq {
                let fb = w.fb;
                let t0 = mono_ns_to_unix_us(ns);
                self.wait = None;
                self.last_captured = fb;
                self.failures = 0;
                return VStep::Capture(fb, t0);
            }
            if now_ms.saturating_sub(w.since_ms) > VBLANK_TIMEOUT_MS {
                let (prev, prev_t0) = (w.prev, w.prev_t0_us);
                self.wait = None;
                self.failures += 1;
                // prefer the frame that certainly retired; else the committed
                // one, which is certainly complete after this long a wait
                let (id, t0) = if prev != 0 && prev != self.last_captured {
                    (prev, prev_t0)
                } else {
                    (self.seen, self.seen_since_us)
                };
                self.last_captured = id;
                return VStep::Capture(id, t0);
            }
            return VStep::None;
        }
        // no pending wait: flips stopped — take the settled current fb
        if self.last_captured != self.seen
            && now_ms.saturating_sub(self.changed_ms) >= SETTLE_MS
        {
            self.last_captured = self.seen;
            return VStep::Capture(self.seen, self.seen_since_us);
        }
        VStep::None
    }

    /// True once vblank has failed enough times in a row to stop trusting it.
    fn give_up(&self) -> bool {
        self.failures >= VBLANK_FAILS_TO_GIVEUP
    }

    /// The import failed: forget the claim so a later poll can retry.
    fn capture_failed(&mut self) {
        self.last_captured = 0;
    }
}

/// Resolved (post-probe) capture policy for this session.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveSync {
    OneBehind,
    Zero,
    Vblank,
}

/// Import `id` and emit a `Scanout`. False = import failed (the fb may already
/// be freed/reused); the caller lets the tracker retry.
fn emit_scanout(fd: &OwnedFd, id: u32, t0: u64, tx: &Sender<CaptureEvent>) -> bool {
    match import_fb(fd, id) {
        Ok(dmabuf) => {
            eprintln!(
                "[capture-drm] fb {id} {}x{} fourcc={:#010x} '{}' modifier={:#x} stride={}",
                dmabuf.format.width,
                dmabuf.format.height,
                dmabuf.format.fourcc,
                crate::fourcc_name(dmabuf.format.fourcc),
                dmabuf.format.modifier,
                dmabuf.format.stride,
            );
            let _ = tx.send(CaptureEvent::Scanout(dmabuf, t0));
            true
        }
        Err(e) => {
            eprintln!("[capture-drm] fb {id} import failed: {e:#}");
            false
        }
    }
}

fn run(card_pref: Option<&str>, sync: CaptureSync, tx: Sender<CaptureEvent>) -> Result<()> {
    eprintln!(
        "[capture-drm] opening card {}",
        card_pref.unwrap_or("(probe /dev/dri/card0..8)")
    );
    let mut card: Option<(OwnedFd, Target)> = None;
    let mut disabled_reported = false;
    let mut tracker = FbTracker::default();
    let mut vtracker = VblankTracker::default();
    let mut ztracker = ZeroTracker::default();
    let mut active = ActiveSync::OneBehind;
    let mut crtc_id: Option<u32> = None;
    let started = std::time::Instant::now();
    // throttles card (re)opens to once a second after a failure
    let mut last_open_attempt: Option<std::time::Instant> = None;
    loop {
        if card.is_none() {
            if let Some(t) = last_open_attempt {
                if t.elapsed() < std::time::Duration::from_secs(1) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            }
            last_open_attempt = Some(std::time::Instant::now());
            match open_card(card_pref) {
                Ok((fdr, target)) => {
                    let fdi = fdr.as_raw_fd();
                    // vblank is per-CRTC; the plane target must resolve its crtc
                    crtc_id = match target {
                        Target::Crtc(id) => Some(id),
                        Target::Plane(id) => get_plane(fdi, id)
                            .ok()
                            .map(|p| p.crtc_id)
                            .filter(|c| *c != 0),
                    };
                    let vblank_probe = crtc_id.map(|c| get_sequence(fdi, c));
                    let vblank_ok = matches!(vblank_probe, Some(Ok(_)));
                    active = match sync {
                        CaptureSync::Zero => ActiveSync::Zero,
                        CaptureSync::OneBehind => ActiveSync::OneBehind,
                        CaptureSync::Vblank | CaptureSync::Auto => {
                            if vblank_ok {
                                ActiveSync::Vblank
                            } else {
                                ActiveSync::OneBehind
                            }
                        }
                    };
                    eprintln!(
                        "[capture-drm] target: {target:?} (crtc {crtc_id:?}); flip sync: {} (requested {sync:?})",
                        match active {
                            ActiveSync::Vblank => "vblank-synced current frame",
                            ActiveSync::OneBehind => "one-behind",
                            ActiveSync::Zero => "zero-delay current frame (TESTING ONLY)",
                        }
                    );
                    if matches!(sync, CaptureSync::Vblank | CaptureSync::Auto) && !vblank_ok {
                        match &vblank_probe {
                            Some(Err(e)) => {
                                eprintln!(
                                    "[capture-drm] vblank unavailable: CRTC_GET_SEQUENCE failed: \
                                     {e:#}; using one-behind"
                                );
                                let unsupported = e
                                    .downcast_ref::<std::io::Error>()
                                    .and_then(|io| io.raw_os_error())
                                    == Some(libc::EOPNOTSUPP);
                                if unsupported {
                                    eprintln!(
                                        "[capture-drm] hint: this driver has no DRM vblank support \
                                         (nvidia-drm disables it by default) — \
                                         `options nvidia_drm vblank=1` in modprobe.d + initramfs \
                                         rebuild enables vblank capture"
                                    );
                                }
                            }
                            _ => eprintln!(
                                "[capture-drm] vblank unavailable (no CRTC for the selected \
                                 target); using one-behind"
                            ),
                        }
                    }
                    card = Some((fdr, target));
                    // a (re)opened card means a fresh scanout: reset the
                    // trackers so the first frame is captured immediately
                    tracker = FbTracker::default();
                    vtracker = VblankTracker::default();
                    ztracker = ZeroTracker::default();
                    disabled_reported = false;
                }
                Err(e) => {
                    eprintln!("[capture-drm] {e:#}; retrying…");
                    continue;
                }
            }
        }
        let (fdr, target) = card.as_ref().unwrap();
        let fdi = fdr.as_raw_fd();
        // vblank mode: sample the sequence BEFORE the fb, so a commit we see
        // now cannot already have been retired by a vblank folded into `seq`
        // (which would make us wait one extra vblank)
        let vblank = if active == ActiveSync::Vblank {
            crtc_id.and_then(|c| get_sequence(fdi, c).ok())
        } else {
            None
        };
        let fb = match *target {
            Target::Crtc(crtc_id) => match get_crtc(fdi, crtc_id) {
                Ok(c) => c.fb_id,
                Err(e) => {
                    eprintln!("[capture-drm] GETCRTC: {e}; rescanning");
                    card = None;
                    last_open_attempt = Some(std::time::Instant::now());
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            },
            Target::Plane(plane_id) => match get_plane(fdi, plane_id) {
                Ok(p) => p.fb_id,
                Err(e) => {
                    eprintln!("[capture-drm] GETPLANE: {e}; rescanning");
                    card = None;
                    last_open_attempt = Some(std::time::Instant::now());
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            },
        };
        let now_ms = started.elapsed().as_millis() as u64;
        let now_us = rdma_display_proto::unix_now_us();
        let disabled = |disabled_reported: &mut bool| {
            // DPMS off / no scanout — tell the server once
            if !*disabled_reported {
                *disabled_reported = true;
                let _ = tx.send(CaptureEvent::Disabled);
            }
        };
        match active {
            ActiveSync::OneBehind => {
                // capture the fb that was just superseded (complete), and the
                // settled current fb once flips stop — never the pending one
                match tracker.update(now_ms, now_us, fb) {
                    FbAction::None => {}
                    FbAction::Disabled => disabled(&mut disabled_reported),
                    FbAction::Capture(id, t0) => {
                        if emit_scanout(&card.as_ref().unwrap().0, id, t0, &tx) {
                            disabled_reported = false;
                        } else {
                            tracker.capture_failed();
                        }
                    }
                }
            }
            ActiveSync::Zero => {
                // TESTING: grab the current fb the moment it changes — no sync
                // at all, so the read may catch the compositor mid-render
                match ztracker.update(now_ms, now_us, fb) {
                    FbAction::None => {}
                    FbAction::Disabled => disabled(&mut disabled_reported),
                    FbAction::Capture(id, t0) => {
                        if emit_scanout(&card.as_ref().unwrap().0, id, t0, &tx) {
                            disabled_reported = false;
                        } else {
                            ztracker.capture_failed();
                        }
                    }
                }
            }
            ActiveSync::Vblank => {
                // capture the CURRENT fb as soon as its vblank retires it
                let (seq, ns) = vblank.unwrap_or((0, 0));
                match vtracker.update(now_ms, now_us, fb, seq, ns) {
                    VStep::None => {}
                    VStep::Disabled => disabled(&mut disabled_reported),
                    VStep::Capture(id, t0) => {
                        if emit_scanout(&card.as_ref().unwrap().0, id, t0, &tx) {
                            disabled_reported = false;
                        } else {
                            vtracker.capture_failed();
                        }
                    }
                }
                if vtracker.give_up() && active == ActiveSync::Vblank {
                    eprintln!(
                        "[capture-drm] vblank never advanced within {VBLANK_TIMEOUT_MS} ms \
                         ({VBLANK_FAILS_TO_GIVEUP}× in a row) — switching to one-behind"
                    );
                    active = ActiveSync::OneBehind;
                    tracker = FbTracker::default();
                }
            }
        }
        // polling cadence: vblank/zero modes need a tight poll (zero is trying
        // to shave every millisecond); one-behind can afford 2 ms
        let poll_ms = if matches!(active, ActiveSync::Vblank | ActiveSync::Zero) {
            1
        } else {
            2
        };
        std::thread::sleep(std::time::Duration::from_millis(poll_ms));
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
        assert_eq!(size_of::<DrmModeObjGetProperties>(), 32);
        assert_eq!(size_of::<DrmModeGetProperty>(), 64);
        assert_eq!(size_of::<DrmModeFbCmd2>(), 104); // u64 modifier[] pads
        assert_eq!(size_of::<DrmPrimeHandle>(), 12);
        assert_eq!(size_of::<DrmGemClose>(), 8);
        assert_eq!(size_of::<DrmCrtcGetSequence>(), 24);
        // ioctl request numbers: _IOWR dir bits = 3 (READ|WRITE). Verified
        // against /usr/include/drm/drm.h macros via gcc (not self-computed!)
        assert_eq!(DRM_IOCTL_MODE_GETRESOURCES, 0xC04064A0);
        assert_eq!(DRM_IOCTL_MODE_GETCRTC, 0xC06864A1);
        assert_eq!(DRM_IOCTL_MODE_GETPLANERESOURCES, 0xC01064B5);
        assert_eq!(DRM_IOCTL_MODE_GETPLANE, 0xC02064B6);
        assert_eq!(DRM_IOCTL_MODE_OBJ_GETPROPERTIES, 0xC02064B9);
        assert_eq!(DRM_IOCTL_MODE_GETPROPERTY, 0xC04064AA);
        assert_eq!(DRM_IOCTL_MODE_GETFB2, 0xC06864CE);
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD, 0xC00C642D);
        // vblank sequence query (drm.h: DRM_IOWR(0x3b, drm_crtc_get_sequence))
        assert_eq!(DRM_IOCTL_CRTC_GET_SEQUENCE, 0xC018643B);
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
    fn fb_tracker_one_behind_and_settle() {
        let mut t = FbTracker::default();
        // first sight: capture the current fb immediately to get a picture up
        assert_eq!(t.update(0, 0, 10), FbAction::Capture(10, 0));
        // A->B: the previous (A) was already captured — nothing new yet
        assert_eq!(t.update(2, 2_000, 11), FbAction::None);
        // B->C: capture the now-complete B, t0 = when B first appeared
        assert_eq!(t.update(4, 4_000, 12), FbAction::Capture(11, 2_000));
        // C->D: capture C, t0 = when C first appeared
        assert_eq!(t.update(6, 6_000, 13), FbAction::Capture(12, 4_000));
        // flips stop: not yet settled
        assert_eq!(t.update(10, 10_000, 13), FbAction::None);
        // settled: capture the final frame with its on-screen time
        assert_eq!(t.update(6 + SETTLE_MS, 26_000, 13), FbAction::Capture(13, 6_000));
        // and never repeat it
        assert_eq!(t.update(1000, 1_000_000, 13), FbAction::None);
        // DPMS off (once), then a fresh first frame on re-enable
        assert_eq!(t.update(1002, 1_002_000, 0), FbAction::Disabled);
        assert_eq!(t.update(1004, 1_004_000, 0), FbAction::None);
        assert_eq!(t.update(1006, 1_006_000, 20), FbAction::Capture(20, 1_006_000));
    }

    #[test]
    fn fb_tracker_failed_import_retries_after_settle() {
        let mut t = FbTracker::default();
        assert_eq!(t.update(0, 0, 5), FbAction::Capture(5, 0));
        // the import failed -> the claim is dropped and the settled fb is
        // captured again on a later poll
        t.capture_failed();
        assert_eq!(t.update(SETTLE_MS + 1, 21_000, 5), FbAction::Capture(5, 0));
    }

    #[test]
    fn vblank_tracker_waits_for_the_retire() {
        let mut t = VblankTracker::default();
        // first commit: hold until the vblank that retires it
        assert_eq!(t.update(0, 0, 10, 100, 0), VStep::None);
        // same vblank: still nothing
        assert_eq!(t.update(1, 1_000, 10, 100, 0), VStep::None);
        // vblank advanced: the commit is on screen and complete -> capture it
        assert!(matches!(
            t.update(2, 2_000, 10, 101, 0),
            VStep::Capture(10, t0) if t0 > 0
        ));
        // a new commit gets its own wait
        assert_eq!(t.update(3, 3_000, 11, 101, 0), VStep::None);
        assert!(matches!(
            t.update(4, 4_000, 11, 102, 0),
            VStep::Capture(11, _)
        ));
        assert_eq!(t.failures, 0);
    }

    #[test]
    fn vblank_tracker_timeout_falls_back_then_gives_up() {
        let mut t = VblankTracker::default();
        // prime: capture 10 at its vblank
        assert_eq!(t.update(0, 0, 10, 100, 0), VStep::None);
        assert!(matches!(
            t.update(1, 1_000, 10, 101, 0),
            VStep::Capture(10, _)
        ));

        // a commit whose vblank never arrives: after the bounded wait take the
        // committed frame (certainly complete by then)
        assert_eq!(t.update(2, 2_000, 11, 101, 0), VStep::None);
        assert!(matches!(
            t.update(2 + VBLANK_TIMEOUT_MS + 1, 70_000, 11, 101, 0),
            VStep::Capture(11, 2_000)
        ));
        assert_eq!(t.failures, 1);
        assert!(!t.give_up());

        // two more consecutive failures cross the give-up threshold
        for (fb, now_ms, now_us) in [(12u32, 80u64, 80_000u64), (13, 160, 160_000)] {
            assert_eq!(t.update(now_ms, now_us, fb, 101, 0), VStep::None);
            assert!(matches!(
                t.update(now_ms + VBLANK_TIMEOUT_MS + 1, now_us + 70_000, fb, 101, 0),
                VStep::Capture(b, _) if b == fb
            ));
        }
        assert!(t.give_up());
    }

    #[test]
    fn zero_tracker_captures_immediately() {
        let mut t = ZeroTracker::default();
        // first observation: capture the current fb at once, no wait
        assert_eq!(t.update(0, 0, 5), FbAction::Capture(5, 0));
        // unchanged: nothing
        assert_eq!(t.update(1, 1_000, 5), FbAction::None);
        // new commit: capture it immediately with its observation time as t0
        assert_eq!(t.update(2, 2_000, 6), FbAction::Capture(6, 2_000));
        // and again, no filtering
        assert_eq!(t.update(3, 3_000, 7), FbAction::Capture(7, 3_000));
        // DPMS off once, then a fresh first frame on re-enable
        assert_eq!(t.update(4, 4_000, 0), FbAction::Disabled);
        assert_eq!(t.update(5, 5_000, 0), FbAction::None);
        assert_eq!(t.update(6, 6_000, 8), FbAction::Capture(8, 6_000));
    }

    #[test]
    fn zero_tracker_failed_import_retries_after_settle() {
        let mut t = ZeroTracker::default();
        assert_eq!(t.update(0, 0, 5), FbAction::Capture(5, 0));
        t.capture_failed();
        assert_eq!(t.update(SETTLE_MS + 1, 21_000, 5), FbAction::Capture(5, 0));
    }
}
