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
//! Known gaps: no cursor-plane blending (hardware cursors are invisible;
//! software cursors are in the fb), no damage info (every flip is full-frame),
//! readback is unfenced (scanout buffers are stable while scanned out; the
//! compositor's next render into the same buffer is ~1-2 flips away).

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::mpsc::Sender;

use anyhow::{bail, Context, Result};

use crate::{CaptureEvent, CaptureSource, DmabufScanout, FrameFormat};

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

// DRM_FORMAT_MOD_INVALID = fourcc_mod_code(NONE, 0x00ffffff...); EGL refuses
// it, and it means "driver gave no real modifier" — treat as linear.
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

pub struct DrmKmsSource {
    /// Card node to capture on. None = probe /dev/dri/card0.. card8 for the
    /// first device with an active CRTC.
    pub card: Option<String>,
}

impl DrmKmsSource {
    pub fn new(card: Option<String>) -> Self {
        Self { card }
    }
}

impl CaptureSource for DrmKmsSource {
    fn name(&self) -> &'static str {
        "drm-kms"
    }

    fn spawn(self: Box<Self>, tx: Sender<CaptureEvent>) -> Result<()> {
        std::thread::Builder::new()
            .name("capture-drm".into())
            .spawn(move || {
                if let Err(e) = run(self.card.as_deref(), tx) {
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
        fds: vec![dmabuf],
        offsets: vec![fb.offsets[0]],
        strides: vec![fb.pitches[0]],
        format,
    })
}

fn run(card_pref: Option<&str>, tx: Sender<CaptureEvent>) -> Result<()> {
    eprintln!(
        "[capture-drm] opening card {}",
        card_pref.unwrap_or("(probe /dev/dri/card0..8)")
    );
    let mut card: Option<(OwnedFd, Target)> = None;
    let mut last_fb: u32 = 0;
    let mut disabled_reported = false;
    let mut last_rescan = std::time::Instant::now();
    loop {
        if card.is_none() {
            if last_fb == 0 || last_fb == u32::MAX {
                // (re)scan at most once a second while hunting for a display
                if last_fb == u32::MAX && last_rescan.elapsed() < std::time::Duration::from_secs(1) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                last_rescan = std::time::Instant::now();
            }
            match open_card(card_pref) {
                Ok(c) => {
                    eprintln!("[capture-drm] target: {:?}", c.1);
                    card = Some(c);
                }
                Err(e) => {
                    eprintln!("[capture-drm] {e:#}; retrying…");
                    last_fb = u32::MAX;
                    continue;
                }
            }
        }
        let (fdr, target) = card.as_ref().unwrap();
        let fdi = fdr.as_raw_fd();
        let fb = match *target {
            Target::Crtc(crtc_id) => match get_crtc(fdi, crtc_id) {
                Ok(c) => c.fb_id,
                Err(e) => {
                    eprintln!("[capture-drm] GETCRTC: {e}; rescanning");
                    card = None;
                    last_fb = 0;
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            },
            Target::Plane(plane_id) => match get_plane(fdi, plane_id) {
                Ok(p) => p.fb_id,
                Err(e) => {
                    eprintln!("[capture-drm] GETPLANE: {e}; rescanning");
                    card = None;
                    last_fb = 0;
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            },
        };
        match fb {
            0 => {
                // DPMS off / no scanout — tell the server once
                if !disabled_reported {
                    disabled_reported = true;
                    let _ = tx.send(CaptureEvent::Disabled);
                }
                last_fb = 0;
            }
            id if id == last_fb => {} // no flip — nothing to send
            id => {
                match import_fb(&card.as_ref().unwrap().0, id) {
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
                        let _ = tx.send(CaptureEvent::Scanout(dmabuf));
                        last_fb = id;
                        disabled_reported = false;
                    }
                    Err(e) => {
                        // retry next tick; keep last_fb so we don't spam
                        eprintln!("[capture-drm] fb {id} import failed: {e:#}");
                        last_fb = id;
                    }
                }
            }
        }
        // polling cadence: compositor flips arrive at refresh rate; a 2 ms
        // poll adds <0.1 ms latency on average and one ioctl costs ~µs
        std::thread::sleep(std::time::Duration::from_millis(2));
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
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x40086409); // _IOW, dir=1
    }
}
