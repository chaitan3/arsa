//! Standalone Rust diagnostic for the DRM two-call enumeration — mirrors
//! drm.rs exactly (same structs, same computed ioctl numbers) and hex-dumps
//! the struct before/after each ioctl so a divergence from the C probe
//! (drmprobe) is pinpointed in one run. Run under sudo.

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
#[derive(Clone, Copy)]
struct DrmModeCrtc {
    set_connectors_ptr: u64,
    count_connectors: u32,
    crtc_id: u32,
    fb_id: u32,
    x: u32,
    y: u32,
    gamma_size: u32,
    mode_valid: u32,
    mode_name: [u8; 68], // drm_mode_modeinfo blob, size-only
}

impl Default for DrmModeCrtc {
    fn default() -> Self {
        // SAFETY: all-zero is the zeroed C struct
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct DrmModeGetPlaneRes {
    plane_id_ptr: u64,
    count_planes: u32,
}

const D: u64 = 'd' as u64;
const fn iowr(nr: u16, size: usize) -> u64 {
    (2 << 30) | ((size as u64) << 16) | (D << 8) | nr as u64
}

const GETRESOURCES: u64 = iowr(0xA0, std::mem::size_of::<DrmModeCardRes>());
const GETCRTC: u64 = iowr(0xA1, std::mem::size_of::<DrmModeCrtc>());
const GETPLANERES: u64 = iowr(0xB5, std::mem::size_of::<DrmModeGetPlaneRes>());

fn ioctl(fd: i32, req: u64, arg: *mut core::ffi::c_void) -> i32 {
    unsafe { libc::ioctl(fd, req as libc::c_ulong, arg) }
}

fn hex(buf: &[u8]) -> String {
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn dump(tag: &str, p: &DrmModeCardRes) {
    let bytes = unsafe {
        std::slice::from_raw_parts(p as *const _ as *const u8, std::mem::size_of::<DrmModeCardRes>())
    };
    println!("{tag}: size={} bytes={}", bytes.len(), hex(bytes));
}

fn main() {
    println!(
        "sizeof card_res={} crtc={} plane_res={} | GETRESOURCES={:#x} GETCRTC={:#x} GETPLANERES={:#x}",
        std::mem::size_of::<DrmModeCardRes>(),
        std::mem::size_of::<DrmModeCrtc>(),
        std::mem::size_of::<DrmModeGetPlaneRes>(),
        GETRESOURCES,
        GETCRTC,
        GETPLANERES,
    );
    let fd = unsafe {
        libc::open(c"/dev/dri/card0".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC)
    };
    if fd < 0 {
        println!("open: {}", std::io::Error::last_os_error());
        return;
    }
    println!("opened card0 fd={fd}");

    // call #1: everything zero
    let mut res = DrmModeCardRes::default();
    dump("call1-in", &res);
    let r = ioctl(fd, GETRESOURCES, &mut res as *mut _ as *mut _);
    dump("call1-out", &res);
    if r < 0 {
        println!("call1 errno: {}", std::io::Error::last_os_error());
        return;
    }
    let n = res.count_crtcs as usize;
    println!("n={n}");

    if n > 0 {
        let mut crtcs: Vec<u32> = vec![0; n];
        let mut res = DrmModeCardRes {
            crtc_id_ptr: crtcs.as_mut_ptr() as u64,
            count_crtcs: n as u32,
            ..Default::default()
        };
        dump("call2-in", &res);
        let r = ioctl(fd, GETRESOURCES, &mut res as *mut _ as *mut _);
        dump("call2-out", &res);
        println!("call2 ret={r} errno-if-neg={}", if r < 0 { std::io::Error::last_os_error().to_string() } else { "-".into() });
        println!("crtcs = {crtcs:?}");

        for &id in crtcs.iter().take(res.count_crtcs as usize) {
            let mut c = DrmModeCrtc::default();
            c.crtc_id = id;
            let r = ioctl(fd, GETCRTC, &mut c as *mut _ as *mut _);
            println!("GETCRTC({id}) ret={r} fb_id={}", c.fb_id);
        }
    }

    // planes, same pattern
    let mut pres = DrmModeGetPlaneRes::default();
    let r = ioctl(fd, GETPLANERES, &mut pres as *mut _ as *mut _);
    println!("GETPLANERES ret={r} count={}", pres.count_planes);
    if r >= 0 && pres.count_planes > 0 {
        let pn = pres.count_planes as usize;
        let mut planes: Vec<u32> = vec![0; pn];
        let mut pres = DrmModeGetPlaneRes {
            plane_id_ptr: planes.as_mut_ptr() as u64,
            count_planes: pn as u32,
        };
        let r = ioctl(fd, GETPLANERES, &mut pres as *mut _ as *mut _);
        println!("GETPLANERES2 ret={r} planes={planes:?}");
    }
}
