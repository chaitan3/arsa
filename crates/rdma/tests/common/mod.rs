//! Shared helpers for the hardware-gated GBM / dma-buf MR tests.
//!
//! These are compiled into each test crate that does `mod common;` and are
//! only exercised by `#[ignore]`d tests run explicitly on real hardware.

#![allow(dead_code)]

use std::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use anyhow::{bail, Context, Result};
use ibverbs::{
    AccessFlags, CompletionQueue, Context as IbvContext, PortState, ProtectionDomain, QueuePair,
    Rc,
};

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;
pub const FORMAT_XR24: u32 = 0x3432_5258; // DRM_FORMAT_XRGB8888

type CreateDevice = unsafe extern "C" fn(i32) -> *mut c_void;
type DeviceDestroy = unsafe extern "C" fn(*mut c_void);
type BoCreateMods =
    unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u64, u32) -> *mut c_void;
type BoCreatePlain = unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32) -> *mut c_void;
type BoI32 = unsafe extern "C" fn(*mut c_void) -> i32;
type BoU32 = unsafe extern "C" fn(*mut c_void) -> u32;
type BoOffset = unsafe extern "C" fn(*mut c_void, i32) -> u32;
type BoU64 = unsafe extern "C" fn(*mut c_void) -> u64;
type BoDestroy = unsafe extern "C" fn(*mut c_void);

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    #[link_name = "munmap"]
    fn munmap_sys(addr: *mut c_void, len: usize) -> i32;
}

const PROT_READ: i32 = 1;
const MAP_SHARED: i32 = 1;

/// A GBM device on a DRM node, plus the libgbm entry points the tests need.
pub struct Gbm {
    // keep the library and the DRM fd alive for the GBM device
    _lib: libloading::Library,
    _drm: std::fs::File,
    dev: *mut c_void,
    device_destroy: DeviceDestroy,
    bo_create: Option<BoCreateMods>,
    bo_create_plain: BoCreatePlain,
    bo_fd: BoI32,
    bo_stride: BoU32,
    bo_offset: BoOffset,
    bo_modifier: BoU64,
    bo_height: BoU32,
    bo_destroy: BoDestroy,
}

/// A linear GBM bo exported as a dma-buf. The owning [`Gbm`] must outlive it.
pub struct GbmBo {
    pub fd: OwnedFd,
    pub offset: u64,
    pub stride: usize,
    pub height: u32,
    pub modifier: u64,
    /// `stride * height` for a linear bo.
    pub size: usize,
    bo: *mut c_void,
    destroy: BoDestroy,
}

impl Drop for GbmBo {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.bo) };
    }
}

impl Drop for Gbm {
    fn drop(&mut self) {
        unsafe { (self.device_destroy)(self.dev) };
    }
}

/// Prefer render nodes, then card nodes. `ARSA_DRM_NODE` overrides the search.
pub fn open_drm_node() -> Option<(std::fs::File, String)> {
    if let Ok(path) = std::env::var("ARSA_DRM_NODE") {
        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        {
            Ok(f) => return Some((f, path)),
            Err(e) => {
                eprintln!("note: cannot open ARSA_DRM_NODE={path}: {e}");
                return None;
            }
        }
    }

    let entries: Vec<_> = std::fs::read_dir("/dev/dri")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    let mut render: Vec<_> = entries
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("renderD"))
        })
        .cloned()
        .collect();
    let mut card: Vec<_> = entries
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card"))
        })
        .cloned()
        .collect();
    render.sort();
    card.sort();

    for path in render.into_iter().chain(card) {
        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        {
            Ok(f) => return Some((f, path.display().to_string())),
            Err(e) => eprintln!("note: cannot open {}: {e}", path.display()),
        }
    }
    None
}

impl Gbm {
    pub fn open(rendernode: Option<&str>) -> Result<(Self, String)> {
        let (drm, path) = if let Some(node) = rendernode {
            match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(node)
            {
                Ok(f) => (f, node.to_string()),
                Err(e) => bail!("cannot open ARSA_DRM_NODE={node}: {e}"),
            }
        } else {
            open_drm_node().context("no usable /dev/dri render or card node")?
        };
        let lib = unsafe { libloading::Library::new("libgbm.so.1") }
            .context("dlopen libgbm.so.1")?;
        unsafe {
            let create_device: CreateDevice = *lib
                .get(b"gbm_create_device")
                .context("libgbm missing gbm_create_device")?;
            let device_destroy: DeviceDestroy = *lib
                .get(b"gbm_device_destroy")
                .context("libgbm missing gbm_device_destroy")?;
            let bo_create: Option<BoCreateMods> = lib
                .get(b"gbm_bo_create_with_modifiers")
                .ok()
                .map(|s| *s);
            let bo_create_plain: BoCreatePlain = *lib
                .get(b"gbm_bo_create")
                .context("libgbm missing gbm_bo_create")?;
            let bo_fd: BoI32 = *lib
                .get(b"gbm_bo_get_fd")
                .context("libgbm missing gbm_bo_get_fd")?;
            let bo_stride: BoU32 = *lib
                .get(b"gbm_bo_get_stride")
                .context("libgbm missing gbm_bo_get_stride")?;
            let bo_offset: BoOffset = *lib
                .get(b"gbm_bo_get_offset")
                .context("libgbm missing gbm_bo_get_offset")?;
            let bo_modifier: BoU64 = *lib
                .get(b"gbm_bo_get_modifier")
                .context("libgbm missing gbm_bo_get_modifier")?;
            let bo_height: BoU32 = *lib
                .get(b"gbm_bo_get_height")
                .context("libgbm missing gbm_bo_get_height")?;
            let bo_destroy: BoDestroy = *lib
                .get(b"gbm_bo_destroy")
                .context("libgbm missing gbm_bo_destroy")?;

            let dev = create_device(drm.as_raw_fd());
            if dev.is_null() {
                bail!(
                    "gbm_create_device({path}) failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            Ok((
                Gbm {
                    _lib: lib,
                    _drm: drm,
                    dev,
                    device_destroy,
                    bo_create,
                    bo_create_plain,
                    bo_fd,
                    bo_stride,
                    bo_offset,
                    bo_modifier,
                    bo_height,
                    bo_destroy,
                },
                path,
            ))
        }
    }

    /// Allocate a linear XR24 bo with at least `min_height` rows. The GBM bo is
    /// a byte buffer here; the caller sizes it by the slot length, not by the
    /// display.
    pub fn alloc_linear(&self, width: u32, min_height: u32) -> Result<GbmBo> {
        unsafe {
            // Request DRM_FORMAT_MOD_LINEAR; fall back to the legacy
            // GBM_BO_USE_LINEAR(16) | GBM_BO_USE_RENDERING(4) allocator.
            let mods = [0u64];
            let mut bo = std::ptr::null_mut();
            if let Some(f) = self.bo_create {
                bo = f(self.dev, width, min_height, FORMAT_XR24, mods.as_ptr(), 1);
            }
            if bo.is_null() {
                bo = (self.bo_create_plain)(self.dev, width, min_height, FORMAT_XR24, 16 | 4);
            }
            if bo.is_null() {
                bail!(
                    "gbm_bo_create(linear) failed: {}",
                    std::io::Error::last_os_error()
                );
            }

            let raw_fd = (self.bo_fd)(bo);
            if raw_fd < 0 {
                (self.bo_destroy)(bo);
                bail!("gbm_bo_get_fd failed (buffer not exportable)");
            }
            let fd = OwnedFd::from_raw_fd(raw_fd);
            let stride = (self.bo_stride)(bo) as usize;
            let height = (self.bo_height)(bo);
            let offset = u64::from((self.bo_offset)(bo, 0));
            let modifier = (self.bo_modifier)(bo);
            let size = stride * height as usize;
            Ok(GbmBo {
                fd,
                offset,
                stride,
                height,
                modifier,
                size,
                bo,
                destroy: self.bo_destroy,
            })
        }
    }
}

/// Read-only CPU view of `len` bytes of a dma-buf starting at `offset`.
pub fn mmap_read(fd: i32, offset: u64, len: usize) -> Option<*const u8> {
    let map_len = offset as usize + len;
    let p = unsafe { mmap(std::ptr::null_mut(), map_len, PROT_READ, MAP_SHARED, fd, 0) };
    if p as isize == -1 {
        None
    } else {
        Some(unsafe { (p as *const u8).add(offset as usize) })
    }
}

/// Release a mapping returned by [`mmap_read`] (the offset must be repeated).
pub fn munmap_read(base: *const u8, offset: u64, len: usize) {
    let map_len = offset as usize + len;
    let base = unsafe { base.sub(offset as usize) } as *mut c_void;
    unsafe { munmap_sys(base, map_len) };
}

/// Open `ARSA_IB_DEVICE`, or the first RDMA device.
pub fn open_rdma() -> Option<(IbvContext, String)> {
    let list = ibverbs::devices().ok()?;
    if list.is_empty() {
        return None;
    }
    let want = std::env::var("ARSA_IB_DEVICE").ok();
    let dev = match &want {
        Some(name) if !name.is_empty() => list
            .iter()
            .find(|d| d.name().is_some_and(|n| n.to_bytes() == name.as_bytes()))?,
        _ => list.iter().next()?,
    };
    let name = dev
        .name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some((dev.open().ok()?, name))
}

/// True when port 1 is ACTIVE or ARMED (the state `create_qp` requires).
pub fn port_active(ctx: &IbvContext) -> bool {
    match ctx.query_port(1) {
        Ok(p) => matches!(p.state(), PortState::Active | PortState::Armed),
        Err(e) => {
            eprintln!("note: query_port(1): {e}");
            false
        }
    }
}

/// Build a self-connected RC queue pair with the given access flags.
pub fn self_qp(
    ctx: &IbvContext,
    pd: &ProtectionDomain,
    cq: &CompletionQueue,
    access: AccessFlags,
) -> QueuePair<Rc> {
    let gid = ctx
        .routable_gid(1)
        .expect("read the GID table")
        .expect("no routable GID on port 1")
        .gid_index;
    let mut builder = pd.create_qp::<Rc>(cq, cq, 1).expect("create_qp");
    builder
        .set_gid_index(gid)
        .set_max_send_wr(16)
        .set_max_recv_wr(16)
        .set_max_send_sge(4)
        .set_max_recv_sge(4)
        .set_access(access);
    let prepared = builder.build().expect("build qp");
    let ep = prepared.endpoint().expect("read the local endpoint");
    prepared.handshake(ep).expect("self-handshake")
}

/// Poll until the completion with `wr_id` arrives; panic on a failed WR or
/// timeout.
pub fn drain_one(cq: &CompletionQueue, wr_id: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let mut comps = cq.poll().expect("poll CQ");
        while let Some(wc) = comps.next() {
            if let Err(e) = wc.ok() {
                panic!("work request {} failed: {e}", wc.wr_id());
            }
            if wc.wr_id() == wr_id {
                return;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for wr_id {wr_id}"
        );
    }
}
