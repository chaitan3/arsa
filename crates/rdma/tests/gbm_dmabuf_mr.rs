//! Retry the GBM → `ibv_reg_dmabuf_mr` client zero-copy route.
//!
//! The C0 experiment found that a linear GBM buffer exported through NVIDIA's
//! static PRIME path was refused by `ibv_reg_dmabuf_mr` with `ENOMEM`. This
//! test re-runs that exact route on the current driver/hardware:
//!
//! 1. allocate a linear XR24 GBM bo on a DRM render node and export it as a
//!    dma-buf;
//! 2. register it as an RDMA MR with `ibv_reg_dmabuf_mr`;
//! 3. self-loopback an RDMA WRITE into it (the same primitive the client's
//!    receive slots use);
//! 4. verify the bytes through a CPU mapping of the dma-buf when the exporter
//!    allows one.
//!
//! It needs a DRM render node and an RDMA device, so it is ignored by default:
//!
//! ```text
//! cargo test -p rdma-display-rdma --test gbm_dmabuf_mr -- --ignored --nocapture
//! ```
//!
//! `ARSA_DRM_NODE` (e.g. `/dev/dri/renderD128`) and `ARSA_IB_DEVICE` (e.g.
//! `mlx5_0`) select a specific node/device. A missing libgbm, DRM node, RDMA
//! device or routable GID is reported as a skip, not a failure. Once all the
//! pieces are present, a failed registration is the result under test and is
//! reported as a test failure.

use std::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::time::{Duration, Instant};

use ibverbs::{AccessFlags, Context, ProtectionDomain, QueuePair, Rc};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FORMAT_XR24: u32 = 0x3432_5258; // DRM_FORMAT_XRGB8888

// libgbm ABI, dlopen'd so the crate keeps no link-time libgbm dependency.
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

/// Prefer render nodes, then card nodes. `ARSA_DRM_NODE` overrides the search.
fn open_drm_node() -> Option<(std::fs::File, String)> {
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

/// Open `ARSA_IB_DEVICE`, or the first RDMA device.
fn open_rdma() -> Option<(Context, String)> {
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

#[test]
#[ignore = "requires a DRM render node and an RDMA device; run with --ignored --nocapture"]
fn gbm_linear_dmabuf_registers_and_receives_rdma_write() {
    let (drm, drm_path) = match open_drm_node() {
        Some(x) => x,
        None => {
            eprintln!("skip: no usable /dev/dri render or card node");
            return;
        }
    };
    let lib = match unsafe { libloading::Library::new("libgbm.so.1") } {
        Ok(l) => l,
        Err(e) => {
            eprintln!("skip: libgbm.so.1 unavailable ({e})");
            return;
        }
    };
    let (ctx, ib_name) = match open_rdma() {
        Some(x) => x,
        None => {
            eprintln!("skip: no RDMA device");
            return;
        }
    };

    unsafe {
        let create_device: CreateDevice =
            *lib.get(b"gbm_create_device").expect("gbm_create_device");
        let device_destroy: DeviceDestroy =
            *lib.get(b"gbm_device_destroy").expect("gbm_device_destroy");
        let bo_create: Option<BoCreateMods> =
            lib.get(b"gbm_bo_create_with_modifiers").ok().map(|s| *s);
        let bo_create_plain: BoCreatePlain = *lib.get(b"gbm_bo_create").expect("gbm_bo_create");
        let bo_fd: BoI32 = *lib.get(b"gbm_bo_get_fd").expect("gbm_bo_get_fd");
        let bo_stride: BoU32 = *lib.get(b"gbm_bo_get_stride").expect("gbm_bo_get_stride");
        let bo_offset: BoOffset = *lib.get(b"gbm_bo_get_offset").expect("gbm_bo_get_offset");
        let bo_modifier: BoU64 = *lib.get(b"gbm_bo_get_modifier").expect("gbm_bo_get_modifier");
        let bo_destroy: BoDestroy = *lib.get(b"gbm_bo_destroy").expect("gbm_bo_destroy");

        let dev = create_device(drm.as_raw_fd());
        if dev.is_null() {
            panic!(
                "gbm_create_device({drm_path}) failed: {}",
                std::io::Error::last_os_error()
            );
        }

        // Request a linear buffer; fall back to the legacy flag-taking alloc.
        let mods = [0u64]; // DRM_FORMAT_MOD_LINEAR
        let mut bo = std::ptr::null_mut();
        if let Some(f) = bo_create {
            bo = f(dev, WIDTH, HEIGHT, FORMAT_XR24, mods.as_ptr(), 1);
        }
        if bo.is_null() {
            // GBM_BO_USE_LINEAR(16) | GBM_BO_USE_RENDERING(4)
            bo = bo_create_plain(dev, WIDTH, HEIGHT, FORMAT_XR24, 16 | 4);
        }
        if bo.is_null() {
            panic!(
                "gbm_bo_create({drm_path}, linear) failed: {}",
                std::io::Error::last_os_error()
            );
        }

        let raw_fd = bo_fd(bo);
        if raw_fd < 0 {
            panic!("gbm_bo_get_fd failed (buffer not exportable)");
        }
        let fd = OwnedFd::from_raw_fd(raw_fd);
        let stride = bo_stride(bo) as usize;
        let offset = u64::from(bo_offset(bo, 0));
        let modifier = bo_modifier(bo);
        let len = stride * HEIGHT as usize;
        eprintln!(
            "gbm: node={drm_path} rdma={ib_name} fd={} offset={offset} stride={stride} \
             modifier={modifier:#x} len={len}",
            fd.as_raw_fd()
        );
        if modifier != 0 {
            eprintln!("skip: GBM returned modifier {modifier:#x}, not linear");
            bo_destroy(bo);
            device_destroy(dev);
            return;
        }

        let pd: ProtectionDomain = ctx.alloc_pd().expect("alloc_pd");
        let cq = ctx.create_cq(64).build().expect("create_cq");
        let gid = match ctx.routable_gid(1).expect("read the GID table") {
            Some(g) => g.gid_index,
            None => {
                eprintln!("skip: no routable GID on port 1");
                drop(cq);
                drop(pd);
                bo_destroy(bo);
                device_destroy(dev);
                return;
            }
        };

        // iova must share `offset`'s page offset; any page-aligned base works.
        let iova = 0x1_0000_0000u64 + offset;
        let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE;
        let dst = match pd.register_dmabuf(fd.as_raw_fd(), offset, len, iova, flags) {
            Ok(m) => m,
            Err(e) => panic!("ibv_reg_dmabuf_mr FAILED for the GBM dma-buf: {e:#}"),
        };
        eprintln!(
            "OK: ibv_reg_dmabuf_mr accepted the GBM dma-buf \
             (addr={:#x} rkey={:#x} lkey={:#x})",
            dst.remote().addr,
            dst.remote().rkey,
            dst.lkey()
        );

        // Prove the NIC can DMA into it with a self-loopback RDMA WRITE, the
        // same primitive the client's receive path uses.
        let mut src = pd.allocate(64, AccessFlags::PERMISSIVE).expect("src MR");
        let pattern = b"arsa-gbm-dmabuf-mr";
        src.bytes_mut()[..pattern.len()].copy_from_slice(pattern);

        let mut builder = pd.create_qp::<Rc>(&cq, &cq, 1).expect("create_qp");
        builder
            .set_gid_index(gid)
            .set_max_send_wr(16)
            .set_max_recv_wr(16)
            .set_max_send_sge(4)
            .set_max_recv_sge(4)
            .set_access(AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_WRITE);
        let prepared = builder.build().expect("build qp");
        let ep = prepared.endpoint().expect("read the local endpoint");
        let mut qp: QueuePair<Rc> = prepared.handshake(ep).expect("self-handshake");

        let remote = dst.remote().slice(..pattern.len());
        let mut batch = qp.start_send();
        batch
            .op()
            .signaled()
            .write(1, &[src.slice(..pattern.len())], remote);
        batch.submit().expect("post RDMA WRITE");

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut comps = cq.poll().expect("poll CQ");
            let mut done = false;
            while let Some(wc) = comps.next() {
                if let Err(e) = wc.ok() {
                    panic!("RDMA WRITE into the GBM dma-buf failed: {e}");
                }
                done |= wc.wr_id() == 1;
            }
            if done {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the RDMA WRITE completion"
            );
        }
        eprintln!("OK: RDMA WRITE completed into the GBM dma-buf");

        // Verify the bytes through a CPU mapping, when the exporter allows one.
        let map_len = offset as usize + pattern.len();
        let p = mmap(
            std::ptr::null_mut(),
            map_len,
            PROT_READ,
            MAP_SHARED,
            fd.as_raw_fd(),
            0,
        );
        if p as isize == -1 {
            eprintln!(
                "note: dma-buf is not CPU-mappable; registration + RDMA WRITE succeeded, \
                 content not checked"
            );
        } else {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let bytes =
                std::slice::from_raw_parts((p as *const u8).add(offset as usize), pattern.len());
            assert_eq!(bytes, pattern, "RDMA WRITE did not land in the GBM buffer");
            munmap_sys(p, map_len);
            eprintln!("OK: content verified through a CPU mapping of the dma-buf");
        }

        drop(qp);
        drop(src);
        drop(dst);
        drop(cq);
        drop(pd);
        drop(fd);
        bo_destroy(bo);
        device_destroy(dev);
    }
}
