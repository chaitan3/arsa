//! CUDA-allocated receive slots (C0 route 2).
//!
//! The GBM route proved NVIDIA's allocator *can* give us linear scanout
//! buffers (modifier 0 via `gbm_bo_create_with_modifiers`), but registering
//! the export with the NIC failed ENOMEM: nvidia-drm's static PRIME export
//! refuses to provide peer-DMA mappings for VRAM — the same wall the server
//! hit from the outside.
//!
//! The CUDA driver API is NVIDIA's documented GPUDirect-RMA source:
//! `cuMemAlloc` memory is linear by construction, and
//! `cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)`
//! exports it through the dedicated P2P dmabuf implementation (BAR1
//! addresses, dynamic pinning) that NICs are meant to attach to. That fd
//! feeds the same `ibv_reg_dmabuf_mr` call the GBM path used.
//!
//! libcuda.so.1 is dlopened; the driver (not the toolkit) is all that's
//! needed, and it is present wherever EGL/GLES on NVIDIA works.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::slots::{mmap_read_fd, SlotAlloc, SlotBuf};

type VoidPtr = *mut core::ffi::c_void;

// cuda.h: CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
const CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD: u32 = 1;
// cuda.h: CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
const CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: i32 = 121;

fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T> {
    unsafe {
        let s: libloading::Symbol<T> = lib
            .get(name.as_bytes())
            .with_context(|| format!("libcuda missing symbol {name}"))?;
        Ok(*s)
    }
}

/// Raw driver-API entry points we need. All return CUresult (i32).
struct Cuda {
    #[allow(dead_code)]
    lib: Arc<libloading::Library>,
    init: unsafe extern "system" fn(u32) -> i32,
    device_get: unsafe extern "system" fn(*mut i32, i32) -> i32,
    device_get_attribute: unsafe extern "system" fn(*mut i32, i32, i32) -> i32,
    primary_ctx_retain: unsafe extern "system" fn(*mut VoidPtr, i32) -> i32,
    ctx_set_current: unsafe extern "system" fn(VoidPtr) -> i32,
    mem_alloc: unsafe extern "system" fn(*mut u64, usize) -> i32,
    mem_free: unsafe extern "system" fn(u64) -> i32,
    mem_get_handle:
        unsafe extern "system" fn(*mut VoidPtr, u64, usize, u32, u64) -> i32,
    memcpy_dto_h: unsafe extern "system" fn(*mut VoidPtr, u64, usize) -> i32,
    get_error_name: unsafe extern "system" fn(i32, *mut *const i8) -> i32,
}

impl Cuda {
    fn load() -> Result<Self> {
        let lib = Arc::new(unsafe {
            libloading::Library::new("libcuda.so.1")
                .or_else(|_| libloading::Library::new("libcuda.so"))
                .context("dlopen libcuda (NVIDIA driver; the CUDA toolkit is NOT required)")?
        });
        Ok(Cuda {
            init: sym(&lib, "cuInit")?,
            device_get: sym(&lib, "cuDeviceGet")?,
            device_get_attribute: sym(&lib, "cuDeviceGetAttribute")?,
            primary_ctx_retain: sym(&lib, "cuDevicePrimaryCtxRetain")?,
            ctx_set_current: sym(&lib, "cuCtxSetCurrent")?,
            mem_alloc: sym(&lib, "cuMemAlloc_v2")
                .or_else(|_| sym(&lib, "cuMemAlloc"))?,
            mem_free: sym(&lib, "cuMemFree_v2").or_else(|_| sym(&lib, "cuMemFree"))?,
            mem_get_handle: sym(&lib, "cuMemGetHandleForAddressRange")?,
            memcpy_dto_h: sym(&lib, "cuMemcpyDtoH_v2")
                .or_else(|_| sym(&lib, "cuMemcpyDtoH"))?,
            get_error_name: sym(&lib, "cuGetErrorName")?,
            lib,
        })
    }

    /// Human-readable CUresult name for error messages.
    fn err(&self, res: i32, what: &str) -> anyhow::Error {
        let mut name = format!("CUresult {res}");
        unsafe {
            let mut p: *const i8 = std::ptr::null();
            if (self.get_error_name)(res, &mut p) == 0 && !p.is_null() {
                name = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
            }
        }
        anyhow::anyhow!("{what} failed: {name}")
    }

    fn ok(&self, res: i32, what: &str) -> Result<()> {
        if res == 0 {
            Ok(())
        } else {
            Err(self.err(res, what))
        }
    }
}

/// Slots allocated as CUDA device memory and exported as dma-bufs.
pub struct CudaSlots {
    cuda: Arc<Cuda>,
    /// kept for diagnostics / primary-ctx reset on teardown
    #[allow(dead_code)]
    device: i32,
}

// single-threaded use (the receiver loop); the primary context is retained
// for the process lifetime on purpose.
unsafe impl Send for CudaSlots {}

impl CudaSlots {
    pub fn open(ordinal: u32) -> Result<Self> {
        let cuda = Arc::new(Cuda::load()?);
        cuda.ok(unsafe { (cuda.init)(0) }, "cuInit")?;
        let mut dev = 0i32;
        cuda.ok(
            unsafe { (cuda.device_get)(&mut dev, ordinal as i32) },
            "cuDeviceGet",
        )?;
        // Gate before touching anything else: is the dmabuf export supported?
        let mut supported = 0i32;
        cuda.ok(
            unsafe {
                (cuda.device_get_attribute)(
                    &mut supported,
                    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                    dev,
                )
            },
            "cuDeviceGetAttribute(DMA_BUF_SUPPORTED)",
        )?;
        if supported == 0 {
            bail!(
                "C0 gate FAILED: CUDA device {ordinal} reports DMA_BUF_SUPPORTED=0 — \
                 GPUDirect-over-dmabuf export is unavailable on this GPU/driver combo"
            );
        }
        // A context must be current for cuMemAlloc; the primary context is the
        // side-effect-free way (no capture of KWin's, shared ownership).
        let mut ctx: VoidPtr = std::ptr::null_mut();
        cuda.ok(
            unsafe { (cuda.primary_ctx_retain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        cuda.ok(unsafe { (cuda.ctx_set_current)(ctx) }, "cuCtxSetCurrent")?;
        log::info!("[slots] cuda: device {ordinal} ready (primary ctx retained, DMA_BUF supported)");
        Ok(CudaSlots { cuda, device: dev })
    }
}

impl SlotAlloc for CudaSlots {
    fn alloc_slot(&self, width: u32, height: u32, stride: u32) -> Result<Box<dyn SlotBuf>> {
        let _ = width;
        // CUDA allocations are linear by construction: size is all we need.
        let size = (stride as usize) * (height as usize);
        let mut dptr = 0u64;
        self.cuda
            .ok(unsafe { (self.cuda.mem_alloc)(&mut dptr, size) }, "cuMemAlloc")?;
        let mut fd_raw: i32 = -1;
        let res = unsafe {
            (self.cuda.mem_get_handle)(
                &mut fd_raw as *mut i32 as *mut VoidPtr,
                dptr,
                size,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                0,
            )
        };
        if res != 0 {
            unsafe { (self.cuda.mem_free)(dptr) };
            return Err(self.cuda.err(res, "cuMemGetHandleForAddressRange(DMA_BUF_FD)"));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd_raw) };
        // best-effort CPU view, same policy as the GBM path
        let map = mmap_read_fd(fd.as_raw_fd(), size);
        if map.is_null() {
            log::warn!(
                "[slots] cuda NOTE: PRIME mmap failed — content checks use DtoH copy \
                 (registration unaffected)"
            );
        }
        log::info!("[slots] cuda slot: dptr={dptr:#x} size={size} fd={}", fd.as_raw_fd());
        Ok(Box::new(CudaSlot {
            cuda: Arc::clone(&self.cuda),
            dptr,
            size,
            fd,
            map,
        }))
    }
}

struct CudaSlot {
    cuda: Arc<Cuda>,
    dptr: u64,
    size: usize,
    fd: OwnedFd,
    map: VoidPtr,
}

unsafe impl Send for CudaSlot {}

impl SlotBuf for CudaSlot {
    fn fd(&self) -> std::os::fd::RawFd {
        self.fd.as_raw_fd()
    }
    fn iova(&self) -> u64 {
        // dmabuf offset 0; iova may be any page-aligned value — use the CPU
        // map when present, else a synthetic low VA (never used locally)
        if self.map.is_null() {
            0x1000_0000
        } else {
            self.map as u64
        }
    }
    fn cpu_bytes(&self, n: usize) -> Option<&[u8]> {
        if self.map.is_null() || n > self.size {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.map as *const u8, n) })
    }
    fn dto_h_vec(&self, n: usize) -> Result<Vec<u8>> {
        // verification-only copy path (C0); the steady-state render path
        // never calls this
        let mut v = vec![0u8; n];
        self.cuda.ok(
            unsafe {
                (self.cuda.memcpy_dto_h)(v.as_mut_ptr() as *mut VoidPtr, self.dptr, n)
            },
            "cuMemcpyDtoH (content verification)",
        )?;
        Ok(v)
    }
}

impl Drop for CudaSlot {
    fn drop(&mut self) {
        // MR deregistration already happened (field order in Slot::Gpu)
        if !self.map.is_null() {
            unsafe { crate::slots::munmap(self.map, self.size) };
        }
        unsafe { (self.cuda.mem_free)(self.dptr) };
        // fd closes via OwnedFd; primary ctx intentionally left retained
    }
}
