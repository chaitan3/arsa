//! GPU-allocated receive slots (C0 zero-copy experiment, client side).
//!
//! Slot buffers live on the *client* GPU, are exported as dma-bufs, and are
//! registered with `ibv_reg_dmabuf_mr` so RDMA WRITEs land directly in GPU
//! memory. The trait pair is deliberately generic so alternative exporters
//! plug in behind `SlotMem`.
//!
//! CUDA backend (`cuda.rs`) is the only survivor of the C0 experiment:
//! - GBM allocation worked (linear via `gbm_bo_create_with_modifiers`) but
//!   its export never registered with the NIC — nvidia-drm's static PRIME
//!   path refuses peer-DMA mappings (ENOMEM).
//! - CUDA's `cuMemGetHandleForAddressRange(DMA_BUF_FD)` export is refused
//!   on GeForce-class GPUs (RM product gate: CUDA_ERROR_NOT_SUPPORTED).
//! Full results in `docs/DEBUG.md`; resurrect on cooperative
//! hardware (open-module pro/data-center GPUs or Mesa GPUs) if ever needed.

use anyhow::{bail, Result};

/// A receive slot backed by GPU memory: an exported dma-buf fd the NIC can
/// register, with optional CPU visibility for content verification.
pub trait SlotBuf {
    /// dma-buf fd for `ibv_reg_dmabuf_mr` (kept open until after MR drop)
    fn fd(&self) -> std::os::fd::RawFd;
    /// byte offset of the slot data inside the fd
    fn offset(&self) -> u64 {
        0
    }
    /// registration iova: must share its page offset with `offset()`
    fn iova(&self) -> u64;
    /// best-effort CPU view (dma-buf mmap), None if refused
    fn cpu_bytes(&self, n: usize) -> Option<&[u8]>;
    /// verification-only device->host copy (Cuda implements; the steady
    /// state render path never calls this)
    fn dto_h_vec(&self, _n: usize) -> Result<Vec<u8>> {
        bail!("allocator has no CPU mapping and no device->host copy path")
    }
}

/// Allocator of GPU slots (currently the CUDA backend).
pub trait SlotAlloc {
    fn alloc_slot(&self, width: u32, height: u32, stride: u32) -> Result<Box<dyn SlotBuf>>;
}

// --- minimal libc shims (no `libc` dep in this crate) ------------------------

extern "C" {
    fn mmap(
        addr: *mut core::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut core::ffi::c_void;
    #[link_name = "munmap"]
    fn munmap_sys(addr: *mut core::ffi::c_void, len: usize) -> i32;
}

const PROT_READ: i32 = 1;
const MAP_SHARED: i32 = 1;

/// PROT_READ|MAP_SHARED mapping of `len` bytes of `fd`; null on failure.
pub(crate) fn mmap_read_fd(fd: i32, len: usize) -> *mut core::ffi::c_void {
    let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_SHARED, fd, 0) };
    if p as isize == -1 {
        std::ptr::null_mut()
    } else {
        p
    }
}

pub(crate) unsafe fn munmap(addr: *mut core::ffi::c_void, len: usize) {
    munmap_sys(addr, len);
}
