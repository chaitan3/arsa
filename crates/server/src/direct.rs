//! Server-side true zero-copy staging for RDMA.
//!
//! Two dma-buf paths, both registered with `ibv_reg_dmabuf_mr` so the NIC
//! reads GPU/scanout memory directly:
//!
//! * [`GpuDirect`] — lossless coded output. A device-local `VkBuffer` is
//!   exported as a dma-buf and registered per staging slot; the Vulkan encoder
//!   writes the coded stream into it and the NIC posts from it.
//! * [`RawDirect`] — raw source. A linear, top-first XRGB8888 scanout plane is
//!   registered as-is; the NIC reads the compositor's buffer with no GPU→host
//!   copy. Only `modifier == 0` and the exact wire byte layout qualify.
//!
//! Every path is best-effort: if export/registration fails the caller keeps
//! using the host-visible Vulkan staging buffers from `gpu.rs`.

use std::collections::{HashSet, VecDeque};
use std::os::fd::AsRawFd;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use ibverbs::{AccessFlags, LocalMemorySlice, MemoryRegion, ProtectionDomain};
use rdma_display_capture::{DmabufScanout, FrameFormat};
use rdma_display_vkcore::{ash::vk, GpuBuffer, VkCore};

/// DRM_FORMAT_XRGB8888 ("XR24"), the wire's BGRX byte order.
const DRM_FORMAT_XRGB8888: u32 = 0x3432_5258;

/// Device-local coded buffers exported as dma-bufs and registered with verbs.
pub struct GpuDirect {
    core: Arc<VkCore>,
    slots: Vec<GpuSlot>,
}

struct GpuSlot {
    /// declared before `buffer` so deregistration happens first on drop
    mr: Option<MemoryRegion<()>>,
    buffer: GpuBuffer,
}

impl GpuDirect {
    /// Create and register `n` coded buffers of `slot_len` bytes. Any failure
    /// unwinds everything created so far and is reported to the caller, which
    /// then uses the host staging path.
    pub fn new(
        core: Arc<VkCore>,
        pd: &ProtectionDomain,
        n: usize,
        slot_len: usize,
    ) -> Result<Self> {
        if slot_len == 0 {
            bail!("zero-sized GPU staging slot");
        }
        let mut out = Self {
            core: core.clone(),
            slots: Vec::with_capacity(n),
        };
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        for i in 0..n {
            let buffer = core
                .create_exportable_buffer(slot_len as u64, usage)
                .with_context(|| format!("create exportable coded buffer {i}"))?;
            let fd = match core.export_dmabuf(buffer.memory) {
                Ok(fd) => fd,
                Err(e) => {
                    destroy_buffer(&core, buffer);
                    return Err(e).context("export coded buffer dma-buf");
                }
            };
            // Unique, page-aligned local IOVA per slot. The dma-buf offset is
            // zero, so the IOVA's page offset must be zero too.
            let iova = 0x4_0000_0000u64 + (i as u64) * 0x4000_0000;
            let mr = pd.register_dmabuf(
                fd,
                0,
                slot_len,
                iova,
                AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ,
            );
            // ibv_reg_dmabuf_mr takes its own reference on success; the fd is
            // ours to close either way.
            unsafe { libc::close(fd) };
            match mr {
                Ok(mr) => out.slots.push(GpuSlot {
                    mr: Some(mr),
                    buffer,
                }),
                Err(e) => {
                    destroy_buffer(&core, buffer);
                    return Err(e)
                        .with_context(|| format!("ibv_reg_dmabuf_mr for coded slot {i}"));
                }
            }
        }
        Ok(out)
    }

    pub fn buffers(&self) -> Vec<vk::Buffer> {
        self.slots.iter().map(|s| s.buffer.buffer).collect()
    }

    /// The local MR slice covering the first `len` coded bytes of `slot`.
    pub fn source(&self, slot: usize, len: usize) -> Option<LocalMemorySlice> {
        self.slots
            .get(slot)
            .and_then(|s| s.mr.as_ref())
            .map(|mr| mr.slice(..len))
    }
}

impl Drop for GpuDirect {
    fn drop(&mut self) {
        unsafe {
            // no GPU work may still reference the buffers; sends are complete
            // before the session is torn down
            let _ = self.core.device.device_wait_idle();
            // deregister all MRs before freeing the exported memory
            for s in &mut self.slots {
                let _ = s.mr.take();
            }
            for s in self.slots.drain(..) {
                destroy_buffer(&self.core, s.buffer);
            }
        }
    }
}

fn destroy_buffer(core: &VkCore, buffer: GpuBuffer) {
    unsafe {
        core.device.destroy_buffer(buffer.buffer, None);
        core.device.free_memory(buffer.memory, None);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RawKey {
    buf_id: u64,
    fourcc: u32,
    modifier: u64,
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
}

/// A small cache of registered raw scanout dma-bufs. The MR keeps the
/// compositor's buffer alive; entries are dropped when the cache grows.
pub struct RawDirect<'a> {
    pd: &'a ProtectionDomain,
    cache: VecDeque<(RawKey, MemoryRegion<()>)>,
    /// Maximum source registrations to keep. This cache holds compositor
    /// scanout buffers, not staging slots; the server sizes the cap as
    /// `--frame-slots + 2` to cover a typical swapchain without pinning many
    /// full-resolution source buffers.
    max_sources: usize,
    /// keys whose registration already failed; do not retry every frame
    failed: HashSet<RawKey>,
    next_iova: u64,
    warned: bool,
}

impl<'a> RawDirect<'a> {
    pub fn new(pd: &'a ProtectionDomain, max_sources: usize) -> Self {
        Self {
            pd,
            cache: VecDeque::new(),
            max_sources: max_sources.max(1),
            failed: HashSet::new(),
            next_iova: 0x8_0000_0000,
            warned: false,
        }
    }

    /// The source MR slice for this scanout, or None when the buffer is not
    /// directly postable (tiled, non-XRGB, bottom-up, unstable identity) or
    /// registration fails. The caller then uses the host staging copy.
    pub fn source(
        &mut self,
        d: &DmabufScanout,
        fmt: &FrameFormat,
        slot_len: usize,
    ) -> Option<LocalMemorySlice> {
        let len = raw_len(d, fmt, slot_len)?;
        let offset = d.offsets.first().copied().unwrap_or(0);
        let key = RawKey {
            buf_id: d.buf_id,
            fourcc: fmt.fourcc,
            modifier: fmt.modifier,
            width: fmt.width,
            height: fmt.height,
            stride: fmt.stride,
            offset,
        };
        if let Some((_, mr)) = self.cache.iter().find(|(k, _)| *k == key) {
            return Some(mr.slice(..len));
        }
        if self.failed.contains(&key) {
            return None;
        }
        // iova must share the dma-buf offset's page offset.
        let iova = self.next_iova | (offset as u64 & 0xfff);
        self.next_iova = self.next_iova.wrapping_add(0x4000_0000);
        let fd = d.fds[0].as_raw_fd();
        match self.pd.register_dmabuf(
            fd,
            offset as u64,
            len,
            iova,
            AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ,
        ) {
            Ok(mr) => {
                let slice = mr.slice(..len);
                self.failed.remove(&key);
                self.cache.push_back((key, mr));
                if self.cache.len() > self.max_sources {
                    self.cache.pop_front();
                }
                Some(slice)
            }
            Err(e) => {
                self.failed.insert(key);
                if !self.warned {
                    log::warn!(
                        "[rdma] raw scanout dma-buf registration failed ({:#x} modifier {:#x}, {}x{}): {e:#}; using the Vulkan copy",
                        fmt.fourcc,
                        fmt.modifier,
                        fmt.width,
                        fmt.height
                    );
                    self.warned = true;
                }
                None
            }
        }
    }
}

/// The scanout's wire length when its dma-buf bytes can be posted as raw
/// XRGB8888: linear, one plane, top-first, exact stride, stable buffer id.
fn raw_len(d: &DmabufScanout, fmt: &FrameFormat, slot_len: usize) -> Option<usize> {
    if d.buf_id == 0
        || d.fds.len() != 1
        || fmt.num_planes != 1
        || fmt.modifier != 0
        || fmt.fourcc != DRM_FORMAT_XRGB8888
        || !fmt.y0_top
        || d.strides.first().copied() != Some(fmt.stride)
    {
        return None;
    }
    let len = fmt.frame_bytes() as usize;
    (len > 0 && len <= slot_len).then_some(len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::OwnedFd;

    fn scanout(fourcc: u32, modifier: u64, y0_top: bool) -> DmabufScanout {
        let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        DmabufScanout {
            fb_id: 1,
            buf_id: 42,
            fds: vec![fd],
            offsets: vec![0],
            strides: vec![256 * 4],
            format: FrameFormat {
                width: 256,
                height: 128,
                stride: 256 * 4,
                fourcc,
                modifier,
                y0_top,
                num_planes: 1,
            },
        }
    }

    #[test]
    fn raw_direct_requires_linear_top_first_xrgb() {
        let slot = 256 * 4 * 128;
        assert_eq!(
            raw_len(
                &scanout(DRM_FORMAT_XRGB8888, 0, true),
                &scanout(DRM_FORMAT_XRGB8888, 0, true).format,
                slot
            ),
            Some(slot)
        );
        for d in [
            scanout(0x3432_5241, 0, true), // AR24
            scanout(DRM_FORMAT_XRGB8888, 1, true), // tiled
            scanout(DRM_FORMAT_XRGB8888, 0, false), // bottom-up
        ] {
            assert_eq!(raw_len(&d, &d.format, slot), None);
        }
    }
}
