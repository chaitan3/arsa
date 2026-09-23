//! Frame materializers: turn capture events into linear bytes for the RDMA
//! staging slots.

use anyhow::{Context, Result};

use crate::{CaptureEvent, DirtyRect, DmabufScanout, FrameFormat};

pub trait FrameMaterializer: Send {
    fn name(&self) -> &'static str;

    /// A new scanout appeared. `dmabuf` is present iff the scanout is
    /// dmabuf-backed; the fds stay owned by the caller.
    fn prepare(&mut self, fmt: &FrameFormat, dmabuf: Option<&DmabufScanout>) -> Result<()>;

    /// Materialize the current frame into `out` (linear, frame_bytes long).
    /// Returns `(changed, times)`: `changed` = frame content differs (worth
    /// sending); `times` = GPU stage durations of this materialization in µs
    /// (fence-wait for the draw, readback, residual finish), server-local.
    fn copy_frame(&mut self, out: &mut [u8]) -> Result<(bool, rdma_display_proto::MatTimes)>;

    /// Optional: bytes to post for the frame just produced (the lossy
    /// materializer writes the coded bitstream itself and reports its length
    /// here). None = the caller owns the encoding (raw/lossless).
    fn coded_len(&self) -> Option<usize> {
        None
    }

    /// Feed an event to the materializer before the next copy_frame. Returns
    /// true if the frame content actually changed (worth sending).
    fn handle_event(&mut self, ev: CaptureEvent) -> Result<bool>;

    /// Optional: format the materializer prefers to deliver (e.g. the plain
    /// materializer cannot change resolution). None = "as the scanout says".
    fn effective_format(&self) -> Option<FrameFormat> {
        None
    }

    /// Force the next copy_frame to produce the current content even if no
    /// new capture event arrived — used when a client connects mid-session
    /// (the internal dirty flag was consumed by the previous session's last
    /// send; the new client needs the current frame immediately).
    fn mark_dirty(&mut self);

    /// Bind zero-copy staging: the materializer allocates `n` staging units
    /// of `frame_bytes` each and returns their (pointer, len) so the server
    /// can register them as RDMA staging slots (`register_from_raw`) — the
    /// GPU readback DMA and the NIC read the same pages. None = no
    /// externally-stageable memory (server falls back to verbs-allocated
    /// slots + direct copy_frame output).
    fn bind_staging(&mut self, _n: usize, _frame_bytes: usize) -> Option<Vec<(*mut u8, usize)>> {
        None
    }

    /// Pipelined phase 1: enqueue the readback into staging slot `slot`
    /// (`out` is the slot's registered memory). Must return promptly — the
    /// GPU work proceeds asynchronously. Returns true when this frame is
    /// real (dirty); false = nothing to send. The slot is then completed by
    /// [`finish_frame`](Self::finish_frame).
    /// Default: the synchronous copy_frame (no pipelining support).
    fn issue_frame(&mut self, _slot: usize, out: &mut [u8]) -> Result<bool> {
        Ok(self.copy_frame(out)?.0)
    }

    /// Pipelined phase 2: complete the slot `issue_frame` started (fence
    /// wait) and return (changed, gpu stage times).
    /// Default: passthrough (the default issue_frame did all the work).
    fn finish_frame(&mut self, _slot: usize, _out: &mut [u8]) -> Result<(bool, rdma_display_proto::MatTimes)> {
        Ok((true, rdma_display_proto::MatTimes::ZERO))
    }

    /// Drop any pipelined state (fences) after a session loss; the next
    /// issue_frame reuses the slots.
    fn drop_frames(&mut self) {}
}

/// The "plain" materializer: QEMU sends host-memory scanouts (requires
/// `gl=off` on the display). Full frames arrive via D-Bus; partial updates
/// patch the rect into our shadow buffer.
pub struct PlainMaterializer {
    frame: Vec<u8>,
    fmt: Option<FrameFormat>,
    dirty: bool,
}

impl PlainMaterializer {
    pub fn new() -> Self {
        Self { frame: Vec::new(), fmt: None, dirty: false }
    }
}

impl FrameMaterializer for PlainMaterializer {
    fn name(&self) -> &'static str {
        "plain"
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn prepare(&mut self, fmt: &FrameFormat, _dmabuf: Option<&DmabufScanout>) -> Result<()> {
        self.frame = vec![0; fmt.frame_bytes() as usize];
        self.fmt = Some(fmt.clone());
        self.dirty = true;
        Ok(())
    }

    fn copy_frame(&mut self, out: &mut [u8]) -> Result<(bool, rdma_display_proto::MatTimes)> {
        let was_dirty = self.dirty;
        self.dirty = false;
        if was_dirty {
            let n = self.frame.len().min(out.len());
            out[..n].copy_from_slice(&self.frame[..n]);
            Ok((true, rdma_display_proto::MatTimes::ZERO))
        } else {
            Ok((false, rdma_display_proto::MatTimes::ZERO))
        }
    }

    fn handle_event(&mut self, ev: CaptureEvent) -> Result<bool> {
        match ev {
            CaptureEvent::PlainUpdate(rect, data, stride, format, _t0) => {
                self.patch_rect(rect, &data, stride, format)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn effective_format(&self) -> Option<FrameFormat> {
        self.fmt.clone()
    }
}

impl PlainMaterializer {
    /// Patch a partial update into the shadow buffer. Only 4-byte-per-pixel
    /// formats are supported (XRGB8888 family); anything else arrives as a
    /// new full `Scanout` anyway.
    fn patch_rect(&mut self, r: DirtyRect, data: &[u8], stride: u32, _format: u32) -> Result<()> {
        let fmt = self.fmt.as_ref().context("patch before scanout")?;
        let frame_stride = fmt.stride as usize;
        let height = fmt.height as usize;
        let width = fmt.width as usize;
        let bpp = 4usize;
        for row in 0..r.h.max(0) as usize {
            let dst_y = r.y.max(0) as usize + row;
            if dst_y >= height {
                break;
            }
            let x0 = r.x.max(0) as usize;
            let x1 = (x0 + r.w.max(0) as usize).min(width);
            if x0 >= width {
                break;
            }
            let src = row * stride as usize;
            let dst = dst_y * frame_stride + x0 * bpp;
            let n = ((x1 - x0) * bpp)
                .min(data.len().saturating_sub(src))
                .min(self.frame.len() - dst);
            self.frame[dst..dst + n].copy_from_slice(&data[src..src + n]);
        }
        self.dirty = true;
        Ok(())
    }
}
