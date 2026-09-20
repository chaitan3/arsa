//! Capture backends for rdma-display.
//!
//! A `CaptureSource` produces two kinds of events:
//!   * `Scanout` — a new scanout appeared (resolution/format changed). The
//!     format is either backed by a dmabuf (`DmabufScanout`, planes passed
//!     with fds) or by host memory (QEMU "plain" scanout, bytes included).
//!   * `Frame` — an update is ready for materialization: either plain bytes
//!     (partial-rect updates patched into a full-frame buffer) or a dmabuf
//!     the materializer should consume.
//!
//! A `FrameMaterializer` turns the source's frame representation into linear
//! bytes the RDMA sender stages into its registered slots. Implemented:
//! `plain` (D-Bus bytes, no GPU), `egl` (GPU blit + readback, default).

pub mod egl;
pub mod materializer;
pub mod qemu;

pub use materializer::FrameMaterializer;
pub use egl::{EglMaterializer, ReadBenchOpts, run_readback_bench};
pub use qemu::QemuDbusSource;

use anyhow::Result;

/// Format of one scanout (dmabuf-backed or plain).
#[derive(Debug, Clone)]
pub struct FrameFormat {
    pub width: u32,
    pub height: u32,
    /// bytes per row (plane 0)
    pub stride: u32,
    /// DRM fourcc (e.g. XRGB8888)
    pub fourcc: u32,
    /// DRM modifier (0 = linear)
    pub modifier: u64,
    /// first scanline at top (false = OpenGL orientation)
    pub y0_top: bool,
    pub num_planes: u32,
}

impl FrameFormat {
    pub fn frame_bytes(&self) -> u32 {
        self.stride * self.height
    }
}

/// A dmabuf scanout with owned (duplicated) plane fds.
#[derive(Debug)]
pub struct DmabufScanout {
    pub fds: Vec<std::os::fd::OwnedFd>,
    pub offsets: Vec<u32>,
    pub strides: Vec<u32>,
    pub format: FrameFormat,
}

impl Drop for DmabufScanout {
    fn drop(&mut self) {
        // OwnedFd drops close the fds; nothing else needed
    }
}

/// One dirty rectangle of the current scanout.
#[derive(Debug, Clone, Copy)]
pub struct DirtyRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Debug)]
pub enum CaptureEvent {
    /// New dmabuf-backed scanout (fds owned by the event).
    Scanout(DmabufScanout),
    /// New plain (host-memory) scanout: format + full-frame bytes.
    /// `t0_unix_us` = when the D-Bus event arrived (latency t0 stamp).
    PlainScanout(FrameFormat, Vec<u8>, u64),
    /// Partial update of the plain scanout (rect + row data).
    PlainUpdate(
        DirtyRect,
        Vec<u8>,
        u32, /* stride */
        u32, /* pixman format */
        u64, /* t0_unix_us */
    ),
    /// A dmabuf frame is ready to be consumed by the materializer.
    /// `t0_unix_us` = when the D-Bus event arrived (latency t0 stamp).
    DmabufFrame(DirtyRect, u64),
    /// The console has no scanout at the moment.
    Disabled,
}

/// A running capture source delivers events until the receiver drops.
/// `input_rx` carries user-input events (proto v2) for injection into the
/// source's console (QEMU D-Bus Keyboard/Mouse interfaces).
pub trait CaptureSource: Send {
    fn name(&self) -> &'static str;
    fn spawn(
        self: Box<Self>,
        tx: std::sync::mpsc::Sender<CaptureEvent>,
        input_rx: std::sync::mpsc::Receiver<rdma_display_proto::InputEvent>,
    ) -> Result<()>;
}
