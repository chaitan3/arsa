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

pub mod drm;
pub mod egl;
mod fence;
pub mod input;
pub mod materializer;
pub mod qemu;

pub use materializer::FrameMaterializer;
pub use egl::{run_readback_bench, EglInitPhases, EglMaterializer, ReadBenchOpts};
pub use input::{InputSink, QemuDbusInput, UinputSink};
pub use drm::DrmKmsSource;
pub use qemu::QemuDbusSource;

use anyhow::Result;

/// Default settle window for [`CaptureSync::Settle`] (milliseconds).
///
/// Measured on this stack: the compositor's per-frame completion fence signals
/// ~0.75 ms (p50), ~1-2 ms (p90), ~8-16 ms (p99) after the commit is observed.
pub const DEFAULT_SETTLE_MS: u64 = 2;

/// How the DRM/KMS source chooses which framebuffer to capture relative to
/// the compositor's flips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSync {
    /// Capture the fb that was just superseded. Always safe — that buffer
    /// scanned out for a full interval and is complete — but the mirrored
    /// image is one compositor frame old.
    OneBehind,
    /// Capture the current fb `N` ms after its commit was observed, instead of
    /// waiting for the next commit (one-behind). The compositor commits with a
    /// non-blocking atomic commit and finishes its GPU work a little later; on
    /// this stack the per-frame completion fence (NVIDIA semsurf) signals ~1 ms
    /// after handover. Small N therefore removes most of the one-behind lag,
    /// but frames whose render outlasts N are read mid-render and can tear.
    /// This is a latency/tearing tradeoff, not a guarantee.
    Settle(u64),
    /// Capture the current fb immediately when its commit is observed — no
    /// vblank wait and no one-behind. **Testing only:** the buffer may still
    /// be mid-render or not yet latched (this stack has no producer fence), so
    /// the image can tear or be a frame early. Exists to measure the latency
    /// floor and to see how badly this stack tears, not for regular use.
    Zero,
    /// Capture the current fb as soon as the vblank that latches it has
    /// retired, so the mirrored image is the one actually on screen now.
    /// Falls back to one-behind for a frame when no vblank arrives in time
    /// (VRR idle, low refresh) and degrades permanently if vblank never
    /// advances.
    Vblank,
    /// Try [`CaptureSync::Fence`], then [`CaptureSync::Vblank`], then degrade
    /// to [`CaptureSync::OneBehind`]. Fence needs root and a flipping
    /// compositor during calibration; each step falls through on failure.
    Auto,
    /// Read the current fb when the compositor's per-frame completion fence
    /// signals, observed through the kernel `dma_fence:dma_fence_signaled`
    /// tracepoint (via tracefs; the same tracepoint an eBPF program would use).
    /// Tear-free at ~1 ms latency on this stack, at the cost of requiring root
    /// and an actively flipping compositor during a short calibration; falls
    /// back to one-behind on any failure. See `capture::fence`.
    Fence,
}

impl CaptureSync {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "onbehind" | "one-behind" | "off" => Some(Self::OneBehind),
            "zero" | "now" => Some(Self::Zero),
            "vblank" => Some(Self::Vblank),
            "auto" => Some(Self::Auto),
            // "bpf" is accepted as an alias: the mechanism is the dma_fence
            // tracepoint, whether consumed via tracefs (here) or an eBPF probe.
            "fence" | "bpf" => Some(Self::Fence),
            _ => {
                let rest = s.strip_prefix("settle")?;
                let ms = match rest.strip_prefix(':').or_else(|| rest.strip_prefix('=')) {
                    None | Some("") => DEFAULT_SETTLE_MS,
                    Some(v) => v.parse().ok()?,
                };
                Some(Self::Settle(ms))
            }
        }
    }
}

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

/// Decode a DRM fourcc into its 4-character name (e.g. 0x3432_5258 → "XR24").
/// The low byte is the first character (`fourcc_code`). Non-printable bytes
/// become `.` so log lines stay readable.
pub fn fourcc_name(fourcc: u32) -> String {
    fourcc
        .to_le_bytes()
        .iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{fourcc_name, CaptureSync, DEFAULT_SETTLE_MS};

    #[test]
    fn fourcc_names() {
        assert_eq!(fourcc_name(0x3432_5258), "XR24");
        assert_eq!(fourcc_name(0x3033_4241), "AB30"); // DRM_FORMAT_ABGR2101010
        assert_eq!(fourcc_name(0), "....");
    }

    #[test]
    fn capture_sync_parse() {
        assert_eq!(CaptureSync::parse("onbehind"), Some(CaptureSync::OneBehind));
        assert_eq!(CaptureSync::parse("off"), Some(CaptureSync::OneBehind));
        assert_eq!(CaptureSync::parse("zero"), Some(CaptureSync::Zero));
        assert_eq!(CaptureSync::parse("now"), Some(CaptureSync::Zero));
        assert_eq!(CaptureSync::parse("vblank"), Some(CaptureSync::Vblank));
        assert_eq!(CaptureSync::parse("auto"), Some(CaptureSync::Auto));
        assert_eq!(CaptureSync::parse("fence"), Some(CaptureSync::Fence));
        assert_eq!(CaptureSync::parse("bpf"), Some(CaptureSync::Fence));
        assert_eq!(CaptureSync::parse("settle"), Some(CaptureSync::Settle(DEFAULT_SETTLE_MS)));
        assert_eq!(CaptureSync::parse("settle:5"), Some(CaptureSync::Settle(5)));
        assert_eq!(CaptureSync::parse("settle=12"), Some(CaptureSync::Settle(12)));
        assert_eq!(CaptureSync::parse("settle:"), Some(CaptureSync::Settle(DEFAULT_SETTLE_MS)));
        assert_eq!(CaptureSync::parse("settle:x"), None);
        assert_eq!(CaptureSync::parse("nope"), None);
    }
}

/// A dmabuf scanout with owned (duplicated) plane fds.
#[derive(Debug)]
pub struct DmabufScanout {
    /// stable DRM framebuffer id for this buffer (the fds are re-imported each
    /// frame, so this is the identifier to cache imports on)
    pub fb_id: u32,
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
    /// `t0_unix_us` = when the frame became visible at the source: for DRM the
    /// flip that put it on screen (the one-behind read captures the previous
    /// fb, so this is one flip earlier than the poll that noticed it), for
    /// QEMU the D-Bus scanout event.
    Scanout(DmabufScanout, u64),
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
/// Input injection is decoupled from capture: see [`input::InputSink`] —
/// the server pairs a source with a sink (`--source qemu` ↔ QEMU D-Bus
/// console injection, `--source drm` ↔ uinput into the host compositor).
pub trait CaptureSource: Send {
    fn name(&self) -> &'static str;
    fn spawn(self: Box<Self>, tx: std::sync::mpsc::Sender<CaptureEvent>) -> Result<()>;
}

/// Current captured screen dimensions (for input sinks: Abs coordinates are
/// frame pixels; uinput needs them as ABS ranges, and they can change with
/// the scanout). Shared cell, updated by the server when a scanout arrives.
#[derive(Debug, Default)]
pub struct ScreenSize {
    pub w: std::sync::atomic::AtomicU32,
    pub h: std::sync::atomic::AtomicU32,
}

impl ScreenSize {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }
    pub fn set(&self, w: u32, h: u32) {
        self.w.store(w, std::sync::atomic::Ordering::Relaxed);
        self.h.store(h, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn get(&self) -> (u32, u32) {
        (
            self.w.load(std::sync::atomic::Ordering::Relaxed),
            self.h.load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}
