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
#[cfg(feature = "portal")]
pub mod portal;
pub mod qemu;

pub use materializer::FrameMaterializer;
pub use rdma_display_lifecycle::{sleep_or_stop, Running, Stop};
pub use egl::{run_readback_bench, EglInitPhases, EglMaterializer, ReadBenchOpts};
pub use input::{InputSink, QemuDbusInput, UinputSink};
pub use drm::DrmKmsSource;
#[cfg(feature = "portal")]
pub use portal::PortalScreenCastSource;
pub use qemu::QemuDbusSource;

use anyhow::Result;

/// Default settle window for [`CaptureSync::Settle`] (milliseconds).
///
/// Measured on this stack: the compositor's per-frame completion fence signals
/// ~0.75 ms (p50), ~1-2 ms (p90), ~8-16 ms (p99) after the commit is observed.
pub const DEFAULT_SETTLE_MS: u64 = 2;

/// How the DRM/KMS source chooses when to capture, relative to the compositor's
/// flips. The DRM path always polls `plane.fb_id` at 1 ms and emits only when it
/// changes; these modes add a wait before the read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSync {
    /// Use the compositor's per-frame completion fence when it is ready, and
    /// fall back to [`CaptureSync::Settle`]`(DEFAULT_SETTLE_MS)` otherwise. The
    /// default.
    Auto,
    /// Capture the current fb `N` ms after its commit was observed. The
    /// compositor commits with a non-blocking atomic commit and finishes the
    /// frame's GPU work afterwards; on this stack the per-frame completion fence
    /// (NVIDIA semsurf) signals ~0.75 ms (p50) after handover, so a small settle
    /// usually captures a complete frame. Frames whose render outlasts the
    /// window are read mid-render and can tear — a latency/tearing tradeoff,
    /// not a guarantee.
    Settle(u64),
    /// Read the current fb when the compositor's per-frame completion fence
    /// signals, observed through the kernel `dma_fence:dma_fence_signaled`
    /// tracepoint (via tracefs; the same tracepoint an eBPF program would use).
    /// Tear-free at ~1 ms latency on this stack, at the cost of requiring root
    /// and an actively flipping compositor during a short calibration; falls
    /// back to [`CaptureSync::Settle`]`(DEFAULT_SETTLE_MS)` until the watch is
    /// ready. See `capture::fence`.
    Fence,
}

impl CaptureSync {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
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
mod lifecycle_tests {
    use super::*;
    use std::sync::mpsc;

    /// A source that just spins until stopped — proves `Running::stop` joins.
    struct Spinner;
    impl CaptureSource for Spinner {
        fn name(&self) -> &'static str {
            "spinner"
        }
        fn spawn(self: Box<Self>, _tx: mpsc::Sender<CaptureEvent>) -> Result<Running> {
            let stop = Stop::new();
            let worker = stop.clone();
            let join = std::thread::spawn(move || {
                while !worker.stopped() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            });
            Ok(Running::new(stop, vec![join]))
        }
    }

    #[test]
    fn running_stop_joins_the_worker() {
        let (tx, _rx) = mpsc::channel();
        let mut running = Box::new(Spinner).spawn(tx).unwrap();
        assert!(!running.is_finished());
        running.stop();
        assert!(running.is_finished());
    }

    #[test]
    fn sleep_or_stop_wakes_early() {
        let stop = Stop::new();
        stop.stop();
        let t = std::time::Instant::now();
        assert!(sleep_or_stop(&stop, std::time::Duration::from_secs(10)));
        assert!(t.elapsed() < std::time::Duration::from_secs(1));
    }
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
    /// DRM framebuffer id that was current when this was grabbed. **Logging
    /// only** — the kernel recycles fb ids (lowest free id), so an id does not
    /// identify the underlying buffer across its lifetime and must not be used
    /// to cache imports.
    pub fb_id: u32,
    /// Identity of the underlying buffer: `(st_dev, st_ino)` of the dma-buf fd.
    /// The kernel keeps exactly one dma-buf per GEM object and gives each a
    /// unique, lifetime-stable inode, so this stays constant while the buffer
    /// lives and changes when the compositor frees it and a new buffer reuses
    /// the fb id. **Cache imports on this, not `fb_id`.** 0 = no stable
    /// identity (e.g. QEMU D-Bus scanouts); callers must not cache those.
    pub buf_id: u64,
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
    /// Start the source. The returned [`Running`] stops and joins the worker
    /// (and stops it on drop), so a source never outlives its handle.
    fn spawn(self: Box<Self>, tx: std::sync::mpsc::Sender<CaptureEvent>) -> Result<Running>;
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
