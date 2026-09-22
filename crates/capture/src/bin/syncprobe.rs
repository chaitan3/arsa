//! Probe whether the scanout dma-buf exposes a usable producer fence via
//! `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`.
//!
//! If the exported sync_file is *pending* at the moment a new fb is committed,
//! the DRM capture backend can `poll` it before reading and therefore read the
//! **current** fb with no tearing and no one-frame lag (i.e. the one-behind
//! workaround can be dropped). If it is always already signaled, the fence is
//! a no-op on this stack and the current approach stands.
//!
//! Usage: `sudo syncprobe [/dev/dri/cardN]`, then move the mouse / type and
//! watch the per-commit lines and the periodic summary. Read-only.

fn main() -> anyhow::Result<()> {
    let card = std::env::args().nth(1);
    rdma_display_capture::drm::sync_probe(card.as_deref())
}
