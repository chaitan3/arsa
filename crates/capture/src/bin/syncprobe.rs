//! Self-driving probe for whether the scanout dma-buf exposes a usable
//! producer fence via `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`.
//!
//! It creates a synthetic uinput mouse (so no physical input device is
//! needed), watches the scanout for `--idle` seconds, then moves the mouse in
//! a bounded circle for `--seconds` seconds while sampling the dma-buf
//! reservation fence on every new fb. It prints a verdict:
//!
//! * a **real** reservation fence (driver name != the kernel's `"stub"`)
//!   means the DRM capture can wait on it and read the CURRENT fb — no tearing
//!   and no one-frame lag, so the one-behind workaround can be removed;
//! * a **stub** fence means there is nothing to wait on and we need the
//!   compositor capture protocol instead.
//!
//! Usage (run as root, on the compositor's host):
//!   sudo syncprobe [/dev/dri/cardN] [--seconds N] [--idle N] [--no-input]

fn main() -> anyhow::Result<()> {
    let mut card: Option<String> = None;
    let mut active = 15u64;
    let mut idle = 5u64;
    let mut inject = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => active = args.next().expect("--seconds needs N").parse()?,
            "--idle" => idle = args.next().expect("--idle needs N").parse()?,
            "--no-input" => inject = false,
            "--help" | "-h" => {
                println!("usage: syncprobe [/dev/dri/cardN] [--seconds N] [--idle N] [--no-input]");
                return Ok(());
            }
            other => card = Some(other.to_string()),
        }
    }
    rdma_display_capture::drm::sync_probe(card.as_deref(), idle, active, inject)
}
