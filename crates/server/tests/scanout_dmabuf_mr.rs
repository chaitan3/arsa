//! Probe whether the **live KMS scanout** dma-buf can be registered with
//! `ibv_reg_dmabuf_mr` on this server.
//!
//! This is the original server zero-copy experiment: if the compositor's
//! scanout export registers, the NIC can DMA-read the scanout directly (raw
//! transport). On the tested NVIDIA stack this previously failed with `ENOMEM`
//! because `nvidia-drm`'s static PRIME path refuses peer-DMA mappings of VRAM.
//! Re-run it after driver/hardware changes.
//!
//! Needs `CAP_SYS_ADMIN` (GETFB2) and an active display, so it is ignored by
//! default:
//!
//! ```text
//! sudo -E cargo test -p rdma-display-server --test scanout_dmabuf_mr \
//!     -- --ignored --nocapture
//! ```
//!
//! A failed registration is the result under test (the test fails). Missing
//! capture permission or no active scanout is reported as a skip.

use std::os::fd::AsRawFd;
use std::time::Duration;

use ibverbs::AccessFlags;
use rdma_display_capture::{CaptureEvent, CaptureSource, DrmKmsSource};

#[test]
#[ignore = "requires CAP_SYS_ADMIN and an active KMS scanout; run as root"]
fn scanout_dmabuf_registers_with_ibv_reg_dmabuf_mr() {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rdma_display_capture=info"),
    )
    .try_init();

    let (tx, rx) = std::sync::mpsc::channel();
    let src: Box<dyn CaptureSource> = Box::new(DrmKmsSource::new(None, None));
    let running = match src.spawn(tx) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skip: cannot start KMS capture ({e:#}); run as root / CAP_SYS_ADMIN");
            return;
        }
    };

    let d = loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(CaptureEvent::Scanout(d, _)) => break d,
            Ok(_) => continue,
            Err(e) => {
                eprintln!(
                    "skip: no dmabuf scanout within 5 s ({e}); \
                     run as root/CAP_SYS_ADMIN and make sure a display is active"
                );
                drop(running);
                return;
            }
        }
    };

    let fmt = &d.format;
    eprintln!(
        "scanout: {}x{} stride={} fourcc={:#010x} modifier={:#x} planes={} offsets={:?} strides={:?}",
        fmt.width,
        fmt.height,
        fmt.stride,
        fmt.fourcc,
        fmt.modifier,
        d.fds.len(),
        d.offsets,
        d.strides
    );
    if fmt.modifier == 0 {
        eprintln!("note: linear scanout (modifier 0)");
    } else {
        eprintln!(
            "note: tiled scanout (modifier {:#x}); a direct raw send would need a \
             modifier-aware client, so only registration is probed here",
            fmt.modifier
        );
    }
    if d.fds.is_empty() {
        eprintln!("skip: scanout has no dma-buf fds");
        drop(running);
        return;
    }

    let ctx = match rdma_display_rdma::open_device(None) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no RDMA device ({e:#})");
            drop(running);
            return;
        }
    };
    let dev = ibverbs::devices()
        .ok()
        .and_then(|list| {
            list.iter()
                .next()
                .and_then(|d| d.name().map(|n| n.to_string_lossy().into_owned()))
        })
        .unwrap_or_default();
    let pd = ctx.alloc_pd().expect("alloc_pd");
    eprintln!("rdma device: {dev}");

    for (i, fd) in d.fds.iter().enumerate() {
        let offset = u64::from(d.offsets.get(i).copied().unwrap_or(0));
        let stride = d.strides.get(i).copied().unwrap_or(fmt.stride);
        // Plane 0 covers the full frame. For a tiled modifier this is a
        // subrange of the GEM object, which is enough to test registration.
        let len = stride as usize * fmt.height as usize;
        // iova must share `offset`'s page offset; any page-aligned base works.
        let iova = 0x1_0000_0000u64 + offset + i as u64 * 0x1000_0000;
        let flags = AccessFlags::LOCAL_WRITE | AccessFlags::REMOTE_READ;
        match pd.register_dmabuf(fd.as_raw_fd(), offset, len, iova, flags) {
            Ok(mr) => eprintln!(
                "OK: plane {i} registered (offset={offset} len={len} addr={:#x} rkey={:#x})",
                mr.remote().addr,
                mr.remote().rkey
            ),
            Err(e) => panic!(
                "ibv_reg_dmabuf_mr FAILED for scanout plane {i} \
                 ({:#010x} modifier {:#x}): {e:#}",
                fmt.fourcc, fmt.modifier
            ),
        }
    }

    drop(running);
}
