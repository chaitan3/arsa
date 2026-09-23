//! readbench: compare ways to move the QEMU scanout dmabuf into host memory.
//!
//! Variants (all sharing one EGL context on the rendernode):
//!   v0  oes-blit-direct — production path via EglMaterializer itself
//!   A1  tex2d-fbo       — dmabuf as renderable TEXTURE_2D, FBO read, no blit
//!   A2  renderbuf       — dmabuf as renderbuffer, FBO read
//!   B   gettexsub       — glGetTextureSubImage from the 2D import (no FBO)
//!   C   pbo-fence       — async PIXEL_PACK + glFenceSync (serial + pipelined)
//!
//! Every variant ends with a full sync, so numbers are "CPU issue → bytes in
//! host memory". Data is checksum-validated against the baseline (direct or
//! v-flipped order).
//!
//! Needs a running QEMU with dmabuf (dbus-display) scanouts on the rendernode.
//! Example:
//!   sudo target/release/readbench --console 0 \
//!       [--dbus-socket /path] [--rendernode /dev/dri/renderD128]

use std::sync::mpsc;

use anyhow::{bail, Context, Result};

use rdma_display_capture::{run_readback_bench, CaptureEvent, QemuDbusSource, ReadBenchOpts};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut iterations = 50usize;
    let mut warmup = 5usize;
    let mut rendernode = None;
    let mut dbus_socket = None;
    let mut console = 0u32;
    let mut load = false;
    let mut events = 0usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--iterations" => iterations = args.next().context("--iterations needs a value")?.parse()?,
            "--warmup" => warmup = args.next().context("--warmup needs a value")?.parse()?,
            "--rendernode" => rendernode = Some(args.next().context("--rendernode needs a value")?),
            "--dbus-socket" => dbus_socket = Some(args.next().context("--dbus-socket needs a value")?),
            "--console" => console = args.next().context("--console needs a value")?.parse()?,
            "--load" => load = true,
            "--events" => events = args.next().context("--events needs a value")?.parse()?,
            "--help" | "-h" => {
                println!(
                    "usage: readbench [--iterations N] [--warmup N] [--rendernode PATH] \
                     [--dbus-socket PATH] [--console N] [--load] [--events N]"
                );
                return Ok(());
            }
            other => bail!("unknown arg {other} (see --help)"),
        }
    }
    let opts = ReadBenchOpts {
        iterations,
        warmup,
        rendernode: rendernode.clone(),
        load,
        events,
    };

    let (tx, rx) = mpsc::channel();
    // --load drives guest repaints through the QEMU D-Bus input path; spawn
    // the (split-off) input sink for it
    let (input_tx, input_rx) = mpsc::channel::<rdma_display_proto::InputEvent>();
    let _input_keepalive = input_tx.clone();
    if load {
        rdma_display_capture::InputSink::spawn(
            Box::new(rdma_display_capture::QemuDbusInput::new(dbus_socket.clone(), console)),
            input_rx,
            std::sync::Arc::new(rdma_display_capture::ScreenSize::default()),
            rdma_display_proto::gamepad::RumbleHub::new(),
        )
        .context("spawn input sink")?;
    }

    rdma_display_capture::CaptureSource::spawn(
        Box::new(QemuDbusSource::new(dbus_socket, console)),
        tx,
    )
    .context("spawn capture source")?;

    println!("waiting for a dmabuf scanout from QEMU (move the mouse in the guest to force one)…");
    let scanout = loop {
        match rx.recv().context("capture source ended")? {
            CaptureEvent::Scanout(d, _) => break d,
            CaptureEvent::PlainScanout(..) | CaptureEvent::PlainUpdate(..) => {
                bail!("guest produces plain (host-memory) scanouts; the readback bench needs dmabuf scanouts (virtio-gpu with dbus-display)")
            }
            _ => {}
        }
    };
    println!(
        "got scanout {}x{}{}",
        scanout.format.width,
        scanout.format.height,
        if opts.events > 0 {
            " — event-driven bench consumes the capture stream"
        } else {
            " — remaining events drained in the background"
        }
    );

    run_readback_bench(&scanout, &opts, Some(input_tx.clone()), rx)?;

    Ok(())
}
