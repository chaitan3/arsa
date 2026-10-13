//! QEMU `-display dbus` capture source.
//!
//! Ported from the M0 spike: connects to QEMU's display D-Bus (session bus or
//! a libvirt-managed private bus given by socket path), registers a console
//! listener, and forwards `Scanout`/`ScanoutDMABUF`/`Update` events.
//! Input injection lives in `input::QemuDbusInput` (same D-Bus console).

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::mpsc::Sender;

use anyhow::{bail, Context, Result};

use crate::{CaptureEvent, CaptureSource, DmabufScanout, DirtyRect, FrameFormat, Running, Stop};

struct Listener {
    tx: Sender<CaptureEvent>,
}

#[async_trait::async_trait]
impl qemu_display::ConsoleListenerHandler for Listener {
    async fn scanout(&mut self, s: qemu_display::Scanout) {
        // plain scanouts are top-first (pixman) host memory; treat any 32bpp
        // pixman format as XRGB8888 byte layout (true for x8r8g8b8/a8r8g8b8)
        let fmt = FrameFormat {
            width: s.width,
            height: s.height,
            stride: s.stride,
            fourcc: 0x3432_5258, // XR24
            modifier: 0,
            y0_top: true,
            num_planes: 1,
        };
        eprintln!(
            "[capture] plain Scanout {}x{} stride={} pixman={:#010x} bytes={} (treating as XR24)",
            s.width,
            s.height,
            s.stride,
            s.format,
            s.data.len()
        );
        let _ = self.tx.send(CaptureEvent::PlainScanout(
            fmt,
            s.data,
            rdma_display_proto::unix_now_us(),
        ));
    }

    async fn update(&mut self, u: qemu_display::Update) {
        eprintln!(
            "[capture] plain Update rect=({},{}) {}x{} stride={} format={:#010x} bytes={}",
            u.x, u.y, u.w, u.h, u.stride, u.format, u.data.len()
        );
        let _ = self.tx.send(CaptureEvent::PlainUpdate(
            DirtyRect {
                x: u.x,
                y: u.y,
                w: u.w,
                h: u.h,
            },
            u.data,
            u.stride,
            u.format,
            rdma_display_proto::unix_now_us(),
        ));
    }

    async fn scanout_dmabuf(&mut self, s: qemu_display::ScanoutDMABUF) {
        eprintln!(
            "[capture] ScanoutDMABUF {}x{} planes={} fourcc={:#010x} modifier={:#x} y0_top={}",
            s.width, s.height, s.num_planes, s.fourcc, s.modifier, s.y0_top
        );
        let n = s.num_planes as usize;
        let fds: Vec<OwnedFd> = s.fd[..n]
            .iter()
            .filter(|&&f| f >= 0)
            .map(|&f| unsafe { OwnedFd::from_raw_fd(libc::dup(f)) })
            .collect();
        let format = FrameFormat {
            width: s.width,
            height: s.height,
            stride: s.stride[0],
            fourcc: s.fourcc,
            modifier: s.modifier,
            y0_top: s.y0_top,
            num_planes: s.num_planes,
        };
        let dmabuf = DmabufScanout {
            // QEMU scanouts have no DRM fb id; 0 disables import caching (the
            // D-Bus fds are not guaranteed stable per buffer)
            fb_id: 0,
            fds,
            offsets: s.offset[..format.num_planes as usize].to_vec(),
            strides: s.stride[..format.num_planes as usize].to_vec(),
            format,
        };
        let _ = self
            .tx
            .send(CaptureEvent::Scanout(dmabuf, rdma_display_proto::unix_now_us()));
    }

    async fn update_dmabuf(&mut self, u: qemu_display::UpdateDMABUF) {
        eprintln!(
            "[capture] UpdateDMABUF rect=({},{}) {}x{}",
            u.x, u.y, u.w, u.h
        );
        let _ = self.tx.send(CaptureEvent::DmabufFrame(
            DirtyRect {
                x: u.x,
                y: u.y,
                w: u.w,
                h: u.h,
            },
            rdma_display_proto::unix_now_us(),
        ));
    }

    async fn disable(&mut self) {
        let _ = self.tx.send(CaptureEvent::Disabled);
    }

    async fn mouse_set(&mut self, _m: qemu_display::MouseSet) {}
    async fn cursor_define(&mut self, _c: qemu_display::Cursor) {}
    fn disconnected(&mut self) {
        eprintln!("[capture] disconnected from QEMU");
    }
    fn interfaces(&self) -> Vec<String> {
        vec!["org.qemu.Display1.Listener".to_string()]
    }
}

/// Open the QEMU display D-Bus and return the console proxy. Shared by the
/// capture source and the input sink (separate D-Bus connections are fine —
/// each opens its own listener/proxy).
pub async fn open_console(
    dbus_socket: &Option<String>,
    console: u32,
) -> Result<qemu_display::Console> {
    let conn: zbus::Connection = if let Some(path) = dbus_socket {
        eprintln!("[capture] connecting to dbus at {path}");
        let stream = std::os::unix::net::UnixStream::connect(path)
            .with_context(|| format!("connect {path}"))?;
        zbus::connection::Builder::async_io_unix_stream(stream)
            .build()
            .await
            .context("dbus connect")?
    } else {
        eprintln!("[capture] connecting to session bus");
        zbus::Connection::session().await.context("session bus")?
    };
    let _display =
        qemu_display::Display::new::<&str>(&conn, None).await.context("no org.qemu")?;
    // leak the connection: the console proxies borrow it; source/input threads
    // live forever anyway
    let conn: &'static zbus::Connection = Box::leak(Box::new(conn));
    qemu_display::Console::new(conn, console).await.context("console proxy")
}

/// Cheap preflight probe for the QEMU source: open the console proxy and read
/// its current size, without registering a listener or streaming. The D-Bus
/// connection is leaked (see `open_console`), so call this at most once per
/// process.
pub fn probe_console(dbus_socket: &Option<String>, console: u32) -> Result<(u32, u32)> {
    async_std::task::block_on(async {
        let c = open_console(dbus_socket, console).await?;
        let w = c.width().await.unwrap_or(0);
        let h = c.height().await.unwrap_or(0);
        if w == 0 || h == 0 {
            bail!("console reported {w}x{h}");
        }
        Ok((w, h))
    })
}

pub struct QemuDbusSource {
    /// `None` → session bus; `Some(path)` → connect as bus client to a private
    /// (libvirt) dbus-daemon listening at that socket.
    pub dbus_socket: Option<String>,
    pub console: u32,
}

impl QemuDbusSource {
    pub fn new(dbus_socket: Option<String>, console: u32) -> Self {
        Self { dbus_socket, console }
    }

    async fn run(self, tx: Sender<CaptureEvent>, stop: Stop) -> Result<()> {
        let console = open_console(&self.dbus_socket, self.console).await?;

        eprintln!(
            "[capture] console {}: {:?} ({}x{})",
            self.console,
            console.label().await.unwrap_or_default(),
            console.width().await.unwrap_or(0),
            console.height().await.unwrap_or(0),
        );
        console
            .register_listener(Listener { tx })
            .await
            .context("RegisterListener")?;
        eprintln!("[capture] listener registered; streaming events");

        // Keep the console (and its D-Bus connection) alive while the events
        // flow; poll the stop token so the source can be torn down. Dropping
        // `console` here unregisters the listener. (The D-Bus connection is
        // still leaked by `open_console`; a per-session connection leak is a
        // known follow-up.)
        while !stop.stopped() {
            async_std::task::sleep(std::time::Duration::from_millis(100)).await;
        }
        eprintln!("[capture] qemu source stopping");
        Ok(())
    }
}

impl CaptureSource for QemuDbusSource {
    fn name(&self) -> &'static str {
        "qemu-dbus"
    }

    fn spawn(self: Box<Self>, tx: Sender<CaptureEvent>) -> Result<Running> {
        let stop = Stop::new();
        let worker_stop = stop.clone();
        let join = std::thread::Builder::new()
            .name("capture-qemu".into())
            .spawn(move || {
                if let Err(e) = async_std::task::block_on(self.run(tx, worker_stop)) {
                    eprintln!("[capture] source failed: {e:#}");
                }
            })
            .context("spawn capture thread")?;
        Ok(Running::new(stop, vec![join]))
    }
}
