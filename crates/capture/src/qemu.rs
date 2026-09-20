//! QEMU `-display dbus` capture source.
//!
//! Ported from the M0 spike: connects to QEMU's display D-Bus (session bus or
//! a libvirt-managed private bus given by socket path), registers a console
//! listener, and forwards `Scanout`/`ScanoutDMABUF`/`Update` events.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::mpsc::Sender;

use anyhow::{Context, Result};

use rdma_display_proto::InputEvent;

use crate::{CaptureEvent, CaptureSource, DmabufScanout, DirtyRect, FrameFormat};

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
            fds,
            offsets: s.offset[..format.num_planes as usize].to_vec(),
            strides: s.stride[..format.num_planes as usize].to_vec(),
            format,
        };
        let _ = self.tx.send(CaptureEvent::Scanout(dmabuf));
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

    async fn run(
        self,
        tx: Sender<CaptureEvent>,
        input_rx: std::sync::mpsc::Receiver<InputEvent>,
    ) -> Result<()> {
        let console = if let Some(path) = &self.dbus_socket {
            eprintln!("[capture] connecting to dbus at {path}");
            let stream = std::os::unix::net::UnixStream::connect(path)
                .with_context(|| format!("connect {path}"))?;
            let conn = zbus::connection::Builder::async_io_unix_stream(stream)
                .build()
                .await
                .context("dbus connect")?;
            let _display =
                qemu_display::Display::new::<&str>(&conn, None).await.context("no org.qemu")?;
            qemu_display::Console::new(&conn, self.console)
                .await
                .context("console proxy")?
        } else {
            eprintln!("[capture] connecting to session bus");
            let conn = zbus::Connection::session().await.context("session bus")?;
            let _display =
                qemu_display::Display::new::<&str>(&conn, None).await.context("no org.qemu")?;
            qemu_display::Console::new(&conn, self.console)
                .await
                .context("console proxy")?
        };

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

        // Input injection: user events from the control channel go out as
        // D-Bus method calls on the console's Keyboard/Mouse interfaces.
        // Plain thread + async_std::block_on per call: each event is one
        // small method invocation, no runtime needed beyond this.
        let kb = console.keyboard.clone();
        let mo = console.mouse.clone();
        // probe the guest's pointing-device mode once (IsAbsolute property)
        if let Ok(abs) = async_std::task::block_on(mo.is_absolute()) {
            eprintln!("[input] guest mouse IsAbsolute={abs}");
        }
        std::thread::Builder::new()
            .name("capture-input".into())
            .spawn(move || {
                while let Ok(ev) = input_rx.recv() {
                    let r = async_std::task::block_on(async {
                        match ev {
                            InputEvent::Key { code, down } => {
                                if down {
                                    kb.press(code).await
                                } else {
                                    kb.release(code).await
                                }
                            }
                            InputEvent::Btn { button, down } => {
                                // D-Bus button values: Left=0 Middle=1 Right=2
                                // WheelUp=3 WheelDown=4 Side=5 Extra=6
                                let btn = match button {
                                    0 => qemu_display::MouseButton::Left,
                                    1 => qemu_display::MouseButton::Middle,
                                    2 => qemu_display::MouseButton::Right,
                                    3 => qemu_display::MouseButton::WheelUp,
                                    4 => qemu_display::MouseButton::WheelDown,
                                    5 => qemu_display::MouseButton::Side,
                                    6 => qemu_display::MouseButton::Extra,
                                    other => {
                                        eprintln!("[input] unknown button {other}");
                                        return Ok(());
                                    }
                                };
                                if down {
                                    mo.press(btn).await
                                } else {
                                    mo.release(btn).await
                                }
                            }
                            InputEvent::Abs { x, y } => mo.set_abs_position(x, y).await,
                            InputEvent::Rel { dx, dy } => mo.rel_motion(dx, dy).await,
                        }
                    });
                    if let Err(e) = r {
                        eprintln!("[input] D-Bus call failed for {ev:?}: {e}");
                    }
                }
                eprintln!("[input] control channel closed; input forwarding stopped");
            })
            .context("spawn input forwarder")?;

        // keep the console (and its D-Bus connection) alive forever —
        // dropping it tears down the listener
        std::future::pending::<()>().await;
        unreachable!()
    }
}

impl CaptureSource for QemuDbusSource {
    fn name(&self) -> &'static str {
        "qemu-dbus"
    }

    fn spawn(
        self: Box<Self>,
        tx: Sender<CaptureEvent>,
        input_rx: std::sync::mpsc::Receiver<InputEvent>,
    ) -> Result<()> {
        std::thread::Builder::new()
            .name("capture-qemu".into())
            .spawn(move || {
                if let Err(e) = async_std::task::block_on(self.run(tx, input_rx)) {
                    eprintln!("[capture] source failed: {e:#}");
                }
            })
            .context("spawn capture thread")?;
        Ok(())
    }
}
