//! XDG Desktop Portal screen-capture source (`--source portal`).
//!
//! The compositor composites, so a captured buffer is complete when it is
//! delivered: there is **no** direct-scanout blank-frame failure mode, no
//! root/`CAP_SYS_ADMIN`, and `--capture-sync` does not apply (the `settle`/
//! `fence` machinery exists only because the KMS path reads a plane mid-flip).
//!
//! Flow: `org.freedesktop.portal.ScreenCast` (`CreateSession` → `SelectSources`
//! → `Start` → `OpenPipeWireRemote`) yields a PipeWire node id + a remote fd;
//! the PipeWire stream negotiates raw video and hands us dmabufs, which we wrap
//! as the same [`DmabufScanout`] the KMS/QEMU sources produce, so the existing
//! materializer / encoder / transport are unchanged.
//!
//! Requires a session bus + a portal backend (GNOME, KDE,
//! `xdg-desktop-portal-wlr`, `xdg-desktop-portal-hyprland`). Feature-gated
//! behind `portal` so the binary still builds and runs without PipeWire.
//!
//! **Input is not touched here.** `--source portal` pairs with `--input uinput`;
//! the portal's `RemoteDesktop`/`InputCapture` (EIS) path is deliberately not
//! used, and gamepads are never covered by the portal anyway.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::mpsc::Sender;
use std::time::Duration;

use anyhow::{anyhow, Context as _, Result};

use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::desktop::PersistMode;
use ashpd::enumflags2::BitFlags;

use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use spa::buffer::DataType;
use spa::param::format::{FormatProperties, MediaSubtype, MediaType};
use spa::param::video::{VideoFormat, VideoInfoRaw};
use spa::pod::Pod;
use spa::utils::{Choice, ChoiceEnum, ChoiceFlags};

/// `SPA_PARAM_BUFFERS_*` property keys (spa/param/buffers.h; not exposed by
/// libspa's Rust API, so they are spelled out here, keeping the C names).
#[allow(non_upper_case_globals)]
const SPA_PARAM_BUFFERS_buffers: u32 = 1;
#[allow(non_upper_case_globals)]
const SPA_PARAM_BUFFERS_blocks: u32 = 2;
#[allow(non_upper_case_globals)]
const SPA_PARAM_BUFFERS_dataType: u32 = 6;

/// `DRM_FORMAT_MOD_INVALID` (0x00ffffffffffffff): "no modifier" — how a producer
/// signals that a buffer is a MemFd rather than a dmabuf.
const DRM_FORMAT_MOD_INVALID: i64 = 0x00ff_ffff_ffff_ffff;
/// `DRM_FORMAT_MOD_LINEAR`.
const DRM_FORMAT_MOD_LINEAR: i64 = 0;
/// `DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D` (the modifier nvidia-drm hands out;
/// the materializer's EGL import already handles it).
const DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D: i64 = 0x0300_0000_0060_6014;

use crate::{CaptureEvent, CaptureSource, DmabufScanout, FrameFormat, Running, Stop};
use rdma_display_proto::unix_now_us;

/// `SelectSources` `types` bitmask (portal `SourceType` values).
pub const SOURCE_MONITOR: u32 = 1;
pub const SOURCE_WINDOW: u32 = 2;
pub const SOURCE_VIRTUAL: u32 = 4;

/// `SelectSources` `cursor_mode`.
pub const CURSOR_HIDDEN: u32 = 1;
pub const CURSOR_EMBEDDED: u32 = 2;
pub const CURSOR_METADATA: u32 = 4;

/// `--source portal` configuration.
#[derive(Debug, Clone)]
pub struct PortalScreenCastSource {
    /// `types` bitmask; default `SOURCE_MONITOR`.
    pub source_types: u32,
    /// `cursor_mode`; default `CURSOR_EMBEDDED` (the compositor draws the
    /// cursor, matching what the KMS path captures).
    pub cursor_mode: u32,
    /// A restore token from a previous session (skips the picker dialog).
    pub restore_token: Option<String>,
    /// `persist_mode`: 0 = none, 1 = transient, 2 = persistent (default).
    pub persist_mode: u32,
}

impl Default for PortalScreenCastSource {
    fn default() -> Self {
        Self {
            source_types: SOURCE_MONITOR,
            cursor_mode: CURSOR_EMBEDDED,
            restore_token: None,
            persist_mode: 2,
        }
    }
}

impl CaptureSource for PortalScreenCastSource {
    fn name(&self) -> &'static str {
        "portal"
    }

    fn spawn(self: Box<Self>, tx: Sender<CaptureEvent>) -> Result<Running> {
        let stop = Stop::new();
        let worker_stop = stop.clone();
        let join = std::thread::Builder::new()
            .name("capture-portal".into())
            .spawn(move || {
                if let Err(e) = run(&self, tx, worker_stop) {
                    log::warn!("[portal] capture ended: {e:#}");
                }
            })
            .context("spawn portal capture thread")?;
        Ok(Running::new(stop, vec![join]))
    }
}

fn run(src: &PortalScreenCastSource, tx: Sender<CaptureEvent>, stop: Stop) -> Result<()> {
    // 1. portal handshake (async API driven to completion on this thread)
    let (node_id, remote_fd, size) =
        async_std::task::block_on(open_portal_stream(src)).context("portal ScreenCast")?;
    log::info!(
        "[portal] ScreenCast stream: node {node_id}, {}x{} (types={:#x}, cursor={:#x})",
        size.0,
        size.1,
        src.source_types,
        src.cursor_mode
    );

    // 2. consume the PipeWire stream until stopped
    pipewire_run(node_id, remote_fd, tx, stop)
}

/// Run the ScreenCast session lifecycle and return `(node_id, pipewire_fd, size)`.
///
/// The portal's `Start` result carries the streams; `OpenPipeWireRemote` gives
/// the fd the PipeWire core must connect through (the portal brokers the
/// connection, so the app needs no PipeWire socket of its own).
async fn open_portal_stream(src: &PortalScreenCastSource) -> Result<(u32, OwnedFd, (i32, i32))> {
    let proxy = Screencast::new()
        .await
        .context("connect to org.freedesktop.portal.ScreenCast")?;
    let session = proxy.create_session().await.context("CreateSession")?;

    let mut types = BitFlags::<SourceType>::empty();
    if src.source_types & SOURCE_MONITOR != 0 {
        types |= SourceType::Monitor;
    }
    if src.source_types & SOURCE_WINDOW != 0 {
        types |= SourceType::Window;
    }
    if src.source_types & SOURCE_VIRTUAL != 0 {
        types |= SourceType::Virtual;
    }
    if types.is_empty() {
        types = SourceType::Monitor.into();
    }
    let cursor = match src.cursor_mode {
        CURSOR_HIDDEN => CursorMode::Hidden,
        CURSOR_METADATA => CursorMode::Metadata,
        _ => CursorMode::Embedded,
    };
    let persist = match src.persist_mode {
        0 => PersistMode::DoNot,
        1 => PersistMode::Application,
        _ => PersistMode::ExplicitlyRevoked,
    };

    proxy
        .select_sources(
            &session,
            cursor,
            types,
            false,
            src.restore_token.as_deref(),
            persist,
        )
        .await
        .context("SelectSources")?;

    let response = proxy
        .start(&session, None)
        .await
        .context("Start")?
        .response()
        .context("Start response")?;
    let stream = response
        .streams()
        .first()
        .ok_or_else(|| anyhow!("the portal returned no stream"))?;
    let node_id = stream.pipe_wire_node_id();
    let size = stream.size().unwrap_or((0, 0));

    let fd = proxy
        .open_pipe_wire_remote(&session)
        .await
        .context("OpenPipeWireRemote")?;

    Ok((node_id, fd, size))
}

/// State carried through the PipeWire callbacks (the fixated format + the sink).
struct State {
    tx: Sender<CaptureEvent>,
    format: VideoInfoRaw,
    have_format: bool,
    /// the non-dmabuf warning is once-only (it would otherwise fire per frame)
    logged_non_dmabuf: bool,
}

impl State {
    /// Wrap a dmabuf plane as a [`DmabufScanout`], dup'ing the fd (the event
    /// owns it; the PipeWire buffer's fd is only valid during `process`).
    fn scanout(&self, fd: RawFd, offset: u32, stride: u32) -> Option<DmabufScanout> {
        let size = self.format.size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        let fourcc = video_fourcc(self.format.format())?;
        let dup = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
        if dup < 0 {
            return None;
        }
        let owned = unsafe { OwnedFd::from_raw_fd(dup) };
        Some(DmabufScanout {
            fb_id: 0,
            buf_id: dmabuf_identity(owned.as_raw_fd()),
            fds: vec![owned],
            offsets: vec![offset],
            strides: vec![stride],
            format: FrameFormat {
                width: size.width,
                height: size.height,
                stride,
                fourcc,
                modifier: self.format.modifier(),
                y0_top: true,
                num_planes: 1,
            },
        })
    }
}

/// Map a PipeWire video format to the DRM fourcc the materializer expects.
/// Only the RGB layouts are offered (the wire format is XR24/BGRX).
fn video_fourcc(f: VideoFormat) -> Option<u32> {
    Some(match f {
        VideoFormat::BGRx => 0x3432_5258, // XR24
        VideoFormat::BGRA => 0x3432_5241, // AR24
        VideoFormat::RGBx => 0x3432_4258, // XB24
        VideoFormat::RGBA => 0x3432_4241, // AB24
        _ => return None,
    })
}

/// `(st_dev, st_ino)` of a dma-buf fd: every dma-buf is an anonymous inode with
/// a globally unique `st_ino`, so this is a stable identity for the buffer (the
/// materializer caches imports on it). Mirrors `drm::dmabuf_identity`.
fn dmabuf_identity(fd: RawFd) -> u64 {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return 0;
    }
    ((st.st_dev as u64) << 32) | (st.st_ino as u64 & 0xffff_ffff)
}

/// Build the `SPA_PARAM_EnumFormat` pod: raw video, a choice of the RGB layouts
/// the materializer can import, any size, any framerate.
fn enum_format_pod() -> Vec<u8> {
    let obj = spa::pod::object!(
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        spa::pod::property!(
            FormatProperties::MediaType,
            Id,
            MediaType::Video
        ),
        spa::pod::property!(
            FormatProperties::MediaSubtype,
            Id,
            MediaSubtype::Raw
        ),
        spa::pod::property!(
            FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            VideoFormat::BGRx,
            VideoFormat::BGRx,
            VideoFormat::BGRA,
            VideoFormat::RGBx,
            VideoFormat::RGBA
        ),
        // the modifier property is what lets the producer allocate *dmabuf*
        // buffers; without it XDPH defaults to MemFd. Accept any modifier the
        // producer might use: MOD_INVALID (a MemFd buffer), LINEAR, and the
        // NVIDIA block-linear modifier the materializer's EGL import handles.
        spa::pod::Property {
            key: FormatProperties::VideoModifier.as_raw(),
            flags: spa::pod::PropertyFlags::MANDATORY
                | spa::pod::PropertyFlags::DONT_FIXATE,
            value: spa::pod::Value::Choice(spa::pod::ChoiceValue::Long(Choice::<i64>(
                ChoiceFlags::empty(),
                ChoiceEnum::<i64>::Enum {
                    default: DRM_FORMAT_MOD_INVALID,
                    alternatives: vec![
                        DRM_FORMAT_MOD_INVALID,
                        DRM_FORMAT_MOD_LINEAR,
                        DRM_FORMAT_MOD_NVIDIA_BLOCK_LINEAR_2D,
                    ],
                },
            ))),
        },
        spa::pod::property!(
            FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            spa::utils::Rectangle {
                width: 0,
                height: 0
            },
            spa::utils::Rectangle {
                width: 1,
                height: 1
            },
            spa::utils::Rectangle {
                width: 16384,
                height: 16384
            }
        ),
        spa::pod::property!(
            FormatProperties::VideoFramerate,
            Fraction,
            spa::utils::Fraction { num: 0, denom: 1 }
        ),
    );
    spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .expect("serialize EnumFormat pod")
    .0
    .into_inner()
}

/// Build the `SPA_PARAM_Buffers` pod: offer **dmabuf** (preferred) and MemFd.
/// Without this the producer (XDPH) hands out MemFd buffers, which this source
/// cannot import — the materializer needs a dmabuf — so MemFd is accepted by
/// the negotiation only so it can be reported and skipped, not captured.
fn buffers_pod() -> Vec<u8> {
    let data_type =
        (1u32 << DataType::DmaBuf.as_raw()) | (1u32 << DataType::MemFd.as_raw());
    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamBuffers.as_raw(),
        id: spa::param::ParamType::Buffers.as_raw(),
        properties: vec![
            spa::pod::Property {
                key: SPA_PARAM_BUFFERS_buffers,
                flags: spa::pod::PropertyFlags::empty(),
                value: spa::pod::Value::Choice(spa::pod::ChoiceValue::Int(Choice::<i32>(
                    ChoiceFlags::empty(),
                    ChoiceEnum::<i32>::Range {
                        default: 8,
                        min: 2,
                        max: 64,
                    },
                ))),
            },
            spa::pod::Property::new(SPA_PARAM_BUFFERS_blocks, spa::pod::Value::Int(1)),
            spa::pod::Property {
                key: SPA_PARAM_BUFFERS_dataType,
                flags: spa::pod::PropertyFlags::MANDATORY,
                value: spa::pod::Value::Int(data_type as i32),
            },
        ],
    };
    spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .expect("serialize Buffers pod")
    .0
    .into_inner()
}

/// Consume the PipeWire stream, emitting a [`CaptureEvent::Scanout`] per frame
/// (the same event the KMS source sends: the server's latest-wins mailbox keeps
/// the newest and the materializer caches imports on the buffer identity).
fn pipewire_run(
    node_id: u32,
    remote_fd: OwnedFd,
    tx: Sender<CaptureEvent>,
    stop: Stop,
) -> Result<()> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None).context("pipewire MainLoop")?;
    let context = pw::context::ContextRc::new(&mainloop, None).context("pipewire Context")?;
    // connect through the portal-brokered remote fd (owns it from here)
    let core = context
        .connect_fd_rc(remote_fd, None)
        .context("pipewire connect_fd (portal remote)")?;

    let stream = pw::stream::StreamBox::new(
        &core,
        "rdma-display",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )
    .context("pipewire Stream")?;

    let state = State {
        tx,
        format: VideoInfoRaw::new(),
        have_format: false,
        logged_non_dmabuf: false,
    };

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .param_changed(|_, state, id, param| {
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                state.have_format = false;
                return;
            };
            match spa::param::format_utils::parse_format(param) {
                Ok((MediaType::Video, MediaSubtype::Raw)) => {}
                _ => return,
            }
            if state.format.parse(param).is_err() {
                log::warn!("[portal] could not parse the fixated video format");
                return;
            }
            state.have_format = true;
            let s = state.format.size();
            log::info!(
                "[portal] negotiated {}x{} format={:?} modifier={:#x}",
                s.width,
                s.height,
                state.format.format(),
                state.format.modifier()
            );
        })
        .process(|stream, state| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() || !state.have_format {
                return;
            }
            let data = &mut datas[0];
            if data.type_() != DataType::DmaBuf {
                // the portal gave us a memfd/memptr buffer: this source only
                // speaks dmabuf (log once, not per frame)
                if !state.logged_non_dmabuf {
                    state.logged_non_dmabuf = true;
                    log::warn!(
                        "[portal] stream is delivering {:?} buffers, not dmabuf; \
                         skipping frames (the modifier/Buffers negotiation did not \
                         get us a dmabuf)",
                        data.type_()
                    );
                }
                return;
            }
            let fd = data.fd();
            let chunk = data.chunk();
            let (offset, stride) = (chunk.offset(), chunk.stride().max(0) as u32);
            if let Some(scanout) = state.scanout(fd, offset, stride) {
                let _ = state.tx.send(CaptureEvent::Scanout(scanout, unix_now_us()));
            }
        })
        .register()
        .context("register pipewire stream listener")?;

    let fmt_bytes = enum_format_pod();
    let buf_bytes = buffers_pod();
    let fmt_pod =
        Pod::from_bytes(&fmt_bytes).ok_or_else(|| anyhow!("invalid EnumFormat pod"))?;
    let buf_pod = Pod::from_bytes(&buf_bytes).ok_or_else(|| anyhow!("invalid Buffers pod"))?;
    let mut params = [fmt_pod, buf_pod];
    stream
        .connect(
            spa::utils::Direction::Input,
            Some(node_id),
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .context("pipewire stream connect")?;

    // drive the loop, checking the stop token between iterations (the loop is
    // single-threaded, so it cannot be quit from another thread)
    let loop_ = mainloop.loop_();
    while !stop.stopped() {
        loop_.iterate(pw::loop_::Timeout::Finite(Duration::from_millis(100)));
    }
    Ok(())
}
