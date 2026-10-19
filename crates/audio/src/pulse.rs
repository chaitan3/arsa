//! dlopen wrapper for libpulse-simple (house style: no build-time native
//! linkage, mirrors eglwin.rs). We speak the pulse-simple API against the
//! host's pipewire-pulse compat socket, so both the capture (server) and
//! playout (client) sides work on PipeWire desktops without linking
//! libpipewire. Signature sources: pulse/simple.h (API frozen since 0.9).

use std::ffi::{c_char, c_void, CString};
use std::sync::Arc;

use anyhow::{bail, Context, Result};

// pa_sample_format_t (pulse/def.h); FLOAT32LE mirrors wire::SAMPLE_FORMAT
const PA_SAMPLE_FLOAT32LE: i32 = 5;
// pa_stream_direction_t (pulse/def.h)
const PA_STREAM_PLAYBACK: i32 = 1;
const PA_STREAM_RECORD: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct PaSampleSpec {
    format: i32,
    rate: u32,
    channels: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BufferAttr {
    maxlength: u32,
    tlength: u32,
    prebuf: u32,
    minreq: u32,
    fragsize: u32,
}

type SimpleNewFn = unsafe extern "C" fn(
    server: *const c_char,
    name: *const c_char,
    dir: i32,
    dev: *const c_char,
    stream_name: *const c_char,
    spec: *const PaSampleSpec,
    map: *const c_void,
    attr: *const BufferAttr,
    error: *mut i32,
) -> *mut c_void;
type SimpleReadFn =
    unsafe extern "C" fn(s: *mut c_void, data: *mut c_void, bytes: usize, error: *mut i32) -> i32;
type SimpleWriteFn =
    unsafe extern "C" fn(s: *mut c_void, data: *const c_void, bytes: usize, error: *mut i32) -> i32;
type SimpleLatencyFn = unsafe extern "C" fn(s: *mut c_void, error: *mut i32) -> u64;
type SimpleFreeFn = unsafe extern "C" fn(s: *mut c_void);

fn err_str(code: i32) -> String {
    // pa_error_code_t (pulse/def.h)
    match code {
        0 => "ok".into(),
        1 => "access failure".into(),
        2 => "unknown command".into(),
        3 => "invalid argument".into(),
        4 => "entity exists".into(),
        5 => "no such entity (bad device name?)".into(),
        6 => "connection refused (no server reachable; under sudo this is the XDG_RUNTIME_DIR ownership check aborting the connect before any socket is opened)".to_string(),
        7 => "protocol error".into(),
        8 => "timeout".into(),
        9 => "no auth key".into(),
        10 => "internal error".into(),
        11 => "connection terminated".into(),
        15 => "bad state".into(),
        19 => "not supported".into(),
        other => format!("pa error {other}"),
    }
}

/// Function table + lib keep-alive, shared by all Simple streams from one
/// dlopen. Cloneable (cheap: two Arc'd fields).
pub struct Pulse {
    _lib: Arc<libloading::Library>,
    simple_new: SimpleNewFn,
    simple_read: SimpleReadFn,
    simple_write: SimpleWriteFn,
    simple_free: SimpleFreeFn,
    simple_get_latency: SimpleLatencyFn,
    simple_flush: SimpleLatencyFn, // same shape: (s, error) -> int
}

impl Clone for Pulse {
    fn clone(&self) -> Self {
        Pulse {
            _lib: self._lib.clone(),
            simple_new: self.simple_new,
            simple_read: self.simple_read,
            simple_write: self.simple_write,
            simple_free: self.simple_free,
            simple_get_latency: self.simple_get_latency,
            simple_flush: self.simple_flush,
        }
    }
}
unsafe impl Send for Pulse {}
unsafe impl Sync for Pulse {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SampleSpec {
    pub rate: u32,
    pub channels: u32,
}

impl Pulse {
    pub fn dlopen() -> Result<Self> {
        let lib = std::sync::Arc::new(unsafe {
            libloading::Library::new("libpulse-simple.so.0")
                .or_else(|_| libloading::Library::new("libpulse-simple.so"))
                .context("dlopen libpulse-simple (pipewire-pulse provides it)")?
        });
        let get = |n: &str| -> anyhow::Result<*mut c_void> {
            unsafe { lib.get::<*mut c_void>(n.as_bytes()) }
                .with_context(|| format!("symbol {n}"))
                .map(|sym| *sym)
        };
        Ok(Pulse {
            simple_new: unsafe { std::mem::transmute(get("pa_simple_new")?) },
            simple_read: unsafe { std::mem::transmute(get("pa_simple_read")?) },
            simple_write: unsafe { std::mem::transmute(get("pa_simple_write")?) },
            simple_free: unsafe { std::mem::transmute(get("pa_simple_free")?) },
            simple_get_latency: unsafe { std::mem::transmute(get("pa_simple_get_latency")?) },
            simple_flush: unsafe { std::mem::transmute(get("pa_simple_flush")?) },
            _lib: lib,
        })
    }

    /// Record stream. `dev`: a source name — a sink's monitor is
    /// "<sink-name>.monitor"; "@DEFAULT_MONITOR@" is the PA ≥15 magic alias
    /// for the default sink's monitor (pipewire-pulse honors it).
    pub fn open_record(
        &self,
        server: Option<&str>,
        dev: &str,
        spec: SampleSpec,
        fragsize_bytes: u32,
    ) -> Result<Simple> {
        self.open(
            server,
            PA_STREAM_RECORD,
            Some(dev),
            "arsa-capture",
            spec,
            BufferAttr {
                // (uint32)-1 = "server default": pipewire-pulse clamps a
                // LITERAL 0 to frame_size (8 bytes) — maxlength=8 then makes
                // every graph cycle hit the overrun-recovery path
                // (skip-to-latest, send one fragsize per cycle), throttling
                // the capture to 8 bytes/cycle ≈ one packet per 2.5 s.
                // PulseAudio proper treats 0 as default; pipewire-pulse
                // (fix_record_buffer_attr) only treats -1 as default.
                maxlength: u32::MAX,
                tlength: 0,
                prebuf: 0,
                minreq: 0,
                fragsize: fragsize_bytes,
            },
        )
        .map_err(|code| PaOpenError { code: code.code })
        .with_context(|| format!("pa record stream open (device {dev:?})"))
    }

    /// Playout stream to the default sink with a fixed server-side buffer —
    /// `tlength` IS the jitter buffer; `prebuf` gates startup on a full
    /// buffer and re-gates after each underrun.
    pub fn open_playback(
        &self,
        server: Option<&str>,
        spec: SampleSpec,
        buffer_ms: u32,
        minreq_bytes: u32,
    ) -> Result<Simple> {
        let tlength = (spec.rate as u64 * buffer_ms as u64 / 1000 * 4 * spec.channels as u64)
            as u32;
        // minreq should track the wire packet size, not a fixed 10 ms: a large
        // minreq forces the server to batch requests and floors the latency
        // above a small tlength.
        let minreq = minreq_bytes.max(1);
        self.open(
            server,
            PA_STREAM_PLAYBACK,
            None, // default sink
            "arsa",
            spec,
            BufferAttr {
                maxlength: tlength.saturating_mul(4),
                tlength,
                prebuf: tlength,
                minreq,
                fragsize: 0,
            },
        )
        .with_context(|| format!("pa playback stream open (buffer {buffer_ms} ms)"))
    }

    fn open(
        &self,
        server: Option<&str>,
        dir: i32,
        dev: Option<&str>,
        name: &str,
        spec: SampleSpec,
        attr: BufferAttr,
    ) -> Result<Simple, PaOpenError> {
        let server_c = server.map(|s| CString::new(s).unwrap()).unwrap_or_else(|| CString::new("").unwrap());
        let dev_c = dev.map(|s| CString::new(s).unwrap()).unwrap_or_else(|| CString::new("").unwrap());
        let name_c = CString::new(name).unwrap();
        let ss = PaSampleSpec {
            format: PA_SAMPLE_FLOAT32LE,
            rate: spec.rate,
            channels: spec.channels as u8,
        };
        let mut err = 0;
        let s = unsafe {
            (self.simple_new)(
                if server.is_some() { server_c.as_ptr() } else { std::ptr::null() },
                name_c.as_ptr(),
                dir,
                if dev.is_some() { dev_c.as_ptr() } else { std::ptr::null() },
                name_c.as_ptr(),
                &ss,
                std::ptr::null(),
                &attr,
                &mut err,
            )
        };
        if s.is_null() {
            return Err(PaOpenError { code: err });
        }
        Ok(Simple { s, pulse: self.clone() })
    }
}

#[derive(Debug)]
pub struct PaOpenError {
    /// pa_error_code_t value (see err_str)
    pub code: i32,
}

impl std::fmt::Display for PaOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pa_simple_new failed: {}", err_str(self.code))
    }
}

impl std::error::Error for PaOpenError {}

/// A live pa_simple stream. libpulse-simple is not thread-safe per stream —
/// each Simple stays on its owning thread; we only promise Send (moved
/// once at spawn).
pub struct Simple {
    s: *mut c_void,
    pulse: Pulse,
}
unsafe impl Send for Simple {}

impl Simple {
    /// Blocking read of exactly `buf.len()` bytes — pa_simple_read fills
    /// the whole request before returning (pulse/simple.h contract).
    pub fn read_exact(&self, buf: &mut [u8]) -> anyhow::Result<()> {
        let mut err = 0;
        let r = unsafe {
            (self.pulse.simple_read)(self.s, buf.as_mut_ptr() as *mut c_void, buf.len(), &mut err)
        };
        if r != 0 {
            bail!("pa_simple_read failed: {}", err_str(err));
        }
        Ok(())
    }

    pub fn write_all(&self, buf: &[u8]) -> anyhow::Result<()> {
        let mut err = 0;
        let r = unsafe {
            (self.pulse.simple_write)(self.s, buf.as_ptr() as *const c_void, buf.len(), &mut err)
        };
        if r != 0 {
            bail!("pa_simple_write failed: {}", err_str(err));
        }
        Ok(())
    }

    /// Data still queued for playout (playback stream).
    pub fn latency_us(&self) -> anyhow::Result<u64> {
        let mut err = 0;
        let us = unsafe { (self.pulse.simple_get_latency)(self.s, &mut err) };
        if us == u64::MAX {
            bail!("pa_simple_get_latency failed: {}", err_str(err));
        }
        Ok(us)
    }

    /// Discard queued playout data (client-side drift resync).
    pub fn flush(&self) -> anyhow::Result<()> {
        let mut err = 0;
        let r = unsafe { (self.pulse.simple_flush)(self.s, &mut err) };
        if r != 0 {
            bail!("pa_simple_flush failed: {}", err_str(err));
        }
        Ok(())
    }
}

impl Drop for Simple {
    fn drop(&mut self) {
        unsafe { (self.pulse.simple_free)(self.s) };
    }
}
