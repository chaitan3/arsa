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
    match code {
        -2 => "bad state".into(),
        -3 => "timeout".into(),
        -4 => "no entity".into(),
        -5 => "connection refused".into(),
        -6 => "invalid (device/server name?)".into(),
        -11 => "not supported".into(),
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
            "rdma-display-capture",
            spec,
            BufferAttr {
                maxlength: 0, // server default
                tlength: 0,
                prebuf: 0,
                minreq: 0,
                fragsize: fragsize_bytes,
            },
        )
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
    ) -> Result<Simple> {
        let tlength = (spec.rate as u64 * buffer_ms as u64 / 1000 * 4 * spec.channels as u64)
            as u32;
        let minreq = (spec.rate as u64 * 10 / 1000 * 4 * spec.channels as u64) as u32;
        self.open(
            server,
            PA_STREAM_PLAYBACK,
            None, // default sink
            "rdma-display",
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
    ) -> Result<Simple> {
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
            bail!("pa_simple_new failed: {}", err_str(err));
        }
        Ok(Simple { s, pulse: self.clone() })
    }
}

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
