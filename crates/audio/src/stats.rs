//! Audio timing diagnostics for the client's `--metrics` run.
//!
//! The playout thread records, for each packet it hands to the sink, how old
//! that packet already was, split into its two halves:
//!
//! ```text
//!   transport = recv_client_us - (host_unix_us - clock_offset)
//!   ring      = write_client_us - recv_client_us
//!   e2e       = transport + ring + sink_latency_us
//! ```
//!
//! i.e. "the server packetized this audio → it comes out of the speaker".
//! `host_unix_us` is stamped when the server's capture read returned (see
//! `capture::capture_into`), so it is the audio analogue of the video's `t0`;
//! the server's capture → packetize latency is the one sliver not on the wire.
//! `transport` is the wire (one-way propagation + the server's send); `ring` is
//! the client's pre-sink backlog, measured on the client clock only, so it is
//! *offset-independent* — a clean read on the playout buffer. `clock_offset`
//! (server − client) is the client's ping/pong estimate, so `transport` carries
//! its ±rtt/2 error — good for spotting a systematic A/V offset, not for
//! sub-millisecond claims.
//!
//! Inert unless the client asked for metrics: `playback::run` takes an
//! `Option<Arc<AudioStats>>` and does no per-packet work when it is `None`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;

/// Rolling window of per-packet samples; 256 packets ≈ 1.3 s at 200 pkt/s.
const WINDOW: usize = 256;
/// `clock_offset_us` before the client's first ping/pong lands. Recording is
/// skipped until then: with no offset the cross-clock subtraction is garbage.
const OFFSET_UNSET: i64 = i64::MIN;

/// One packet handed to the sink, all in µs: `transport` is the wire, `ring`
/// the client's pre-sink backlog, `e2e` the two plus the sink's own buffer.
#[derive(Clone, Copy)]
struct AudioSample {
    e2e: u64,
    transport: u64,
    ring: u64,
}

/// Shared, cheap-to-update audio timing stats. Written by the playout thread
/// (per packet), the client's clock sync (offset) and its metrics collector
/// (read); all fields are atomics or a short-lived lock.
pub struct AudioStats {
    /// server_time − client_time (µs), from the client's clock sync.
    clock_offset_us: AtomicI64,
    /// most recent sink latency (µs) — sampled at the report cadence.
    sink_us: AtomicU64,
    /// packets handed to the sink (since the last reset)
    packets: AtomicU64,
    /// ring overflow drops the reader has counted (cumulative)
    dropped: AtomicU64,
    /// convergence trims the playout has performed (cumulative)
    trimmed: AtomicU64,
    /// adaptive ring jitter target (ms), mirrored from the playout
    jitter_ms: AtomicU64,
    /// manual A/V alignment delay (ms), mirrored from the playout
    align_ms: AtomicU64,
    samples: Mutex<VecDeque<AudioSample>>,
}

impl AudioStats {
    pub fn new() -> Self {
        Self {
            clock_offset_us: AtomicI64::new(OFFSET_UNSET),
            sink_us: AtomicU64::new(0),
            packets: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            trimmed: AtomicU64::new(0),
            jitter_ms: AtomicU64::new(0),
            align_ms: AtomicU64::new(0),
            samples: Mutex::new(VecDeque::with_capacity(WINDOW)),
        }
    }

    /// server_time − client_time (µs), pushed by the client's clock sync.
    pub fn set_clock_offset(&self, us: i64) {
        self.clock_offset_us.store(us, Ordering::Relaxed);
    }

    /// Latest sink latency (µs), sampled where the drift guard already queries
    /// it (every report interval) and used by `record` until it changes.
    pub fn set_sink_us(&self, us: u64) {
        self.sink_us.store(us, Ordering::Relaxed);
    }

    /// Cumulative ring overflow drops, mirrored from the reader so the overlay
    /// can show whether the client is actually falling behind.
    pub fn set_dropped(&self, total: u64) {
        self.dropped.store(total, Ordering::Relaxed);
    }

    /// Cumulative convergence trims, mirrored from the playout.
    pub fn set_trimmed(&self, total: u64) {
        self.trimmed.store(total, Ordering::Relaxed);
    }

    /// Adaptive jitter target (ms), mirrored from the playout.
    pub fn set_jitter_ms(&self, ms: u64) {
        self.jitter_ms.store(ms, Ordering::Relaxed);
    }

    /// Manual A/V alignment delay (ms), mirrored from the playout.
    pub fn set_align_ms(&self, ms: u64) {
        self.align_ms.store(ms, Ordering::Relaxed);
    }

    /// Record one packet handed to the sink: its server stamp, the client clock
    /// when the reader received it, and the client clock at the write. No-op
    /// until the clock offset is known.
    pub fn record(&self, host_unix_us: u64, recv_client_us: u64, write_client_us: u64) {
        let off = self.clock_offset_us.load(Ordering::Relaxed);
        if off == OFFSET_UNSET {
            return;
        }
        let sent_client = host_unix_us as i64 - off;
        let transport = (recv_client_us as i64 - sent_client).max(0) as u64;
        let ring = (write_client_us as i64 - recv_client_us as i64).max(0) as u64;
        let e2e = transport + ring + self.sink_us.load(Ordering::Relaxed);
        self.packets.fetch_add(1, Ordering::Relaxed);
        let mut w = self.samples.lock().unwrap();
        if w.len() == WINDOW {
            w.pop_front();
        }
        w.push_back(AudioSample { e2e, transport, ring });
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn trimmed(&self) -> u64 {
        self.trimmed.load(Ordering::Relaxed)
    }

    pub fn jitter_ms(&self) -> u64 {
        self.jitter_ms.load(Ordering::Relaxed)
    }

    pub fn align_ms(&self) -> u64 {
        self.align_ms.load(Ordering::Relaxed)
    }

    pub fn sink_us(&self) -> u64 {
        self.sink_us.load(Ordering::Relaxed)
    }

    /// (p50, p95) of the current e2e window, in µs; (0, 0) until packets arrive.
    pub fn e2e_p50_p95(&self) -> (u64, u64) {
        let w = self.samples.lock().unwrap();
        if w.is_empty() {
            return (0, 0);
        }
        let mut v: Vec<u64> = w.iter().map(|s| s.e2e).collect();
        v.sort_unstable();
        (v[v.len() / 2], v[(v.len() * 95 / 100).min(v.len() - 1)])
    }

    /// Median (transport, ring), in µs; (0, 0) until packets arrive. `ring` is
    /// clock-offset-independent, so it is the honest pre-sink backlog.
    pub fn split_p50(&self) -> (u64, u64) {
        let w = self.samples.lock().unwrap();
        if w.is_empty() {
            return (0, 0);
        }
        let mut t: Vec<u64> = w.iter().map(|s| s.transport).collect();
        let mut r: Vec<u64> = w.iter().map(|s| s.ring).collect();
        t.sort_unstable();
        r.sort_unstable();
        (t[t.len() / 2], r[r.len() / 2])
    }
}
