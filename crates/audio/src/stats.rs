//! Audio timing diagnostics for the client's `--metrics` run.
//!
//! The playout thread records, for each packet it hands to the sink, how old
//! that packet already was:
//!
//! ```text
//!   e2e = (write_client_us - (host_unix_us - clock_offset)) + sink_latency_us
//! ```
//!
//! i.e. "the server packetized this audio → it comes out of the speaker".
//! `host_unix_us` is stamped when the server's capture read returned (see
//! `capture::capture_into`), so it is the audio analogue of the video's `t0`;
//! the server's capture → packetize latency is the one sliver not on the wire.
//! `clock_offset` (server − client) is the client's ping/pong estimate, so the
//! number carries its ±rtt/2 error — good for spotting a systematic A/V offset,
//! not for sub-millisecond claims.
//!
//! Inert unless the client asked for metrics: `playback::run` takes an
//! `Option<Arc<AudioStats>>` and does no per-packet work when it is `None`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;

/// Rolling window of per-packet e2e samples; 256 packets ≈ 1.3 s at 200 pkt/s.
const WINDOW: usize = 256;
/// `clock_offset_us` before the client's first ping/pong lands. Recording is
/// skipped until then: with no offset the cross-clock subtraction is garbage.
const OFFSET_UNSET: i64 = i64::MIN;

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
    /// ring drops the reader has counted (cumulative)
    dropped: AtomicU64,
    e2e: Mutex<VecDeque<u64>>,
}

impl AudioStats {
    pub fn new() -> Self {
        Self {
            clock_offset_us: AtomicI64::new(OFFSET_UNSET),
            sink_us: AtomicU64::new(0),
            packets: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            e2e: Mutex::new(VecDeque::with_capacity(WINDOW)),
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

    /// Cumulative ring drops, mirrored from the reader so the overlay can show
    /// whether the client is actually falling behind.
    pub fn set_dropped(&self, total: u64) {
        self.dropped.store(total, Ordering::Relaxed);
    }

    /// Record one packet handed to the sink: its server stamp and the client
    /// clock at the write. No-op until the clock offset is known.
    pub fn record(&self, host_unix_us: u64, write_client_us: u64) {
        let off = self.clock_offset_us.load(Ordering::Relaxed);
        if off == OFFSET_UNSET {
            return;
        }
        let sent_client = host_unix_us as i64 - off;
        let age = (write_client_us as i64 - sent_client).max(0) as u64;
        let e2e = age + self.sink_us.load(Ordering::Relaxed);
        self.packets.fetch_add(1, Ordering::Relaxed);
        let mut w = self.e2e.lock().unwrap();
        if w.len() == WINDOW {
            w.pop_front();
        }
        w.push_back(e2e);
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn sink_us(&self) -> u64 {
        self.sink_us.load(Ordering::Relaxed)
    }

    /// (p50, p95) of the current e2e window, in µs; (0, 0) until packets arrive.
    pub fn e2e_p50_p95(&self) -> (u64, u64) {
        let w = self.e2e.lock().unwrap();
        if w.is_empty() {
            return (0, 0);
        }
        let mut v: Vec<u64> = w.iter().copied().collect();
        v.sort_unstable();
        (v[v.len() / 2], v[(v.len() * 95 / 100).min(v.len() - 1)])
    }
}
