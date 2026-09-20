//! Client-side latency/bandwidth diagnostics (`--metrics`).
//!
//! Measurement model:
//!
//! ```text
//! t0 (server) QEMU D-Bus event arrived in the capture thread
//! t1 (server) materializer readback finished (glFinish done, slot populated)
//! t2 (server) RDMA WRITE posted
//! t3 (client) frame completion polled from the recv CQ
//! t4 (client) glTexImage2D upload returned
//! t5 (client) eglSwapBuffers returned
//! ```
//!
//! Stages (rolling median/p95 over the last N frames):
//!   capture = t1-t0, split into:
//!     q   = t0d-t0  (pacing gate + event-queue wait in the server loop)
//!     sw  = t1s-t0d (staging-slot credit wait; nonzero => client lagging)
//!     mat = t1-t1s  (dmabuf import + blit + readback + glFinish)
//!   xfer    = t2c-t2  (RDMA WRITE transfer; both stamps same-host, exact)
//!   poll    = t3-t2c  (client poll granularity; uses the clock offset)
//!   upload  = t4-t3  (texImage2D from pinned RAM over PCIe)
//!   present = t5-t4  (blit + swap submission; compositor lag not included)
//!   e2e     = t5-t0, corrected by the ping-pong clock offset estimate
//!
//! Cross-host comparability: the offset estimate comes from `Ping`/`Pong`
//! on the control channel — offset = s1 - (c0+c2)/2 taken from the **min-RTT**
//! sample of the recent window (least queuing => tightest ±rtt/2 bound),
//! refreshed every couple of seconds so real drift is tracked. Only `e2e`
//! and `poll` use it; `xfer` is same-host and exact. Known honesty limit:
//! t5 is SwapBuffers return, not the monitor's scanout.

use std::collections::{HashMap, VecDeque};

use rdma_display_proto::FrameStamp;
use rdma_display_rdma::ControlEvent;

/// Rolling sample window for medians/p95.
const WINDOW: usize = 64;
/// Max frames kept waiting for their server stamp (staleness guard).
const PENDING_CAP: usize = 256;
/// Overlay geometry: 8px font, COLS columns wide, LINES lines tall.
const COLS: usize = 48;
const LINES: usize = 8;
pub const OVERLAY_W: u32 = (COLS * 8 + 8) as u32;
pub const OVERLAY_H: u32 = (LINES * 8 + 8) as u32;
/// Milliseconds between overlay text refreshes.
pub const REFRESH_MS: u64 = 250;
/// Seconds between clock-sync pings.
pub const PING_SECS: u64 = 2;
/// e2e above this prints an immediate stderr spike line ([metrics-spike]).
const SPIKE_E2E_US: u64 = 40_000;

#[derive(Clone, Copy)]
struct Sample {
    q: u64,    // pacing + event-queue wait (t0d - t0)
    sw: u64,   // staging-slot credit wait (t1s - t0d)
    mat: u64,  // materialize total (t1 - t1s)
    blit: u64, // GPU executed our draw (fence wait) — from the server
    read: u64, // readback DMA into the staging slot — from the materializer
    fin: u64,  // residual glFinish — from the materializer
    xfer: u64, // RDMA WRITE transfer (t2c - t2; same-host, exact)
    poll: u64, // completion -> client poll (t3 - t2c, offset-corrected)
    upl: u64,
    pres: u64,
    e2e: u64, // clock-corrected
}

pub struct Metrics {
    samples: VecDeque<Sample>,
    /// presented frames awaiting their server stamp: (seq, t3, t4, t5)
    local: VecDeque<(u64, u64, u64, u64)>,
    /// server stamps that arrived before their frame was presented
    stamps: HashMap<u64, FrameStamp>,
    frame_bytes: usize,
    last_seq: Option<u64>,
    skipped: u64,
    frames_since_refresh: u64,
    last_refresh: Option<std::time::Instant>,
    last_ping: Option<std::time::Instant>,
    clock: ClockSync,
    text: String,
}

struct ClockSync {
    /// recent (offset, rtt) samples; offset = server_time - client_time.
    /// The reported offset comes from the min-RTT sample (least queuing =>
    /// tightest ±rtt/2 error bound; classic NTP trick).
    samples: VecDeque<(i64, u64)>,
}

impl ClockSync {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(16),
        }
    }

    fn add(&mut self, c0: u64, s1: u64, c2: u64) {
        let rtt = c2.saturating_sub(c0);
        let offset = s1 as i64 - ((c0 + c2) / 2) as i64;
        if self.samples.len() == 16 {
            self.samples.pop_front();
        }
        self.samples.push_back((offset, rtt));
    }

    /// Offset from the min-RTT sample; `None` until the first pong.
    fn offset(&self) -> Option<i64> {
        self.samples
            .iter()
            .min_by_key(|s| s.1)
            .map(|s| s.0)
    }

    /// Smallest RTT seen recently (the offset's uncertainty bound: ±rtt/2).
    fn rtt_min(&self) -> Option<u64> {
        self.samples.iter().map(|s| s.1).min()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(WINDOW),
            local: VecDeque::with_capacity(4),
            stamps: HashMap::new(),
            frame_bytes: 0,
            last_seq: None,
            skipped: 0,
            frames_since_refresh: 0,
            last_refresh: None,
            last_ping: None,
            clock: ClockSync::new(),
            text: String::new(),
        }
    }

    pub fn set_frame_bytes(&mut self, bytes: usize) {
        self.frame_bytes = bytes;
    }

    /// One presented frame: the presenter loop stamps t3 when the RDMA
    /// completion is polled and t4/t5 inside the GL calls. The sample is
    /// only counted once the server's stamp for `seq` arrives.
    pub fn record(&mut self, seq: u64, t3: u64, t4: u64, t5: u64) {
        // sequence gaps (imm carries seq mod 2^24) => dropped/skipped frames
        if let Some(last) = self.last_seq {
            let diff = seq.wrapping_sub(last) & 0x00ff_ffff;
            if diff > 1 && diff < 0x1000 {
                self.skipped += (diff - 1) as u64;
            }
        }
        self.last_seq = Some(seq);
        self.frames_since_refresh += 1;

        if self.local.len() >= PENDING_CAP {
            self.local.pop_front();
        }
        self.local.push_back((seq, t3, t4, t5));
        self.try_match();
    }

    /// Drain one control event (server stamp or pong).
    pub fn on_control(&mut self, ev: ControlEvent) {
        match ev {
            ControlEvent::Stamp(s) => {
                if self.stamps.len() >= PENDING_CAP {
                    self.stamps.clear();
                }
                self.stamps.insert(s.seq, s);
                self.try_match();
            }
            ControlEvent::Pong { c0, s1, c2 } => self.clock.add(c0, s1, c2),
        }
    }

    /// Emit samples whose stamp arrived. Frames are presented in order and
    /// stamps arrive in order, so the oldest unmatched local entry decides.
    fn try_match(&mut self) {
        while let Some(front) = self.local.front().copied() {
            let (seq, t3, t4, t5) = front;
            match self.stamps.remove(&seq) {
                Some(s) => {
                    self.local.pop_front();
                    let off = self.clock.offset().unwrap_or(0);
                    // poll: client saw the completion at t3 (client clock);
                    // t2c is server clock, client clock = server clock - off,
                    // so the true gap is t3 - (t2c - off) = t3 - t2c + off
                    let poll = (t3 as i64 + off as i64 - s.t2c as i64).max(0) as u64;
                    let sample = Sample {
                        q: s.t0d.saturating_sub(s.t0),
                        sw: s.t1s.saturating_sub(s.t0d),
                        mat: s.t1.saturating_sub(s.t1s),
                        blit: s.blit_us,
                        read: s.read_us,
                        fin: s.fin_us,
                        xfer: s.t2c.saturating_sub(s.t2),
                        poll,
                        upl: t4.saturating_sub(t3),
                        pres: t5.saturating_sub(t4),
                        e2e: (t5 as i64 + off as i64 - s.t0 as i64).max(0) as u64,
                    };
                    // sanity filter: clock garbage or clamped-stale stamps
                    // (shouldn't happen post-fix) must not poison medians
                    if sample.e2e > 3_000_000 || sample.poll > 3_000_000 {
                        continue;
                    }
                    if self.samples.len() == WINDOW {
                        self.samples.pop_front();
                    }
                    self.samples.push_back(sample);
                    if sample.e2e > SPIKE_E2E_US {
                        eprintln!(
                            "[metrics-spike] seq={} e2e {:.2} ms | q {:.2} sw {:.2} mat {:.2} (b {:.2} r {:.2} f {:.2}) | xfer {:.2} poll {:.2} upl {:.2} pre {:.2}",
                            seq,
                            sample.e2e as f64 / 1000.0,
                            sample.q as f64 / 1000.0,
                            sample.sw as f64 / 1000.0,
                            sample.mat as f64 / 1000.0,
                            sample.blit as f64 / 1000.0,
                            sample.read as f64 / 1000.0,
                            sample.fin as f64 / 1000.0,
                            sample.xfer as f64 / 1000.0,
                            sample.poll as f64 / 1000.0,
                            sample.upl as f64 / 1000.0,
                            sample.pres as f64 / 1000.0,
                        );
                    }
                }
                None => break,
            }
        }
    }

    /// Ping due? The caller sends one and pings again on the next interval.
    pub fn maybe_ping(&mut self) -> bool {
        let due = match self.last_ping {
            None => true,
            Some(t) => t.elapsed().as_secs() >= PING_SECS,
        };
        if due {
            self.last_ping = Some(std::time::Instant::now());
        }
        due
    }

    /// Recompute the overlay text; returns Some when it changed (the caller
    /// rasterizes and uploads it).
    pub fn refresh(&mut self) -> Option<String> {
        let now = std::time::Instant::now();
        let (fps, mbps) = match self.last_refresh {
            None => (0.0, 0.0),
            Some(t) => {
                let dt = now.duration_since(t).as_secs_f64().max(1e-6);
                let fps = self.frames_since_refresh as f64 / dt;
                (fps, self.frame_bytes as f64 * fps * 8.0 / 1e6)
            }
        };
        self.last_refresh = Some(now);
        self.frames_since_refresh = 0;

        let stat = |pick: fn(&Sample) -> u64| -> (f64, f64) {
            let mut v: Vec<u64> = self.samples.iter().map(pick).collect();
            if v.is_empty() {
                return (0.0, 0.0);
            }
            v.sort_unstable();
            let med = v[v.len() / 2] as f64 / 1000.0;
            let p95 = v[(v.len() * 95 / 100).min(v.len() - 1)] as f64 / 1000.0;
            (med, p95)
        };
        let (e2e_m, e2e_p) = stat(|s| s.e2e);
        let cap = |s: &Sample| s.q + s.sw + s.mat;
        let (cap_m, cap_p) = stat(cap);
        let (q_m, _) = stat(|s| s.q);
        let (sw_m, _) = stat(|s| s.sw);
        let (mat_m, _) = stat(|s| s.mat);
        let (b_m, _) = stat(|s| s.blit);
        let (r_m, _) = stat(|s| s.read);
        let (f_m, _) = stat(|s| s.fin);
        let (net_m, net_p) = stat(|s| s.xfer + s.poll);
        let (xf_m, xf_p) = stat(|s| s.xfer);
        let (po_m, _) = stat(|s| s.poll);
        let (upl_m, upl_p) = stat(|s| s.upl);
        let (pre_m, pre_p) = stat(|s| s.pres);

        let mut t = String::with_capacity(COLS * LINES);
        t.push_str(&format!(
            "seq {:>6}  fps {:>5.1}  {:>6.1} Mb/s",
            self.last_seq.unwrap_or(0) & 0x00ff_ffff,
            fps,
            mbps
        ));
        t.push_str(&format!("\ne2e  {:>6.2} {:>6.2} ms", e2e_m, e2e_p));
        t.push_str(&format!("\ncap  {:>6.2} {:>6.2}", cap_m, cap_p));
        t.push_str(&format!(
            "\n  q {:>5.2} sw {:>5.2} mat {:>5.2}",
            q_m, sw_m, mat_m
        ));
        t.push_str(&format!(
            "\n  b {:>5.2} r {:>5.2} f {:>5.2}",
            b_m, r_m, f_m
        ));
        t.push_str(&format!(
            "\nxfer {:>5.2} {:>5.2} poll {:>5.2} (net {:>5.2} {:>5.2})",
            xf_m, xf_p, po_m, net_m, net_p
        ));
        t.push_str(&format!("\nupl  {:>6.2} {:>6.2}", upl_m, upl_p));
        t.push_str(&format!("\npre  {:>6.2} {:>6.2}", pre_m, pre_p));
        let clk = match (self.clock.offset(), self.clock.rtt_min()) {
            (Some(o), Some(r)) => format!("skip {}  clk {:+} us rttn {} us", self.skipped, o, r),
            _ => format!("skip {}  clk syncing", self.skipped),
        };
        t.push_str(&format!("\n{}", clk));
        // pad to the line budget so the raster size is stable
        while t.matches('\n').count() < LINES - 1 {
            t.push('\n');
        }
        if t != self.text {
            self.text = t.clone();
            Some(t)
        } else {
            None
        }
    }
}

/// Headless (`--output none`) one-line stats print for `--metrics`.
pub fn stderr_line(m: &mut Metrics) -> String {
    m.refresh().unwrap_or_else(|| m.text.clone())
}

/// Current overlay text without forcing a refresh (presenter mode: the
/// overlay owns the refresh cadence; the 5s stderr dump just snapshots it).
pub fn snapshot(m: &Metrics) -> &str {
    &m.text
}

// ---------------------------------------------------------------------------
// 8x8 bitmap font rasterizer (font8x8, public domain, dhepper/font8x8)
// ---------------------------------------------------------------------------

/// Rasterize `text` into an RGBA8 buffer of [`OVERLAY_W`]x[`OVERLAY_H`]:
/// translucent black background, white glyphs, 8px monospace grid.
pub fn rasterize(text: &str) -> Vec<u8> {
    let mut buf = vec![0u8; (OVERLAY_W * OVERLAY_H * 4) as usize];
    // background
    for px in buf.chunks_exact_mut(4) {
        px[0] = 0;
        px[1] = 0;
        px[2] = 0;
        px[3] = 150;
    }
    for (row, line) in text.split('\n').take(LINES).enumerate() {
        for (col, ch) in line.chars().take(COLS).enumerate() {
            let glyph = &FONT8X8[(ch as usize).min(127)];
            let ox = 4 + col * 8;
            let oy = 4 + row * 8;
            for gy in 0..8 {
                let bits = glyph[gy];
                for gx in 0..8 {
                    if (bits >> gx) & 1 == 0 {
                        continue;
                    }
                    let x = ox + gx;
                    let y = oy + gy;
                    if x >= OVERLAY_W as usize || y >= OVERLAY_H as usize {
                        continue;
                    }
                    let o = (y * OVERLAY_W as usize + x) * 4;
                    buf[o] = 255;
                    buf[o + 1] = 255;
                    buf[o + 2] = 255;
                    buf[o + 3] = 255;
                }
            }
        }
    }
    buf
}

/// font8x8 basic latin (U+0000..U+007F), public domain (dhepper/font8x8).
/// One byte per row; bit N of a row = pixel column N (LSB = leftmost).
const FONT8X8: [[u8; 8]; 128] = {
    let mut f = [[0u8; 8]; 128];
    let mut i = 0;
    while i < RAW.len() {
        f[i] = RAW[i];
        i += 1;
    }
    f
};

const RAW: [[u8; 8]; 128] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // space
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00], // !
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // "
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00], // #
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00], // $
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00], // %
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00], // &
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00], // '
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00], // (
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00], // )
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // *
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00], // +
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ,
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00], // -
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00], // .
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00], // /
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00], // 0
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00], // 1
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00], // 2
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00], // 3
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00], // 4
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00], // 5
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00], // 6
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00], // 7
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00], // 8
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00], // 9
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00], // :
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ;
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00], // <
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00], // =
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // >
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00], // ?
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00], // @
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00], // A
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00], // B
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00], // C
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00], // D
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00], // E
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00], // F
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00], // G
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00], // H
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // I
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00], // J
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00], // K
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00], // L
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00], // M
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00], // N
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00], // O
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00], // P
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00], // Q
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00], // R
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00], // S
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // T
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00], // U
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // V
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00], // W
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00], // X
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00], // Y
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00], // Z
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00], // [
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00], // backslash
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00], // ]
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00], // ^
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // _
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // `
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00], // a
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00], // b
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00], // c
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00], // d
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00], // e
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00], // f
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F], // g
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00], // h
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // i
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E], // j
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00], // k
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // l
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00], // m
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00], // n
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00], // o
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F], // p
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78], // q
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00], // r
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00], // s
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00], // t
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00], // u
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // v
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00], // w
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00], // x
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F], // y
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00], // z
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00], // {
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // |
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00], // }
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ~
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // DEL
];
