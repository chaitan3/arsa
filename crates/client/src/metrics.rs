//! Client-side latency/bandwidth diagnostics (`--metrics`).
//!
//! Measurement model
//! =================
//!
//! Every reported stage is a duration measured on ONE host, so no stage
//! depends on the two machines sharing a clock:
//!
//! Server (all `unix_now_us` on the server; differences are exact):
//!   t0  = capture event latched (QEMU D-Bus arrival / DRM scanout poll)
//!   t0d = the sender picked the frame up (pacing + event-queue wait done)
//!   t0p = the scanout dma-buf import / materializer prepare finished
//!   t1s = a staging slot was free (after any credit wait)
//!   t1  = materializer readback finished (slot holds the pixels)
//!   t2  = RDMA WRITE posted to the NIC
//!   t2c = send CQ reported the WRITE complete (frame written on the client)
//!
//!   queue    = t0d - t0    capture queue / pacing
//!   prepare  = t0p - t0d   scanout import / materializer prepare
//!   slot     = t1s - t0p   staging-slot credit wait
//!   material = t1 - t1s    blit + readback + fence (codec-dependent)
//!   post     = t2 - t1     WQE build + post
//!   transfer = t2c - t2    RDMA WRITE as the SENDER saw it (one-sided)
//!
//! Client (all `unix_now_us` on the client; differences are exact):
//!   t3  = recv CQ completion observed for the frame
//!   t4  = Vulkan decode/upload done; render submission starting
//!   t5  = vkQueuePresentKHR returned (frame submitted to the compositor)
//!   tp  = compositor's on-screen timestamp (wp_presentation, client clock)
//!
//!   upload  = t4 - t3      upload call
//!   present = t5 - t4      draw + swap submission
//!   display = tp - t5      compositor queue + scanout (0 = feedback pending)
//!
//! Network latency is measured independently on each side — the server
//! brackets its own WRITE (t2 -> t2c) and the client measures a ping/pong
//! round trip entirely on its own clock (one-way estimate = rtt/2). No
//! network number subtracts timestamps taken on different machines.
//!
//! End to end is assembled from those per-host pieces rather than from a
//! cross-host timestamp subtraction (so clock skew cannot poison it):
//!
//!   e2e = (t2c - t0) + (client_end - t3) + net
//!       = queue+slot+material+post+transfer+net + upload+present+display
//!
//! where `net` is 0 for RDMA (its `transfer` = t2c-t2 is the sender's WRITE
//! completion, i.e. "written on the client", so the transit is already in it)
//! and the ping/pong min-RTT one-way estimate for TCP (whose `transfer` is
//! only the local socket write). The only unmeasured sliver is the client's
//! CQE observation delay between the server's send completion and t3
//! (polling granularity, sub-ms). The min-RTT ping/pong offset is still
//! tracked for diagnostics, but no *same-host* stage depends on it. Known
//! honesty limit: without wp_presentation (`display n/a`) the client span
//! ends at SwapBuffers return, not scanout.
//!
//! Audio (`--audio`, the default) is the one stage that cannot be assembled
//! from same-host spans: the playout thread ages each packet against the
//! server's stamp using the ping/pong clock offset, so its e2e carries that
//! estimate's ±rtt/2 error. The age is split into `xfer` (the wire, carries the
//! offset error) and `ring` (the client's pre-sink backlog, client-clock only,
//! so offset-independent); `av` = audio_e2e − video_e2e is the A/V offset to
//! watch (positive = audio lags video). All inert unless metrics are on (see
//! `audio::stats`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use rdma_display_audio::stats::AudioStats;
use rdma_display_proto::{FrameStamp, MatKind, MAT_STAGES};
use rdma_display_rdma::ControlEvent;

/// Rolling sample window for medians/p95.
const WINDOW: usize = 64;
/// Max frames kept waiting for their server stamp (staleness guard).
const PENDING_CAP: usize = 256;
/// wp_presentation_feedback.presented flag bits (wayland presentation-time).
const PRES_VSYNC: u32 = 0x1;
const PRES_ZERO_COPY: u32 = 0x8;
/// Overlay geometry: 8x8 bitmap glyphs scaled by `FONT_SCALE` (2 = twice the
/// old UI in each dimension), `COLS` columns, `LINES` lines.
const FONT_SCALE: usize = 2;
const COLS: usize = 50;
/// 13 video/session lines + 3 audio lines (e2e + xfer/ring/sink + av).
const LINES: usize = 16;
pub const OVERLAY_W: u32 = ((COLS * 8 + 8) * FONT_SCALE) as u32;
pub const OVERLAY_H: u32 = ((LINES * 8 + 8) * FONT_SCALE) as u32;
/// Milliseconds between overlay text refreshes.
pub const REFRESH_MS: u64 = 250;
/// Seconds between clock-sync pings.
pub const PING_SECS: u64 = 2;
/// e2e above this prints an immediate stderr spike line ([metrics-spike]).
const SPIKE_E2E_US: u64 = 40_000;

#[derive(Clone, Copy)]
struct Sample {
    seq: u64,
    // --- server-side durations (same host, exact) ------------------------
    q: u64,     // t0d - t0   capture queue / pacing
    prep: u64,  // t0p - t0d  scanout import / materializer prepare
    sw: u64,    // t1s - t0p  staging-slot credit wait
    mat: u64,   // t1 - t1s   materialize / encode span
    /// per-codec breakdown of `mat` (µs) + the labels that belong to it
    mat_kind: MatKind,
    mat_us: [u64; MAT_STAGES],
    post: u64,  // t2 - t1    WQE build + post
    xfer: u64,  // t2c - t2   RDMA WRITE as the sender saw it
    // --- client-side durations (same host, exact) ------------------------
    t5: u64,    // swap return (client clock; anchor for `disp`)
    upl: u64,   // t4 - t3    upload call
    pres: u64,  // t5 - t4    draw + swap submission
    disp: u64,  // tp - t5    compositor queue + scanout (0 = unknown)
    // --- clock-free end to end -------------------------------------------
    e2e: u64, // (t2c - t0) + (client_end - t3) + net (TCP one-way)
    fb_id: Option<u64>,
}

/// One presented frame whose server stamp has not arrived yet.
#[derive(Clone, Copy)]
struct Local {
    seq: u64,
    t3: u64,
    t4: u64,
    t5: u64,
    fb_id: Option<u64>,
    /// presentation timestamp that beat the server stamp here
    presented: Option<u64>,
}

pub struct Metrics {
    samples: VecDeque<Sample>,
    /// presented frames awaiting their server stamp: see [`Local`]
    local: VecDeque<Local>,
    /// server stamps that arrived before their frame was presented
    stamps: HashMap<u64, FrameStamp>,
    /// presentation timestamps whose frame was already matched: fb_id -> us
    presented: HashMap<u64, u64>,
    /// bytes actually received since the last refresh (the real wire transfer,
    /// summed over every received frame — superseded ones included)
    rx_bytes_since_refresh: u64,
    last_seq: Option<u64>,
    skipped: u64,
    frames_since_refresh: u64,
    last_refresh: Option<std::time::Instant>,
    last_ping: Option<std::time::Instant>,
    clock: ClockSync,
    /// audio playout timing (None when audio is off or metrics are off)
    audio: Option<Arc<AudioStats>>,
    /// whether the presenter bound wp_presentation (display stage meaningful)
    onscreen: bool,
    /// TCP transport: `xfer` does not include the network transit, so the
    /// min-RTT one-way estimate is added to `e2e`
    tcp: bool,
    /// presentation-feedback flags over the current window
    pres_count: u64,
    zero_copy_count: u64,
    vsync_count: u64,
    /// min/max refresh interval (ns) the compositor reported this window
    refresh_min: u32,
    refresh_max: u32,
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

    /// Smallest RTT seen recently (the tightest bound on true propagation and
    /// the basis of the offset estimate: ±rtt/2).
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
            presented: HashMap::new(),
            rx_bytes_since_refresh: 0,
            last_seq: None,
            skipped: 0,
            frames_since_refresh: 0,
            last_refresh: None,
            last_ping: None,
            clock: ClockSync::new(),
            audio: None,
            onscreen: false,
            tcp: false,
            pres_count: 0,
            zero_copy_count: 0,
            vsync_count: 0,
            refresh_min: u32::MAX,
            refresh_max: 0,
            text: String::new(),
        }
    }

    /// Tell the metrics whether the video transport is TCP, so the network
    /// one-way estimate is added to the end-to-end total (RDMA's `transfer`
    /// already includes the transit).
    pub fn set_tcp(&mut self, tcp: bool) {
        self.tcp = tcp;
    }

    /// Attach the audio playout stats. The metrics pushes the clock offset into
    /// them (on every pong) and folds their e2e into the overlay.
    pub fn set_audio_stats(&mut self, stats: Arc<AudioStats>) {
        self.audio = Some(stats);
    }

    /// Count one received frame's bytes — the actual bytes the NIC/TCP moved
    /// for it (the coded length, not the uncompressed frame size). Superseded
    /// frames count too: they crossed the wire even though they weren't shown.
    pub fn add_rx_bytes(&mut self, bytes: usize) {
        self.rx_bytes_since_refresh = self.rx_bytes_since_refresh.saturating_add(bytes as u64);
    }

    /// Tell the overlay whether wp_presentation is bound, so the `display`
    /// stage reads a real number instead of a misleading 0.00.
    pub fn set_onscreen(&mut self, supported: bool) {
        self.onscreen = supported;
    }

    /// One presented frame: the presenter loop stamps t3 when the RDMA
    /// completion is observed and t4/t5 inside the GL calls; `fb_id` is the
    /// wp_presentation feedback requested for this commit. The sample is only
    /// counted once the server's stamp for `seq` arrives.
    pub fn record(&mut self, seq: u64, t3: u64, t4: u64, t5: u64, fb_id: Option<u64>) {
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
        self.local.push_back(Local {
            seq,
            t3,
            t4,
            t5,
            fb_id,
            presented: None,
        });
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
            ControlEvent::Pong { c0, s1, c2 } => {
                self.clock.add(c0, s1, c2);
                // keep the playout thread's clock estimate fresh, so its
                // per-packet aging uses the tightest offset we have
                if let (Some(a), Some(off)) = (&self.audio, self.clock.offset()) {
                    a.set_clock_offset(off);
                }
            }
            // routed to the gamepad rumble player by the caller before this
            ControlEvent::Rumble(_) => {}
            // handled by the presenter (window/meta apply), not metrics
            ControlEvent::Meta(_) => {}
            // the pre-handshake mode result is consumed by connect_with
            ControlEvent::ModeReply(_) => {}
        }
    }

    /// Attach a wp_presentation result to its frame: the frame may still be
    /// waiting for its server stamp (stash it) or already be matched (update
    /// the `display` stage and the end-to-end total in place).
    pub fn on_presented(&mut self, fb_id: u64, unix_us: u64, presented: bool, flags: u32, refresh: u32) {
        if !presented {
            // discarded (superseded / not shown): leave `display` unknown
            return;
        }
        // wp_presentation_feedback flags: direct scanout (zero-copy) and
        // vsync. Aggregated per window so the overlay can show whether the
        // compositor is scanning out directly (and whether VRR is dropping
        // vsync), independent of the stamp matching below.
        self.pres_count += 1;
        if flags & PRES_ZERO_COPY != 0 {
            self.zero_copy_count += 1;
        }
        if flags & PRES_VSYNC != 0 {
            self.vsync_count += 1;
        }
        // the refresh interval the compositor used; it varies under VRR
        if refresh > 0 {
            self.refresh_min = self.refresh_min.min(refresh);
            self.refresh_max = self.refresh_max.max(refresh);
        }
        if let Some(l) = self.local.iter_mut().find(|l| l.fb_id == Some(fb_id)) {
            l.presented = Some(unix_us);
            return;
        }
        if let Some(s) = self.samples.iter_mut().find(|s| s.fb_id == Some(fb_id)) {
            set_disp(s, unix_us);
            return;
        }
        if self.presented.len() >= PENDING_CAP {
            self.presented.clear();
        }
        self.presented.insert(fb_id, unix_us);
    }

    /// Emit samples whose stamp arrived. Frames are presented in order and
    /// stamps arrive in order, so the oldest unmatched local entry decides.
    fn try_match(&mut self) {
        while let Some(front) = self.local.front().copied() {
            let Some(s) = self.stamps.remove(&front.seq) else {
                break;
            };
            self.local.pop_front();
            // on-screen time may have beaten or trailed the server stamp
            let tp = front
                .presented
                .or_else(|| front.fb_id.and_then(|id| self.presented.remove(&id)));
            // Server span (t2c - t0) and client span (client_end - t3) are
            // each same-host differences: no cross-clock subtraction.
            let q = s.t0d.saturating_sub(s.t0);
            let prep = s.t0p.saturating_sub(s.t0d);
            let sw = s.t1s.saturating_sub(s.t0p);
            let mat = s.t1.saturating_sub(s.t1s);
            let post = s.t2.saturating_sub(s.t1);
            let xfer = s.t2c.saturating_sub(s.t2);
            let upl = front.t4.saturating_sub(front.t3);
            let pres = front.t5.saturating_sub(front.t4);
            let disp = tp.map(|us| us.saturating_sub(front.t5)).unwrap_or(0);
            // TCP: `xfer` is only the local socket write, so add the ping/pong
            // one-way estimate to account for the network transit (RDMA's
            // `xfer` already covers it).
            let net = if self.tcp {
                self.clock.rtt_min().unwrap_or(0) / 2
            } else {
                0
            };
            let e2e = q + prep + sw + mat + post + xfer + net + upl + pres + disp;
            let sample = Sample {
                seq: front.seq,
                q,
                prep,
                sw,
                mat,
                mat_kind: s.mat.kind,
                mat_us: s.mat.us,
                post,
                xfer,
                t5: front.t5,
                upl,
                pres,
                disp,
                e2e,
                fb_id: front.fb_id,
            };
            // sanity filter: a garbage sample must not poison the medians
            if sample.e2e > 3_000_000 {
                continue;
            }
            if self.samples.len() == WINDOW {
                self.samples.pop_front();
            }
            self.samples.push_back(sample);
            if sample.e2e > SPIKE_E2E_US {
                eprintln!(
                    "[metrics-spike] seq={} end-to-end {:.2} ms | queue {:.2} prep {:.2} slot {:.2} material {:.2} ({}) | post {:.2} transfer {:.2} | upload {:.2} present {:.2} display {:.2}",
                    sample.seq,
                    sample.e2e as f64 / 1000.0,
                    sample.q as f64 / 1000.0,
                    sample.prep as f64 / 1000.0,
                    sample.sw as f64 / 1000.0,
                    sample.mat as f64 / 1000.0,
                    stage_str(sample.mat_kind, &sample.mat_us),
                    sample.post as f64 / 1000.0,
                    sample.xfer as f64 / 1000.0,
                    sample.upl as f64 / 1000.0,
                    sample.pres as f64 / 1000.0,
                    sample.disp as f64 / 1000.0,
                );
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
                // actual transferred bytes over the interval, not
                // frame_size × fps: for lossless/lossy the coded frame is much
                // smaller than the raw one.
                let mbps = self.rx_bytes_since_refresh as f64 * 8.0 / dt / 1e6;
                (fps, mbps)
            }
        };
        self.last_refresh = Some(now);
        self.frames_since_refresh = 0;
        self.rx_bytes_since_refresh = 0;

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
        let cap = |s: &Sample| s.q + s.prep + s.sw + s.mat;
        let (cap_m, cap_p) = stat(cap);
        let (q_m, _) = stat(|s| s.q);
        let (prep_m, _) = stat(|s| s.prep);
        let (sw_m, _) = stat(|s| s.sw);
        let (mat_m, _) = stat(|s| s.mat);
        // materialize breakdown: use the codec of the most recent sample and
        // take the median of each populated stage slot independently
        let kind = self
            .samples
            .iter()
            .next_back()
            .map(|s| s.mat_kind)
            .unwrap_or(MatKind::None);
        let labels = kind.labels();
        let mut stage_med = [0.0f64; MAT_STAGES];
        for (i, m) in stage_med.iter_mut().enumerate() {
            let mut v: Vec<u64> = self.samples.iter().map(|s| s.mat_us[i]).collect();
            if !v.is_empty() {
                v.sort_unstable();
                *m = v[v.len() / 2] as f64 / 1000.0;
            }
        }
        let send = |s: &Sample| s.post + s.xfer;
        let (send_m, send_p) = stat(send);
        let (po_m, _) = stat(|s| s.post);
        let (xf_m, _) = stat(|s| s.xfer);
        let cli = |s: &Sample| s.upl + s.pres + s.disp;
        let (cli_m, cli_p) = stat(cli);
        let (upl_m, _) = stat(|s| s.upl);
        let (pre_m, _) = stat(|s| s.pres);
        let (disp_m, _) = stat(|s| s.disp);
        // network: client-side ping/pong round trip (no cross-clock math);
        // the min-RTT sample is the tightest bound on the true propagation
        let rtt_min = self.clock.rtt_min().unwrap_or(0) as f64 / 1000.0;

        let mut t = String::with_capacity(COLS * LINES);
        t.push_str(&format!(
            "frame {:>6}   fps {:>5.1}   {:>6.1} Mbit/s",
            self.last_seq.unwrap_or(0) & 0x00ff_ffff,
            fps,
            mbps
        ));
        t.push_str(&format!(
            "\nend-to-end   p50 {:>6.2}   p95 {:>6.2} ms",
            e2e_m, e2e_p
        ));
        t.push_str(&format!(
            "\ncapture ({}) p50 {:>6.2}   p95 {:>6.2} ms",
            kind.name(),
            cap_m,
            cap_p
        ));
        t.push_str(&format!(
            "\n  queue {:>5.2}  prep {:>5.2}  slot {:>5.2}  mat {:>5.2}",
            q_m, prep_m, sw_m, mat_m
        ));
        // per-codec stage breakdown; always reserve two lines so the sections
        // below do not move between modes
        let mut line = String::from("  ");
        let mut n = 0usize;
        let mut emitted = 0usize;
        for i in 0..MAT_STAGES {
            if labels[i].is_empty() {
                continue;
            }
            line.push_str(&format!("{} {:>5.2}  ", labels[i], stage_med[i]));
            n += 1;
            if n == 3 {
                t.push('\n');
                t.push_str(line.trim_end());
                line = String::from("  ");
                n = 0;
                emitted += 1;
            }
        }
        if n > 0 {
            t.push('\n');
            t.push_str(line.trim_end());
            emitted += 1;
        }
        for _ in emitted..2 {
            t.push('\n');
        }
        t.push_str(&format!(
            "\nserver send  p50 {:>6.2}   p95 {:>6.2} ms",
            send_m, send_p
        ));
        t.push_str(&format!(
            "\n  post {:>5.2}  transfer {:>5.2}",
            po_m, xf_m
        ));
        t.push_str(&format!(
            "\nnetwork      rtt {:>5.2}  one-way {:>5.2} ms{}",
            rtt_min,
            rtt_min / 2.0,
            if self.tcp { " (e2e)" } else { "" }
        ));
        // presentation-feedback flags: zero-copy => the compositor scanned our
        // buffer out directly; vsync absent => VRR did not force a vblank; the
        // refresh range widens when VRR is actually varying the interval.
        let pc = self.pres_count;
        let zc = self.zero_copy_count;
        let vs = self.vsync_count;
        let pct = |n: u64| if pc > 0 { n as f64 * 100.0 / pc as f64 } else { 0.0 };
        let refstr = if self.refresh_max > 0 {
            format!(
                "{:.1}-{:.1}",
                self.refresh_min as f64 / 1e6,
                self.refresh_max as f64 / 1e6
            )
        } else {
            "n/a".to_string()
        };
        t.push_str(&format!(
            "\nscanout     zc {:>3.0}% vsync {:>3.0}% ref {} n={}",
            pct(zc),
            pct(vs),
            refstr,
            pc
        ));
        self.pres_count = 0;
        self.zero_copy_count = 0;
        self.vsync_count = 0;
        self.refresh_min = u32::MAX;
        self.refresh_max = 0;
        t.push_str(&format!(
            "\nclient       p50 {:>6.2}   p95 {:>6.2} ms",
            cli_m, cli_p
        ));
        let disp_field = if self.onscreen {
            format!("{:>5.2}", disp_m)
        } else {
            "  n/a".to_string()
        };
        t.push_str(&format!(
            "\n  upload {:>5.2}  present {:>5.2}  display {}",
            upl_m, pre_m, disp_field
        ));
        let clk = match self.clock.offset() {
            Some(o) => format!(
                "session      skipped {}   clock {:+} us",
                self.skipped, o
            ),
            None => format!("session      skipped {}   clock syncing", self.skipped),
        };
        t.push_str(&format!("\n{}", clk));
        // audio playout: e2e is "server packetize -> speaker" (see audio::stats),
        // split into the wire (`xfer`) and the client's pre-sink backlog
        // (`ring`); `av` = audio_e2e - video_e2e (positive = audio lags video).
        // Always reserve three lines so the raster size does not shift when
        // audio is off.
        if let Some(a) = &self.audio {
            let (am, ap) = a.e2e_p50_p95();
            let (aud_m, aud_p) = (am as f64 / 1000.0, ap as f64 / 1000.0);
            let (xm, rm) = a.split_p50();
            t.push_str(&format!(
                "\naudio        p50 {:>6.2}   p95 {:>6.2} ms",
                aud_m, aud_p
            ));
            t.push_str(&format!(
                "\n  xfer {:>5.2}  ring {:>5.2}  jit {:>2}  sink {:>5.2}",
                xm as f64 / 1000.0,
                rm as f64 / 1000.0,
                a.jitter_ms(),
                a.sink_us() as f64 / 1000.0
            ));
            t.push_str(&format!(
                "\n  av {:+7.2} ms  align {}  drops {}  trim {}",
                aud_m - e2e_m,
                a.align_ms(),
                a.dropped(),
                a.trimmed()
            ));
        } else {
            t.push_str("\naudio        (off)");
            t.push_str("\n  xfer    n/a  ring    n/a  jit n/a  sink    n/a");
            t.push_str("\n  av     n/a ms  align 0  drops 0  trim 0");
        }
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

/// Fold a presentation timestamp into a sample's `display` stage and keep the
/// end-to-end total consistent.
fn set_disp(s: &mut Sample, presented: u64) {
    let d = presented.saturating_sub(s.t5);
    if d != s.disp {
        s.e2e = s.e2e.saturating_sub(s.disp).saturating_add(d);
        s.disp = d;
    }
}

/// Render a sample's materialize breakdown with its codec's labels, e.g.
/// `" activate 0.01 encode 2.07 ref 0.08"`. Used by the spike line.
fn stage_str(kind: MatKind, us: &[u64; MAT_STAGES]) -> String {
    let labels = kind.labels();
    let mut s = String::new();
    for i in 0..MAT_STAGES {
        if labels[i].is_empty() {
            continue;
        }
        s.push_str(&format!(" {} {:.2}", labels[i], us[i] as f64 / 1000.0));
    }
    s
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
/// translucent black background, white glyphs, 8px monospace grid scaled by
/// [`FONT_SCALE`].
pub fn rasterize(text: &str) -> Vec<u8> {
    let w = OVERLAY_W as usize;
    let h = OVERLAY_H as usize;
    let mut buf = vec![0u8; w * h * 4];
    // background
    for px in buf.chunks_exact_mut(4) {
        px[0] = 0;
        px[1] = 0;
        px[2] = 0;
        px[3] = 150;
    }
    let s = FONT_SCALE;
    for (row, line) in text.split('\n').take(LINES).enumerate() {
        for (col, ch) in line.chars().take(COLS).enumerate() {
            let glyph = &FONT8X8[(ch as usize).min(127)];
            let ox = (4 + col * 8) * s;
            let oy = (4 + row * 8) * s;
            for gy in 0..8 {
                let bits = glyph[gy];
                for gx in 0..8 {
                    if (bits >> gx) & 1 == 0 {
                        continue;
                    }
                    for sy in 0..s {
                        let y = oy + gy * s + sy;
                        if y >= h {
                            continue;
                        }
                        for sx in 0..s {
                            let x = ox + gx * s + sx;
                            if x >= w {
                                continue;
                            }
                            let o = (y * w + x) * 4;
                            buf[o] = 255;
                            buf[o + 1] = 255;
                            buf[o + 2] = 255;
                            buf[o + 3] = 255;
                        }
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rdma_display_proto::MatTimes;
    use rdma_display_rdma::ControlEvent;

    fn stamp(seq: u64) -> FrameStamp {
        FrameStamp {
            seq,
            t0: 1_000,
            t0d: 1_100,
            t0p: 1_100,
            t1s: 1_150,
            t1: 3_000,
            t2: 3_050,
            t2c: 5_050,
            mat: MatTimes::new(MatKind::Raw, [100, 1_500, 50, 0, 0, 0]),
            meta_gen: 0,
        }
    }

    /// server span = queue(100) + slot(50) + material(1850) + post(50)
    /// + transfer(2000); client span = upload(200) + present(300)
    const SERVER: u64 = 100 + 50 + 1850 + 50 + 2000;
    const CLIENT: u64 = 200 + 300;

    #[test]
    fn e2e_is_sum_of_same_host_spans() {
        let mut m = Metrics::new();
        m.record(1, 10_000, 10_200, 10_500, Some(7));
        m.on_control(ControlEvent::Stamp(stamp(1)));
        assert_eq!(m.samples.len(), 1);
        let s = m.samples.front().unwrap();
        assert_eq!(s.e2e, SERVER + CLIENT);
        assert_eq!(s.disp, 0);
        assert_eq!(s.t5, 10_500);
    }

    #[test]
    fn presented_before_stamp_is_folded_in() {
        let mut m = Metrics::new();
        m.record(1, 10_000, 10_200, 10_500, Some(7));
        m.on_presented(7, 10_900, true, 0, 0); // arrives before the server stamp
        m.on_control(ControlEvent::Stamp(stamp(1)));
        let s = m.samples.front().unwrap();
        assert_eq!(s.disp, 400);
        assert_eq!(s.e2e, SERVER + CLIENT + 400);
    }

    #[test]
    fn presented_after_match_updates_in_place() {
        let mut m = Metrics::new();
        m.record(1, 10_000, 10_200, 10_500, Some(7));
        m.on_control(ControlEvent::Stamp(stamp(1)));
        assert_eq!(m.samples.front().unwrap().e2e, SERVER + CLIENT);
        m.on_presented(7, 11_000, true, 0, 0);
        let s = m.samples.front().unwrap();
        assert_eq!(s.disp, 500);
        assert_eq!(s.e2e, SERVER + CLIENT + 500);
    }

    #[test]
    fn discarded_leaves_display_unknown() {
        let mut m = Metrics::new();
        m.record(1, 10_000, 10_200, 10_500, Some(7));
        m.on_presented(7, 0, false, 0, 0); // discarded: no timestamp
        m.on_control(ControlEvent::Stamp(stamp(1)));
        let s = m.samples.front().unwrap();
        assert_eq!(s.disp, 0);
        assert_eq!(s.e2e, SERVER + CLIENT);
    }

    /// Every rendered line must fit the overlay grid, or the UI truncates.
    #[test]
    fn overlay_lines_fit_columns() {
        let mut m = Metrics::new();
        m.set_onscreen(true);
        // worst-ish case: 3-digit millisecond stages
        for i in 0..4u64 {
            m.record(
                100 + i,
                1_000_000 + i,
                1_000_200 + i,
                1_000_500 + i,
                Some(i),
            );
            let mut s = stamp(100 + i);
            s.t0 = 1_000_000;
            s.t0d = 1_000_300;
            s.t0p = 1_000_320;
            s.t1s = 1_000_350;
            s.t1 = 1_002_500;
            s.t2 = 1_002_560;
            s.t2c = 1_005_600;
            m.on_presented(i, 1_001_100 + i, true, 0, 0);
            m.on_control(ControlEvent::Stamp(s));
        }
        let t = m.refresh().expect("text changed");
        for line in t.split('\n') {
            let n = line.chars().count();
            assert!(n <= COLS, "overlay line {n} > {COLS} cols: {line:?}");
        }
        assert_eq!(t.matches('\n').count(), LINES - 1);
    }

    /// The audio block renders the playout e2e and the A/V offset, and keeps
    /// the overlay within the grid.
    #[test]
    fn audio_lines_render_av_offset() {
        let mut m = Metrics::new();
        m.set_onscreen(true);
        let a = Arc::new(AudioStats::new());
        a.set_clock_offset(0);
        a.set_sink_us(20_000);
        // 15 ms wire + 15 ms ring + 20 ms sink => 50 ms audio e2e
        for i in 0..8u64 {
            a.record(1_000_000 + i, 1_015_000 + i, 1_030_000 + i);
        }
        a.set_dropped(21);
        a.set_trimmed(4);
        a.set_jitter_ms(15);
        a.set_align_ms(0);
        m.set_audio_stats(a);
        // one video frame: server span + client span = 4.55 ms
        m.record(1, 10_000, 10_200, 10_500, None);
        m.on_control(ControlEvent::Stamp(stamp(1)));
        let t = m.refresh().expect("text changed");
        assert!(t.contains("audio        p50"), "{t}");
        // the e2e split: 15 ms on the wire, 15 ms of pre-sink backlog
        assert!(t.contains("xfer 15.00"), "{t}");
        assert!(t.contains("ring 15.00"), "{t}");
        // the adaptive jitter target and the (default-off) alignment term
        assert!(t.contains("jit 15"), "{t}");
        assert!(t.contains("align 0"), "{t}");
        assert!(t.contains("trim 4"), "{t}");
        // audio (50.00) lags the video (4.55): a positive offset
        assert!(t.contains("av  +45.45"), "{t}");
        for line in t.split('\n') {
            assert!(line.chars().count() <= COLS, "line too wide: {line:?}");
        }
        assert_eq!(t.matches('\n').count(), LINES - 1);
    }

    /// The materialize breakdown must use the codec's own stage labels, not a
    /// fixed raw-oriented set (lossless used to render its whole GPU encode
    /// under "blit" with "readback 0.00").
    #[test]
    fn breakdown_is_labeled_per_codec() {
        let mut m = Metrics::new();
        m.record(1, 10_000, 10_200, 10_500, Some(7));
        let mut s = stamp(1);
        s.mat = MatTimes::new(MatKind::Lossless, [11, 0, 2069, 77, 2, 23]);
        m.on_control(ControlEvent::Stamp(s));
        let t = m.refresh().expect("text changed");
        assert!(t.contains("capture (lossless)"), "{t}");
        for want in ["activate", "encode", "ref", "copy", "restore"] {
            assert!(t.contains(want), "missing {want:?} in:\n{t}");
        }
        assert!(!t.contains("readback"), "raw label leaked into lossless:\n{t}");
    }

    /// The raw materializer keeps its blit/readback/sync split.
    #[test]
    fn raw_breakdown_is_labeled() {
        let mut m = Metrics::new();
        m.record(1, 10_000, 10_200, 10_500, Some(7));
        m.on_control(ControlEvent::Stamp(stamp(1)));
        let t = m.refresh().expect("text changed");
        assert!(t.contains("capture (raw)"), "{t}");
        for want in ["blit", "readback", "sync"] {
            assert!(t.contains(want), "missing {want:?} in:\n{t}");
        }
    }
}
