//! Client side: connect to the audio side-channel and play it out.
//!
//! Two threads per session:
//!   * a socket READER that owns the TCP stream and pushes each decoded PCM
//!     packet into a bounded ring. It never blocks on the audio sink, so the
//!     server's sender never backpressures and its mailbox never drops (which
//!     is what produced the `seq gap` lines);
//!   * the audio thread (here) that pops the ring and does the blocking
//!     `pa_simple_write`. The ring *is* the jitter buffer: its depth adapts to
//!     the measured arrival jitter ([`Ring::observe`]), and it also keeps TCP
//!     drained while the write blocks. The pulse buffer (`tlength`/`prebuf` =
//!     `--audio-buffer-ms`) stays minimal.
//!
//! Under `--metrics` an `Arc<AudioStats>` is threaded through: the reader keeps
//! each packet's server stamp and the client clock at receive, and the audio
//! thread records the packet's age when it reaches the sink (see
//! [`super::stats`]), split into wire (`transport`) and pre-sink backlog
//! (`ring`). The receive stamp is taken either way — the jitter estimate needs
//! it — but the per-packet `record` is skipped when metrics are off.

use std::collections::VecDeque;
use std::io::Read;
use std::net::TcpStream;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::pulse::{Pulse, SampleSpec};
use super::stats::AudioStats;
use super::wire;

/// The ring is the client's one jitter buffer (the pulse `tlength` stays one
/// packet), so its depth adapts to the measured arrival jitter: after each
/// window the target becomes `jitter peak-to-peak + margin`, clamped into this
/// band. The floor keeps a couple of packets of headroom even on a perfect
/// link; the cap is the default latency ceiling the buffer may add
/// (`--audio-target-clamp-ms` overrides it).
const RING_JITTER_FLOOR_MS: u64 = 10;
pub const RING_JITTER_CAP_MS: u64 = 20;
const RING_JITTER_MARGIN_MS: u64 = 5;
/// Packets per jitter-estimate window (~2.6 s at 200 pkt/s).
const JITTER_WINDOW: u32 = 512;
/// Only trim once the backlog exceeds the target + this slack, so ordinary
/// jitter does not cause repeated trims.
const RING_SLACK_MS: u64 = 20;
/// Hard cap: at this depth the reader starts dropping the oldest packet. It is
/// a stall guard, never the steady state.
const RING_MAX_MS: u64 = 200;

fn packets_for(ms: u64) -> usize {
    let packet_ms = (wire::PACKET_FRAMES as u64 * 1000 / wire::RATE as u64).max(1);
    (ms / packet_ms).max(1) as usize
}

fn ring_cap() -> usize {
    packets_for(RING_MAX_MS).max(8)
}

/// Bounded packet queue shared between the socket reader and the audio thread.
/// Each entry is `(server packetize stamp, client receive stamp, PCM payload)`;
/// the receive stamp also feeds the jitter estimate, and the pair feeds
/// `--metrics`.
struct Ring {
    q: VecDeque<(u64, u64, Vec<u8>)>,
    /// Set by the reader when the connection ends; the error to surface.
    closed: Option<String>,
    dropped: u64,
    trimmed: u64,
    /// manual A/V alignment (ms): extra audio delay on top of the jitter target
    align_ms: u64,
    /// upper clamp on the adaptive jitter target (ms)
    clamp_ms: u64,
    /// jitter estimate: running range of (recv_client - host) over the window
    jitter_lo: i64,
    jitter_hi: i64,
    jitter_n: u32,
    /// adaptive jitter target (ms), recomputed at each window boundary
    jitter_ms: u64,
}

impl Ring {
    fn new(align_ms: u64, clamp_ms: u64) -> Self {
        let clamp_ms = clamp_ms.max(RING_JITTER_FLOOR_MS);
        Ring {
            q: VecDeque::new(),
            closed: None,
            dropped: 0,
            trimmed: 0,
            align_ms,
            clamp_ms,
            jitter_lo: i64::MAX,
            jitter_hi: i64::MIN,
            jitter_n: 0,
            // start at the cap and adapt down, so the first window is never
            // shallower than the link turns out to need
            jitter_ms: clamp_ms,
        }
    }

    /// Fold one arrival into the jitter estimate. `transit` is
    /// `recv_client - host_unix`: its *range* over the window is the arrival
    /// jitter (the constant clock offset cancels), which is exactly the depth
    /// the buffer must cover. On a full window, recompute the target.
    fn observe(&mut self, transit: i64) {
        self.jitter_lo = self.jitter_lo.min(transit);
        self.jitter_hi = self.jitter_hi.max(transit);
        self.jitter_n += 1;
        if self.jitter_n >= JITTER_WINDOW {
            let p2p_ms = (self.jitter_hi - self.jitter_lo).max(0) as u64 / 1000;
            self.jitter_ms = (p2p_ms + RING_JITTER_MARGIN_MS)
                .clamp(RING_JITTER_FLOOR_MS, self.clamp_ms);
            self.jitter_lo = i64::MAX;
            self.jitter_hi = i64::MIN;
            self.jitter_n = 0;
        }
    }

    /// Ring depth to converge to: the adaptive jitter target plus the manual
    /// alignment delay.
    fn target(&self) -> usize {
        packets_for(self.jitter_ms + self.align_ms).max(2)
    }

    /// Push a packet, dropping the oldest when full. Returns true if a packet
    /// was dropped, so the caller can log/count it.
    fn push(&mut self, host_unix_us: u64, recv_us: u64, pkt: Vec<u8>) -> bool {
        self.observe(recv_us as i64 - host_unix_us as i64);
        let dropped = self.q.len() >= ring_cap();
        if dropped {
            self.q.pop_front();
            self.dropped += 1;
        }
        self.q.push_back((host_unix_us, recv_us, pkt));
        dropped
    }

    fn pop(&mut self) -> Option<(u64, u64, Vec<u8>)> {
        self.q.pop_front()
    }

    /// Converge a standing backlog back to the target: once the queue is deeper
    /// than target + slack, drop the oldest packets down to the target. The
    /// playout owns the consumer side, so this is where the offset accumulated
    /// by a startup burst or a stall is paid back. Returns how many were
    /// trimmed (0 in steady state).
    fn trim(&mut self) -> u64 {
        let limit = self.target() + packets_for(RING_SLACK_MS);
        if self.q.len() <= limit {
            return 0;
        }
        let target = self.target();
        let mut n = 0;
        while self.q.len() > target {
            self.q.pop_front();
            self.trimmed += 1;
            n += 1;
        }
        n
    }

    fn close(&mut self, reason: String) {
        if self.closed.is_none() {
            self.closed = Some(reason);
        }
    }
}

/// Retry loop wrapper — an audio session failure never takes down video;
/// we log once per failure and reconnect. `stats` is `Some` only under
/// `--metrics`.
pub fn run(
    addr: std::net::SocketAddr,
    buffer_ms: u32,
    align_ms: u32,
    clamp_ms: u32,
    stats: Option<Arc<AudioStats>>,
) {
    loop {
        match session(addr, buffer_ms, align_ms, clamp_ms, stats.clone()) {
            Ok(()) => unreachable!("audio session loop returned"),
            Err(e) => {
                log::info!("[audio] session ended: {e:#}; reconnecting in 2 s");
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Mark the ring closed and wake the audio thread.
fn close(shared: &Arc<(Mutex<Ring>, Condvar)>, reason: String) {
    let (lock, cv) = &**shared;
    lock.lock().unwrap().close(reason);
    cv.notify_one();
}

/// Socket reader: read header + payload, check seq order, push to the ring.
/// It never touches pulse, so it keeps draining TCP even while the audio thread
/// is blocked inside `pa_simple_write`.
fn reader(
    mut sock: TcpStream,
    shared: Arc<(Mutex<Ring>, Condvar)>,
    stats: Option<Arc<AudioStats>>,
) {
    let mut hdr = [0u8; wire::PACKET_HDR_LEN];
    let mut last_seq: Option<u32> = None;
    loop {
        match sock.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                log::warn!("[audio] no packets for 10 s (silent capture — nothing playing into the server's sink?); still waiting");
                continue;
            }
            Err(e) => return close(&shared, format!("audio: read packet header: {e}")),
        }
        let (seq, host_us, frames) = wire::decode_packet_header(&hdr);
        if let Err(e) = wire::frames_plausible(frames) {
            return close(&shared, format!("{e:#} (framing desync; reconnecting)"));
        }
        let mut payload = vec![0u8; wire::payload_len(frames).unwrap_or(0)];
        match sock.read_exact(&mut payload) {
            Ok(()) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return close(
                    &shared,
                    "audio: payload read timed out mid-frame (framing desync; reconnecting)".into(),
                );
            }
            Err(e) => return close(&shared, format!("audio: read packet payload: {e}")),
        }
        if let Some(prev) = last_seq {
            if seq != prev.wrapping_add(1) {
                log::debug!("[audio] seq gap {} -> {}", prev, seq);
            }
        }
        last_seq = Some(seq);

        // client receive stamp: feeds the adaptive jitter estimate (always) and
        // the --metrics e2e split
        let recv_us = wire::unix_now_us();

        let (lock, cv) = &*shared;
        let mut g = lock.lock().unwrap();
        if g.push(host_us, recv_us, payload) {
            let d = g.dropped;
            if d == 1 || d % 1000 == 0 {
                log::debug!("[audio] playback ring overflow: dropped {d} packets");
            }
            if let Some(s) = &stats {
                s.set_dropped(d);
            }
        }
        cv.notify_one();
    }
}

/// Shut the socket down when the audio session ends, so the reader thread's
/// blocking read returns and it can exit (otherwise the server would keep the
/// old audio connection open and refuse the next one).
struct SocketGuard(TcpStream);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = self.0.shutdown(std::net::Shutdown::Both);
    }
}

fn session(
    addr: std::net::SocketAddr,
    buffer_ms: u32,
    align_ms: u32,
    clamp_ms: u32,
    stats: Option<Arc<AudioStats>>,
) -> Result<()> {
    let mut sock = TcpStream::connect(addr).with_context(|| format!("audio connect {addr}"))?;
    sock.set_nodelay(true)?;
    // idle tolerance: a silent monitor delivers nothing for as long as the
    // game is quiet — the server keeps the stream open and waits, so read
    // timeouts here are warnings, not session-killers (a mid-frame timeout can
    // only happen after packets already stopped, where reconnecting via the
    // normal path is the fallback)
    sock.set_read_timeout(Some(Duration::from_secs(10)))?;
    use std::io::Write;
    sock.write_all(&[wire::TAG_AUDIO]).context("audio: send tag")?;

    let mut fmt = [0u8; wire::FORMAT_LEN];
    sock.read_exact(&mut fmt).context("audio: read format record")?;
    wire::decode_format(&fmt)?;

    let pulse = Pulse::dlopen()?;
    let stream = pulse.open_playback(
        None,
        SampleSpec { rate: wire::RATE, channels: wire::CHANNELS },
        buffer_ms,
        wire::PACKET_BYTES as u32,
    )?;
    log::info!(
        "[audio] playing (f32le {} Hz, {}ch, {} ms buffer via the default sink)",
        wire::RATE,
        wire::CHANNELS,
        buffer_ms
    );

    let shared = Arc::new((
        Mutex::new(Ring::new(align_ms as u64, clamp_ms as u64)),
        Condvar::new(),
    ));
    let reader_shared = shared.clone();
    let reader_sock = sock.try_clone().context("clone audio socket for the reader")?;
    let reader_stats = stats.clone();
    std::thread::Builder::new()
        .name("audio-reader".into())
        .spawn(move || reader(reader_sock, reader_shared, reader_stats))
        .context("spawn audio reader thread")?;
    // from here the audio thread owns `sock` only to shut it down on exit
    let _sock_guard = SocketGuard(sock);

    let mut next_report = Instant::now() + Duration::from_secs(5);
    // seed the sink latency so the first e2e window is not biased low (it is
    // otherwise only refreshed at the report interval below)
    if let Some(s) = &stats {
        if let Ok(us) = stream.latency_us() {
            s.set_sink_us(us);
        }
    }
    // The sink's own latency is fixed and can be large (a Bluetooth/USB sink, or
    // a pulse `tsched` buffer); only *growth* means the two clocks are drifting.
    // Track the running minimum as the baseline, so the guard fires on real
    // drift instead of on a constant sink latency.
    let mut latency_floor_us: u64 = 0; // 0 = not measured yet
    loop {
        let (host_us, recv_us, payload) = {
            let (lock, cv) = &*shared;
            let mut g = lock.lock().unwrap();
            loop {
                if let Some(reason) = g.closed.clone() {
                    return Err(anyhow::anyhow!(reason));
                }
                // converge a standing backlog (startup burst / stall) back to
                // the ring target before taking the next packet
                let n = g.trim();
                if n > 0 {
                    let (d, t) = (g.dropped, g.trimmed);
                    log::debug!(
                        "[audio] trimmed {n} packets to converge the ring \
                         ({t} trimmed, {d} overflow-dropped)"
                    );
                    if let Some(s) = &stats {
                        s.set_trimmed(t);
                    }
                }
                if let Some(p) = g.pop() {
                    break p;
                }
                // 10 s of silence is a warning, not a failure
                let (g2, _) = cv
                    .wait_timeout(g, Duration::from_secs(10))
                    .unwrap_or_else(|e| e.into_inner());
                g = g2;
            }
        };
        stream.write_all(&payload)?;
        if let Some(s) = &stats {
            s.record(host_us, recv_us, wire::unix_now_us());
        }
        if Instant::now() > next_report {
            next_report += Duration::from_secs(5);
            let (drops, trimmed, jitter_ms, align_ms, clamp_ms) = {
                let (lock, _) = &*shared;
                let g = lock.lock().unwrap();
                (g.dropped, g.trimmed, g.jitter_ms, g.align_ms, g.clamp_ms)
            };
            if let Some(s) = &stats {
                s.set_dropped(drops);
                s.set_trimmed(trimmed);
                s.set_jitter_ms(jitter_ms);
                s.set_align_ms(align_ms);
            }
            match stream.latency_us() {
                Ok(us) => {
                    if let Some(s) = &stats {
                        s.set_sink_us(us);
                    }
                    // drift guard: the playout latency should sit at the sink's
                    // own floor. If it grows well above the floor, the two pulse
                    // streams' clocks are drifting apart — flush and re-baseline
                    // rather than accumulating latency.
                    if latency_floor_us == 0 || us < latency_floor_us {
                        latency_floor_us = us;
                    }
                    let drift_us = us.saturating_sub(latency_floor_us);
                    if drift_us > 150_000 {
                        log::debug!(
                            "[audio] playout latency {} ms, {} ms above the {}-ms floor, \
                             flushing to resync (ring jitter {jitter_ms}ms clamp {clamp_ms}ms \
                             align {align_ms}ms drops {drops} trims {trimmed})",
                            us / 1000,
                            drift_us / 1000,
                            latency_floor_us / 1000
                        );
                        stream.flush()?;
                        latency_floor_us = 0; // re-baseline after the flush
                    } else {
                        log::debug!(
                            "[audio] playout latency ~{} ms ({} ms above the {}-ms floor; \
                             ring jitter {jitter_ms}ms clamp {clamp_ms}ms align {align_ms}ms drops {drops} trims {trimmed})",
                            us / 1000,
                            drift_us / 1000,
                            latency_floor_us / 1000
                        );
                    }
                }
                Err(e) => log::debug!("[audio] latency query failed: {e:#}"),
            }
        }
    }
}
