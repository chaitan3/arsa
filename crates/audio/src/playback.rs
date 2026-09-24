//! Client side: connect to the audio side-channel and play it out.
//!
//! Two threads per session:
//!   * a socket READER that owns the TCP stream and pushes each decoded PCM
//!     packet into a bounded ring. It never blocks on the audio sink, so the
//!     server's sender never backpressures and its mailbox never drops (which
//!     is what produced the `seq gap` lines);
//!   * the audio thread (here) that pops the ring and does the blocking
//!     `pa_simple_write`. The pulse buffer (`tlength`/`prebuf` =
//!     `--audio-buffer-ms`) remains the jitter buffer; the ring only absorbs
//!     the bursty write-blocking and keeps TCP drained.

use std::collections::VecDeque;
use std::io::Read;
use std::net::TcpStream;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::pulse::{Pulse, SampleSpec};
use super::wire;

/// How much audio the ring holds before it starts dropping the oldest packet.
/// This is a safety valve for a stalled sink, not the jitter buffer (that is
/// the pulse `tlength`); it only has to cover the bursty blocking of
/// `pa_simple_write`, so a few hundred ms is plenty.
const RING_TARGET_MS: u64 = 200;

fn ring_cap() -> usize {
    let packet_ms = (wire::PACKET_FRAMES as u64 * 1000 / wire::RATE as u64).max(1);
    (RING_TARGET_MS / packet_ms).max(8) as usize
}

/// Bounded packet queue shared between the socket reader and the audio thread.
struct Ring {
    q: VecDeque<Vec<u8>>,
    /// Set by the reader when the connection ends; the error to surface.
    closed: Option<String>,
    dropped: u64,
}

impl Ring {
    fn new() -> Self {
        Ring { q: VecDeque::new(), closed: None, dropped: 0 }
    }

    /// Push a packet, dropping the oldest when full. Returns true if a packet
    /// was dropped, so the caller can log/count it.
    fn push(&mut self, pkt: Vec<u8>) -> bool {
        let dropped = self.q.len() >= ring_cap();
        if dropped {
            self.q.pop_front();
            self.dropped += 1;
        }
        self.q.push_back(pkt);
        dropped
    }

    fn pop(&mut self) -> Option<Vec<u8>> {
        self.q.pop_front()
    }

    fn close(&mut self, reason: String) {
        if self.closed.is_none() {
            self.closed = Some(reason);
        }
    }
}

/// Retry loop wrapper — an audio session failure never takes down video;
/// we log once per failure and reconnect.
pub fn run(addr: std::net::SocketAddr, buffer_ms: u32) {
    loop {
        match session(addr, buffer_ms) {
            Ok(()) => unreachable!("audio session loop returned"),
            Err(e) => {
                eprintln!("[audio] session ended: {e:#}; reconnecting in 2 s");
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
fn reader(mut sock: TcpStream, shared: Arc<(Mutex<Ring>, Condvar)>) {
    let mut hdr = [0u8; wire::PACKET_HDR_LEN];
    let mut last_seq: Option<u32> = None;
    loop {
        match sock.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                eprintln!("[audio] no packets for 10 s (silent capture — nothing playing into the server's sink?); still waiting");
                continue;
            }
            Err(e) => return close(&shared, format!("audio: read packet header: {e}")),
        }
        let (seq, _host_us, frames) = wire::decode_packet_header(&hdr);
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
                eprintln!("[audio] seq gap {} -> {}", prev, seq);
            }
        }
        last_seq = Some(seq);

        let (lock, cv) = &*shared;
        let mut g = lock.lock().unwrap();
        if g.push(payload) {
            let d = g.dropped;
            if d == 1 || d % 1000 == 0 {
                eprintln!("[audio] playback ring overflow: dropped {d} packets");
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

fn session(addr: std::net::SocketAddr, buffer_ms: u32) -> Result<()> {
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
    eprintln!(
        "[audio] playing (f32le {} Hz, {}ch, {} ms buffer via the default sink)",
        wire::RATE,
        wire::CHANNELS,
        buffer_ms
    );

    let shared = Arc::new((Mutex::new(Ring::new()), Condvar::new()));
    let reader_shared = shared.clone();
    let reader_sock = sock.try_clone().context("clone audio socket for the reader")?;
    std::thread::Builder::new()
        .name("audio-reader".into())
        .spawn(move || reader(reader_sock, reader_shared))
        .context("spawn audio reader thread")?;
    // from here the audio thread owns `sock` only to shut it down on exit
    let _sock_guard = SocketGuard(sock);

    let mut next_report = Instant::now() + Duration::from_secs(5);
    loop {
        let payload = {
            let (lock, cv) = &*shared;
            let mut g = lock.lock().unwrap();
            loop {
                if let Some(p) = g.pop() {
                    break p;
                }
                if let Some(reason) = g.closed.clone() {
                    return Err(anyhow::anyhow!(reason));
                }
                // 10 s of silence is a warning, not a failure
                let (g2, _) = cv
                    .wait_timeout(g, Duration::from_secs(10))
                    .unwrap_or_else(|e| e.into_inner());
                g = g2;
            }
        };
        stream.write_all(&payload)?;
        if Instant::now() > next_report {
            next_report += Duration::from_secs(5);
            let drops = {
                let (lock, _) = &*shared;
                lock.lock().unwrap().dropped
            };
            match stream.latency_us() {
                Ok(us) => {
                    // drift guard: if backlog exceeds the buffer by a wide
                    // margin (clock skew between the two pulse streams),
                    // flush and resync rather than accumulating latency
                    let backlog_us = us as i64 - (buffer_ms as i64) * 1000;
                    if backlog_us > 150_000 {
                        eprintln!("[audio] playout latency {} ms > buffer, flushing to resync (ring drops {drops})", us / 1000);
                        stream.flush()?;
                    } else {
                        eprintln!("[audio] playout latency ~{} ms (ring drops {drops})", us / 1000);
                    }
                }
                Err(e) => eprintln!("[audio] latency query failed: {e:#}"),
            }
        }
    }
}
