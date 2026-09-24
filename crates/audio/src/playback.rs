//! Client side: connect to the audio side-channel and play it out.
//!
//! One thread per session: TCP recv → pa_simple_write. The pulse
//! server-side buffer (`tlength`/`prebuf` = wire::BUFFER_MS) IS the jitter
//! buffer: playout starts once it is full, after every underrun it
//! re-gates, and network backlog beyond ~150 ms is flushed (log + resync)
//! so drift can never accumulate. No ring, no second thread.

use std::io::Read;
use std::net::TcpStream;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use super::pulse::{Pulse, SampleSpec};
use super::wire;

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

fn session(addr: std::net::SocketAddr, buffer_ms: u32) -> Result<()> {
    let mut sock = TcpStream::connect(addr).with_context(|| format!("audio connect {addr}"))?;
    sock.set_nodelay(true)?;
    // idle tolerance: a silent monitor delivers nothing for as long as the
    // game is quiet — the server keeps the stream open and waits, so read
    // timeouts here are warnings, not session-killers (a mid-frame timeout
    // can only happen after packets already stopped, where reconnecting
    // via the normal path is the fallback)
    sock.set_read_timeout(Some(Duration::from_secs(10)))?;
    use std::io::Write;
    sock.write_all(&[wire::TAG_AUDIO])
        .context("audio: send tag")?;

    let mut fmt = [0u8; wire::FORMAT_LEN];
    sock.read_exact(&mut fmt).context("audio: read format record")?;
    wire::decode_format(&fmt)?;

    let pulse = Pulse::dlopen()?;
    let stream = pulse.open_playback(
        None,
        SampleSpec { rate: wire::RATE, channels: wire::CHANNELS },
        buffer_ms,
    )?;
    eprintln!(
        "[audio] playing (f32le {} Hz, {}ch, {} ms buffer via the default sink)",
        wire::RATE,
        wire::CHANNELS,
        buffer_ms
    );

    let mut hdr = [0u8; wire::PACKET_HDR_LEN];
    let mut next_report = Instant::now() + Duration::from_secs(5);
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
            Err(e) => return Err(e).context("audio: read packet header"),
        }
        let (seq, _host_us, frames) = wire::decode_packet_header(&hdr);
        wire::frames_plausible(frames)?;
        let mut payload = vec![0u8; wire::payload_len(frames)?];
        match sock.read_exact(&mut payload) {
            Ok(()) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                bail!("audio: payload read timed out mid-frame (framing desync; reconnecting)")
            }
            Err(e) => return Err(e).context("audio: read packet payload"),
        }
        stream.write_all(&payload)?;
        if let Some(prev) = last_seq {
            if seq != prev.wrapping_add(1) {
                eprintln!("[audio] seq gap {} -> {}", prev, seq);
            }
        }
        last_seq = Some(seq);
        if Instant::now() > next_report {
            next_report += Duration::from_secs(5);
            match stream.latency_us() {
                Ok(us) => {
                    // drift guard: if backlog exceeds the buffer by a wide
                    // margin (clock skew between the two pulse streams),
                    // flush and resync rather than accumulating latency
                    let backlog_us = us as i64 - (buffer_ms as i64) * 1000;
                    if backlog_us > 150_000 {
                        eprintln!("[audio] playout latency {} ms > buffer, flushing to resync", us / 1000);
                        stream.flush()?;
                    } else {
                        eprintln!("[audio] playout latency ~{} ms", us / 1000);
                    }
                }
                Err(e) => eprintln!("[audio] latency query failed: {e:#}"),
            }
        }
    }
}
