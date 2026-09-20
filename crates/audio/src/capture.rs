//! Server side: keep a pulse record stream alive against a sink monitor
//! and serve it to audio connections.
//!
//! Threading model (plain threads, house rule):
//! - supervisor thread: (re)opens the record stream, reads fixed 10 ms
//!   chunks, timestamps them, pushes packets into a latest-wins mailbox.
//!   Stream death (device vanished, PW restart) → log + reopen every 2 s.
//! - per-connection sender thread: format record + packet loop off the
//!   same mailbox. Socket errors kill only that connection; the client
//!   reconnects on its own schedule.
//!
//! The mailbox is the realtime valve: while nobody consumes (or the client
//! stalls), newer packets overwrite older ones — audio never queues up and
//! never blocks the capture thread beyond one packet.

use std::io::Write;
use std::net::TcpStream;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};

use super::pulse::{Pulse, SampleSpec};
use super::wire::{self, AudioPacket};

/// Drop-oldest mailbox: `put` overwrites; `take_wait` blocks until the
/// next packet.
#[derive(Clone)]
pub struct Mailbox {
    inner: Arc<(Mutex<Option<wire::AudioPacket>>, Condvar)>,
}

impl Mailbox {
    fn new() -> Self {
        Mailbox {
            inner: Arc::new((Mutex::new(None), Condvar::new())),
        }
    }

    #[allow(dead_code)] // kept for future multi-consumer use
    fn put(&self, pkt: AudioPacket) {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        *g = Some(pkt);
        cv.notify_one();
    }

    /// drop-oldest: returns true if a previous packet was overwritten
    fn put_overwrites(&self, pkt: AudioPacket) -> bool {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        let replaced = g.is_some();
        *g = Some(pkt);
        cv.notify_one();
        replaced
    }

    fn take_wait(&self, timeout: Duration) -> Option<AudioPacket> {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(p) = g.take() {
                return Some(p);
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            let (g2, _res) = cv
                .wait_timeout(g, deadline - now)
                .unwrap_or_else(|e| e.into_inner());
            g = g2;
        }
    }
}

/// Start the capture supervisor thread. Fails fast on a missing libpulse
/// (deployment error) but tolerates a missing/vanished device in the
/// background (log + reopen loop). `dev`: source name — "<sink>.monitor"
/// for a specific sink, "@DEFAULT_MONITOR@" for the default sink's mix.
pub fn start_capture(dev: Option<&str>) -> Result<Mailbox> {
    let dev = dev.unwrap_or("@DEFAULT_MONITOR@").to_string();
    let pulse = Pulse::dlopen()?;
    let mbox = Mailbox::new();
    let m2 = mbox.clone();
    std::thread::Builder::new()
        .name("audio-capture".into())
        .spawn(move || {
            let spec = SampleSpec { rate: wire::RATE, channels: wire::CHANNELS };
            let mut seq: u32 = 0;
            let mut drops: u64 = 0;
            loop {
                match pulse.open_record(None, &dev, spec, wire::PACKET_BYTES as u32) {
                    Ok(stream) => {
                        eprintln!("[audio] capturing {dev:?} (f32le {} Hz, {}ch, 10 ms packets)", wire::RATE, wire::CHANNELS);
                        capture_loop(&pulse, &stream, &m2, &mut seq, &mut drops);
                        eprintln!("[audio] capture stream ended; reopening in 2 s");
                    }
                    Err(e) => {
                        eprintln!("[audio] capture open failed ({dev:?}): {e:#}; retrying in 2 s");
                    }
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        })
        .context("spawn audio-capture thread")?;
    Ok(mbox)
}

/// blocking read → timestamp → mailbox, forever; returns on stream error
fn capture_loop(
    _pulse: &Pulse,
    stream: &super::pulse::Simple,
    mbox: &Mailbox,
    seq: &mut u32,
    drops: &mut u64,
) {
    loop {
        let mut buf = vec![0u8; wire::PACKET_BYTES];
        if let Err(e) = stream.read_exact(&mut buf) {
            eprintln!("[audio] capture read failed: {e:#}");
            return;
        }
        let pkt = AudioPacket {
            seq: *seq,
            host_unix_us: wire::unix_now_us(),
            frames: wire::PACKET_FRAMES,
            data: buf,
        };
        *seq = seq.wrapping_add(1);
        if mbox.put_overwrites(pkt) {
            *drops += 1;
            if *drops == 1 || *drops % 1000 == 0 {
                eprintln!("[audio] capture side has dropped {drops} packets (no consumer or stalled client)");
            }
        }
    }
}

/// Per-connection sender: format record, then packets off the mailbox
/// until the socket dies (client gone / mid-session resync). Reconnects
/// are the client's job.
pub fn serve_connection(mut stream: TcpStream, mbox: &Mailbox) -> Result<()> {
    stream
        .write_all(&wire::encode_format())
        .and_then(|_| stream.flush())
        .map_err(|e| anyhow::anyhow!("audio socket: format send failed: {e}"))?;
    loop {
        let pkt = mbox
            .take_wait(Duration::from_millis(500))
            .ok_or_else(|| anyhow::anyhow!("audio stream idle for 500 ms (capture thread gone)"))?;        let hdr = wire::encode_packet_header(pkt.seq, pkt.host_unix_us, pkt.frames);
        stream
            .write_all(&hdr)
            .and_then(|_| stream.write_all(&pkt.data))
            .map_err(|e| anyhow::anyhow!("audio socket write failed: {e}"))?;
    }
}
