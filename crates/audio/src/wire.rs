//! Audio side-channel wire format.
//!
//! Transport: a second connection to the main TCP port. The first byte of
//! every connection tags its kind — audio connections send [`TAG_AUDIO`]
//! and then only ever receive; control connections carry the proto
//! handshake (magic "RDDS", first byte 0x53), so routing is a one-byte
//! peek with zero change to the control-connection bytes.
//!
//! Server → client, right after the tag byte: one fixed [`FORMAT_LEN`]
//! byte format record, then a stream of packets (16-byte header + raw
//! PCM). Nothing flows client → server.

use anyhow::{ensure, Result};

pub const TAG_AUDIO: u8 = 0x02;

/// PA_SAMPLE_FLOAT32LE (we dlopen libpulse-simple, whose enum this mirrors)
pub const SAMPLE_FORMAT: u32 = 5;
pub const RATE: u32 = 48000;
pub const CHANNELS: u32 = 2;
/// 5 ms packetization (240 frames @ 48 kHz) — the same default Moonlight/
/// Sunshine use for lowest latency. At f32 stereo that is a 1920 B payload
/// against a 16 B header (~0.8% overhead), i.e. 200 packets/s.
pub const PACKET_FRAMES: u32 = 240;
pub const BYTES_PER_FRAME: usize = CHANNELS as usize * 4; // f32 stereo
pub const PACKET_BYTES: usize = PACKET_FRAMES as usize * BYTES_PER_FRAME;

pub const FORMAT_LEN: usize = 20;
pub const PACKET_HDR_LEN: usize = 16;
/// Client playout buffer depth (also the pulse `tlength`/`prebuf` target).
/// The audio is a separate steady ~200 pkt/s TCP stream, *not* subject to the
/// video flow-control, so its arrival jitter is small and one packet (5 ms) has
/// proven sufficient (`ring drops 0` across runs). Note this is not where the
/// playout latency lives: the sink's own buffer dominates it by an order of
/// magnitude. Raise it (or pass `--audio-buffer-ms`) only if you hear dropouts.
pub const BUFFER_MS: u32 = 5;

/// One PCM packet on the wire.
pub struct AudioPacket {
    pub seq: u32,
    pub host_unix_us: u64,
    pub frames: u32,
    pub data: Vec<u8>,
}

pub fn unix_now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// One-time format record: `AUD1` + format/rate/channels/packet_frames.
pub fn encode_format() -> [u8; FORMAT_LEN] {
    let mut v = [0u8; FORMAT_LEN];
    v[0..4].copy_from_slice(b"AUD1");
    v[4..8].copy_from_slice(&SAMPLE_FORMAT.to_le_bytes());
    v[8..12].copy_from_slice(&RATE.to_le_bytes());
    v[12..16].copy_from_slice(&CHANNELS.to_le_bytes());
    v[16..20].copy_from_slice(&PACKET_FRAMES.to_le_bytes());
    v
}

pub fn decode_format(buf: &[u8; FORMAT_LEN]) -> Result<()> {
    ensure!(&buf[0..4] == b"AUD1", "bad audio format magic");
    ensure!(
        u32::from_le_bytes(buf[4..8].try_into()?) == SAMPLE_FORMAT,
        "unexpected audio sample format"
    );
    ensure!(u32::from_le_bytes(buf[8..12].try_into()?) == RATE, "unexpected rate");
    ensure!(
        u32::from_le_bytes(buf[12..16].try_into()?) == CHANNELS,
        "unexpected channel count"
    );
    ensure!(
        u32::from_le_bytes(buf[16..20].try_into()?) == PACKET_FRAMES,
        "unexpected packetization"
    );
    Ok(())
}

/// Packet header: seq u32, host unix-us u64, frames-per-channel u32.
/// No client-facing timestamps in v1 — the host stamp exists so the queued
/// AV-sync rate-trim lever can use it without another format change.
pub fn encode_packet_header(seq: u32, host_unix_us: u64, frames: u32) -> [u8; PACKET_HDR_LEN] {
    let mut v = [0u8; PACKET_HDR_LEN];
    v[0..4].copy_from_slice(&seq.to_le_bytes());
    v[4..12].copy_from_slice(&host_unix_us.to_le_bytes());
    v[12..16].copy_from_slice(&frames.to_le_bytes());
    v
}

pub fn decode_packet_header(buf: &[u8; PACKET_HDR_LEN]) -> (u32, u64, u32) {
    (
        u32::from_le_bytes(buf[0..4].try_into().unwrap()),
        u64::from_le_bytes(buf[4..12].try_into().unwrap()),
        u32::from_le_bytes(buf[12..16].try_into().unwrap()),
    )
}

/// sanity cap for a declared frame count: at most 10 packets' worth
pub fn frames_plausible(frames: u32) -> Result<()> {
    ensure!(
        (0..=PACKET_FRAMES * 10).contains(&frames),
        "audio packet frame count out of range: {frames}"
    );
    Ok(())
}

pub fn payload_len(frames: u32) -> Result<usize> {
    frames_plausible(frames)?;
    Ok(frames as usize * BYTES_PER_FRAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_record_round_trip() {
        decode_format(&encode_format()).unwrap();
        // flipped magic must be rejected
        let mut bad = encode_format();
        bad[0] = b'X';
        assert!(decode_format(&bad).is_err());
    }

    #[test]
    fn packet_header_round_trip() {
        let h = encode_packet_header(0xDEAD_BEEF, 1_234_567_890, PACKET_FRAMES);
        let (seq, us, frames) = decode_packet_header(&h);
        assert_eq!((seq, us, frames), (0xDEAD_BEEF, 1_234_567_890, PACKET_FRAMES));
        assert!(frames_plausible(PACKET_FRAMES).is_ok());
        assert!(frames_plausible(PACKET_FRAMES * 11).is_err());
        assert_eq!(payload_len(PACKET_FRAMES).unwrap(), PACKET_BYTES);
    }
}
