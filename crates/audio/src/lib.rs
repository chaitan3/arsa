//! Audio side-channel: raw PCM streaming alongside the RDMA video path.
//!
//! Architecture (discussed and pinned 2026-09):
//! - Capture is a HOST concern, not a capture-source concern: whatever is
//!   being streamed (DRM/KMS screen or a QEMU guest via the audiodev→sink
//!   recipe), the audio ends up in a PipeWire sink whose monitor we
//!   capture through the pulse-simple API against the host's
//!   pipewire-pulse compat socket.
//! - Transport: a SECOND connection to the main TCP port, tagged by the
//!   first byte (see `wire`); the audio thread owns its socket outright —
//!   no locks shared with control/RDMA, no application-level acks, and
//!   realtime drop-oldest everywhere so a stalled client can never exert
//!   backpressure onto the video path.
//! - Wire format: fixed self-describing header (AUD1) + raw f32le 48 kHz
//!   stereo packets. No codec: the LAN link eats 3.07 Mbit/s without
//!   noticing, and a codec dependency would only add latency and vendoring
//!   pain. Revisit Opus only if WAN use ever appears.
//! - Native libs are dlopened (house style, mirrors eglwin.rs): we speak
//!   pulse-simple against `libpulse-simple`, which the pipewire-pulse
//!   compat layer services. Native PipeWire streams (pipewire-rs) would
//!   buy a little latency control and is a known follow-up lever.
//!
//! Server side: `capture::start_capture` (one supervisor thread, bound to
//! a sink monitor) + `capture::serve_connection` per audio connection.
//! Client side: `playback::run` (connect/retry loop + playout thread).
//! `stats::AudioStats` carries the client-side playout timing used by the
//! `--metrics` overlay (A/V offset); it is inert when metrics are off.
//!
//! TODO(host-play-off): an opt-in server mode that routes host audio into a
//! virtual sink (PipeWire/PulseAudio `module-null-sink`, e.g. `rdma-display`)
//! and makes it the default while a session is active, so the host speakers
//! stay silent and only the client hears the audio; restore the previous
//! default sink on disconnect and on SIGINT/SIGTERM/SIGQUIT. Off by default —
//! the default stays the `@DEFAULT_MONITOR@` tap, which keeps the host audible.
//! Muting the physical sink is NOT a substitute: a sink's monitor is post
//! volume/mute, so muting it silences the capture too. Plan: a new `route.rs`
//! that shells out to `pactl` (pipewire-pulse compat) like display.rs does for
//! hyprctl/kscreen-doctor; under root the commands must run as the session
//! user, so reuse the setuid-down helper machinery in `capture.rs`. Switch on
//! session start, restore on session end + via `server::signals`.

pub mod capture;
pub mod playback;
pub mod pulse;
pub mod stats;
pub mod wire;
