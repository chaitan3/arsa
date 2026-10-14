//! Server side: keep a pulse record stream alive against a sink monitor
//! and feed audio connections.
//!
//! Threading model (plain threads, house rule):
//! - ONE supervisor thread owns capture and feeds a drop-oldest mailbox.
//! - Per-connection sender threads pop packets and write them to the
//!   client. Socket errors kill only that connection; the client
//!   reconnects on its own schedule.
//!
//! The uid problem (verified against libpulse source, core-util.c
//! `pa_get_runtime_dir`): a root process with XDG_RUNTIME_DIR pointing at
//! a user's session is refused BEFORE any connection is attempted — the
//! check compares the DIRECTORY owner against our REAL uid and aborts
//! while resolving the default server path (error 6 =
//! PA_ERR_CONNECTIONREFUSED; the device name is never queried). Since our
//! capture server must stay root for DRM/KMS, we spawn a helper CHILD that
//! drops privileges to the runtime-dir owner and runs the pulse stream
//! there, streaming raw PCM packets back over a pipe:
//!
//!   fork + (child: PDEATHSIG, setgroups(0), setresgid, setresuid,
//!           HOME=<target user's home>, dup2(pipe, stdout), exec our own
//!           binary in `--audio-helper` mode)
//!
//! fork+exec (not fork alone) so the exec resets the multithreaded
//! parent's inherited locks — a fork-without-exec child calling into
//! libpulse/malloc would risk lock deadlocks.
//!
//! When we already run as the runtime-dir owner (a normal user session),
//! capture runs in-process — the helper is sudo-only machinery.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::MetadataExt as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};

use super::pulse::{Pulse, SampleSpec};
use super::wire::{self, AudioPacket};

use rdma_display_lifecycle::{sleep_or_stop, Running, Stop};

/// Bounded drop-oldest mailbox: realtime valve. While nobody consumes (no
/// client connected, or a stalled client) newer packets eventually overwrite
/// the oldest — audio never queues without bound and never exerts
/// backpressure. The depth is a small safety margin above the single packet a
/// sender is working on, so a brief socket-write stall no longer drops audio.
const MAILBOX_DEPTH: usize = 8;

#[derive(Clone)]
pub struct Mailbox {
    inner: Arc<(Mutex<VecDeque<AudioPacket>>, std::sync::Condvar)>,
}

impl Mailbox {
    fn new() -> Self {
        Mailbox {
            inner: Arc::new((Mutex::new(VecDeque::new()), std::sync::Condvar::new())),
        }
    }

    /// Push at the back, dropping the oldest when full.
    /// Returns true if a packet was dropped.
    fn put_overwrites(&self, pkt: AudioPacket) -> bool {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        let dropped = g.len() >= MAILBOX_DEPTH;
        if dropped {
            g.pop_front();
        }
        g.push_back(pkt);
        cv.notify_one();
        dropped
    }

    fn take_wait(&self, timeout: Duration) -> Option<AudioPacket> {
        let (lock, cv) = &*self.inner;
        let mut g = lock.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(p) = g.pop_front() {
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

    /// Block up to `timeout` for the next captured packet. Returns false on
    /// timeout. Used by the server's startup preflight to prove the capture
    /// path actually produces audio, then stop.
    pub fn wait_first(&self, timeout: Duration) -> bool {
        self.take_wait(timeout).is_some()
    }
}

/// A running audio capture. Holds the packet mailbox and the worker handle;
/// `stop` (or dropping) ends capture — including killing the setuid helper
/// child when the helper path is in use.
pub struct AudioCapture {
    pub mailbox: Mailbox,
    handle: Running,
}

impl AudioCapture {
    /// Stop capture and join the worker (and kill the setuid helper child on
    /// the helper path). Idempotent.
    pub fn stop(&mut self) {
        self.handle.stop();
    }
}

/// Start the capture supervisor. Fails fast on an obviously broken setup
/// (no XDG_RUNTIME_DIR as root, missing libpulse); a vanished device is
/// tolerated in the background (log + reopen loop).
///
/// `dev`: a SOURCE name — a sink's monitor is "<sink-name>.monitor"
/// ("my-sink.monitor"); "@DEFAULT_MONITOR@" is the PA ≥15 alias for the
/// default sink's mix (pipewire-pulse honors it).
pub fn start_capture(dev: Option<&str>) -> Result<AudioCapture> {
    let dev = dev.unwrap_or("@DEFAULT_MONITOR@").to_string();
    let mbox = Mailbox::new();

    // The setuid helper exists only for the root/sudo case: root runs with the
    // user's XDG_RUNTIME_DIR (owned by another uid), and libpulse refuses to
    // resolve a socket it doesn't own, so a helper child dropped to that uid is
    // the only way through. A non-root process — including a CAP_SYS_ADMIN
    // setcap binary, whose uid is still the user — owns/uses its own session
    // socket directly and could not drop to another uid anyway, so it always
    // captures in-process (no fork).
    let helper = if unsafe { libc::getuid() } == 0 {
        match std::env::var("XDG_RUNTIME_DIR") {
            Ok(dir) => {
                let st = stat_dir(&dir).with_context(|| format!("stat {dir}"))?;
                if st.uid() == 0 {
                    None // root's own session (unusual): in-process
                } else {
                    Some((st.uid(), st.gid()))
                }
            }
            Err(_) => {
                // no runtime dir: as a regular user pulse can fall back to its
                // legacy $HOME paths; as root there is no sane fallback at all
                bail!("--audio: XDG_RUNTIME_DIR is not set; run with sudo -E (or XDG_RUNTIME_DIR=/run/user/<uid>) so the user's pipewire-pulse session can be reached");
            }
        }
    } else {
        None
    };

    let pulse = Pulse::dlopen()?;
    let stop = Stop::new();
    let worker_stop = stop.clone();
    let join = match helper {
        None => {
            let m = mbox.clone();
            std::thread::Builder::new()
                .name("audio-capture".into())
                .spawn(move || in_process_loop(pulse, dev, m, worker_stop))
                .context("spawn audio-capture thread")?
        }
        Some((uid, gid)) => {
            let exe = std::env::current_exe().context("current_exe (for the audio helper)")?;
            let m = mbox.clone();
            std::thread::Builder::new()
                .name("audio-capture-sup".into())
                .spawn(move || helper_loop(pulse, exe, dev, uid, gid, m, worker_stop))
                .context("spawn audio-capture supervisor")?
        }
    };
    Ok(AudioCapture {
        mailbox: mbox,
        handle: Running::new(stop, vec![join]),
    })
}

/// Normal case: this process's uid matches the session owner.
fn in_process_loop(pulse: Pulse, dev: String, mbox: Mailbox, stop: Stop) {
    let mut seq: u32 = 0;
    let mut drops: u64 = 0;
    let mut fed: u64 = 0;
    let mut last_report = std::time::Instant::now();
    loop {
        if stop.stopped() {
            return;
        }
        match open_capture_stream(&pulse, &dev) {
            Ok(stream) => {
                log::info!(
                    "[audio] capturing {dev:?} (f32le {} Hz, {}ch, {:.1} ms packets)",
                    wire::RATE,
                    wire::CHANNELS,
                    wire::PACKET_FRAMES as f64 * 1000.0 / wire::RATE as f64
                );
                match capture_into(&stream, |pkt| {
                    fed += 1;
                    if mbox.put_overwrites(pkt) {
                        drops += 1;
                        if drops == 1 || drops % 10000 == 0 {
                            log::debug!("[audio] capture dropped {drops} packets (no consumer or stalled client)");
                        }
                    }
                    if last_report.elapsed() >= Duration::from_secs(10) {
                        log::debug!("[audio] capture feed alive: {fed} packets so far");
                        last_report = std::time::Instant::now();
                    }
                }, &mut seq, &stop) {
                    Ok(()) => log::debug!("[audio] capture stream ended; reopening in 2 s"),
                    Err(e) => log::debug!("[audio] capture read failed: {e:#}; reopening in 2 s"),
                }
            }
            Err(e) => log::debug!("[audio] capture open failed ({dev:?}): {e:#}; retrying in 2 s"),
        }
        if stop.stopped() {
            return;
        }
        if sleep_or_stop(&stop, Duration::from_secs(2)) {
            return;
        }
    }
}

/// Read wire packets from a pulse record stream and hand each (freshly
/// stamped) to `sink` until the stream dies or `stop` is set.
fn capture_into(
    stream: &super::pulse::Simple,
    mut sink: impl FnMut(AudioPacket),
    seq: &mut u32,
    stop: &Stop,
) -> Result<()> {
    loop {
        if stop.stopped() {
            return Ok(());
        }
        let mut buf = vec![0u8; wire::PACKET_BYTES];
        stream.read_exact(&mut buf)?;
        sink(AudioPacket {
            seq: *seq,
            host_unix_us: wire::unix_now_us(),
            frames: wire::PACKET_FRAMES,
            data: buf,
        });
        *seq = seq.wrapping_add(1);
    }
}

/// Sudo case: keep a helper child alive and drain its pipe into the
/// mailbox. The helper does the pulse work; we only re-spawn it when it
/// dies (device vanished, perms changed, ...).
fn helper_loop(
    _pulse: Pulse,
    exe: std::path::PathBuf,
    dev: String,
    uid: u32,
    gid: u32,
    mbox: Mailbox,
    stop: Stop,
) {
    loop {
        if stop.stopped() {
            return;
        }
        match spawn_helper(&exe, &dev, uid, gid) {
            Ok((read_fd, pid)) => {
                let mut f = unsafe { File::from_raw_fd(read_fd) };
                log::debug!("[audio] helper process running (uid {uid}) capturing {dev:?}");
                let mut seq: u32 = 0;
                let mut drops: u64 = 0;
                let mut fed: u64 = 0;
                let mut last_report = std::time::Instant::now();
                let mut buf = vec![0u8; wire::PACKET_BYTES];
                loop {
                    if stop.stopped() {
                        break;
                    }
                    // poll before the blocking read so a silent helper does not
                    // delay shutdown
                    let mut p = libc::pollfd {
                        fd: f.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    let n = unsafe { libc::poll(&mut p, 1, 50) };
                    if n == 0 {
                        continue;
                    }
                    match f.read_exact(&mut buf) {
                        Ok(()) => {
                            fed += 1;
                            let pkt = AudioPacket {
                                seq,
                                host_unix_us: wire::unix_now_us(),
                                frames: wire::PACKET_FRAMES,
                                data: buf.clone(),
                            };
                            seq = seq.wrapping_add(1);
                            if mbox.put_overwrites(pkt) {
                                drops += 1;
                                if drops == 1 || drops % 10000 == 0 {
                                    log::debug!("[audio] capture dropped {drops} packets (no consumer or stalled client)");
                                }
                            }
                            if last_report.elapsed() >= Duration::from_secs(10) {
                                log::debug!("[audio] capture feed alive: {fed} packets so far");
                                last_report = std::time::Instant::now();
                            }
                        }
                        Err(e) => {
                            log::debug!("[audio] helper pipe ended: {e:#}; respawning in 2 s");
                            break;
                        }
                    }
                }
                drop(f);
                unsafe {
                    let mut status = 0;
                    if stop.stopped() {
                        // tearing down: don't leave the helper capturing
                        libc::kill(pid, libc::SIGKILL);
                    }
                    libc::waitpid(pid, &mut status, 0);
                }
                if stop.stopped() {
                    return;
                }
            }
            Err(e) => log::debug!("[audio] helper spawn failed: {e:#}; retrying in 2 s"),
        }
        if sleep_or_stop(&stop, Duration::from_secs(2)) {
            return;
        }
    }
}

/// fork + drop-privileges + exec our own binary in `--audio-helper` mode
/// with stdout wired to a fresh pipe (one CLOEXEC pair; dup2 clears the
/// flag on fd 1 so the write end survives exec).
fn spawn_helper(exe: &std::path::Path, dev: &str, uid: u32, gid: u32) -> Result<(i32, i32)> {
    let mut fds = [0i32; 2];
    let rc = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
    if rc != 0 {
        bail!("pipe2 failed: {}", std::io::Error::last_os_error());
    }
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        unsafe { libc::close(fds[0]); libc::close(fds[1]); }
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        unsafe {
            // die when the server dies (prctl survives exec)
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0);
            if libc::getppid() == 1 {
                libc::_exit(1); // parent died between fork and prctl
            }
            // drop to the session owner: no supplementary groups, no root
            libc::setgroups(0, std::ptr::null());
            if libc::setresgid(gid, gid, gid) != 0 || libc::setresuid(uid, uid, uid) != 0 {
                libc::_exit(3);
            }
            // cookie lookup needs the target user's HOME
            let pw = libc::getpwuid(uid);
            if !pw.is_null() {
                let dir = std::ffi::CStr::from_ptr((*pw).pw_dir);
                if let Ok(h) = dir.to_str() {
                    if let Ok(c) = std::ffi::CString::new(h) {
                        libc::setenv(c"HOME".as_ptr(), c.as_ptr(), 1);
                    }
                }
            }
            // stdout = the pipe; the CLOEXEC'd original write end closes at exec
            libc::dup2(fds[1], 1);
            let exe_c = std::ffi::CString::new(exe.as_os_str().as_bytes()).unwrap();
            let dev_c = std::ffi::CString::new(dev).unwrap();
            let a1 = c"--audio-helper".to_bytes_with_nul().as_ptr();
            let a2 = c"--audio-device".to_bytes_with_nul().as_ptr();
            let argv: [*const std::ffi::c_char; 5] = [
                exe_c.as_ptr(),
                a1.cast(),
                a2.cast(),
                dev_c.as_ptr(),
                std::ptr::null(),
            ];
            libc::execv(exe_c.as_ptr(), argv.as_ptr());
            log::debug!("[audio] helper exec failed");
            libc::_exit(4);
        }
    }
    unsafe { libc::close(fds[1]) }; // parent keeps only the read end
    Ok((fds[0], pid))
}

/// --audio-helper mode (spawned BY us, running as the session user):
/// pulse capture → stdout, nothing else on stdout. Dies with its parent
/// (PDEATHSIG); the parent respawns on pipe EOF.
pub fn run_helper(dev: &str) -> Result<()> {
    let pulse = Pulse::dlopen()?;
    let mut out = std::mem::ManuallyDrop::new(unsafe {
        File::from_raw_fd(1) // parent's pipe, unbuffered
    });
    // the helper is torn down by the parent (SIGKILL on stop, PDEATHSIG on
    // parent death); it never sets its own stop token
    let stop = Stop::new();
    loop {
        match open_capture_stream(&pulse, dev) {
            Ok(stream) => {
                log::info!("[audio-helper] capturing {dev:?} (f32le {} Hz, {}ch)", wire::RATE, wire::CHANNELS);
                match capture_into(&stream, |pkt| {
                    let _ = out.write_all(&pkt.data);
                }, &mut 0, &stop) {
                    Ok(()) => {}
                    Err(e) => log::debug!("[audio-helper] capture read failed: {e:#}"),
                }
            }
            Err(e) => {
                log::debug!("[audio-helper] capture open failed ({dev:?}): {e:#}");
                std::process::exit(1); // parent will respawn after backoff
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Open the record stream with a usability fallback: PA record streams
/// take a SOURCE name, but everyone naturally passes a SINK name
/// (Sunshine's `audio_sink = my-sink`) — when the lookup fails with
/// NOENTITY, retry once against the sink's monitor source `<name>.monitor`.
fn open_capture_stream(pulse: &Pulse, dev: &str) -> Result<super::pulse::Simple> {
    let spec = SampleSpec { rate: wire::RATE, channels: wire::CHANNELS };
    match pulse.open_record(None, dev, spec, wire::PACKET_BYTES as u32) {
        Ok(s) => Ok(s),
        Err(e) => {
            let code = e
                .downcast_ref::<super::pulse::PaOpenError>()
                .map(|p| p.code)
                .unwrap_or(-1);
            if code == 5 && !dev.ends_with(".monitor") && !dev.starts_with('@') {
                let alt = format!("{dev}.monitor");
                match pulse.open_record(None, &alt, spec, wire::PACKET_BYTES as u32) {
                    Ok(s) => {
                        log::info!(
                            "[audio] {dev:?} is a sink name (sinks can't be recorded directly); capturing its monitor source {alt:?} instead"
                        );
                        return Ok(s);
                    }
                    Err(e2) => return Err(e2.context(format!("pa record stream open (device {dev:?})"))),
                }
            }
            Err(e)
        }
    }
}

/// Per-connection sender: format record, then packets off the mailbox.
///
/// A silent monitor (nothing playing into the sink) delivers NO packets —
/// that is not an error and must never close the connection; when the
/// game starts playing, packets resume and the client hears them. So the
/// stream idles open indefinitely, warning only (the old 500 ms idle kill
/// caused the "reset by peer" reconnect loop on silent monitors).
pub fn serve_connection(mut stream: TcpStream, mbox: &Mailbox) -> Result<()> {
    stream
        .write_all(&wire::encode_format())
        .and_then(|_| stream.flush())
        .map_err(|e| anyhow::anyhow!("audio socket: format send failed: {e}"))?;
    let mut warned = false;
    loop {
        match mbox.take_wait(Duration::from_secs(10)) {
            Some(pkt) => {
                warned = false;
                let hdr = wire::encode_packet_header(pkt.seq, pkt.host_unix_us, pkt.frames);
                stream
                    .write_all(&hdr)
                    .and_then(|_| stream.write_all(&pkt.data))
                    .map_err(|e| anyhow::anyhow!("audio socket write failed: {e}"))?;
            }
            None => {
                if !warned {
                    log::info!(
                        "[audio] no capture data for 10 s (silent monitor? is anything playing into the sink?); keeping the stream open"
                    );
                    warned = true;
                }
            }
        }
    }
}

fn stat_dir(path: &str) -> Result<std::fs::Metadata> {
    std::fs::metadata(path).with_context(|| format!("stat {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_drops_oldest_when_full() {
        let m = Mailbox::new();
        for i in 0..MAILBOX_DEPTH as u32 {
            let p = AudioPacket { seq: i, host_unix_us: 0, frames: 96, data: vec![] };
            assert!(!m.put_overwrites(p), "packet {i} should fit");
        }
        // full: this put drops the oldest (seq 0)
        let p = AudioPacket { seq: MAILBOX_DEPTH as u32, host_unix_us: 0, frames: 96, data: vec![] };
        assert!(m.put_overwrites(p));
        assert_eq!(m.take_wait(Duration::ZERO).unwrap().seq, 1);
    }

    #[test]
    fn mailbox_take_times_out_when_empty() {
        let m = Mailbox::new();
        assert!(m.take_wait(Duration::from_millis(5)).is_none());
    }
}
