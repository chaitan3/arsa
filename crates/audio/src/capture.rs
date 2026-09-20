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
//! there, streaming raw 10 ms packets back over a pipe:
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

use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::io::FromRawFd;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::MetadataExt as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};

use super::pulse::{Pulse, SampleSpec};
use super::wire::{self, AudioPacket};

/// Drop-oldest mailbox: realtime valve. While nobody consumes (no client
/// connected, or a stalled client), newer packets overwrite older ones —
/// audio never queues up and never exerts backpressure.
#[derive(Clone)]
pub struct Mailbox {
    inner: Arc<(Mutex<Option<AudioPacket>>, std::sync::Condvar)>,
}

impl Mailbox {
    fn new() -> Self {
        Mailbox {
            inner: Arc::new((Mutex::new(None), std::sync::Condvar::new())),
        }
    }

    /// returns true if a previous packet was overwritten (dropped)
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

/// Start the capture supervisor. Fails fast on an obviously broken setup
/// (no XDG_RUNTIME_DIR as root, missing libpulse); a vanished device is
/// tolerated in the background (log + reopen loop).
///
/// `dev`: a SOURCE name — a sink's monitor is "<sink-name>.monitor"
/// ("my-sink.monitor"); "@DEFAULT_MONITOR@" is the PA ≥15 alias for the
/// default sink's mix (pipewire-pulse honors it).
pub fn start_capture(dev: Option<&str>) -> Result<Mailbox> {
    let dev = dev.unwrap_or("@DEFAULT_MONITOR@").to_string();
    let mbox = Mailbox::new();

    // sudo case: XDG_RUNTIME_DIR points at the user's session but is owned
    // by another uid — libpulse refuses to even resolve the socket path.
    // A helper child running as that uid is the only way through.
    let helper = match std::env::var("XDG_RUNTIME_DIR") {
        Ok(dir) => {
            let st = stat_dir(&dir).with_context(|| format!("stat {dir}"))?;
            let my_uid = unsafe { libc::getuid() } as u32;
            if st.uid() == my_uid {
                None // normal user run: in-process capture
            } else {
                Some((st.uid(), st.gid()))
            }
        }
        Err(_) => {
            // no runtime dir: as a regular user pulse can fall back to its
            // legacy $HOME paths; as root there is no sane fallback at all
            if unsafe { libc::getuid() } == 0 {
                bail!("--audio: XDG_RUNTIME_DIR is not set; run with sudo -E (or XDG_RUNTIME_DIR=/run/user/<uid>) so the user's pipewire-pulse session can be reached");
            }
            None
        }
    };

    let pulse = Pulse::dlopen()?;
    match helper {
        None => {
            let m = mbox.clone();
            std::thread::Builder::new()
                .name("audio-capture".into())
                .spawn(move || in_process_loop(pulse, dev, m))
                .context("spawn audio-capture thread")?;
        }
        Some((uid, gid)) => {
            let exe = std::env::current_exe().context("current_exe (for the audio helper)")?;
            let m = mbox.clone();
            std::thread::Builder::new()
                .name("audio-capture-sup".into())
                .spawn(move || helper_loop(pulse, exe, dev, uid, gid, m))
                .context("spawn audio-capture supervisor")?;
        }
    }
    Ok(mbox)
}

/// Normal case: this process's uid matches the session owner.
fn in_process_loop(pulse: Pulse, dev: String, mbox: Mailbox) -> ! {
    let mut seq: u32 = 0;
    let mut drops: u64 = 0;
    loop {
        match open_capture_stream(&pulse, &dev) {
            Ok(stream) => {
                eprintln!("[audio] capturing {dev:?} (f32le {} Hz, {}ch, 10 ms packets)", wire::RATE, wire::CHANNELS);
                match capture_into(&stream, |pkt| {
                    if mbox.put_overwrites(pkt) {
                        drops += 1;
                        if drops == 1 || drops % 1000 == 0 {
                            eprintln!("[audio] capture dropped {drops} packets (no consumer or stalled client)");
                        }
                    }
                }, &mut seq) {
                    Ok(()) => eprintln!("[audio] capture stream ended; reopening in 2 s"),
                    Err(e) => eprintln!("[audio] capture read failed: {e:#}; reopening in 2 s"),
                }
            }
            Err(e) => eprintln!("[audio] capture open failed ({dev:?}): {e:#}; retrying in 2 s"),
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Read 10 ms packets from a pulse record stream and hand each (freshly
/// stamped) to `sink` until the stream dies.
fn capture_into(
    stream: &super::pulse::Simple,
    mut sink: impl FnMut(AudioPacket),
    seq: &mut u32,
) -> Result<()> {
    loop {
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
) -> ! {
    loop {
        match spawn_helper(&exe, &dev, uid, gid) {
            Ok((read_fd, pid)) => {
                let mut f = unsafe { File::from_raw_fd(read_fd) };
                eprintln!("[audio] helper process running (uid {uid}) capturing {dev:?}");
                let mut seq: u32 = 0;
                let mut drops: u64 = 0;
                let mut buf = vec![0u8; wire::PACKET_BYTES];
                loop {
                    match f.read_exact(&mut buf) {
                        Ok(()) => {
                            let pkt = AudioPacket {
                                seq,
                                host_unix_us: wire::unix_now_us(),
                                frames: wire::PACKET_FRAMES,
                                data: buf.clone(),
                            };
                            seq = seq.wrapping_add(1);
                            if mbox.put_overwrites(pkt) {
                                drops += 1;
                                if drops == 1 || drops % 1000 == 0 {
                                    eprintln!("[audio] capture dropped {drops} packets (no consumer or stalled client)");
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[audio] helper pipe ended: {e:#}; respawning in 2 s");
                            break;
                        }
                    }
                }
                drop(f);
                unsafe {
                    let mut status = 0;
                    libc::waitpid(pid, &mut status, 0);
                }
            }
            Err(e) => eprintln!("[audio] helper spawn failed: {e:#}; retrying in 2 s"),
        }
        std::thread::sleep(Duration::from_secs(2));
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
            eprintln!("[audio] helper exec failed");
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
    loop {
        match open_capture_stream(&pulse, dev) {
            Ok(stream) => {
                eprintln!("[audio-helper] capturing {dev:?} (f32le {} Hz, {}ch)", wire::RATE, wire::CHANNELS);
                match capture_into(&stream, |pkt| {
                    let _ = out.write_all(&pkt.data);
                }, &mut 0) {
                    Ok(()) => {}
                    Err(e) => eprintln!("[audio-helper] capture read failed: {e:#}"),
                }
            }
            Err(e) => {
                eprintln!("[audio-helper] capture open failed ({dev:?}): {e:#}");
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
                        eprintln!(
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

/// Per-connection sender: format record, then packets off the mailbox
/// until the socket dies. Reconnects are the client's job.
pub fn serve_connection(mut stream: TcpStream, mbox: &Mailbox) -> Result<()> {
    stream
        .write_all(&wire::encode_format())
        .and_then(|_| stream.flush())
        .map_err(|e| anyhow::anyhow!("audio socket: format send failed: {e}"))?;
    loop {
        let pkt = mbox
            .take_wait(Duration::from_millis(500))
            .ok_or_else(|| anyhow::anyhow!("audio stream idle for 500 ms (capture thread gone)"))?;
        let hdr = wire::encode_packet_header(pkt.seq, pkt.host_unix_us, pkt.frames);
        stream
            .write_all(&hdr)
            .and_then(|_| stream.write_all(&pkt.data))
            .map_err(|e| anyhow::anyhow!("audio socket write failed: {e}"))?;
    }
}

fn stat_dir(path: &str) -> Result<std::fs::Metadata> {
    std::fs::metadata(path).with_context(|| format!("stat {path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_overwrites_oldest() {
        let m = Mailbox::new();
        assert!(!m.put_overwrites(AudioPacket { seq: 1, host_unix_us: 0, frames: 480, data: vec![] }));
        assert!(m.put_overwrites(AudioPacket { seq: 2, host_unix_us: 0, frames: 480, data: vec![] }));
        assert_eq!(m.take_wait(Duration::ZERO).unwrap().seq, 2);
    }

    #[test]
    fn mailbox_take_times_out_when_empty() {
        let m = Mailbox::new();
        assert!(m.take_wait(Duration::from_millis(5)).is_none());
    }
}
