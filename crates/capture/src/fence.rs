//! Fence-triggered capture (prototype): wake the capture loop when the
//! compositor's per-frame completion fence signals, instead of polling.
//!
//! On this stack (Hyprland + nvidia-drm, explicit sync) the compositor's frame
//! completion is an NVIDIA "semsurf" (semaphore-surface) `dma_fence`: created
//! once per frame through `nv_drm_semsurf_fence_create_ioctl` and signalled from
//! the semaphore-surface completion callback. It is *not* on the scanout
//! dma-buf's reservation, so userspace cannot wait on it via
//! `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` (that is why `syncprobe` reports a stub
//! fence). But it is a real `dma_fence`, so the kernel's
//! `dma_fence:dma_fence_signaled` tracepoint fires for it.
//!
//! This module consumes that tracepoint through tracefs - the same tracepoint
//! an eBPF program would attach to, without the libbpf/BPF-object dependency
//! (and consistent with the hand-rolled style of the DRM capture). It:
//!
//!   1. mounts tracefs if needed;
//!   2. **calibrates in the background**: watches `dma_fence_init` plus an
//!      atomic-commit kprobe for a short window and picks the fence context
//!      whose init count tracks the commit count (the per-frame completion
//!      fence). An idle screen has no commits to calibrate against, so this
//!      retries until activity appears instead of failing the mode;
//!   3. filters `dma_fence_signaled` to that context and reports each signal's
//!      timestamp over a channel, setting [`FenceWatch::ready`].
//!
//! Until `ready`, the caller falls back to one-behind; calibration needs an
//! actively flipping compositor (something moving on screen), but that can
//! happen at any time after startup. If tracefs cannot be mounted at all,
//! [`start`] returns an error and the caller falls back permanently.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{bail, Context, Result};

/// Shared state the worker publishes and the capture source reads.
#[derive(Default)]
struct FenceShared {
    /// latest signal: `seq` increments per signal, `latest_us` is its time
    seq: AtomicU64,
    latest_us: AtomicU64,
    /// true once calibration succeeded and signals are being forwarded
    ready: AtomicBool,
}

/// The kernel's per-frame completion-fence signals, as a process-wide
/// singleton: the watched context is the compositor's and tracefs is global,
/// so exactly one worker may own it. The capture source (re-spawned per
/// session) only reads the latest published signal, so a session restart can
/// never race another watch over tracefs.
pub struct FenceWatch {
    shared: Arc<FenceShared>,
}

impl FenceWatch {
    /// Whether calibration has finished and the completion fence is being
    /// watched. While false the caller should use its own fallback (one-behind).
    pub fn ready(&self) -> bool {
        self.shared.ready.load(Ordering::Relaxed)
    }

    /// The newest signal as `(seq, unix_us)`, or `None` if none arrived since
    /// `since_seq`. The caller captures with `unix_us` as `t0` when it changes.
    pub fn signal_since(&self, since_seq: u64) -> Option<(u64, u64)> {
        let s = self.shared.seq.load(Ordering::Relaxed);
        if s != since_seq {
            Some((s, self.shared.latest_us.load(Ordering::Relaxed)))
        } else {
            None
        }
    }
}

static WATCH: OnceLock<FenceWatch> = OnceLock::new();

const TRACEFS_CANDIDATES: [&str; 2] = ["/sys/kernel/tracing", "/sys/kernel/debug/tracing"];

fn find_tracefs() -> Option<String> {
    TRACEFS_CANDIDATES
        .iter()
        .find(|p| std::path::Path::new(&format!("{p}/events")).exists())
        .map(|s| s.to_string())
}

fn write_str(path: &str, val: &str) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .with_context(|| format!("open {path}"))?;
    f.write_all(val.as_bytes())
        .with_context(|| format!("write {path}"))?;
    Ok(())
}

fn ensure_tracefs() -> Result<String> {
    if let Some(p) = find_tracefs() {
        return Ok(p);
    }
    for p in TRACEFS_CANDIDATES {
        let _ = std::fs::create_dir_all(p);
        // Mount in-process via mount(2): a setcap server's CAP_SYS_ADMIN is
        // effective here, whereas a child `mount` process would not inherit a
        // file capability and would fail with "must be superuser".
        let Ok(target) = std::ffi::CString::new(p) else {
            continue;
        };
        let rc = unsafe {
            libc::mount(
                c"tracefs".as_ptr(),
                target.as_ptr(),
                c"tracefs".as_ptr(),
                0,
                std::ptr::null(),
            )
        };
        if rc == 0 {
            if let Some(found) = find_tracefs() {
                return Ok(found);
            }
        }
    }
    bail!("tracefs unavailable (mounting it needs CAP_SYS_ADMIN; also needs a tracefs-capable kernel)")
}

/// Parse the digits following `key` in a human-readable ftrace line.
fn field_u64(line: &str, key: &str) -> Option<u64> {
    let i = line.find(key)? + key.len();
    let rest = &line[i..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Watch `dma_fence_init` + the atomic-commit kprobe for `calib`, pick the
/// per-frame completion context, and leave tracefs idle for the caller.
fn calibrate(tracefs: &str, calib: Duration) -> Result<u32> {
    // reset
    let _ = write_str(&format!("{tracefs}/tracing_on"), "0\n");
    let _ = std::fs::write(format!("{tracefs}/trace"), b"");
    let _ = write_str(&format!("{tracefs}/events/enable"), "0\n");
    let _ = std::fs::write(format!("{tracefs}/kprobe_events"), b"");

    // fence creation + commit count
    write_str(
        &format!("{tracefs}/events/dma_fence/dma_fence_init/enable"),
        "1\n",
    )
    .context("enable dma_fence_init (needs CAP_DAC_OVERRIDE, or mount tracefs with uid=)")?;
    let _ = std::fs::OpenOptions::new()
        .append(true)
        .open(format!("{tracefs}/kprobe_events"))
        .and_then(|mut f| f.write_all(b"p:rdma_handover drm_atomic_set_fb_for_plane\n"));
    let _ = write_str(&format!("{tracefs}/events/kprobes/enable"), "1\n");

    write_str(&format!("{tracefs}/tracing_on"), "1\n")?;
    std::thread::sleep(calib);
    let _ = write_str(&format!("{tracefs}/tracing_on"), "0\n");

    let mut dump = String::new();
    std::fs::File::open(format!("{tracefs}/trace"))
        .context("open trace")?
        .read_to_string(&mut dump)
        .context("read trace")?;

    // stop calibrating
    let _ = write_str(&format!("{tracefs}/events/enable"), "0\n");
    let _ = std::fs::write(format!("{tracefs}/kprobe_events"), b"");

    let mut handovers = 0usize;
    let mut inits: HashMap<u32, usize> = HashMap::new();
    for line in dump.lines() {
        if line.contains("dma_fence_init:") {
            if let Some(c) = field_u64(line, "context=") {
                *inits.entry(c as u32).or_default() += 1;
            }
        } else if line.contains("rdma_handover:") {
            handovers += 1;
        }
    }
    if handovers < 8 {
        bail!("only {handovers} commits in {:?}", calib);
    }
    // the per-frame completion fence is created ~once per commit; pick the
    // context whose creation count is closest to the commit count
    let mut best: Option<(u32, usize)> = None;
    for (ctx, n) in inits {
        if n * 2 < handovers {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, bn)) => n.abs_diff(handovers) < bn.abs_diff(handovers),
        };
        if better {
            best = Some((ctx, n));
        }
    }
    let (ctx, n) = best.context("no fence context tracked the commits")?;
    log::debug!(
        "[capture-drm] fence watch: completion context {ctx} \
         ({n} fence inits vs {handovers} commits in {:?})",
        calib
    );
    Ok(ctx)
}

/// Point the signal tracepoint at `ctx` and turn tracing on.
fn configure_signal(tracefs: &str, ctx: u32) -> Result<()> {
    let _ = write_str(
        &format!("{tracefs}/events/dma_fence/dma_fence_signaled/filter"),
        &format!("context == {ctx}\n"),
    );
    write_str(
        &format!("{tracefs}/events/dma_fence/dma_fence_signaled/enable"),
        "1\n",
    )
    .context("enable dma_fence_signaled")?;
    write_str(&format!("{tracefs}/tracing_on"), "1\n")?;
    Ok(())
}

/// Read `trace_pipe` until it errors/EOF, publishing one timestamp per
/// completion-fence signal. Returns when the stream ends (caller recalibrates).
fn read_signals(tracefs: &str, shared: &FenceShared) {
    let Ok(pipe) = std::fs::OpenOptions::new()
        .read(true)
        .open(format!("{tracefs}/trace_pipe"))
    else {
        return;
    };
    let mut br = BufReader::new(pipe);
    let mut line = String::new();
    loop {
        line.clear();
        match br.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if line.contains("dma_fence_signaled:") {
                    shared.latest_us.store(rdma_display_proto::unix_now_us(), Ordering::Relaxed);
                    shared.seq.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(_) => break,
        }
    }
}

/// Background loop: calibrate (retrying until the screen is active), then
/// forward completion-fence signals. Runs for the process lifetime.
fn worker_loop(tracefs: &str, calib: Duration, shared: Arc<FenceShared>) {
    let mut attempts: u32 = 0;
    loop {
        match calibrate(tracefs, calib) {
            Ok(ctx) => {
                if let Err(e) = configure_signal(tracefs, ctx) {
                    attempts += 1;
                    if attempts <= 3 {
                        log::debug!(
                            "[capture-drm] fence watch: configure signalled failed: {e:#}; \
                             retrying"
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1000));
                    continue;
                }
                log::debug!(
                    "[capture-drm] fence watch: switching to fence-triggered capture \
                     (context {ctx})"
                );
                shared.ready.store(true, Ordering::Relaxed);
                read_signals(tracefs, &shared);
                shared.ready.store(false, Ordering::Relaxed);
                log::debug!("[capture-drm] fence watch: signal stream ended; recalibrating");
            }
            Err(e) => {
                attempts += 1;
                // the usual cause is an idle screen at boot: no commits to
                // calibrate against. One-behind is used until activity appears.
                if attempts <= 2 || attempts % 20 == 0 {
                    log::debug!(
                        "[capture-drm] fence watch: no activity to calibrate yet ({e:#}); \
                         using one-behind until the screen changes"
                    );
                }
                std::thread::sleep(Duration::from_millis(750));
            }
        }
    }
}

/// Mount tracefs and start the process-wide fence watch (idempotent). Returns
/// an error only if tracefs itself is unavailable; an idle screen is fine (the
/// caller uses one-behind until [`FenceWatch::ready`]).
pub fn start(calib: Duration) -> Result<&'static FenceWatch> {
    if let Some(w) = WATCH.get() {
        return Ok(w);
    }
    let tracefs = ensure_tracefs()?;
    let shared = Arc::new(FenceShared::default());
    let worker_shared = shared.clone();
    std::thread::Builder::new()
        .name("fence-watch".into())
        .spawn(move || worker_loop(&tracefs, calib, worker_shared))
        .context("spawn fence-watch thread")?;
    Ok(WATCH.get_or_init(|| FenceWatch { shared }))
}
