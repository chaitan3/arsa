//! Cooperative shutdown for background workers.
//!
//! The server starts capture sources, input sinks, and audio capture on demand
//! and must be able to stop them again (on client disconnect, and for a clean
//! idle state). A worker checks its [`Stop`] token in its loop (and wakes from
//! blocking waits promptly) and returns; the owning [`Running`] signals the
//! token and joins, so no worker thread or the resources it holds can outlive
//! the handle.
//!
//! `Running` also stops on drop, so the failure mode is "too early" rather
//! than "leaked thread holding the DRM card / a GL context / a pulse stream".

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// A clonable stop signal. `stopped()` is checked by workers.
#[derive(Clone, Default)]
pub struct Stop(Arc<AtomicBool>);

impl Stop {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request shutdown. Idempotent.
    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn stopped(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// A running set of worker threads that share one [`Stop`] token. `stop`
/// signals them and joins; dropping the handle does the same, so workers can
/// never outlive it.
pub struct Running {
    stop: Stop,
    joins: Vec<JoinHandle<()>>,
}

impl Running {
    pub fn new(stop: Stop, joins: Vec<JoinHandle<()>>) -> Self {
        Self { stop, joins }
    }

    /// The token to hand to worker loops.
    pub fn stop_signal(&self) -> &Stop {
        &self.stop
    }

    /// Signal every worker and wait for them to exit.
    pub fn stop(&mut self) {
        self.stop.stop();
        for j in self.joins.drain(..) {
            let _ = j.join();
        }
    }

    /// True once every worker thread has exited on its own.
    pub fn is_finished(&self) -> bool {
        self.joins.iter().all(|j| j.is_finished())
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Sleep up to `dur`, waking early when `stop` is set. Returns true if stopped.
/// Lets a poll/retry sleep not delay shutdown.
pub fn sleep_or_stop(stop: &Stop, dur: std::time::Duration) -> bool {
    let step = std::time::Duration::from_millis(5);
    let deadline = std::time::Instant::now() + dur;
    while !stop.stopped() {
        let now = std::time::Instant::now();
        if now >= deadline {
            return false;
        }
        std::thread::sleep(step.min(deadline - now));
    }
    true
}
