//! One signal watcher for the whole process.
//!
//! More than one feature needs to undo something before the process dies:
//! `--gpu-power` releases the clock floors it set, and `--display-control`
//! restores the display mode it changed. Installing a handler per feature would
//! simply replace the previous one, so instead each registers a hook here. The
//! first registration installs SIGINT/SIGTERM/SIGQUIT handlers and a single
//! watcher thread that runs every hook, in registration order, on a signal,
//! then exits.
//!
//! The handlers only record the signal number (async-signal-safe); the actual
//! work happens on the watcher thread.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, Once, OnceLock};

type Hook = Box<dyn FnOnce(i32) + Send>;

static GOT_SIGNAL: AtomicI32 = AtomicI32::new(0);
static HOOKS: OnceLock<Mutex<Vec<Hook>>> = OnceLock::new();
static INSTALL: Once = Once::new();

fn hooks() -> &'static Mutex<Vec<Hook>> {
    HOOKS.get_or_init(|| Mutex::new(Vec::new()))
}

extern "C" fn record_signal(sig: libc::c_int) {
    GOT_SIGNAL.store(sig, Ordering::SeqCst);
}

/// Register a cleanup hook. The first call installs the signal handlers and the
/// watcher thread; later calls just append. With no hooks registered the
/// default signal behaviour (immediate termination) is left in place.
pub fn on_signal(hook: Hook) {
    hooks().lock().unwrap().push(hook);
    INSTALL.call_once(|| {
        // sigaction rather than signal(): some libcs reset the handler on
        // delivery.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            let handler: extern "C" fn(libc::c_int) = record_signal;
            sa.sa_sigaction = handler as usize;
            sa.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut sa.sa_mask);
            for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGQUIT] {
                libc::sigaction(sig, &sa, std::ptr::null_mut());
            }
        }
        std::thread::Builder::new()
            .name("signal-watch".into())
            .spawn(|| {
                let sig = loop {
                    let s = GOT_SIGNAL.load(Ordering::SeqCst);
                    if s != 0 {
                        break s;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                };
                let mut hs = hooks().lock().unwrap();
                for h in hs.drain(..) {
                    h(sig);
                }
                drop(hs);
                std::process::exit(128 + sig);
            })
            .expect("spawn signal watcher");
    });
}
