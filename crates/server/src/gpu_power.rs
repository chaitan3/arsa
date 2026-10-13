//! Best-effort NVIDIA GPU clock flooring for `--gpu-power`.
//!
//! `nvidia-smi --lock-gpu-clocks=<min>,<max>` constrains the graphics clock to
//! the inclusive range [min, max]: with `min == max` it pins the clock, but with
//! `min < max` the driver may still boost freely up to `max`. We only ever want
//! the floor. The lossless encode path on the reference box is bandwidth bound
//! and the driver was caught dropping to the idle P8 state (mem 405 MHz)
//! mid-session, swinging `encode` between ~0.3 ms and ~4 ms; we want to stop
//! that drop without capping the boost. So every preset is a *minimum* and the
//! ceiling is always the card's `clocks.max.*`.
//!
//! A floor survives this process, so `install_cleanup()` catches SIGINT/SIGTERM/
//! SIGQUIT and releases exactly the floors we set before exiting. SIGKILL and a
//! reboot still leave nothing behind (the driver resets on load).
//!
//! nvidia-smi (not nvidia-settings) is used deliberately: these commands work
//! headless, with no X/NV-CONTROL display.
//!
//! Everything here is advisory. Clock floors need root and a supporting GPU; a
//! missing nvidia-smi, a permission error, or an unsupported clock is logged and
//! the session runs on with the driver's own power management. The two failure
//! modes seen on the reference 2080 Ti are handled explicitly: `-lgc` needs root
//! and prints "does not have permission" to *stdout* with a non-zero exit, while
//! `-lmc` is not supported on GeForce at all and prints "not supported" to
//! stdout with exit 0 — so `lock_range()` inspects the output, not just the exit
//! status.

use anyhow::{bail, Context, Result};

/// The P5 operating point measured on the reference GPU (nvidia-smi
/// `clocks.sm` / `clocks.mem` while the memprobe reported the fast state).
/// These are *floors*: the GPU is free to boost above them. Override with
/// `--gpu-power <sm>[,<mem>]` on different hardware. The memory half is
/// rejected on GeForce cards (the reference 2080 Ti included); there the
/// graphics floor alone has to hold PowerMizer out of P8.
const P5_SM_MHZ: u32 = 435;
const P5_MEM_MHZ: u32 = 810;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuPower {
    /// Leave the driver's power management alone (default).
    Off,
    /// Floor at the observed P5 operating point, boost allowed.
    P5,
    /// Floor at the card's maximum (effectively pinned there).
    Max,
    /// Floor at explicit `<sm>[,<mem>]` MHz, boost allowed; memory is optional.
    Custom { sm: u32, mem: Option<u32> },
}

/// Which clock floors were actually applied, so cleanup releases only those.
#[derive(Clone, Copy, Debug, Default)]
pub struct Applied {
    sm: bool,
    mem: bool,
}

impl Applied {
    fn any(self) -> bool {
        self.sm || self.mem
    }

    /// Release the floors we set. A card that rejected one never gets a reset
    /// for it, so a clock set by someone else is not clobbered.
    fn reset(self) {
        if self.sm {
            match nvidia_smi(&["-rgc"]) {
                Ok(_) => eprintln!("[gpu-power] graphics clock floor released"),
                Err(e) => eprintln!("[gpu-power] releasing graphics clock floor failed: {e:#}"),
            }
        }
        if self.mem {
            match nvidia_smi(&["-rmc"]) {
                Ok(_) => eprintln!("[gpu-power] memory clock floor released"),
                Err(e) => eprintln!("[gpu-power] releasing memory clock floor failed: {e:#}"),
            }
        }
    }
}

impl GpuPower {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "off" | "none" => Ok(Self::Off),
            "p5" | "5" => Ok(Self::P5),
            "p0" | "max" => Ok(Self::Max),
            other => {
                let (sm, mem) = match other.split_once(',') {
                    Some((sm, mem)) => (sm, Some(mem)),
                    None => (other, None),
                };
                let sm: u32 = sm.trim().parse().map_err(|_| {
                    anyhow::anyhow!(
                        "--gpu-power must be off|p5|p0|max|<sm>[,<mem>], got {other:?}"
                    )
                })?;
                let mem = match mem {
                    Some(m) => Some(m.trim().parse().with_context(|| {
                        format!("--gpu-power memory clock must be MHz, got {m:?}")
                    })?),
                    None => None,
                };
                Ok(Self::Custom { sm, mem })
            }
        }
    }

    /// Apply the floor. Returns an error for display only: the caller logs it
    /// and keeps streaming, so this is never fatal.
    pub fn apply(self) -> Result<Applied> {
        match self {
            Self::Off => Ok(Applied::default()),
            Self::P5 => pin(P5_SM_MHZ, Some(P5_MEM_MHZ), max_clocks()?, "p5"),
            Self::Max => {
                let ceil = max_clocks()?;
                pin(ceil.sm, Some(ceil.mem), ceil, "p0/max")
            }
            Self::Custom { sm, mem } => pin(sm, mem, max_clocks()?, "custom"),
        }
    }
}

/// Release the clock floors on SIGINT (ctrl-c), SIGTERM or SIGQUIT. No-op when
/// nothing was floored, so an unpinned run keeps the default signal behavior.
pub fn install_cleanup(applied: Applied) {
    if !applied.any() {
        return;
    }
    // the shared watcher runs this alongside any other cleanup (display mode)
    crate::signals::on_signal(Box::new(move |sig| {
        eprintln!("[gpu-power] quit signal ({sig}); releasing clock floors");
        applied.reset();
    }));
}

/// The card's boost clocks, used as the ceiling so a floor never caps boost.
#[derive(Clone, Copy, Debug)]
struct MaxClocks {
    sm: u32,
    mem: u32,
}

/// Run nvidia-smi against GPU 0. stdout and stderr are merged because some
/// failures ("not supported", "does not have permission") are printed to
/// stdout; a non-zero exit is reported with that combined text.
fn nvidia_smi(args: &[&str]) -> Result<String> {
    let out = std::process::Command::new("nvidia-smi")
        .arg("-i")
        .arg("0")
        .args(args)
        .output()
        .context("running nvidia-smi (NVIDIA driver installed?)")?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.trim().is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(err.trim());
    }
    let text = text.trim().to_string();
    if !out.status.success() {
        bail!("nvidia-smi {} failed: {text}", args.join(" "));
    }
    Ok(text)
}

/// `clocks.max.sm` / `clocks.max.mem` for GPU 0.
fn max_clocks() -> Result<MaxClocks> {
    let s = nvidia_smi(&[
        "--query-gpu=clocks.max.sm,clocks.max.mem",
        "--format=csv,noheader,nounits",
    ])?;
    let line = s.lines().next().unwrap_or("").trim();
    let (sm, mem) = line
        .split_once(',')
        .with_context(|| format!("unexpected nvidia-smi clock output: {line:?}"))?;
    Ok(MaxClocks {
        sm: sm.trim().parse().context("parsing clocks.max.sm")?,
        mem: mem.trim().parse().context("parsing clocks.max.mem")?,
    })
}

/// Apply one `-lgc`/`-lmc` range. nvidia-smi reports "not supported" and "does
/// not have permission" as a *successful* exit with the reason on stdout
/// (GeForce cards reject the memory range this way), so the exit status alone
/// does not mean the clock moved.
fn lock_range(flag: &str, min: u32, max: u32) -> Result<()> {
    let out = nvidia_smi(&[flag, &format!("{min},{max}")])?;
    let low = out.to_ascii_lowercase();
    for bad in [
        "not supported",
        "does not have permission",
        "permission denied",
        "insufficient permission",
        "terminating early",
    ] {
        if low.contains(bad) {
            bail!("{out}");
        }
    }
    Ok(())
}

/// Floor the graphics (and optionally memory) clock at `floor`, leaving the
/// ceiling at the card maximum so the GPU may still boost. Each clock is
/// attempted independently so a card that rejects one still gets the other;
/// only a complete failure is reported.
fn pin(sm_floor: u32, mem_floor: Option<u32>, ceil: MaxClocks, label: &str) -> Result<Applied> {
    // persistence mode keeps the driver resident so the floors survive idle;
    // best-effort — a permission error here is not fatal on its own
    if let Err(e) = nvidia_smi(&["-pm", "1"]) {
        eprintln!("[gpu-power] {label}: persistence mode not set ({e:#})");
    }

    let mut applied = Applied::default();
    match lock_range("-lgc", sm_floor, ceil.sm) {
        Ok(()) => {
            applied.sm = true;
            eprintln!(
                "[gpu-power] {label}: graphics clock floored at {sm_floor} MHz (may boost to {})",
                ceil.sm
            );
        }
        Err(e) => eprintln!("[gpu-power] {label}: graphics clock floor failed: {e:#}"),
    }
    if let Some(mem_floor) = mem_floor {
        match lock_range("-lmc", mem_floor, ceil.mem) {
            Ok(()) => {
                applied.mem = true;
                eprintln!(
                    "[gpu-power] {label}: memory clock floored at {mem_floor} MHz (may boost to {})",
                    ceil.mem
                );
            }
            Err(e) => eprintln!("[gpu-power] {label}: memory clock floor failed: {e:#}"),
        }
    }
    if !applied.any() {
        bail!("{label}: no clock floor could be set (need root / a supported GPU)");
    }
    if let Ok(state) = nvidia_smi(&[
        "--query-gpu=pstate,clocks.sm,clocks.mem",
        "--format=csv,noheader",
    ]) {
        eprintln!("[gpu-power] {label}: floored -> {state}");
    }
    eprintln!("[gpu-power] floors will be released on exit (manual: nvidia-smi -i 0 -rgc -rmc)");
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_presets() {
        assert_eq!(GpuPower::parse("off").unwrap(), GpuPower::Off);
        assert_eq!(GpuPower::parse("p5").unwrap(), GpuPower::P5);
        assert_eq!(GpuPower::parse("5").unwrap(), GpuPower::P5);
        assert_eq!(GpuPower::parse("p0").unwrap(), GpuPower::Max);
        assert_eq!(GpuPower::parse("max").unwrap(), GpuPower::Max);
    }

    #[test]
    fn parses_explicit_floors() {
        assert_eq!(
            GpuPower::parse("435,810").unwrap(),
            GpuPower::Custom {
                sm: 435,
                mem: Some(810)
            }
        );
        assert_eq!(
            GpuPower::parse("1200").unwrap(),
            GpuPower::Custom {
                sm: 1200,
                mem: None
            }
        );
    }

    #[test]
    fn rejects_junk() {
        assert!(GpuPower::parse("p9").is_err());
        assert!(GpuPower::parse("435,fast").is_err());
        assert!(GpuPower::parse("").is_err());
    }

    #[test]
    fn cleanup_applies_only_to_what_was_set() {
        assert!(!Applied::default().any());
        assert!(Applied { sm: true, mem: false }.any());
        assert!(Applied { sm: false, mem: true }.any());
        assert!(Applied { sm: true, mem: true }.any());
    }
}
