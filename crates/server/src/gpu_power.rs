//! Best-effort NVIDIA GPU clock pinning for `--gpu-power`.
//!
//! The lossless encode path on the reference box is memory-bandwidth bound, and
//! the driver was caught toggling between the fast P5 state (mem 810 MHz) and
//! the slow idle P8 state (mem 405 MHz) mid-session, swinging `encode` between
//! ~0.3 ms and ~4 ms. nvidia-smi cannot select a P-state directly, so we pin
//! the clock limits instead: `p5` uses the operating point we measured, `p0`
//! pins the card's maximum, and `<sm>[,<mem>]` takes explicit MHz. nvidia-smi
//! (not nvidia-settings) is used deliberately: the lock commands work headless,
//! with no X/NV-CONTROL display.
//!
//! Everything here is advisory. Clock locking needs root and a supporting GPU;
//! a missing nvidia-smi, a permission error, or an unsupported clock is logged
//! and the session runs on with the driver's own power management. The two
//! failure modes seen on the reference 2080 Ti are handled explicitly:
//! `-lgc` needs root and prints "does not have permission" to *stdout* with a
//! non-zero exit, while `-lmc` is not supported on GeForce at all and prints
//! "not supported" to stdout with exit 0 — so `lock()` inspects the output
//! rather than trusting the status alone.

use anyhow::{bail, Context, Result};

/// The P5 operating point measured on the reference GPU (nvidia-smi
/// `clocks.sm` / `clocks.mem` while the memprobe reported the fast state).
/// Override with `--gpu-power <sm>[,<mem>]` on different hardware. Note the
/// memory half is rejected on GeForce cards (the reference 2080 Ti included);
/// there the graphics lock alone has to hold PowerMizer out of P8.
const P5_SM_MHZ: u32 = 435;
const P5_MEM_MHZ: u32 = 810;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuPower {
    /// Leave the driver's power management alone (default).
    Off,
    /// Pin to the observed P5 operating point.
    P5,
    /// Pin to the card's maximum graphics and memory clocks (P0).
    Max,
    /// Pin to explicit `<sm>[,<mem>]` MHz; memory is optional.
    Custom { sm: u32, mem: Option<u32> },
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

    /// Apply the pin. Returns an error for display only: the caller logs it and
    /// keeps streaming, so this is never fatal.
    pub fn apply(self) -> Result<()> {
        match self {
            Self::Off => Ok(()),
            Self::P5 => pin(P5_SM_MHZ, Some(P5_MEM_MHZ), "p5"),
            Self::Max => {
                let (sm, mem) = max_clocks()?;
                pin(sm, Some(mem), "p0/max")
            }
            Self::Custom { sm, mem } => pin(sm, mem, "custom"),
        }
    }
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

/// Apply one `-lgc`/`-lmc` pin. nvidia-smi reports "not supported" and "does
/// not have permission" as a *successful* exit with the reason on stdout
/// (GeForce cards reject the memory lock this way), so the exit status alone
/// does not mean the clock moved.
fn lock(flag: &str, mhz: u32) -> Result<()> {
    let out = nvidia_smi(&[flag, &format!("{mhz},{mhz}")])?;
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

/// `clocks.max.sm` / `clocks.max.mem` for GPU 0.
fn max_clocks() -> Result<(u32, u32)> {
    let s = nvidia_smi(&[
        "--query-gpu=clocks.max.sm,clocks.max.mem",
        "--format=csv,noheader,nounits",
    ])?;
    let line = s.lines().next().unwrap_or("").trim();
    let (sm, mem) = line
        .split_once(',')
        .with_context(|| format!("unexpected nvidia-smi clock output: {line:?}"))?;
    Ok((
        sm.trim().parse().context("parsing clocks.max.sm")?,
        mem.trim().parse().context("parsing clocks.max.mem")?,
    ))
}

/// Lock the graphics (and optionally memory) clock to a fixed value. Each lock
/// is attempted independently so a card that rejects one still gets the other;
/// only a complete failure is reported.
fn pin(sm: u32, mem: Option<u32>, label: &str) -> Result<()> {
    // persistence mode keeps the driver resident so the locks survive idle;
    // best-effort — a permission error here is not fatal on its own
    if let Err(e) = nvidia_smi(&["-pm", "1"]) {
        eprintln!("[gpu-power] {label}: persistence mode not set ({e:#})");
    }

    let mut locked = Vec::new();
    match lock("-lgc", sm) {
        Ok(()) => {
            locked.push(format!("sm {sm} MHz"));
            eprintln!("[gpu-power] {label}: locked graphics clock to {sm} MHz");
        }
        Err(e) => eprintln!("[gpu-power] {label}: graphics clock lock failed: {e:#}"),
    }
    if let Some(mem) = mem {
        match lock("-lmc", mem) {
            Ok(()) => {
                locked.push(format!("mem {mem} MHz"));
                eprintln!("[gpu-power] {label}: locked memory clock to {mem} MHz");
            }
            Err(e) => eprintln!("[gpu-power] {label}: memory clock lock failed: {e:#}"),
        }
    }
    if locked.is_empty() {
        bail!("{label}: no clock could be locked (need root / a supported GPU)");
    }
    if let Ok(state) = nvidia_smi(&[
        "--query-gpu=pstate,clocks.sm,clocks.mem",
        "--format=csv,noheader",
    ]) {
        eprintln!(
            "[gpu-power] {label}: pinned ({}) -> {state}",
            locked.join(", ")
        );
    }
    // otherwise the pins outlive this process; say how to undo them
    eprintln!("[gpu-power] release with: nvidia-smi -i 0 -rgc -rmc");
    Ok(())
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
    fn parses_explicit_clocks() {
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
}
