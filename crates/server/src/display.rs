//! Compositor output-mode control for the client's pre-handshake resolution
//! request (the client's `--set-mode` / `--native`).
//!
//! The server applies the request to its *own* compositor before starting
//! capture, so the DRM scanout — and therefore the session's slots — comes up
//! at the requested size. There is no mid-session resize and no reconnect: the
//! mode change happens between `Hello` and `Meta`.
//!
//! Backends shell out to the compositor's own CLI:
//!
//!   * `hyprctl`        — Hyprland (`HYPRLAND_INSTANCE_SIGNATURE` is set)
//!   * `kscreen-doctor` — KWin/KDE (`XDG_CURRENT_DESKTOP` contains KDE)
//!
//! The request names a size and refresh; the server picks the *nearest* mode
//! the output actually advertises ([`nearest`]). Everything is best-effort: a
//! missing tool, an unknown output, or a mode the display does not offer is
//! reported as a failed [`ModeReply`] and the session runs at the server's
//! current resolution.
//!
//! Changing the mode needs the user's session (`WAYLAND_DISPLAY` +
//! `XDG_RUNTIME_DIR`), which the setcap-as-user server already has; under
//! `sudo` it needs `sudo -E`.

use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};

/// A display mode: size plus refresh (mHz, 0 = unknown/any).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    pub w: u32,
    pub h: u32,
    pub refresh_mhz: u32,
}

impl Mode {
    /// `"WxH@R"` (or `"WxH"`), human-readable.
    pub fn label(&self) -> String {
        if self.refresh_mhz > 0 {
            format!(
                "{}x{}@{:.2}",
                self.w,
                self.h,
                self.refresh_mhz as f64 / 1000.0
            )
        } else {
            format!("{}x{}", self.w, self.h)
        }
    }

    /// `"WxH@R"` with a rounded integer refresh — the form both tools accept
    /// (`kscreen-doctor` matches `WxH@qRound(rate)`).
    fn spec(&self) -> String {
        if self.refresh_mhz > 0 {
            format!(
                "{}x{}@{}",
                self.w,
                self.h,
                (self.refresh_mhz as f64 / 1000.0).round() as u32
            )
        } else {
            format!("{}x{}", self.w, self.h)
        }
    }
}

/// A compositor's output control. One instance targets one output: the first
/// one, or the name given with `--display-output`.
pub trait DisplayControl: Send + Sync {
    /// Backend name, for logs.
    fn name(&self) -> &'static str;
    /// The target output's name.
    fn output(&self) -> Result<String>;
    /// The modes the target output advertises.
    fn modes(&self) -> Result<Vec<Mode>>;
    /// The target output's current mode.
    fn current(&self) -> Result<Mode>;
    /// Apply `mode` to the target output.
    fn set(&self, mode: Mode) -> Result<()>;
}

/// The advertised mode closest to `(w, h, refresh)`: an exact W×H wins, else the
/// smallest pixel-count difference, tie-broken by refresh distance (ignored
/// when `refresh_mhz == 0` or the candidate has no refresh).
pub fn nearest(modes: &[Mode], w: u32, h: u32, refresh_mhz: u32) -> Option<Mode> {
    modes.iter().copied().min_by_key(|m| {
        let dp = (m.w as i64 * m.h as i64 - w as i64 * h as i64).unsigned_abs();
        let dr = if refresh_mhz == 0 || m.refresh_mhz == 0 {
            0
        } else {
            (m.refresh_mhz as i64 - refresh_mhz as i64).unsigned_abs()
        };
        (dp, dr)
    })
}

/// Choose a backend. `pref` is `auto|hyprctl|kscreen|none`; `output` pins the
/// target output name. `auto` detects from the session environment; `none`
/// disables the feature.
pub fn detect(pref: &str, output: Option<&str>) -> Result<Option<Arc<dyn DisplayControl>>> {
    let forced = match pref {
        "none" | "off" => return Ok(None),
        "hyprctl" | "hyprland" => Some(Backend::Hyprctl),
        "kscreen" | "kscreen-doctor" | "kde" => Some(Backend::Kscreen),
        "auto" => None,
        other => bail!("--display-control must be auto|hyprctl|kscreen|none, got {other:?}"),
    };
    let backend = match forced {
        Some(b) => b,
        None => {
            if hyprland_signature().is_some() {
                Backend::Hyprctl
            } else if std::env::var("XDG_CURRENT_DESKTOP")
                .map(|d| d.to_ascii_uppercase().contains("KDE"))
                .unwrap_or(false)
            {
                Backend::Kscreen
            } else {
                return Ok(None);
            }
        }
    };
    // `hyprctl` locates the instance through HYPRLAND_INSTANCE_SIGNATURE; if the
    // caller did not set it, fill it in from the runtime dir so both `auto` and
    // an explicit `--display-control hyprctl` reach a live instance.
    if matches!(backend, Backend::Hyprctl)
        && std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_none()
    {
        if let Some(sig) = infer_hyprland_signature() {
            std::env::set_var("HYPRLAND_INSTANCE_SIGNATURE", sig);
        }
    }
    let ctl: Arc<dyn DisplayControl> = match backend {
        Backend::Hyprctl => Arc::new(Hyprctl {
            output: output.map(str::to_string),
        }),
        Backend::Kscreen => Arc::new(Kscreen {
            output: output.map(str::to_string),
        }),
    };
    Ok(Some(ctl))
}

/// The Hyprland instance signature: `HYPRLAND_INSTANCE_SIGNATURE` when set,
/// otherwise the newest instance inferred from the runtime dir.
fn hyprland_signature() -> Option<String> {
    match std::env::var("HYPRLAND_INSTANCE_SIGNATURE") {
        Ok(sig) if !sig.is_empty() => Some(sig),
        _ => infer_hyprland_signature(),
    }
}

/// Infer the Hyprland instance signature from `$XDG_RUNTIME_DIR/hypr/<sig>`:
/// each live instance owns a `.socket.sock` there. Picks the newest if several.
fn infer_hyprland_signature() -> Option<String> {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")?;
    let hypr = std::path::Path::new(&runtime).join("hypr");
    let mut best: Option<(std::time::SystemTime, String)> = None;
    for entry in std::fs::read_dir(&hypr).ok()?.flatten() {
        let Ok(meta) = std::fs::metadata(entry.path().join(".socket.sock")) else {
            continue;
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
            best = Some((mtime, name));
        }
    }
    best.map(|(_, name)| name)
}

#[derive(Clone, Copy)]
enum Backend {
    Hyprctl,
    Kscreen,
}

// ---------------------------------------------------------------------------
// restore-on-signal
// ---------------------------------------------------------------------------

static RESTORE: Mutex<Option<(Arc<dyn DisplayControl>, Mode)>> = Mutex::new(None);

/// Remember the mode to put back on a quit signal (and the backend to do it
/// with). Called when a session changes the mode.
pub fn arm_restore(ctl: Arc<dyn DisplayControl>, original: Mode) {
    *RESTORE.lock().unwrap() = Some((ctl, original));
}

/// Forget the armed restore (the mode has been put back, or the session ended
/// cleanly).
pub fn disarm_restore() {
    *RESTORE.lock().unwrap() = None;
}

/// Register the shared signal watcher's display-restore hook. Call once, after
/// [`detect`] returned a backend.
pub fn install_restore_watcher() {
    crate::signals::on_signal(Box::new(|sig| {
        let target = RESTORE.lock().unwrap().take();
        if let Some((ctl, m)) = target {
            eprintln!("[display] signal {sig}; restoring mode {}", m.label());
            if let Err(e) = ctl.set(m) {
                eprintln!("[display] restore failed: {e:#}");
            }
        }
    }));
}

// ---------------------------------------------------------------------------
// hyprctl (Hyprland)
// ---------------------------------------------------------------------------

struct Hyprctl {
    output: Option<String>,
}

impl Hyprctl {
    fn monitors(&self) -> Result<Vec<serde_json::Value>> {
        let out = run("hyprctl", &["-j", "monitors"])?;
        let v: serde_json::Value =
            serde_json::from_str(&out).context("parsing hyprctl -j monitors")?;
        Ok(v.as_array().cloned().unwrap_or_default())
    }

    fn pick<'a>(&self, mons: &'a [serde_json::Value]) -> Result<&'a serde_json::Value> {
        match &self.output {
            Some(name) => mons
                .iter()
                .find(|m| m["name"].as_str() == Some(name.as_str()))
                .with_context(|| format!("hyprctl: no monitor named {name:?}")),
            None => mons.first().context("hyprctl: no monitors"),
        }
    }

    fn mode_of(m: &serde_json::Value) -> Result<Mode> {
        let w = m["width"].as_u64().context("hyprctl: monitor width")? as u32;
        let h = m["height"].as_u64().context("hyprctl: monitor height")? as u32;
        let r = m["refreshRate"].as_f64().unwrap_or(0.0);
        Ok(Mode {
            w,
            h,
            refresh_mhz: (r * 1000.0).round() as u32,
        })
    }
}

impl DisplayControl for Hyprctl {
    fn name(&self) -> &'static str {
        "hyprctl"
    }

    fn output(&self) -> Result<String> {
        let mons = self.monitors()?;
        let m = self.pick(&mons)?;
        m["name"]
            .as_str()
            .map(str::to_string)
            .context("hyprctl: monitor name")
    }

    fn modes(&self) -> Result<Vec<Mode>> {
        let mons = self.monitors()?;
        let m = self.pick(&mons)?;
        let mut out = Vec::new();
        if let Some(arr) = m["availableModes"].as_array() {
            for s in arr {
                if let Some(mode) = s.as_str().and_then(parse_mode_token) {
                    if !out.contains(&mode) {
                        out.push(mode);
                    }
                }
            }
        }
        if out.is_empty() {
            // some builds omit availableModes: at least offer the current mode
            out.push(Self::mode_of(m)?);
        }
        Ok(out)
    }

    fn current(&self) -> Result<Mode> {
        let mons = self.monitors()?;
        let m = self.pick(&mons)?;
        Self::mode_of(m)
    }

    fn set(&self, mode: Mode) -> Result<()> {
        let mons = self.monitors()?;
        let m = self.pick(&mons)?;
        let name = m["name"].as_str().context("hyprctl: monitor name")?;
        // keep the monitor's scale; let Hyprland place it
        let scale = m["scale"].as_f64().unwrap_or(1.0);
        let spec = format!("{name},{},auto,{scale}", mode.spec());
        run("hyprctl", &["keyword", "monitor", &spec]).map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// kscreen-doctor (KWin / KDE)
// ---------------------------------------------------------------------------

struct Kscreen {
    output: Option<String>,
}

/// One output parsed from `kscreen-doctor -o`.
#[derive(Debug, PartialEq, Eq)]
struct KOut {
    id: String,
    name: String,
    modes: Vec<Mode>,
    current: Option<Mode>,
}

impl Kscreen {
    fn listing(&self) -> Result<Vec<KOut>> {
        let text = run("kscreen-doctor", &["-o"])?;
        Ok(parse_kscreen(&text))
    }

    fn pick<'a>(&self, outs: &'a [KOut]) -> Result<&'a KOut> {
        match &self.output {
            Some(name) => outs
                .iter()
                .find(|o| &o.name == name || &o.id == name)
                .with_context(|| format!("kscreen-doctor: no output named {name:?}")),
            None => outs.first().context("kscreen-doctor: no outputs"),
        }
    }
}

impl DisplayControl for Kscreen {
    fn name(&self) -> &'static str {
        "kscreen-doctor"
    }

    fn output(&self) -> Result<String> {
        let outs = self.listing()?;
        Ok(self.pick(&outs)?.name.clone())
    }

    fn modes(&self) -> Result<Vec<Mode>> {
        let outs = self.listing()?;
        Ok(self.pick(&outs)?.modes.clone())
    }

    fn current(&self) -> Result<Mode> {
        let outs = self.listing()?;
        let o = self.pick(&outs)?;
        o.current
            .or_else(|| o.modes.first().copied())
            .context("kscreen-doctor: no current mode")
    }

    fn set(&self, mode: Mode) -> Result<()> {
        let outs = self.listing()?;
        let o = self.pick(&outs)?;
        let target = if o.name.is_empty() { &o.id } else { &o.name };
        let arg = format!("output.{target}.mode.{}", mode.spec());
        run("kscreen-doctor", &[&arg]).map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// parsing helpers
// ---------------------------------------------------------------------------

/// Parse a mode token as printed by either tool: `"1920x1080@144.15Hz"` (hyprctl
/// `availableModes`), `"1:1920x1080@144*"` (kscreen `Modes:` list), or
/// `"2880x1800@120.00"` (kscreen multi-line list). Trailing `*` (current) / `!`
/// (preferred) markers and a leading `N:` index are ignored.
fn parse_mode_token(tok: &str) -> Option<Mode> {
    let tok = tok.trim().trim_end_matches(['*', '!', ',']);
    let tok = tok.rsplit(':').next().unwrap_or(tok);
    let (wh, r) = tok.split_once('@')?;
    let (w, h) = wh.split_once('x')?;
    let w: u32 = w.trim().parse().ok()?;
    let h: u32 = h.trim().parse().ok()?;
    // drop a trailing unit ("Hz")
    let r = r.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let rf: f64 = r.trim().parse().ok()?;
    Some(Mode {
        w,
        h,
        refresh_mhz: (rf * 1000.0).round() as u32,
    })
}

/// Parse `kscreen-doctor -o`. Handles both the multi-line layout (one
/// `Output:` header then indented `Modes:` / `Geometry:` lines) and the
/// single-line layout (everything after `Output: <id> <name>` on one line).
fn parse_kscreen(text: &str) -> Vec<KOut> {
    let mut outs: Vec<KOut> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("Output:") {
            let mut it = rest.split_whitespace();
            let id = it.next().unwrap_or("").to_string();
            let name = it.next().unwrap_or("").to_string();
            outs.push(KOut {
                id,
                name,
                modes: Vec::new(),
                current: None,
            });
            // fall through: the same line may carry the mode list
        }
        let Some(cur) = outs.last_mut() else {
            continue;
        };
        for tok in line.split_whitespace() {
            if let Some(m) = parse_mode_token(tok) {
                if !cur.modes.contains(&m) {
                    cur.modes.push(m);
                }
                if tok.trim_end_matches(',').ends_with('*') {
                    cur.current = Some(m);
                }
            }
        }
        // "Geometry: 0,0 1920x1080" — the current size when no `*` marker
        if line.contains("Geometry:") && cur.current.is_none() {
            for tok in line.split_whitespace() {
                if let Some((w, h)) = tok.split_once('x') {
                    if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                        cur.current = Some(Mode {
                            w,
                            h,
                            refresh_mhz: 0,
                        });
                        break;
                    }
                }
            }
        }
    }
    outs
}

/// Run a helper, merging stderr into the text (some tools print their reason on
/// stderr) and failing on a non-zero exit.
fn run(cmd: &str, args: &[&str]) -> Result<String> {    let out = Command::new(cmd)
        .args(args)
        .output()
        .with_context(|| format!("running {cmd} (installed and on PATH?)"))?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.trim().is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(err.trim());
    }
    if !out.status.success() {
        bail!("{cmd} {} failed: {}", args.join(" "), text.trim());
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mode_tokens() {
        assert_eq!(
            parse_mode_token("1920x1080@144.15Hz"),
            Some(Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 144_150
            })
        );
        assert_eq!(
            parse_mode_token("1:1920x1080@144*"),
            Some(Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 144_000
            })
        );
        assert_eq!(
            parse_mode_token("2880x1800@120.00"),
            Some(Mode {
                w: 2880,
                h: 1800,
                refresh_mhz: 120_000
            })
        );
        assert_eq!(parse_mode_token("Geometry:"), None);
        assert_eq!(parse_mode_token("0,0"), None);
        assert_eq!(parse_mode_token("1920x1080"), None); // no refresh
    }

    #[test]
    fn nearest_prefers_exact_size_then_refresh() {
        let modes = [
            Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 60_000,
            },
            Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 144_000,
            },
            Mode {
                w: 2560,
                h: 1440,
                refresh_mhz: 60_000,
            },
        ];
        // exact size, closest refresh
        assert_eq!(
            nearest(&modes, 1920, 1080, 143_000),
            Some(Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 144_000
            })
        );
        // no exact size: smallest pixel-count difference
        assert_eq!(
            nearest(&modes, 2500, 1400, 0),
            Some(Mode {
                w: 2560,
                h: 1440,
                refresh_mhz: 60_000
            })
        );
        // refresh 0 = any: first exact-size candidate wins
        assert_eq!(
            nearest(&modes, 1920, 1080, 0),
            Some(Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 60_000
            })
        );
        assert_eq!(nearest(&[], 1920, 1080, 0), None);
    }

    #[test]
    fn kscreen_multiline() {
        let text = "\
Output: 1 eDP-1 8e5f5d9c-uuid
\tpriorities: 1
\tenabled
\tconnected
\tModes: 
\t\t1: 2880x1800@120.00
\t\t2: 2880x1800@60.00
\t\t3: 1920x1080@120.00
\tGeometry: 0,0 2880x1800
\tScale: 1.5
";
        let outs = parse_kscreen(text);
        assert_eq!(outs.len(), 1);
        let o = &outs[0];
        assert_eq!(o.id, "1");
        assert_eq!(o.name, "eDP-1");
        assert_eq!(o.modes.len(), 3);
        assert_eq!(o.current, Some(Mode { w: 2880, h: 1800, refresh_mhz: 0 }));
    }

    #[test]
    fn kscreen_single_line() {
        let text = "Output: 1 DVI-D-1 enabled connected priority 1 DVI Modes: 0:1920x1080@60! 1:1920x1080@144* 2:1920x1080@120 Geometry: 0,0 1920x1080 Scale: 1 Rotation: 1\n";
        let outs = parse_kscreen(text);
        assert_eq!(outs.len(), 1);
        let o = &outs[0];
        assert_eq!(o.name, "DVI-D-1");
        assert_eq!(o.modes.len(), 3);
        assert_eq!(
            o.current,
            Some(Mode {
                w: 1920,
                h: 1080,
                refresh_mhz: 144_000
            })
        );
    }

    #[test]
    fn mode_spec_rounds_refresh() {
        let m = Mode {
            w: 2560,
            h: 1440,
            refresh_mhz: 143_981,
        };
        assert_eq!(m.spec(), "2560x1440@144");
        assert_eq!(m.label(), "2560x1440@143.98");
    }
}
