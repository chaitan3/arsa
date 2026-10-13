# Display mode control (client-requested resolution)

Status: implemented (2026-10-13)

## Goal

Let a client ask the server to change its compositor output resolution before a
session starts — either to the client's own ("native") mode or to an explicit
`WxH@R`. This makes the client pixel-for-pixel 1:1 with the server scanout
(no compositor-side scaling) and is a prerequisite for reliable direct scanout.

## Why it happens at connect (not mid-session)

A resolution change changes the DRM scanout size, which changes
`frame_bytes`, the codec geometry, the RDMA slot size, the input coordinate
mapping and the client presenter. Today a mid-session size change is not
negotiated: the server drops frames larger than the slot and the client's
`apply_meta` refuses a `frame_bytes` change.

Instead, the request is applied **between `Hello` and `Meta`**, while the
server is still idle:

```
client                                server
  |-- Hello ------------------------->|   negotiate video transport
  |-- SetMode { native|WxH@R } ------>|   apply to the compositor
  |                                   |   re-probe the DRM scanout
  |<---------------------- ModeReply -|   ok / error + mode now in effect
  |-- (RDMA endpoint exchange) ------>|
  |<---------------------------- Meta -|   sized for the *new* geometry
  |-- Buffers (sized for Meta) ------>|
  |<--------------------------- Ready -|
```

No reconnect, no in-session resize. The capture source starts only after the
mode is in effect, and the session's slots are sized from a fresh
`probe_display`.

## Protocol (v13)

`crates/proto`:

- `SetMode` (client → server): `Keep` | `Set { width, height, refresh_mhz }` |
  `Reset`.
- `ModeReply` (server → client): `ok`, the mode now in effect, and a message.

The client always sends exactly one `SetMode` right after the `Hello` reply and
blocks on the `ModeReply` before the RDMA endpoint exchange. `Keep` means "no
change" (the default; a plain `rdma-display-client` run does this).

## Server backends

`crates/server/src/display.rs`. `--display-control auto|hyprctl|kscreen|none`
(default `auto`; only meaningful for `--source drm`).

| Backend        | Compositor     | List                | Set                                              |
| -------------- | -------------- | ------------------- | ------------------------------------------------ |
| `hyprctl`      | Hyprland       | `hyprctl -j monitors` | `hyprctl keyword monitor <name>,<WxH@R>,auto,<scale>` |
| `kscreen-doctor` | KWin / KDE   | `kscreen-doctor -o` | `kscreen-doctor output.<name>.mode.<WxH@R>`      |

`auto` detects Hyprland from `HYPRLAND_INSTANCE_SIGNATURE`, KDE from
`XDG_CURRENT_DESKTOP`. Both need the user's session (`WAYLAND_DISPLAY` +
`XDG_RUNTIME_DIR`), which the setcap-as-user server already has; under `sudo`
use `sudo -E`. `--display-output NAME` pins the target output (default: the
first).

The requested size is mapped to the **nearest advertised mode** (exact W×H,
else smallest pixel-count difference, tie-broken by refresh). If the output
advertises nothing suitable — or there is no backend — the reply is an error
and the session runs at the server's current resolution.

## Restore on disconnect

The mode in effect before the first change is remembered. It is put back when
the session ends (including a client that requested a change and then never
established a session), and on `SIGINT`/`SIGTERM`/`SIGQUIT` via the shared
signal watcher (`crates/server/src/signals.rs`, shared with `--gpu-power`).

## Client

`crates/client`:

- `--native` — read this client's own output mode (`wl_output` `CURRENT` mode)
  and request it.
- `--set-mode WxH[@R]` — request an explicit size/refresh (`R` in Hz,
  fractional allowed).
- The reply is logged as `[rdma] display mode: ... (ok|failed; ...)`.

`--native` opens a short-lived Wayland connection before the server handshake
to read the mode; a failure just means no change is requested.

## Limits / future work

- Only at connect. A mid-session change still needs a reconnect (or the
  in-session `Resize`/`Buffers` re-handshake that was deliberately deferred).
- `hyprctl`/`kscreen-doctor` can only pick from the modes the output
  advertises; forcing an arbitrary size needs a virtual output (Hyprland can
  create one; KWin's virtual outputs are headless-only).
- `wlr-output-management` (a native protocol backend, with `set_custom_mode`)
  is not implemented; it would cover wlroots compositors without shelling out.
- Single output.
