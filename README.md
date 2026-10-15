# rdma-display

A low-latency remote display system for Linux. A **server** captures a
display (either the host's own KMS scanout or a QEMU guest's D-Bus display),
codes each frame, and ships it to a **client** over RDMA (RoCE) or TCP. The
client presents it in a Wayland window (EGL) or dumps it headless, and sends
keyboard/mouse/gamepad input back to the server, which injects it via
`uinput` (host) or the QEMU D-Bus console (guest).

The design goal is a short, mostly zero-copy pipeline and honest end-to-end
latency metrics. It is a Sunshine/Parsec-style remote desktop for two Linux
boxes on a fast link, not a browser-streaming service.

```
 server                                            client
 ┌───────────────────────────┐                    ┌──────────────────────────┐
 │ capture: drm (KMS) or qemu │                    │ receive slot             │
 │   → EGL import / blit      │                    │   → EGL upload           │
 │   → codec (raw/lossless/   │   RDMA WRITE+imm   │   → letterboxed draw     │
 │     lossy) into a slot MR  │ ─────────────────► │   → eglSwapBuffers       │
 │   → RDMA WRITE / TCP       │                    │                          │
 │                            │ ◄─── credit ───────│ wl_buffer released        │
 │ input sink: uinput / qemu  │ ◄─── input ────────│ wl_keyboard / wl_pointer  │
 │ audio: PipeWire monitor    │ ──► audio ────────►│ PulseAudio/PipeWire sink  │
 └───────────────────────────┘                    └──────────────────────────┘
        control channel: TCP (handshake, mode request, credits, stamps)
```

## Features

- **Two capture sources**: host KMS/DRM scanout (`--source drm`) or a QEMU
  guest's D-Bus display (`--source qemu`).
- **Two transports**: RDMA/RoCE (preferred when a device and a routable GID
  exist) or TCP; negotiated per session, `--transport auto|rdma|tcp`.
- **Three codecs**: `raw`, `lossless` (tile-delta, GPU-encoded), `lossy`
  (PyroWave intra-only GPU codec).
- **Wayland/EGL client presenter** with fullscreen, letterbox scaling,
  compositor-direct-scanout hints, and a live latency overlay (`--metrics`).
- **Input forwarding**: keyboard, absolute + relative pointer, and gamepads
  (rumble back-channel). A `Ctrl+Alt+G` grab (or `-g`) hands the desktop's own
  shortcuts to the remote side.
- **Audio side-channel** (server monitor → client sink).
- **Client-requested resolution**: the client can ask the server to switch its
  output to the client's native mode (`--native`) or an explicit `WxH@R`
  (`--set-mode`) before the session starts.
- **Latency metrics** end-to-end (capture → send → transfer → upload →
  present → on-screen), with cross-host clock sync.

## Requirements

- Linux on both ends. Kernel ≥ 5.8 for KMS capture (`GETFB2`), plus
  `CAP_SYS_ADMIN` on the server (or run it as root).
- Rust 1.85+ and a C toolchain to build.
- **Client**: a Wayland compositor (for `--output egl`); EGL/GLES.
- **Server, KMS capture**: an active DRM scanout. NVIDIA was the primary test
  platform; a `plain` materializer covers host-linear scanouts.
- **RDMA (optional)**: an RDMA NIC with RoCE configured (a routable GID) on
  both ends; otherwise use TCP.
- **Audio (optional)**: PipeWire/PulseAudio with the session environment
  available to the server.
- **Lossy codec (optional)**: `libpyrowave-shared.so` next to the binaries
  (see `scripts/build-pyrowave-docker.sh`); needed on both server (encode) and
  client (decode).

## Building

The one-shot script builds both binaries in release mode and gives the server
`CAP_SYS_ADMIN` so it can read the host scanout without running the whole
process as root:

```bash
./install.sh
```

It must be run as your **normal user** (it calls `sudo setcap` internally).
File capabilities live on the inode, so **re-run `install.sh` after every
rebuild** — otherwise the server loses the capability.

Manual equivalent:

```bash
cargo build --release -p rdma-display-server -p rdma-display-client
sudo setcap cap_sys_admin+ep target/release/rdma-display-server
```

Widen the capability set if you use the root-only features (see the comments
in `install.sh`):

```bash
# --input uinput needs /dev/uinput (root or the `input` group)
# --capture-sync fence needs cap_sys_admin (mount tracefs) + cap_dac_override (writes)
# --gpu-power needs root (nvidia-smi)
CAPS=cap_sys_admin,cap_dac_override,cap_dac_read_search ./install.sh
```

## Running

### Server

```bash
# host KMS capture, uinput injection, everything auto (the default)
./target/release/rdma-display-server

# explicit, with the common knobs
./target/release/rdma-display-server \
    --source drm --input uinput \
    --codec lossless --transport auto \
    --listen 0.0.0.0:4791
```

Run it from inside your graphical session (or with `sudo -E`) so it inherits
`WAYLAND_DISPLAY` + `XDG_RUNTIME_DIR`; audio capture, the KDE/Hyprland
resolution backend, and `--gpu-power` all need the session environment.
`sudo` without `-E` strips it.

For a QEMU guest instead of the host:

```bash
./target/release/rdma-display-server --source qemu --dbus-socket /run/.../dbus.sock
```

### Client

```bash
# Wayland window, fullscreen, grab from the start
./target/release/rdma-display-client 192.168.1.10 -f -g

# headless sanity check: 10 seconds to a PPM
./target/release/rdma-display-client 192.168.1.10 --output none --seconds 10 --dump out.ppm

# ask the server to match this monitor's resolution before starting
./target/release/rdma-display-client 192.168.1.10 -f --native
```

The host is the first positional argument (`HOST` or `HOST:PORT`; port 4791 is
assumed). `-s`/`--server` is the same thing.

### Groups and permissions

The server is designed to run as your normal user with `CAP_SYS_ADMIN` (set by
`install.sh`), not as root. That covers the KMS `GETFB2` call, but several
device nodes are still group-restricted, so add your user to the relevant
groups and log out / back in for the change to take effect:

| Group | Needed for | Device nodes |
| --- | --- | --- |
| `input` | server `--input uinput`; client gamepad capture | `/dev/uinput`; `/dev/input/event*` (read+write — write is for rumble) |
| `video` | server KMS capture (`--source drm`) | `/dev/dri/card*` |
| `render` | server EGL materializer / GPU encode (`--rendernode`) | `/dev/dri/renderD*` |
| `rdma` | RDMA transport, on both ends (where the distribution restricts it) | `/dev/infiniband/*` |

```bash
sudo usermod -aG input,video,render "$USER"   # add rdma too if you use the RDMA transport
# log out and back in (group membership is fixed at login)
```

Group names and node ownership vary by distribution; if a node is already
world-accessible you can skip it. `--capture-sync fence` no longer needs root —
it mounts tracefs itself with the server's `cap_sys_admin` and writes it with
`cap_dac_override` (or pre-mount tracefs with `uid=<you>`). Only `--gpu-power`
(nvidia-smi) genuinely needs root.

### Firewall / networking

The server binds **one TCP port** (default `0.0.0.0:4791`, `--listen`). The
client opens **four** TCP connections to that same port, distinguished by a
one-byte tag: control, video (TCP transport only), input, and audio. So a
single inbound TCP rule on the server is enough:

```bash
# ufw
sudo ufw allow 4791/tcp

# firewalld
sudo firewall-cmd --add-port=4791/tcp --permanent && sudo firewall-cmd --reload

# nftables
sudo nft add rule inet filter input tcp dport 4791 accept
```

If you use the **RDMA** transport, the endpoints (QPN/PSN/GID) are exchanged
over the control TCP connection, so there is **no extra TCP port** — but the
RoCE traffic itself must be allowed between the two hosts on the fabric
(RoCEv2 is UDP; it conventionally uses destination port 4791). If your
firewall or switch does not permit RoCE, force TCP with `--transport tcp` on
both ends.

## Common options

### Server

| Option | Meaning |
| --- | --- |
| `--source drm\|qemu` | Capture the host KMS scanout (default) or a QEMU guest. |
| `--input qemu\|uinput` | Injection sink. Default: paired with the source (`drm`→`uinput`, `qemu`→`qemu-dbus`). |
| `--listen ADDR:PORT` | Bind address (default `0.0.0.0:4791`). May also be given as the first positional argument, e.g. `rdma-display-server 0.0.0.0:4791`; a bare host gets port 4791. |
| `--codec auto\|raw\|lossless\|lossy` | Frame coding. `auto` (default) honours the client's requested codec, else lossless; an explicit value forces it. |
| `--transport auto\|rdma\|tcp` | Video transport (default `auto`: RDMA when possible). |
| `--fps N` | Pacing cap; `0` = consumer-paced by client credits (default). |
| `--pace auto\|latency\|smooth` | Frame pacing: how many frames the server keeps in flight ahead of the client's display (the flow-control window). `auto` (default) sizes it from the measured link; `latency` = 1 (present-paced, lowest latency); `smooth` = the transport maximum (smoothest). |
| `--capture-sync MODE` | Which fb to capture: `onbehind` (default), `vblank`, `settle[:ms]`, `fence`, `zero`, `auto`. |
| `--drm-card /dev/dri/cardN` | Pin the KMS card (default: first card with an active scanout). |
| `--display-control auto\|hyprctl\|kscreen\|none` | Backend for the client's resolution request (default `auto`). |
| `--display-output NAME` | DRM connector name to capture and to apply the resolution request to (default: the compositor's primary output). |
| `--audio` / `--no-audio` / `--audio-device SRC` | Audio side-channel (default on; `@DEFAULT_MONITOR@` = default sink). |
| `--gpu-power off\|p5\|p0\|max\|<sm>[,<mem>]` | Floor NVIDIA clocks (needs root). |
| `--frame-slots N` | Receive/staging slots, each holding one coded frame; sets the receive-buffer pool and the maximum flow-control window (default 3; `--slots` accepted as an alias). |
| `--idle-metrics N` | Log RSS/CPU every N seconds. |
| `-v`, `--verbose` | Debug-level logging for the project's crates (per-frame/per-event detail). `RUST_LOG` overrides. |

### Client

| Option | Meaning |
| --- | --- |
| `HOST[:PORT]` / `-s HOST` | Server address (default `127.0.0.1:4791`). |
| `--output egl\|none` | Present in a Wayland window (default) or headless. |
| `-f`/`--fullscreen` | Fullscreen on the current output (letterboxed, aspect kept). |
| `-g`/`--grab` | Start with the input grab on (same as `Ctrl+Alt+G`). |
| `--native` / `--set-mode WxH[@R]` | Ask the server to change its output resolution. |
| `--codec auto\|raw\|lossless\|lossy` | Request a codec (honoured under the server's `--codec auto`). Default `auto`: the server picks. |
| `--metrics` | Latency overlay + clock-sync pings. |
| `--scale-filter auto\|nearest\|linear\|cubic\|area` | Letterbox sampler. |
| `--transport auto\|rdma\|tcp` | Video transport preference. |
| `--slot-mem host\|cuda` | Receive memory (`host` = pinned RAM, the production path). |
| `--audio` / `--no-audio` / `--audio-buffer-ms N` | Audio playback (default on). |
| `--no-controller` / `--no-controller-grab` / `--fake-gamepad` | Gamepad forwarding. |
| `--seconds N` / `--dump FILE` | Headless capture to a PPM for debugging. |
| `-v`, `--verbose` | Debug-level logging for the project's crates (per-frame/per-event detail). `RUST_LOG` overrides. |

**Shortcut:** `Ctrl+Alt+G` (or `-g` at startup) grabs keyboard+mouse for the
remote side — the local compositor's own shortcuts (Alt+F4, Alt+Tab, …) are
forwarded too. `Ctrl+Alt+G` again is the only way to give control back.

## Supported combinations

| Server capture | Server input | Client | Notes |
| --- | --- | --- | --- |
| `--source drm` (host KMS) | `--input uinput` | `--output egl` (Wayland) | The main configuration. Needs `CAP_SYS_ADMIN`. |
| `--source drm` | `--input uinput` | `--output none --dump` | Headless/debug. |
| `--source qemu` | `--input qemu` | `--output egl` | Guest over QEMU's D-Bus display. |
| `--source qemu` | `--input qemu` | `--output none` | Guest, headless. |

Transport, codec, and presenter are independent axes:

- **Transport** — `rdma` requires a RoCE-capable NIC + routable GID on *both*
  ends; `tcp` works anywhere; `auto` picks RDMA when both sides advertise it.
- **Codec** — any client works with `raw` and `lossless`; `lossy` additionally
  needs `libpyrowave-shared.so` on both ends.
- **Presenter** — `egl` needs a Wayland compositor; `none` is headless.
- **Resolution control** — the server's compositor must be **Hyprland**
  (`hyprctl`) or **KDE/KWin** (`kscreen-doctor`) for `--native`/`--set-mode`
  to do anything; otherwise the request is reported as failed and the session
  runs at the server's current resolution.
- **GPUs** — the server capture path was developed and tested on NVIDIA
  (KMS + EGL dmabuf import); a `plain` materializer handles host-linear
  scanouts. The client presenter is GPU-agnostic (EGL upload from host RAM).
  VRAM zero-copy paths are NVIDIA-gated and currently dormant.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design and architecture:
  the data path, capture sources, transport/protocol, codecs, presenter,
  input, display-mode control.
- [`docs/DEBUG.md`](docs/DEBUG.md) — debugging and performance: the metrics
  model, the capture-sync/handover investigation, measured numbers, and
  troubleshooting.

## License

See the repository for license information.
