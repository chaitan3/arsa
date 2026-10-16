# rdma-display — Architecture

Design and internals. For the metrics model, measurements, the capture-sync
investigation, and troubleshooting see [`DEBUG.md`](DEBUG.md). For build/run
instructions see the top-level [`README.md`](../README.md).

## 1. Overview

rdma-display moves display frames from a Linux **server** to a Linux
**client** with a short, mostly zero-copy pipeline, and sends input back the
other way. Two capture sources are supported behind one `CaptureSource`
trait:

- **KMS/DRM** (`--source drm`): the host's own scanout framebuffer, read via
  `GETFB2`/`GETCRTC` (needs `CAP_SYS_ADMIN`, kernel ≥ 5.8). This is the primary
  path.
- **QEMU D-Bus** (`--source qemu`): a QEMU guest's display via
  `org.qemu.Display1` scanout signals.

A frame is captured, optionally coded, and placed in a staging slot; the slot
is RDMA-WRITEn (or sent over TCP) into a client receive slot; the client
uploads it to the GPU and draws it in a Wayland window. Credits return as the
client frees slots. Input events, audio, and control messages ride separate
tagged TCP connections to the same server port.

The protocol crate (`rdma-display-proto`) is the single source of truth for
the wire format and is versioned (`VERSION`, currently 13). **Build the server
and client together** — a mismatch is rejected at `Hello`.

## 2. Data path

```
server                                              client
────────────────────────────────────────────────────────────────────────────
capture source            (drm poll / qemu D-Bus signal)
   │  dmabuf / raw frame
   ▼
materializer (EGL)        import scanout dmabuf, blit to linear, read back
   │  staging slot (pinned host RAM)
   ▼
codec                     raw | lossless (GPU) | lossy (GPU)
   │  coded bytes in the slot
   ▼
RDMA WRITE + immediate ────────────────────────────► recv slot
   (or TCP length-prefixed frame)                     │
                                                      ▼
                                            presenter (EGL upload → window)
                                                      │
   ◄────────────── credit (slot free) ────────────────┘
```

The client also runs:
- an **input thread** (`TAG_INPUT`) that forwards `InputEvent`s;
- an **audio thread** (`TAG_AUDIO`) that plays the server's monitor capture;
- the **control channel** (`TAG_CONTROL`) for the handshake, mode request,
  credits, per-frame stamps, and clock-sync pings.

## 3. Workspace layout

```
rdma-display/
├── crates/
│   ├── proto/        # wire structs, framing, MsgType, fourcc, InputEvent, keymap
│   ├── capture/      # CaptureSource trait; drm + qemu impls; InputSink; materializers
│   ├── codec/        # codec ids + the CPU reference (lossless format lives here)
│   ├── codec-gpu/    # GLES compute lossless encoder/decoder
│   ├── egl/          # shared surfaceless EGL context (server + client bring-up)
│   ├── rdma/         # verbs setup, QP bring-up, slot MRs, WRITE loop, slots
│   ├── audio/        # PipeWire/Pulse monitor capture + client playback, wire format
│   ├── lifecycle/    # lazy per-session bring-up/teardown helpers
│   ├── client/       # binary: control handshake, presenter, input, metrics
│   └── server/       # binary: wires capture → codec → transport + control channel
├── docs/             # ARCHITECTURE.md, DEBUG.md
├── scripts/          # build-pyrowave-docker.sh
├── install.sh        # release build + setcap
└── Cargo.toml        # workspace
```

## 4. Capture

### 4.1 KMS/DRM source (`crates/capture/src/drm.rs`)

`open_card` probes `/dev/dri/card0..8` (or `--drm-card`) and finds the active
scanout (or resolves a named `--display-output` connector to its CRTC — see §9).
On nvidia-drm the framebuffer pointer lives on the **plane** state, not the
legacy CRTC (atomic-only drivers do not populate `crtc.fb_id`), so the capture
picks the **first bound, non-cursor plane** in `GETPLANERESOURCES` order
(Sunshine's selection) and reads its `fb_id`.
`DRM_CLIENT_CAP_UNIVERSAL_PLANES` + `DRM_CLIENT_CAP_ATOMIC` are enabled first,
or the legacy getters report stale/zero state; if no plane is bound the
connector's CRTC is used instead. `probe_display` imports the current fb
(`GETFB2` → dmabuf) and returns a
`FrameFormat { width, height, stride, fourcc, modifier }`.

The capture loop polls `plane.fb_id` every 1 ms and emits a scanout only when it
changes — one frame per flip, silent while the screen is idle. The export
(`GETFB2` → `PRIME_HANDLE_TO_FD`) runs in the same iteration as the read, so the
compositor cannot destroy the fb in between. The *sync mode* adds a wait before
the read (§4.5).

### 4.2 QEMU D-Bus source (`crates/capture/src/qemu.rs`)

Connects to QEMU's D-Bus display server (session bus or a private
`--dbus-socket`), registers an `org.qemu.Display1.Listener`, and consumes:

- `ScanoutDMABUF(width, height, stride, fourcc, modifier, y0_top, fd)` — a new
  scanout (fd passed over D-Bus).
- `UpdateDMABUF(x, y, w, h)` — guest repaint of the current scanout.
- `Fence(...)` — an in-fence so the NIC does not read a half-written frame.

The listener thread and the RDMA/send thread communicate over a small mpsc
channel of frame events.

### 4.3 Portal ScreenCast source (`crates/capture/src/portal.rs`)

`--source portal` captures through the compositor instead of the KMS scanout.
It is the source to prefer on a Wayland desktop: the compositor composites, so a
captured buffer is complete when it is delivered — there is **no**
direct-scanout blank-frame failure mode, no root/`CAP_SYS_ADMIN`, and
`--capture-sync` does not apply (the `settle`/`fence` machinery exists only
because the KMS path reads a plane mid-flip).

"no root" has a wrinkle: `install.sh` setcaps the binary for `--source drm`, and
a process carrying file capabilities runs with `suid_dumpable=0` — **non-dumpable**
— so the portal, which identifies the caller by opening `/proc/<pid>/root`,
refuses it ("Unable to open /proc/&lt;pid&gt;/root"). The portal path therefore
drops its capabilities and calls `prctl(PR_SET_DUMPABLE, 1)` at startup.

`org.freedesktop.portal.ScreenCast` (`CreateSession` → `SelectSources` →
`Start` → `OpenPipeWireRemote`, driven through `ashpd`) yields a PipeWire node id
plus a remote fd. The PipeWire stream negotiates raw video and hands us dmabufs,
wrapped as the same `DmabufScanout` the KMS/QEMU sources produce, so the
materializer, codecs and transport are unchanged. The compositor draws the
cursor (`cursor_mode = Embedded`). The `restore_token` returned by `Start` is
cached in `$XDG_DATA_HOME/rdma-display/portal_token` and replayed on the next
run (with `persist_mode`), so the picker dialog only appears once — or again
whenever the token is revoked or the source changes.

It needs a session bus and a portal backend (GNOME, KDE,
`xdg-desktop-portal-wlr`, `xdg-desktop-portal-hyprland`). The `portal` feature
(`ashpd` + PipeWire) is on by default, so there is a single server binary;
`--no-default-features` builds without PipeWire. The portal has **no mode
control**, so
`--set-mode`/`--display-output` do not apply — the stream size is whatever the
compositor (or the user's picker choice) gives.

The portal covers display capture only. **Audio** stays on the pipewire-pulse
monitor (§8) and **gamepads** are never covered by the portal (`RemoteDesktop`
is keyboard/pointer/touchscreen only), so `--input` stays `uinput` and input is
deliberately not routed through the portal.

### 4.4 Frame materialization (`crates/capture/src/{egl,materializer}.rs`)

The materializer turns a source frame into the linear wire bytes. `--materialize`
selects:

- **`egl`** (default): import the scanout dmabuf with `eglCreateImageKHR`, blit
  to a linear RGBA8 target, read back through persistent-coherent PBOs. Foreign
  (tiled) dmabufs bind as `GL_TEXTURE_EXTERNAL_OES`; linear ones as
  `GL_TEXTURE_2D`. The context is surfaceless on the EGL device platform.
- **`plain`**: for scanouts that are already host-linear; no EGL.

NVIDIA specifics are load-bearing and encoded in `egl.rs`: the modifier
attributes are `EGL_DMA_BUF_PLANE0_MODIFIER_LO/HI_EXT = 0x3443/0x3444` (not
`0x3275/0x3276`), foreign dmabufs must bind to `EXTERNAL_OES`, and the readback
rows are copied in order (no extra flip). `--readback pbo|direct` only affects
the readback path. The wire format is always `XR24`/BGRX bytes, top-first.

When the GPU lossless codec is active it encodes **straight from the imported
dmabuf** — no full-frame readback, no re-upload (§6).

### 4.5 Capture sync modes (`--capture-sync`)

The loop always polls `plane.fb_id` at 1 ms and emits only when it changes. The
sync mode adds a wait before the read, trading latency against tearing:

| Mode | Behaviour |
| --- | --- |
| `auto` (default) | Use `fence` when the completion-fence watch is ready, else fall back to `settle:2ms`. |
| `settle[:ms]` | Read the current frame `ms` after its commit was observed (default 2 ms). |
| `fence` (alias `bpf`) | Read the current frame when the compositor's per-frame completion fence signals (tracefs; mounts it via `mount(2)` needing `CAP_SYS_ADMIN`, writes needing `CAP_DAC_OVERRIDE` — or pre-mount with `uid=`; falls back to `settle:2ms` until the watch is ready). |

The reasoning behind these (and why there is no userspace completion fence on
nvidia-drm) is in [`DEBUG.md`](DEBUG.md#4-capture-sync--handover-investigation).

## 5. Transport and protocol

### 5.1 Slots and flow control

The client allocates `--frame-slots` (default 3) receive slots of the codec's
worst-case size, registers them as memory regions, and advertises
`{rkey, addr, len}` per slot in `Buffers`. The server posts one RDMA WRITE
(single SGE) per frame into a slot, with the immediate word encoding
`{frame_seq, slot_index, flags}`. The immediate arrives as a completion on the
client's recv QP — the "frame N is in slot k" notification.

Flow control is a **frame window**: the client returns a `Credit(slot)` for
every frame it consumes (presented *or* superseded), and the server keeps at
most `window` frames sent-but-uncredited. On RDMA a credit also frees the
staging slot; with TCP there are no staging slots, so the credit is the *only*
backpressure (frames ride a length-prefixed stream on `TAG_VIDEO`). Either way
the client's present rate paces the server's sends.

### 5.2 Control protocol (`crates/proto`)

Framing: every message is `u32 payload_len | u32 msg_type | payload`,
little-endian. The first byte of every connection is a tag so the server routes
explicitly: `TAG_CONTROL=0x01`, `TAG_AUDIO=0x02`, `TAG_INPUT=0x03`,
`TAG_VIDEO=0x04`.

Handshake:

```
client                                         server
  │── Hello {magic, version, caps} ───────────►│   negotiate transport (CAP_RDMA)
  │── SetMode {Keep|Set|Reset} ───────────────►│   apply to the compositor (§9)
  │◄───────────────────── ModeReply ───────────│
  │◄────────────────────────── Meta ───────────│   display geometry
  │◄───────────────────────── Codec ───────────│   codec id + slot size + chroma
  │── (RDMA endpoint exchange: QPN/PSN/GID) ───►│   over the control connection
  │── Buffers {slot rkeys/addrs} ─────────────►│
  │◄───────────────────────── Ready ───────────│
  │◄──────────────── frame RDMA WRITEs ────────│
  │── Credit(slot) ───────────────────────────►│
```

`MsgType`: `Hello=1, Meta=2, QpEndpoint=3, Buffers=4, Ready=5, Credit=6,
Bye=7, Input=8, Stamp=9, Ping=10, Pong=11, Rumble=12, Codec=13, SetMode=14,
ModeReply=15`.

`Stamp` carries the per-frame server pipeline timestamps (`t0`, `t0d`, `t0p`,
`t1s`, `t1`, `t2`, `t2c`, …) on a cross-host comparable wall clock; `Ping`/
`Pong` do clock sync so the client can compute an end-to-end figure. The
metrics model is documented in [`DEBUG.md`](DEBUG.md#1-metrics-model).

### 5.3 RDMA data path and zero-copy findings

One RC QP per session; RDMA WRITE with immediate; the completion loop busy-polls
`ibv_poll_cq` on a dedicated thread (no async runtime). `set_access(PERMISSIVE)`
is required on the QP (MR flags alone do not permit inbound writes), and
`RLIMIT_MEMLOCK` must cover the pinned pages.

The original goal was to register the source dmabuf directly as an MR
(`ibv_reg_dmabuf_mr`) so the NIC DMA-reads out of GPU memory. On the tested
NVIDIA stack this fails: `nvidia-drm` exports scanouts through the *static*
PRIME path (no pin/unpin ops), and peer-DMA mapping of VRAM returns `ENOMEM`.
The **copy path is therefore the default**: import → blit → readback into a
pinned host MR → WRITE. Zero-copy dmabuf-MR remains implemented behind the same
config for hosts with a cooperative exporter (Intel/AMD rendernode). The
client-side equivalent investigation (GBM/CUDA slots) is in
[`DEBUG.md`](DEBUG.md#5-client-zero-copy-c0-findings).

### 5.4 Frame pacing (`--pace`)

Two clocks meet here: the capture is slaved to the *host* compositor's flips,
the present to the *client's* vsync (or VRR). Nothing couples them, so the
server in the middle is the only place a frame is ever deliberately held. Its
flow-control window (§5.1) is that lever: too small and the pipeline stalls (the
rate drops below the source); too large and it queues (latency creeps).

The right size is the **bandwidth-delay product counted in frames**. The server
measures the send -> client-credit *turnaround* (an EMA; it already includes the
client's decode and its wait for the next present) and the **source** frame
*period* (the capture cadence), and sets

```
window = ceil(turnaround / source_period)   # floored at 1, capped at --frame-slots
```

The source period is a **low percentile (p25) of a sliding window of capture
intervals**, not an EMA of them: the raw interval is bimodal — the trace
reader's scheduling jitter on the short side, and the compositor's
damage-driven idle gaps on the long side — so an EMA wobbled between ~16 and
~155 ms and made the window oscillate 1..3. The percentile rejects both tails
and tracks the *active* frame period (the rate the pipeline must sustain during
a burst).

The denominator must be the *source* period, **never the send interval**. The
send interval is a consequence of the window — in steady state
`send_interval == turnaround / window` — so dividing by it makes the formula
`ceil(turnaround / (turnaround/window)) == window`: a neutral fixed point that
just holds whatever it first computed and can never climb out of a bad value.
(That is exactly how the window collapsed to 1 on a 30 ms-RTT link and capped
the rate at `1/turnaround`.) The source period is independent of the window, so
the formula is stable and correct.

The turnaround is timestamped when the **control reader receives the credit**,
not when the sender loop later consumes it, so the server's own loop latency is
not billed to the measurement.

`--pace` selects the policy:

| Mode | Window | Use |
| --- | --- | --- |
| `auto` (default) | `ceil(turnaround / source_period)`, capped at `--frame-slots` | Keeps the pipe full without queueing; adapts to the link. |
| `latency` | `1` | Present-paced: the client's credit gates every frame. Lowest latency; only for `turnaround` ≪ the frame period (RDMA/same-host) — on a WAN it caps the rate at `1/turnaround`. |
| `smooth` | `--frame-slots` | Never gate: buffer as far as the transport allows. Smoothest cadence, highest latency. |

The chosen window, the turnaround, the source period and the send interval are
logged (debug) on every change **and every 2 s**, so the measurements are visible
after startup.

Both transports obey it. RDMA already had credits (the window was implicitly
`--frame-slots`); the window now *narrows* it. TCP previously had no frame-level flow
control at all, so the server free-ran into the socket buffer and the client
subsampled — the window is new there. Beyond the window, frames are dropped
**latest-wins** at both the server's capture mailbox and the client's drain, so
nothing queues and no GPU work is spent on a frame that will not be shown.

## 6. Codecs (`crates/codec`, `crates/codec-gpu`)

Selected with the server's `--codec auto|raw|lossless|lossy` (default `auto`).
The client sends its preference in `Hello.codec_pref` (`CODEC_ANY` = none); the
server makes the final choice and reports it in the `Codec` message, which the
client uses to size its slots and build its decoder. With `auto` the client may
pick any codec it can decode (raw/lossless/lossy); an explicit `--codec`
on the server overrides the client's request (logged). Ids:

| id | codec | notes |
| --- | --- | --- |
| 0 | `raw` | no header; the slot holds the raw frame (`stride × height`). |
| 1 | `lossless` | tile-delta against the previous frame; GPU-encoded. |
| 2 | `lossy` | PyroWave intra-only GPU codec; needs `libpyrowave-shared.so`. |

### Lossless (v2)

- The frame is cut into `tile × tile` tiles (`tile = 1 << tile_log2`, default
  64 px). Each frame is **full** or **delta** against the previously decoded
  frame (only changed tiles).
- A changed tile picks the cheapest of `RAW` (valid pixels), `REF` (runs of
  `cur ^ ref`), or `DPCM` (runs of the horizontal predictor
  `cur[x] - cur[x-1]`); ties prefer `RAW`, then `REF`, then `DPCM`, so the coded
  total is deterministic even though the GPU packs payloads in atomic order.
- The payload carries a CRC32, so a desync or transport fault is caught by the
  decoder instead of showing wrong pixels.
- Because the codec is lossless, the server's and client's references are the
  same pixels and cannot drift. The server resets the reference on every new
  session, so the first frame is always full.

Wire layout (see `crates/codec/src/lib.rs` for the reference):

```text
offset  size          field
  0     1             magic 0xC0
  1     1             codec (1 = lossless)
  2     1             version = 2
  3     1             flags: bit0 = delta/full
  4     1             tile_log2
  5     3             reserved
  8     4  LE         width
 12     4  LE         height
 16     4  LE         stride
 20     4  LE         changed tile count
 24     4  LE         index capacity (tiles_x * tiles_y)
 28     4  LE         crc32 of the payload
 32     4  LE         total payload words
 36    12             reserved
 48     cap*12 LE     tile entries {tile_id, off_words, (mode<<16)|len_words}
 48+cap*12  payload    packed tile payloads, in write order
```

**Reference-chain invariant:** a delta references the previous frame, so the
client must decode *every* received frame in order, even ones it will not
present (it presents only the newest). Skipping one desyncs the chain and trips
the CRC.

**Encoding cost.** The scanout dmabuf is imported into the codec's own GLES 3.1
compute context and encoded there; only the coded delta is copied into the
staging slot. The import (`eglCreateImageKHR` + bind) is the cost, so it is
**cached per buffer** (keyed by `fd, offset, stride, fourcc, modifier, w, h`,
evicted past 4 entries) — a compositor swaps a handful of buffers, so steady
state pays only the encode. Linear scanouts bind as `GL_TEXTURE_2D`
(`texelFetch`); tiled ones as `EXTERNAL_OES`. The external path must swizzle
`.bgra` (the dma-buf sampler returns real `(R,G,B)`, the coded word is
`B,G,R,A`). Measured numbers are in
[`DEBUG.md`](DEBUG.md#3-performance).

### Lossy (PyroWave)

Intra-only GPU codec. The server imports the scanout dmabuf into a Vulkan image
and runs PyroWave's `encode_gpu_scaled_synchronous` (RGB → YCbCr + scale), then
packetizes the bitstream into the staging slot — no GL, no full-frame readback.
The client decodes on the GPU into Y/U/V planes and converts to BGRA. Because
it is intra-only there is no reference-chain invariant and no CRC.
`--lossy-bitrate MBPS` sets the bitstream budget in Mbit/s, which the server
converts to PyroWave's per-frame byte cap at a nominal 60 fps
(`MBPS * 1e6 / 8 / 60`; slot = cap + 1 MiB). The client can request its own
value in `Hello` (`--lossy-bitrate`), which overrides the server's default.
`--lossy-chroma 420|444` picks subsampling. The `.so` is `dlopen`ed from beside
the binaries (`$ORIGIN` rpath); build it against the lab's glibc with
`scripts/build-pyrowave-docker.sh`.

## 7. Client presenter (`crates/client`)

### 7.1 Slot memory

`--slot-mem host` (pinned RAM) is the production path. The client receives into
the slot, uploads the frame to the GPU with `glTexImage2D`, and draws it. The
dormant `--slot-mem cuda` (CUDA GPUDirect-over-dmabuf) is kept for future
cooperative hardware; see [`DEBUG.md`](DEBUG.md#5-client-zero-copy-c0-findings).

### 7.2 EGL-upload presenter (`--output egl`)

Raw wayland-client + wayland-protocols (no sctk/winit). An xdg_toplevel at the
frame size; each frame is uploaded from the slot's CPU pointer with
`glTexImage2D` (the GPU reads host RAM — no dma-buf import anywhere), a shader
blit with a `.bgra` swizzle draws it letterboxed, and `eglSwapBuffers` presents
it. Credits return immediately after the upload call (the client pointer is
consumed per spec); `eglSwapInterval(1)` gives vsync pacing. `--output none`
keeps the headless stats/dump path.

Event integration stays on plain threads: `event_queue.prepare_read()` →
`poll(fd)` → `read()` → `dispatch_pending`, interleaved with RDMA completions.

**GPU lossless decode.** With `--codec lossless --output egl`, the client
decodes deltas into a persistent `GL_RGBA8` texture and the presenter draws
*that texture* — no per-frame full-frame upload. Every received frame is
decoded in order (reference chain), only the newest is presented. On a `Meta`
geometry change the frame is decoded with the CPU decoder and the stale texture
is dropped.

### 7.3 Direct scanout and presentation latency

To let the compositor bypass composition, the client marks the surface fully
opaque (`wl_surface.set_opaque_region`, updated on every size change), uses an
alpha-free (XRGB) EGL window config, and runs `--fullscreen`. That makes the
buffer eligible for **direct scanout** to the primary plane; otherwise the
compositor composites and adds a buffered stage. `wp_presentation` feedback
gives the true on-screen timestamp (`display = tp − t5`), which is the
compositor's queue-to-scanout, not our work. See
[`DEBUG.md`](DEBUG.md#6-presentation-latency).

### 7.4 Input capture and grab

Keyboard and pointer are read from the Wayland seat. Two pointer modes:

- **Absolute (default)**: `wl_pointer.Motion` → device pixels → un-letterboxed
  guest pixels → `InputEvent::Abs`.
- **Grab** (`Ctrl+Alt+G`, or `-g`/`--grab` at startup): the pointer is locked
  with `zwp_pointer_constraints_v1` and raw deltas come from
  `zwp_relative_pointer_v1` → `InputEvent::Rel` (local cursor hidden). The
  keyboard is grabbed with `zwp_keyboard_shortcuts_inhibit_v1`, so the
  compositor's own shortcuts (Alt+F4, Alt+Tab, …) reach the guest as plain
  keys. If the global is missing, the grab degrades to mouse-only.

`Ctrl+Alt+G` is the only way out; a focus shuffle deliberately does not release
the grab (the pieces self-heal: the lock re-acquires on pointer entry / on the
`unlocked` event, the inhibitor re-engages on keyboard focus). Held keys and
buttons are flushed on focus loss so the guest never keeps a key down.

Gamepads are read from `/dev/input/event*` (filtered by `BTN_GAMEPAD`),
optionally `EVIOCGRAB`ed so the local compositor does not double-consume them,
and forwarded as `InputEvent::GamepadBtn/Axis`. Rumble returns from the server.

## 8. Input injection (server, `crates/capture/src/input.rs`)

An `InputSink` consumes `InputEvent`s, paired with the source:

- **`qemu-dbus`**: injects into the guest console (keyboard/mouse/rel) via the
  qemu-display D-Bus proxy.
- **`uinput`** (host compositor): creates a virtual **keyboard**, an **ABS
  pointer** (`INPUT_PROP_POINTER` + `ABS_X/Y` over inputtino's fixed
  0..19200/0..12000 ranges; client pixels scaled against the current captured
  size, so a resolution change needs no device recreation), a **REL pointer**
  for grab-mode deltas, and **gamepads** that impersonate a real Xbox One pad
  (xpad identity, 11-button set, FF/rumble handshake). Buttons go to whichever
  pointer device saw the last motion.

## 9. Display-mode control (`crates/server/src/display.rs`)

The client can request a server output-mode change before the session starts
(`--native` = the client's own `wl_output` mode, or `--set-mode WxH[@R]`). The
request is applied **between `Hello` and `Meta`** — no reconnect and no
mid-session resize — so the scanout and the session's slots come up at the
requested geometry.

Backends shell out to the compositor's CLI:

| `--display-control` | Compositor | List | Set |
| --- | --- | --- | --- |
| `hyprctl` | Hyprland | `hyprctl -j monitors` | `hyprctl keyword monitor <name>,<WxH@R>,auto,<scale>` |
| `kscreen` | KDE/KWin | `kscreen-doctor -o` | `kscreen-doctor output.<name>.mode.<WxH@R>` |

`auto` detects Hyprland from `HYPRLAND_INSTANCE_SIGNATURE` (or by scanning
`$XDG_RUNTIME_DIR/hypr/<sig>/.socket.sock`, setting the variable if absent) and
KDE from `XDG_CURRENT_DESKTOP`. The requested size maps to the **nearest
advertised mode** (exact W×H, else smallest pixel-count difference, tie-broken
by refresh); if nothing suitable is advertised — or there is no backend — the
reply is an error and the session runs at the current resolution.

### Output selection (multiple outputs)

When the compositor has several outputs, one selection drives the mode request:

- `--display-output NAME` forces a **DRM connector name** (`DP-1`, `HDMI-A-1`,
  `eDP-1` — the same string sysfs and the compositors use) for the mode request.
  Capture picks the first bound, non-cursor plane (§4.1) regardless of the
  connector, so with several active outputs make sure only the intended one is
  scanning out.
- With no `--display-output`, the default is the compositor's **primary**
  output: Hyprland's focused monitor, or KScreen's priority-1 output. With no
  compositor backend, capture falls back to the first active CRTC (there is no
  "primary" in KMS).

Hotplug/reconfiguration is not re-resolved mid-session yet — the target is
resolved once at startup.

Because the compositor commits the KMS modeset asynchronously, the server
re-probes the DRM scanout in a bounded wait until it matches the reply before
sizing the session (otherwise every frame overflows the old slots). The mode in
effect before the change is restored on disconnect and on
`SIGINT`/`SIGTERM`/`SIGQUIT` (`crates/server/src/signals.rs`).

## 10. Audio (`crates/audio`)

The server captures a PipeWire/Pulse monitor source (default
`@DEFAULT_MONITOR@`) and streams f32le PCM to the client over `TAG_AUDIO`; the
client plays it to the default sink with a configurable buffer depth
(`--audio-buffer-ms`). Running as root uses a setuid helper; the setcap-as-user
server captures in-process (it needs `XDG_RUNTIME_DIR` + `WAYLAND_DISPLAY`).
Under `--metrics` the playout thread also records each packet's end-to-end age
(against the ping/pong clock offset) in `audio::stats`, split into the wire
(`xfer`) and the client's pre-sink backlog (`ring`), which the overlay folds
into an audio e2e and an A/V offset; it is inert without `--metrics`. The
playout ring itself converges to a small target (`RING_TARGET_MS`), trimming a
standing backlog left by a startup burst or a stall so it does not become a
permanent A/V offset.

## 11. Design decisions and limits

- **Copy path is the default**; zero-copy dmabuf-MR is implemented but only
  viable with a cooperative exporter (see §5.3).
- **One session at a time**, single output, single console. The accept path is
  single-threaded, so a per-session bring-up (including the mode-settle wait)
  briefly delays a second client.
- **Resolution changes only at connect**; a mid-session change still needs a
  reconnect.
- **Input injection needs privileges**: `uinput` needs `/dev/uinput`
  (root or the `input` group); `uinput` also creates real kernel devices.
- **The presenter does not import dmabufs**: KWin 6.7.5 + NVIDIA refuses
  foreign system dmabuf imports, so the client uploads from host RAM.
- **`--slot-mem cuda` and VRAM zero-copy are dormant** — RM-gated on
  GeForce-class NVIDIA.
