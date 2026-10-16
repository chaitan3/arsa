# rdma-display — Debugging & Performance

Measurements, the capture-sync/handover investigation, and troubleshooting.
For design and architecture see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. Metrics model

Every stage is a difference of stamps taken on **one host**, so no number
depends on the two clocks agreeing. The absolute unix stamps travel on the wire
(`Stamp`) only so the client can choose which pairs to subtract; `Ping`/`Pong`
give a min-RTT one-way estimate for the TCP case.

Server stamps (`crates/proto`, `FrameStamp`):

| stamp | meaning |
| --- | --- |
| `t0` | capture event latched (QEMU D-Bus arrival / DRM scanout poll) |
| `t0d` | the sender picked the frame up (pacing + event-queue wait done) |
| `t0p` | the scanout dmabuf import / materializer prepare finished |
| `t1s` | a staging slot was available (after any credit wait) |
| `t1` | the capture → staging-slot work finished (slot holds payload) |
| `t2` | RDMA WRITE posted |
| `t2c` | send CQ reported the WRITE fully written on the client |

Client stamps: `t3` = frame received (drain), `t4` = `glTexImage2D`/decode
returned, `t5` = `eglSwapBuffers` returned, `tp` = compositor on-screen time
(`wp_presentation`).

Derived stages:

```
queue    = t0d - t0    capture queue / pacing
prepare  = t0p - t0d   scanout import / materializer prepare
slot     = t1s - t0p   staging-slot credit wait
material = t1  - t1s   blit + readback + fence (codec-dependent)
post     = t2  - t1    WQE build + post
transfer = t2c - t2    RDMA WRITE as the sender saw it (one-sided)
upload   = t4  - t3    upload call
present  = t5  - t4    draw + swap submission
display  = tp  - t5    compositor queue + scanout (0 = feedback pending)

e2e = (t2c - t0) + (client_end - t3) + net
    = queue+slot+material+post+transfer+net + upload+present+display
```

`net` is 0 for RDMA (its `transfer` already covers the wire); for TCP it is the
ping/pong min-RTT one-way estimate. Without `wp_presentation` the `display`
stage reads `n/a` and the end-to-end figure is a lower bound. `e2e` above a
threshold prints a `[metrics-spike]` line.

The `material` span is codec-dependent and labelled by `MatKind`
(`Raw`/`Lossless`/`Lossy`); every codec has a different set of stages inside
it, so reporting them under one fixed label was misleading.

## 2. Overlays, logs, and diagnostics

Both binaries log through `log` + `env_logger`. By default they print the
user-facing lines — init, connection/disconnect, and feature-disabling failures
(`info`/`warn`). `-v`/`--verbose` adds per-frame/per-event/setup/timing detail
(`debug`). The `RUST_LOG` environment variable overrides either default, e.g.
`RUST_LOG=rdma_display_capture=debug`. Output format is `[LEVEL] message` (the
`[tag]` prefix is kept inside the message).

- **`--metrics`** (client): per-stage latency overlay + clock-sync pings.
- **`--idle-metrics N`** (server): log RSS/CPU/threads every N seconds.
- **`--input-test`** (server): synthesize a test input sequence.
- `RDMA_DISPLAY_LOSSY_DEBUG=1`: verbose lossy dma-buf import diagnostics.
- `RDMA_DISPLAY_LOSSY_CPU=1`: force the lossy CPU download/upload path.
- Capture-sync logs (`[capture-drm] …`) report when the fence watch is
  unavailable and the mode falls back to `settle`.

## 3. Performance

### Lossless codec (4K, RTX 2080 Ti, `codecbench` synthetic desktop)

| | v1 (raw tiles) | v2 (ref/dpcm runs) |
| --- | --- | --- |
| 10-frame coded total | 34.7 MiB (9.1x) | **7.9 MiB (40.3x)** |
| steady delta | 325 KB/frame (102x) | **46 KB/frame (719x)** |
| full first frame | 31.9 MiB (1.0x) | **7.8 MiB (4.2x)** |
| GPU encode (steady) | 0.18 ms | 0.20 ms |
| GPU decode (steady) | 0.07 ms | 0.04 ms |

Per-frame compute: `cpu 6.6 / gpu 0.2 ms` for a delta, `cpu 9.3 / gpu 8.1 ms`
for a full frame. On high-motion content that compresses poorly, `raw` is still
cheaper.

**The import path dominates.** The scanout import (`eglCreateImageKHR` + target
bind) costs ~3 ms/frame uncached vs 0.17–0.33 ms for the encode — so it is
cached per buffer (keyed by `fd, offset, stride, fourcc, modifier, w, h`,
evicted past 4 entries). Steady state then pays only the encode. The external
(tiled) sampler is not the bottleneck once cached.

`codecbench` benchmarks all variants (`[texture]`, `[external]`, `[fetch]`,
`[import]`) and requires them byte-identical to the SSBO encode and
CPU-decodable.

## 4. Capture sync / handover investigation

KMS capture polls the plane's `fb_id` every 1 ms and emits only when it changes.
There is no producer fence on the scanout dma-buf, so the read timing is the
only lever; the default (`auto`) reads on the compositor's per-frame completion
fence when it is available and falls back to `settle:2ms`. A `vblank`-synced
read would avoid the wait entirely but is dead on this stack:
`CRTC_GET_SEQUENCE` returns `Ok` with a frozen counter (`1`, `ns 0`).

Kernel surface on the test host (6.12.74, driver 610.57.04): the `dma_fence_*`
and `drm_vblank_event*` tracepoints exist, and generic DRM functions are
visible — but **NVIDIA's flip entry points are `notrace`** and cannot be probed.
Observation must go through the generic DRM functions NVIDIA calls into.

### Phase 1 — does a vblank/flip event fire?

532 atomic commits over ~20 s of motion:

| counter | value |
| --- | --- |
| `drm_mode_atomic_ioctl` | 532 |
| `drm_atomic_set_fb_for_plane` | 532 (exactly one per commit) |
| `drm_vblank_event{,_queued,_delivered}` | 0 |
| `drm_crtc_arm_vblank_event` / `send_vblank_event` | 0 |
| `drm_send_event` / `_locked` | 0 |
| `drm_syncobj_timeline_wait_ioctl` | 59 |
| `drm_syncobj_wait_ioctl` | 0 |

**Verdict:** the flip-event hypothesis is falsified — the compositor arms no
page-flip event and delivers none to userspace. But the handover *is* precisely
observable: `drm_atomic_set_fb_for_plane` fires once per commit with the fb id.
Explicit sync is confirmed (`so_tl_wait > 0`, `so_wait == 0`): the compositor
syncs via `drm_syncobj` timeline points, not the dma-buf reservation.

### Phase 2 — is the frame complete at handover?

401 handovers over ~40 s:

| observation | value |
| --- | --- |
| atomic ioctl duration | 401/401 in [0,1) ms (non-blocking) |
| handover cadence | 391/401 in 16–32 ms |
| `@signaled[6]` | 401 = handover count (a per-frame fence) |
| `@signaled[433]` | 1598 (~4/frame) |
| `@tl_wait` | 7, all 2–8 µs |

The compositor hands the fb to KMS **before its GPU work is complete**;
context 6 signals ~6–11 ms later. Context 6 is the safe-read clock, and it is
**not** on the scanout dma-buf's reservation (not reachable via
`EXPORT_SYNC_FILE`).

### Phase 3 — settle-time distribution

`signal6 - handover`: p50 ≈ **0.75 ms**, mean ≈ **1.3 ms**, p90 ~1–2 ms, p99
**~8–16 ms**. A 10x heavier workload did not move the median.

### Phase 4 — whose fence is context 6?

Kernel stacks show it is NVIDIA's **semsurf (semaphore-surface) fence**:
created per frame by the compositor through the private
`nv_drm_semsurf_fence_create_ioctl`, signalled from NVIDIA's semaphore-surface
completion callback. It is a GPU-completion signal, owned by the compositor's
process/fd, and **not reachable from our capture**. `emit=0` for both contexts
is a red herring — NVIDIA's fence ops do not call `dma_fence_emit`.

### Decision and implementation

There is no userspace wait for completion on this stack. Options:

| option | latency | tearing | deps |
| --- | --- | --- | --- |
| `settle:2ms` (the `auto` fallback) | ~2 ms | when render outlasts settle | none |
| eBPF-triggered read on the semsurf signal | ~1 ms | none | eBPF + root |
| `ext-image-copy-capture` (screencopy) | ~1 compositor frame | none | Wayland capture |

Two modes implement this; `auto` (the default) is `fence` with a `settle:2ms`
fallback:

- **`--capture-sync settle[:ms]`** (default 2 ms): read the current fb `ms`
  after its commit was observed, with a bounded delay queue. No dependencies;
  tearing is possible whenever a frame's render outlasts the window.
- **`--capture-sync fence`** (alias `bpf`): read the current fb when the
  compositor's per-frame completion fence signals, via the
  `dma_fence:dma_fence_signaled` tracepoint through tracefs (no libbpf). It
  calibrates the completion context by matching `dma_fence_init` counts against
  the atomic-commit count, in a background thread that retries until the screen
  is active (an idle desktop has no commits), settling meanwhile. Root is
  required; a tracefs mount failure falls back to `settle:2ms`.

Benchmark with the client overlay (`queue prep slot mat`) over `settle:2`,
`settle:4`, and `fence`.

## 5. Client zero-copy (C0) findings

The client-side question was whether `ibv_reg_dmabuf_mr` accepts a client-
allocated GPU dmabuf (the server's failure was on *foreign* nvidia-drm
exports). On the test client (RTX-class NVIDIA, open module 615.71, KWin
6.7.5):

- Diagnostics: open kernel module ✓, BAR1 = 16 GB ✓, IOMMU translated, GPU and
  NIC on the same host bridge.
- **GBM route**: `gbm_bo_create_with_modifiers([LINEAR])` returns a genuinely
  linear XB24 bo (modifier 0, stride 7680) — the legacy `gbm_bo_create(USE_LINEAR)`
  silently returned block-linear, caught by an assert. But `ibv_reg_dmabuf_mr`
  → **ENOMEM**: the exporter is nvidia-drm's *static* PRIME path, which refuses
  peer-DMA mappings for VRAM. Buffer ownership did not matter; exporter flavour
  did.
- **CUDA route** (`cuMemGetHandleForAddressRange`): `DMA_BUF_SUPPORTED=1`
  advertised, but the export call returned `CUDA_ERROR_NOT_SUPPORTED` — refused
  by RM at export, product-gated on GeForce-class GPUs.
- **dma-heap route** (`/dev/dma_heap/system`): system RAM that is a dma-buf from
  birth — the NIC registers it as plain host DMA (no GDR gate), and the same fd
  is EGL-importable/presentable. This is the route that kept the zero-copy
  structure while abandoning VRAM landing; in the end the production path is
  simply `--slot-mem host` (pinned RAM).

**PCIe ordering note.** With a GPU-BAR destination, the CQE (a posted write to
host RAM) can land before the last frame bytes (posted writes to the BAR) —
PCIe does not order posted writes across destinations. The mitigation is a
sentinel: split the frame into a payload WRITE (no immediate) and an 8-byte
trailer at `slot_end - 8` *with* the immediate, so a visible sentinel implies
the payload is in the buffer. With host-RAM destinations, payload and CQE go to
the same destination and are ordered, so no sentinel is needed.

## 6. Presentation latency

`display = tp − t5` (compositor on-screen time − swap return) dominates the
client span and is the compositor's queue-to-scanout, not our work. At a fixed
refresh it is quantized to vblanks and inflated by the compositor's render loop
and triple buffering: expect ~1–2 frames (8–16 ms at 120 Hz). To reduce it:

- Mark the surface **fully opaque** (`wl_surface.set_opaque_region`, updated on
  every size change), use an **alpha-free (XRGB)** EGL config, and run
  **`--fullscreen`** — that makes the buffer eligible for direct scanout, which
  removes the compositing stage.
- **VRR** removes the fixed-vblank quantization (the compositor flips on
  demand); without it, `display` is bounded by the compositor's scheduling, not
  by our pipeline.
- On KDE/KWin, `KWIN_DRM_DISABLE_TRIPLE_BUFFERING=1` removes the 2-frame
  allowance; `KWIN_USE_OVERLAYS=0` / `KWIN_DRM_USE_COLOR_PIPELINE=0` help rule
  out plane/color-pipeline interaction when VRR will not engage.

## 7. Troubleshooting

**Server capture fails / "drm capture needs root".** KMS capture needs
`CAP_SYS_ADMIN` and an active scanout. Re-run `install.sh` after every rebuild
(file capabilities live on the inode). Confirm with `getcap
target/release/rdma-display-server`.

**Video freezes or goes black when a fullscreen app/game starts (audio keeps
playing).** Under **direct scanout** the compositor points the plane at the
client's own buffer, which on this driver is not readable through the scanout
path, so the captured frame is blank. Disable direct scanout in the compositor
(e.g. Hyprland) or keep the client composited. Separately, the compositor can
replace/destroy its scanout buffers, so the fb the capture loop is reading stops
existing (`GETFB2 … ENOENT`); after a few consecutive failures the loop re-picks
the plane and logs `GETFB2 failed Nx in a row … re-picking the plane`.

**Audio / resolution control / `--gpu-power` do nothing under `sudo`.** The
server needs the session environment (`XDG_RUNTIME_DIR`, `WAYLAND_DISPLAY`).
Use `sudo -E`, or run the setcap binary directly as your user. Audio uses a
setuid helper only when running as root.

**`--native`/`--set-mode` reports failure.** `--display-control auto` only
detects **Hyprland** (`HYPRLAND_INSTANCE_SIGNATURE`, or an inferred instance
under `$XDG_RUNTIME_DIR/hypr/`) or **KDE** (`XDG_CURRENT_DESKTOP` containing
`KDE`). Force it with `--display-control hyprctl|kscreen`; the tool still needs
the session env to reach the compositor. Also confirm the requested size is an
advertised mode (the server picks the nearest).

**Session sized wrong / "scanout exceeds negotiated slot".** The compositor
commits a mode change asynchronously; the server waits (bounded) for the DRM
scanout to match the reply before sizing. If it times out it logs the mismatch.

**`fence` sync fell back to `settle`.** `fence` mounts tracefs itself (needs
`CAP_SYS_ADMIN`) and writes it (`CAP_DAC_OVERRIDE`, or pre-mount tracefs with
`uid=<you>`) on a tracefs-capable kernel, and only engages once its calibration
sees an actively flipping screen. Until then (and if tracefs is unavailable) it
falls back to `settle:2ms` and says so.

**Client has no window / `wayland connect` error.** The client needs
`WAYLAND_DISPLAY` + `XDG_RUNTIME_DIR` (e.g. under `sudo -E`).

**Input not reaching the server.** `uinput` needs `/dev/uinput` (root or the
`input` group). Gamepad capture needs read+write on `/dev/input/event*`.
If `keyboard-shortcuts-inhibit` is missing, the grab is mouse-only.

**`lossy` fails to `dlopen`.** `libpyrowave-shared.so` must sit next to the
binaries and be built against the lab's glibc
(`scripts/build-pyrowave-docker.sh`); a newer host bakes in symbol versions the
lab lacks.

**RDMA not used.** `--transport auto` needs a routable GID on port 1 on both
ends (`[rdma] no routable GID`). Check `RLIMIT_MEMLOCK` covers the pinned pages
and that the QP uses `PERMISSIVE` access. Force `--transport tcp` to isolate.

**Handshake rejects the peer.** The wire format is versioned
(`VERSION`, currently 13). Rebuild the server and client together.

## 8. Useful commands

```bash
# what capability the server binary actually has
getcap target/release/rdma-display-server

# 10 s headless capture to a PPM (no compositor needed)
./target/release/rdma-display-client HOST --output none --seconds 10 --dump out.ppm

# latency overlay + clock sync
./target/release/rdma-display-client HOST -f --metrics

# which sync mode the server chose and why
./target/release/rdma-display-server --capture-sync auto   # watch [capture-drm] lines

# advertise the current monitor mode to the server
./target/release/rdma-display-client HOST -f --native
```
