# RDMA Display — Implementation Plan

Status: draft, v1 (discussed 2026-09-19)

## 1. Goal

Transmit display frames from a host running QEMU's D-Bus display backend to a
remote client over RDMA, with a zero-copy data path: the dmabuf produced by the
display source is registered directly as an RDMA memory region (MR), and the NIC
DMA-reads the frame straight out of the GPU/exporter's memory. No per-frame
staging copy through host memory.

Out of scope (for now): video encoding, cursor transport, input injection
(though the protocol reserves space for it). The client render target is a
stub in early milestones.

## 2. Decisions made

| Topic               | Decision                                        |
| ------------------- | ----------------------------------------------- |
| First capture src   | QEMU D-Bus display (`-display dbus`)            |
| Second capture src  | KMS plane capture (later)                       |
| Data path           | Zero-copy: dmabuf registered as MR via `ibv_reg_dmabuf_mr` |
| RDMA transport      | RC (Reliable Connection) + RDMA WRITE with immediate |
| Control channel     | Plain TCP / Unix socket (no RDMA CM)            |
| Runtime             | Plain std threads; polling completion loop, no tokio |
| Language            | Rust; `ibverbs` crate (jonhoo/rust-ibverbs)     |
| Test hardware       | Two physical hosts with RDMA NICs               |

## 3. Key research findings

- `ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access)` exists in
  rdma-core (symbol `IBVERBS_1.12`, rdma-core >= ~36). Kernel support landed
  in 5.12+ (`drivers/infiniband/core/umem_dmabuf.c`); the NIC driver must
  implement `reg_user_mr_dmabuf` (mlx5 does; irdma, efa, bnxt_re have it too).
- The safe `ibverbs` crate wraps it: `ProtectionDomain::register_dmabuf()`,
  returning a `MemoryRegion` usable directly in SGEs. It also exposes
  `MemoryRegion::as_raw()` as an escape hatch for provider-specific
  extensions (e.g. `mlx5dv_reg_dmabuf_mr`) if ever needed.
- Kernel-side, dmabuf MR registration attaches to the dmabuf with
  `move_notify` (dynamic attach). This is the crux of the zero-copy design:
  the exporter must support dynamic attach/pinning. Failure mode is
  `EOPNOTSUPP`/`EINVAL` at registration time.
- Allowed access flags for dmabuf MRs: LOCAL_WRITE, REMOTE_WRITE,
  REMOTE_READ, REMOTE_ATOMIC, RELAXED_ORDERING.
- The iova must have the same page offset as the MR offset within the dmabuf;
  we will simply use `iova = offset` (i.e. iova 0 for offset 0).

## 4. Architecture

```
Server (QEMU host)                          Client (display host)
+--------------------------------+          +--------------------------------+
| QEMU (dbus display backend)    |          | control ch client (TCP/Unix)   |
|   org.qemu.Display1.Listener   |          |   recv meta, ack buffers       |
|     ScanoutDMABUF / UpdateDMABUF|         |                                |
|     / Fence  (+ dmabuf fd)     |          | ibverbs: RC QP, target MRs     |
|          | fd                  |          |   (plain malloc-registered     |
|          v                     |          |    host buffers, 2-3 slots)    |
| capture: hold fd + format info |          |          ^                     |
|          |                     |          |          | remote writes       |
| rdma:    | register dmabuf -> MR (zero-copy, server side)  |                    |
|          | post RDMA WRITE + immediate  ===================> |                    |
+--------------------------------+          +--------------------------------+
        |                                                                      |
        +------------------ control channel (TCP or Unix socket) --------------+
        QPN/GID/PSN/MTU exchange, display format metadata, buffer-slot rkeys
```

Zero-copy applies to the **server side** (capture dmabuf -> NIC). The client
receives into ordinary registered host memory; client-side zero-copy into a
GPU dmabuf is a possible later extension but is not needed to satisfy the goal
(client render pipeline is a stub for now).

### 4.1 Capture (QEMU D-Bus backend)

- Start/connect to QEMU's D-Bus display server; for each console, register a
  `org.qemu.Display1.Listener` object.
- Signals/methods consumed:
  - `ScanoutDMABUF(width, height, stride, fourcc, modifier, y0_top, fd)` —
    new scanout: remember format + dmabuf fd (fd passing over D-Bus).
  - `UpdateDMABUF(x, y, w, h)` — guest repaint of the current scanout.
  - `Fence(...)` (QEMU >= 8.1) — in-fences; acquire the sync_file so the GPU
    has finished writing before the NIC reads.
  - `Scanout(width, height, stride, fourcc)` — non-dmabuf fallback scanout;
    treat as "copy path only" or reject in v1.
- Scanout changes (resolution change) mean a new dmabuf: deregister old MR,
  register new one, notify client (control channel), re-exchange buffer size.
- Rust stack: `zbus` for the D-Bus client + fd passing
  (`zvariant::Fd`). The listener thread is independent of the RDMA thread;
  they communicate via a small mpsc channel of `FrameEvent`s.

### 4.2 RDMA data path

- One RC QP server<->client per console.
- Flow control by triple buffering:
  - Client allocates N (2–3) receive slots of `frame_bytes`, registers them as
    MRs, and sends `{rkey, addr, len}` per slot plus N over the control
    channel.
  - Server posts one RDMA WRITE (single SGE) per frame to slot `i`, with the
    immediate word encoding `{frame_seq, slot_index, flags}`.
  - The immediate arrives as a RECV completion on the client's recv QP — this
    is the client's "frame N is in slot k" notification. Client is free to
    render slot k while the server keeps writing newer frames into other
    slots; the server may reuse a slot only after the client acks it (tiny
    send/recv credit message) or, for v1, tracks the oldest in-flight slot
    conservatively.
- On the server, the SGE points into the registered **dmabuf MR** (`lkey`
  from `register_dmabuf`); `wr.rkey/remote_addr` target the client slot.
- Ordering: writes with the same QP are delivered in order; fences on the
  capture side guarantee the frame is complete before the WRITE is posted.
- Completion loop: busy-poll `ibv_poll_cq` on a dedicated thread; no async
  runtime. All user interaction happens on the control channel thread
  (blocking reads; cheap).

### 4.3 Control channel protocol (v1)

Length-prefixed messages over TCP (or Unix socket for local testing).
Everything is fixed-layout little-endian structs from a shared crate.

Handshake (server -> client initiated by client connect):

1. `HELLO { magic, version, features }`
2. `DISPLAY_META { width, height, stride, fourcc, modifier, y0_top }`
3. `QP_EXCHANGE { qpn, psn, mtu, gid_index, gid[16] }` (both directions)
4. Client: `BUFFER_OFFER { slot_count, len, rkey[i], addr[i] }`
5. Server: `READY` — data path begins.

Runtime:

- `SCANOUT_CHANGED { meta }` — resize/new dmabuf; client re-offers buffers.
- `CREDIT { slot }` — client returned a slot (v2; v1 uses conservative reuse).
- `BYE { reason }`.

The `fourcc`/`modifier` pair is passed verbatim from the capture source
(DRM format/modifier codes), so the client — even in stub form — always
knows exactly how to interpret the bytes, including tiled layouts.

## 5. Workspace layout

```
rdma-display/
├── crates/
│   ├── proto/        # fixed-layout wire structs, fourcc/modifier types, codec
│   ├── capture/      # trait `CaptureSource`; qemu-dbus impl; kms impl later
│   ├── rdma/         # verbs setup, QP bring-up, dmabuf MR mgmt, write loop
│   ├── server/       # binary: wires capture -> rdma + control channel
│   └── client/       # binary: control handshake, QP, recv slots, stub sink
├── docs/
│   └── plan.md
└── Cargo.toml        # workspace
```

Dependencies: `ibverbs` (vendor rdma-core if not installed system-wide),
`zbus` + `zvariant`, `nix` (fd/fcntl helpers), `thiserror`, `tracing`.
Control channel uses plain `std::net` or `std::os::unix::net`.

## 6. Risks & mitigations

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | **Exporter lacks dynamic attach/pin support** (e.g. virtio-gpu dmabufs, some GPU drivers) — `ibv_reg_dmabuf_mr` fails | Blocks zero-copy for that source | Spike #1, first work item. If it fails for QEMU: test QEMU's `virtio-gpu-rutabaga`/venus paths, a different guest exporter, or KMS source; keep a hidden copy-path fallback behind a trait so the rest of the system still works |
| 2 | Race: GPU writes frame while NIC DMA-reads it | Corrupt/tearing frames | Only post WRITE after `Fence` (sync_file) is signaled or, for KMS, on vblank event; also rely on dmabuf `dma_resv` write barriers handled by the kernel MR path |
| 3 | Scanout changes (resize) invalidate MR mid-transfer | Faults / lost frames | Quiesce the QP during scanout transition; re-register; notify client; reclaim in-flight slots |
| 4 | Tiled modifiers make client rendering hard | Client stub complexity | v1 client can still blit via EGL importing the dmabuf semantics (it has fourcc+modifier); no manual decode needed |
| 5 | MR lifecycle vs dmabuf lifetime (QEMU may reuse fd or close it) | Use-after-free / EBUSY | Capture layer owns fd validity events; rdma layer deregisters before drop; add an integration test for resize storms |
| 6 | Flow control stall (server outpaces client rendering) | Frame drops or deadlock | Triple buffer + conservative slot reuse in v1; credits in v2 |
| 7 | rdma-core version mismatch on hosts | Build/runtime failure | Vendor rdma-core via the `ibverbs` crate build (supported), or require >= v36 in docs |

## 7. Milestones

**M0 — Hardware/stack spike: COMPLETE (2026-09-19).** Verdict from the
`spike-dmabuf-mr` experiment (crate since deleted from the tree — see git
history) on the real hosts:

- Capture path fully works: libvirt private D-Bus bus → QEMU console
  `video0` → `ScanoutDMABUF` (XRGB8888, NVIDIA block-linear modifier).
- NIC: ConnectX-4 Lx (`rocep5s0`); ODP supported (general + RC WRITE/READ) —
  the kernel dmabuf-MR prerequisite is satisfied.
- Plain host-memory MR registration: works (control experiment).
- `ibv_reg_dmabuf_mr` on QEMU scanout dmabufs: **fails with ENOMEM**, for any
  length (incl. single page). Evidence chain: `exp_name: drm` ⇒ exporter is
  nvidia-drm via the generic static PRIME path (`get_sg_table` only, **no
  pin/unpin ops**); the 6.12 path's silent-ENOMEM site is
  `dma_buf_map_attachment` (NVIDIA sgt + P2P DMA mapping of VRAM pages for
  the mlx5 device). Zero-copy dmabuf-MR is therefore NOT viable with an
  NVIDIA rendernode, independent of the NIC.

Decision: **copy path is the default data path** (`mode = copy` config);
zero-copy dmabuf-MR remains implemented behind the same config (`mode =
zero`) for hosts with a cooperative exporter (Intel/AMD rendernode, future
NIC/driver combos). Both modes share the entire capture/RDMA/protocol stack;
only the frame-materialization step differs.

**M1 — Protocol crate + control channel** (`proto`, `client`/`server`
skeletons): handshake, metadata exchange, QP bring-up over TCP control
channel. Test: two processes establish a RC QP and echo a message.

**M2 — RDMA loop with host-memory MRs: VERIFIED on real hardware
(2026-09-19).** Testcard server + stub client, two ConnectX-4 Lx hosts over
RoCE. One RC QP, 3 triple-buffered slots, WRITE+immediate, TCP credits:
~21 fps sustained for 80 s with seq lockstep server<->client, zero drops.
(The 21 fps ceiling is the server's *software* testcard fill, not the
transport; see M7.) First-run fixes: QP access flags must include
REMOTE_WRITE (`set_access(PERMISSIVE)`) — MR flags alone are not enough for
inbound writes; and RLIMIT_MEMLOCK must cover pinned pages (run elevated or
raise the limit on both hosts).

**M3 — QEMU D-Bus capture** (`capture`): listener, scanout metadata, fence
handling, scanout-change events.

**M4 — Copy-path frame materialization** (default `mode=copy`): import scanout
dmabuf into an EGL context on the rendernode, blit to linear, read into a
registered host MR, RDMA WRITE. (A `mode=zero` server variant was planned but
never implemented; the spike's verdict — no cooperative exporter on this
stack — settled it as a copy-path architecture. See M4 status above.)

### M4 status (verified end-to-end on hardware)

QEMU D-Bus capture -> EGL blit -> readback -> RDMA WRITE -> client PPM works
(1920x1080, XB24/AB24 fourccs, NVIDIA modifier 0x0300000000e08014). NVIDIA
quirks learned on the way, all encoded in `capture/src/egl.rs`:

- The dmabuf importer is `eglCreateImageKHR` with the **EGLint** attribute
  list including `EGL_DMA_BUF_PLANE0_MODIFIER_LO/HI_EXT` = **0x3443/0x3444**.
  (An early build used 0x3275/0x3276 — nonexistent attr names — which made
  every modifier import fail with EGL_BAD_ATTRIBUTE and silently fall back to
  no-modifier import: import "succeeded" but the block-linear buffer was
  sampled as linear → diagonal-shear garble. GBM `FD_MODIFIER` import of these
  nvidia-drm-exported buffers also fails; both detours were removed.)
- Foreign dmabufs bind as `GL_TEXTURE_EXTERNAL_OES` (GLES-only target);
  binding them to `GL_TEXTURE_2D` is invalid (0x0502). The blit shader is
  `#version 310 es` with `samplerExternalOES`.
- Context: surfaceless (EGL_KHR_surfaceless_context) on the EGL device
  platform for the rendernode; `eglBindAPI(EGL_OPENGL_API)` + version-3.1
  context works with the ES shaders on NVIDIA (ARB_ES3 compatibility path) —
  left exactly as tested, do not "fix" without hardware to retest.
- Orientation: OES sampling puts scanout row0 at FBO row0 for y0_top=false,
  so readback rows are copied in order (no extra flip).
- Readback via persistent-coherent PBO + RGBA8 linear target; the wire format
  is always XR24/BGRX bytes, top-first rows.
- The working materializer set is `egl` (default) and `plain` (host-memory
  scanouts); the GBM materializer experiment was removed.

**M5 — Client presenter**: Wayland/EGL window. Plan in
`docs/client-display.md`: C0 spike first (will `ibv_reg_dmabuf_mr` accept our
own client-side GBM allocations? the server's ENOMEM was on *foreign*
nvidia-drm exports, the client allocates its own slots), then C1 Wayland
presenter on the proven copy path, then C2 full zero-copy
(NIC → GPUDirect → GPU dmabuf slot → wl_buffer → scanout, no CPU copies,
compositor-release-paced credits). PPM dump remains as headless debug mode.

**M6 — KMS capture source** (second backend) behind the same
`CaptureSource` trait; adds vblank pacing + modifier from DRM.

**M7 — Credits, resize storms, long-run stability, benchmarks.**

## 8. Open questions

- Hardware/stack confirmed: ConnectX-4 Lx (mlx5, RoCE), QEMU 11.0, Linux
  6.12 on both hosts. Still to verify: QEMU guest GPU setup on the capture
  host (virtio-gpu-gl vs virtio-gpu-rutabaga/venus) and that dmabuf scanout
  is actually produced (the spike prints this).
- Multi-console/multi-head in v1, or single head only? (Plan assumes one
  console per process instance.)
- Does the client need to survive mid-session scanout format changes (e.g.
  guest switching XRGB8888 <-> tiled modifier), or restart on change?
