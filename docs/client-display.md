# M5 — Client presentation (Wayland/EGL), zero-copy first

> **Post-C1 cleanup (2026-09-19):** the client now ships exactly two slot
> memories and one presenter: `--slot-mem host` (pinned RAM, the production
> path) with `--output egl` (GPU upload), plus a **dormant** `--slot-mem cuda`
> kept for future cooperative hardware. Removed as dead experiments: the
> dmabuf-attach presenter (`--output wayland`) and the `heap`/`gbm` slot
> allocators, along with all wl_buffer/release-credit machinery and the
> `linux_dmabuf` bind — KWin 6.7.5+NVIDIA refuses foreign system dmabuf
> imports (`import_failed`), GBM exports never registered with the NIC
> (ENOMEM), and CUDA's export is RM-gated on GeForce. Full findings below;
> resurrect via git history on cooperative hardware (Mesa GPUs, open-module
> pro/data-center NVIDIA).

Goal: received frames land in the client GPU and get on screen with **no CPU
copies**: NIC → (GPUDirect RDMA) → GPU dmabuf slot → wl_buffer → scanout.
Fallback: host-memory slots + one H2D blit (the copy path, mirroring the
server's decision tree).

```
server                                          client
materializer → slot MR ─ RDMA WRITE+imm ─────► CQ event → slot ready
   (host mem, today)                          │            │
                                              │            ├─ attach as wl_buffer (zero-copy)
   ◄────────── TCP: credit ───────────────────┘            └─ or EGL import → blit to window
   (credit returned on wl_buffer.release)
```

The protocol (`proto`) does not change: the client already advertises
`addr/len/rkey` per slot in `BUFFERS`; whether that memory is host RAM or GPU
VRAM is the client's business. Credits flow back over TCP; we only change
*when* a credit is returned (on compositor release instead of on dump).

## Phase C0 — viability gate (spike, half a day)

Everything hinges on one question: **will `ibv_reg_dmabuf_mr` accept a dmabuf
that we allocated on the client GPU?** On the server it failed (ENOMEM) because
those were NVIDIA-allocated scanout buffers exported through nvidia-drm's
static PRIME path. The client stack differs in our favor, but it is still
NVIDIA — so measure before building.

Diagnostics to collect on the client host first:

```bash
# 1. open vs proprietary kernel module — GPUDirect-over-dmabuf needs 'open'
modinfo nvidia | grep ^license          # "MIT" => open module; "NVIDIA" => proprietary
cat /proc/driver/nvidia/version
# 2. BAR size (peer writes land in BAR1; small BAR = tiny usable window)
sudo lspci -vv -s <gpu> | grep -E "Region 1|Memory at"
# 3. PCIe topology: NIC and GPU under the same root complex/switch?
sudo lspci -t; readlink /sys/class/infiniband/rocep5s0/device
# 4. compositor + driver version
echo $XDG_SESSION_TYPE; weston-info 2>/dev/null || loginctl show-session $XDG_SESSION_ID
```

Spike: **implemented as `rdma-display-client --slot-mem gbm`** (allocation +
export + `ibv_reg_dmabuf_mr` live in `crates/rdma/src/slots.rs` /
`ClientLink::connect_with`). Gates enforced there, in order:

1. `gbm_create_device` on the client rendernode succeeds (fails loudly if no
   GBM backend — NVIDIA needs the open kernel module + libnvidia-egl-gbm).
2. `gbm_bo_create(XB24, GBM_BO_USE_LINEAR | GBM_BO_USE_SCANOUT)` succeeds and
   `gbm_bo_get_modifier == 0` — the allocator honoring USE_LINEAR is the key
   NVIDIA unknown; a tiled slot gets hard-rejected (NIC writes are
   layout-oblivious; this is the server garble lesson applied preemptively).
   Also asserts gbm stride == wire stride.
3. `gbm_bo_get_fd` exports, and `ibv_reg_dmabuf_mr` accepts (registered with
   LOCAL_WRITE|REMOTE_WRITE only — deliberately **no** RELAXED_ORDERING, see
   the C2 ordering note).
4. Content verification: the testcard server writes its usual pattern into
   the GPU slots; the client best-effort CPU-mmaps the dmabuf for
   `--dump`/checksums. If PRIME mmap of VRAM is refused (plausible), the
   client says so and content must be judged GPU-side (EGL import + on-screen
   later in C1, or an EGL readback spike if we want proof before that).

### C0 field results (client: RTX-class NVIDIA, open module 615.71, KWin 6.7.5)

- Diagnostics: open kernel module ✓, BAR1 = 16 GB (full-VRAM resizable BAR) ✓,
  AMD-Vi IOMMU in Translated mode, GPU `00:01.1` / NIC `00:02.2` same host bridge.
- GBM route: **gates #1–2 PASS** — `gbm_bo_create_with_modifiers([LINEAR])`
  returns a genuinely linear XB24 bo (modifier 0x0, stride 7680). The legacy
  `gbm_bo_create(USE_LINEAR)` silently returned block-linear
  `0x300000000e08014` (same tiling as server scanouts) — caught by the assert.
- GBM route gate #3 **FAILED**: `ibv_reg_dmabuf_mr` → ENOMEM. Mechanism: the
  exporter is nvidia-drm's *static* PRIME path, which refuses peer-DMA
  mappings for VRAM — identical wall to the server-side registration failure,
  from the opposite side. Buffer ownership didn't matter; exporter flavor does.
- ⇒ Route 2 implemented as `--slot-mem cuda` (`crates/rdma/src/cuda.rs`):
  `cuMemAlloc` + `cuMemGetHandleForAddressRange(DMA_BUF_FD)` — NVIDIA's
  dedicated GPUDirect-over-dmabuf exporter (dynamic P2P mappings).
  **Result: DMA_BUF_SUPPORTED=1 advertised, but the export call itself
  returned CUDA_ERROR_NOT_SUPPORTED** — refused by RM at export, before NIC
  or IOMMU involvement ⇒ GPUDirect-to-VRAM is product-gated on this GPU
  (GeForce-class). VRAM zero-copy is dead on this box; no userland unlock.
- ⇒ Route 3 `--slot-mem heap` (`crates/rdma/src/heap.rs`): system dma-heap
  (`/dev/dma_heap/system`) slots — system RAM that IS a dma-buf from birth.
  NIC registers it as plain host DMA (no GDR gate; exporter is the kernel),
  the same fd is EGL-importable / presentable as a wl_buffer (C1+), and CPU
  never touches frame bytes. This keeps the actual zero-copy goal while
  abandoning only the VRAM-landing implementation. malloc would work
  equally for the NIC — the heap exists purely because malloc has no fd for
  the GPU/Wayland participants.
- Ordering note: with RAM-destination slots (host or heap), payload and CQE
  are posted writes to the SAME destination ⇒ in-order ⇒ the sentinel
  protocol from C2 is unnecessary on this route.

Run order on the client host:

```bash
# 0. environment facts (module flavor, BAR size, NIC/GPU topology, GBM stack)
./docs/c0-client-checks.sh        # paste full output

# 1. baseline sanity (host slots against testcard server)
sudo rdma-display-client 192.168.4.2:4791 --device rocep5s0 --seconds 10 --dump host.ppm

# 2. the C0 gate itself
sudo rdma-display-client 192.168.4.2:4791 --device rocep5s0 \
  --slot-mem gbm --rendernode /dev/dri/renderD128 --seconds 10 --dump gbm.ppm
```

**Decision table:**

| C0 result | Path |
|---|---|
| dmabuf MR registers, WRITE lands, linear honored | **C2 full zero-copy** (NIC→VRAM→scanout) |
| MR registers but WRITE lands in host pages / BAR issues | **C1 copy-presenter** (host MR + H2D), revisit GDR later |
| MR fails (likely if proprietary module) | **C1**; optionally reinstall `nvidia-driver-*-open` and retry C0 |
| Client GPU is actually Intel/AMD (display head) | dmabuf MR + `USE_LINEAR` work there; do **C2** as-is |

## Phase C1 — Wayland presenter (IMPLEMENTED — first hardware test pending)

Shape changed after C0's outcome: since `heap` slots passed the zero-copy
gate, the presenter is built **directly on them** — no EGL context on the
client at all. `crates/client/src/wayland.rs` (~350 lines, raw
wayland-client 0.31 + wayland-protocols 0.32, no sctk/winit):

- xdg_toplevel, fixed 1:1 window at frame size; initial commit maps it.
- Each heap slot fd wrapped ONCE into a persistent `wl_buffer` via
  `zwp_linux_dmabuf_v1` v2+ `create_immed` (XR24, stride, modifier 0).
  The object data of each buffer is its slot index ⇒ `Release` events map
  1:1 to slots.
- Frame path: CQ completion → `attach + damage + commit`. Zero pixel writes.
- Credits returned on `wl_buffer.release` (compositor-paced). RDMA crate
  gained `try_next_frame` (non-blocking) and split `rearm_slot`/`credit_slot`
  from `ack_frame` to support this.
- Event integration without an async runtime: `prepare_read()` guard →
  `poll(guard.connection_fd(), 1ms)` → `read()` → `dispatch_pending` — the
  exact sequence wayland-backend's docs prescribe; loop drains all RDMA
  completions per iteration.
- `--output none|wayland`; headless stats/dump path untouched; KWin import
  failure surfaces as `[wayland] COMPOSITOR REFUSED DMABUF IMPORT
  (params::failed)` — the C0 open question answers itself with a log line.

- New dep: `wayland-client` + `wayland-protocols` (std-only, no C toolchain;
  we keep the dlopen approach for EGL). Event loop stays plain threads:
  the receiver loop `poll()`s over
  `[ibv comp-channel fd, wayland fd, control TCP fd, timerfd(fps cap)]`.
  Wayland integration: `event_queue.prepare_read()` → read → `dispatch_pending`.
- `crates/client/src/present.rs`:
  - connect to `$XDG_RUNTIME_DIR/wayland-0`, bind `wl_compositor`,
    `zwlr... or zwp_linux_dmabuf_v1`, `wl_shm` (window fallback).
  - single window (xdg-shell via `wayland-protocols`' xdg-shell module),
    size = frame dims from `META`; resize → recreate surface.
  - slot state machine: `Free(credit owed)` / `Filled(seq)` / `OnDisplay`
    (compositor holds it). **The wl_buffer `release` event is what sends the
    TCP `CREDIT`** — that gives us compositor-paced backpressure for free and
    replaces the "dumped" notion. 3 slots cover a 1-deep queue: 1 scanning,
    1 queued, 1 being written.
  - copy-path slots: keep today's host `ibv_reg_mr` buffers, but the wl_buffer
    is created per-frame from a persistent `GBM_BO_USE_SCANOUT|LINEAR` buffer:
    `wl_dmabuf_feedback`-free path: CPU `memcpy` host MR → gbm bo (via
    `gbm_bo_map` write or EGL `glTexImage2D` upload + FBO draw). On the copy
    path, prefer EGL upload: `eglCreateImageKHR` is *not* needed; just
    `glTexImage2D(GL_BGRA)` + blit into the window surface; `eglSwapBuffers`
    owns the wl_buffer lifecycle, and credits return on surface `frame`/present
    events (approximate: return credit on swap completion).
  - `--output ppm` (today's behavior) stays available for headless debugging.
- Orientation: the wire is top-first (server guarantees), wayland wants
  top-left origin for dmabuf buffers — no flip anywhere on the client.

Status after the KWin `import_failed` (protocol error 7) verdict:
the dmabuf-attach presenter (`--output wayland`, kept — it should work on
Mesa compositors/GPUs) is joined by the **EGL-upload presenter
(`--output egl`, now the intended default path)**: NIC → pinned host slot
(`--slot-mem host`, mirroring the server staging) → `glTexImage2D` from the
slot's CPU pointer (GPU reads host RAM; **no dma-buf import anywhere**) →
shader blit with `.bgra` swizzle → `eglSwapBuffers` on an EGL window surface
whose backings are driver-native. Credits return immediately after the upload
call (glTexImage2D consumes the client pointer per spec), with vsync pacing
via swap-interval throttling. Server-side: steps 1+2 already realized the
"import → convert → transfer to host buffer" target; steps 3+4 (heap
staging / render-into-imported) dropped as dead ends on this hardware —
the malloc MR and heap MR are the same pages+DMA for the NIC, and every
foreign-dmabuf import door tested on NVIDIA so far has refused.

Acceptance: live desktop in a window at ≥50 fps with the current server,
`--output wayland`, credits pacing naturally (no tearing if compositor does
page flipping).

## Input grab (Ctrl+Alt+M)

The EGL window forwards keyboard/pointer to the guest whenever it is
focused, but the local desktop is still in charge of its own shortcuts.
Ctrl+Alt+M (client-local, never forwarded to the guest) toggles an input
grab that hands *everything* to the guest — keys and mouse:

- **keys** — `keyboard-shortcuts-inhibit-unstable-v1` (KWin: v1). While the
  inhibitor is up the compositor must not act on its own bindings and sends
  them over as plain key events, so **alt-f4, alt-tab, alt-f1 and every other
  desktop shortcut reach the guest** instead of the client desktop. There is
  no client-side substitute for this protocol: if the global is missing, the
  grab still takes the mouse and logs `keyboard grab UNAVAILABLE`.
- **mouse** — `zwp_pointer_constraints_v1` lock + `zwp_relative_pointer_v1`
  (unaccelerated deltas → `InputEvent::Rel`), local cursor hidden. A lock
  requested without hover stays dormant until the pointer enters the window.

**Ctrl+Alt+M is the only way out.** A focus shuffle deliberately no longer
releases the grab (that was the hole through which alt-tab and friends
dumped the user out of relative mode mid-play); the pieces self-heal instead:
the pointer lock is re-acquired on pointer re-entry and whenever the
compositor breaks it (`zwp_locked_pointer_v1.unlocked`), and the shortcut
inhibitor re-engages by itself on keyboard focus. The chorded M press+release
is swallowed entirely and only the *initial* press toggles (key repeats
don't).

Protocol caveats worth one lab check (see `crates/client/src/wayland.rs`,
`set_grab`/`grab_keys`):

- the compositor "is under no obligation to disable all of its shortcuts"
  and may keep one escape combo for itself (it then sends `inactive`, logged
  as `keyboard grab INACTIVE`);
- KWin's 2020 implementation disabled all global shortcuts but not the
  shortcut "spies" (e.g. the Meta key opening the app launcher) — that was
  to be fixed later, so verify Meta on current Plasma.

## Phase C2 — zero-copy slots (NIC → GPU dmabuf → scanout)

Only allocation changes; wire/protocol untouched.

- `crates/client/src/slots.rs` (new): `SlotAllocator` trait with impls
  `Host` (today) and `Gbm` (zero-copy). `Gbm` allocates each slot as a
  `gbm_bo` (XB24, LINEAR|SCANOUT), `gbm_bo_get_fd` → `ibv_reg_dmabuf_mr`
  (offset = `gbm_bo_get_offset(bo, 0)`) → advertises rkey in `BUFFERS`.
  Client keeps `zwp_linux_buffer_params` dmabuf buffers alive per slot:
  same fd, fourcc `XB24`, modifier `0`, stride from META/bo.
- Per-frame path: CQ event (imm carries seq+slot) → slot is *display-ready* →
  `attach_and_commit`. Nothing touches pixels between NIC and compositor.
- **PCIe ordering (the real zero-copy hazard):** the CQE is a posted write to
  host RAM; frame data are posted writes to GPU BAR. PCIe does *not* order
  posted writes from one source to two different destinations — the CQE can
  land before the last of the data. Mitigation, server-side and cheap:
  - server splits each frame into two RDMA WRITEs: payload (no imm) then an
    8-byte trailer `sentinel = seq` at `slot_end - 8` **with imm**;
    writes to the *same destination* (the GPU BAR) are ordered, so a visible
    sentinel ⟹ all payload bytes that precede it are in the buffer;
  - client CPU-spin (with timeout + fallback re-poll) reads the sentinel via
    the mmap'd dmabuf before committing; bail to `--copy` fallback on timeout.
  - measure the spin cost first — often ~0 (the CQE usually arrives late
    anyway). Alternative if sentinel reads are ugly: `wr_flush` / an extra
    READ-after-WRITE roundtrip, or `IBV_ACCESS_RELAXED_ORDERING` **off**
    (it is off by default; keep it off).
- Resize/renegotiation: on `META` change, destroy wl_buffers + bos, reallocate,
  send new `BUFFERS` (server already handles mid-session re-advertising? if
  not, wire a `RESET` to control channel — small proto addition).
- Fences: QEMU's guest-side `Fence` fd is *not* needed on the client; by
  sentinel+CQE the data is complete. (Server-internal fence sync stays a
  separate M6 item.)

Acceptance: same visual result as C1, with `perf stat`/bandwidth showing no
per-frame CPU memcpy, and ≥ the same fps. Compare both modes in the docs.

## Code reuse with the server side

- The server's proven EGL import recipe (`egl.rs`) is exactly what the client
  needs *if* we ever blit imported slots; extract the minimal pieces into
  `crates/egl-util` (libEGL/GL dlopen, device display, surfaceless context,
  `eglCreateImageKHR` + `EXTERNAL_OES` bind + shader blit) shared by
  `capture` and `client`. Do this during C1 — the presenter needs the same
  three functions, and duplicating 400 lines of FFI twice is how we got the
  0x3275 bug in the first place.
- GBM: do **not** resurrect the deleted server-side `gbm.rs` materializer; the
  client needs GBM allocation (not import), for which the maintained `gbm`
  crate (wayland-rs family) is preferable to hand-rolled dlopen.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Proprietary nvidia module on client → dmabuf MR fails | C2 dead on arrival | C0 gate; `nvidia-driver-5xx-open` package swap; or Intel/AMD display GPU does C2 anyway |
| NVIDIA GBM ignores `USE_LINEAR` (tiled slots) | NIC writes garble | C0 step 2 asserts modifier==linear; if tiled: C2-with-blit (NIC→tiled VRAM is pointless) ⇒ fall to C1 |
| Resizable BAR / BAR1 too small for 3×8MB slots | registration ok, writes fail | C0 checks BAR; slots = stride×h ≤ BAR anyway (24MB fine on 256MB BAR1) |
| PCIe write reordering (CQE beats data) | 1-frame tearing, rare | sentinel write + CPU check (C2) |
| Compositor (mutter/GNOME on NVIDIA) import quirks for our linear XB24 bos | present fails | C0 spike also exports one bo to a weston window test; kwin/weston first if GNOME misbehaves |
| NIC and GPU on different root complexes | P2P slow (cross-socket) or fails | C0 topology check; document that GDR wants same-complex placement |

## Order of work

1. C0 diagnostics + spike on the client host (needs your hardware; I'll write
   the code, you run it) → decide path.
2. Extract `egl-util`; build C1 presenter on the copy path; ship day-to-day
   usable display.
3. C2 zero-copy slots + sentinel write (proto: add trailer WRITE; server flag
   `--slot-sentinel`), A/B against C1.
4. Only then: server-side QEMU `Fence` sync + mid-session renegotiation
   (M6/M7 backlog), which the presenter will want anyway for tearing-free
   capture pacing.
