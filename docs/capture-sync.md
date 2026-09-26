# Capture sync / handover tracing

Status: investigation in progress (2026-09-26). Companion scripts in
`scripts/capture-handover/`.

## Problem

KMS capture polls `GETCRTC` and, with no producer fence on the scanout dma-buf
(`syncprobe`: stub fences only), uses **one-behind** — tear-free but one refresh
of latency (`queue ~16.5 ms` at 60 Hz). The `vblank` mode that would avoid it is
dead: `CRTC_GET_SEQUENCE` returns `Ok` with a frozen sequence (`1`, `ns 0`).

## Kernel surface found on this host (6.12.74+deb13, driver 610.57.04)

- Tracepoints compiled in: `dma_fence_{init,emit,enable_signal,signaled,wait_start,wait_end,destroy}`,
  `drm_vblank_event{,_queued,_delivered}`.
- Generic DRM: `drm_atomic_set_fb_for_plane`, `drm_mode_atomic_ioctl`,
  `drm_atomic_helper_wait_for_fences`, `drm_syncobj_{wait,timeline_wait}_ioctl`,
  `drm_mode_getcrtc`, `drm_crtc_send_vblank_event`, `drm_flip_work_commit`.
- NVIDIA: `nv_drm_handle_flip_event`, `nv_drm_crtc_send_vblank_event`,
  `nv_drm_atomic_commit`, `nv_drm_gem_prime_fence_attach_ioctl`,
  `nv_drm_prime_fence_context_create_ioctl`, `nv_drm_semsurf_fence_{create,attach}_ioctl`.
- BTF present for the base kernel and the `drm`/`nvidia*` modules (CO-RE possible).
- **NVIDIA's flip entry points are `notrace`**: `nv_drm_handle_flip_event` can be
  neither kprobed nor traced (`not traceable ... attaching to it will likely
  fail`). Observation must go through the generic DRM functions/tracepoints that
  NVIDIA calls into.

## Hypothesis (Phase 1)

The frozen **counter** and the flip **event** are different mechanisms. The event
path is `nv_drm_handle_flip_event -> nv_drm_crtc_send_vblank_event ->
drm_crtc_send_vblank_event -> tracepoint drm_vblank_event_delivered`. If it
fires per presented frame, the signal the vblank mode wanted is available.

## Plan

| Phase | Goal | Tool |
|---|---|---|
| 1 | Does `drm_vblank_event_delivered` fire, at what rate, with what `seq`? | `scripts/capture-handover/phase1-flip-event.{sh,bt}` |
| 2 | Map the sync model: explicit (syncobj/semsurf) vs implicit (dma_resv); measure `T_commit`, `T_flip`, `T_ready`, `T_obs` | bpftrace/ftrace on `drm_atomic_set_fb_for_plane`, `drm_syncobj_*`, `nv_drm_*fence*`, `dma_fence_signaled`, `drm_mode_getcrtc` |
| 3 | Correlate with the server's `t0`/`t0d`/`t0p` on CLOCK_MONOTONIC; split `queue` into one-behind vs observation lag | server diagnostic + trace |
| 4 | Decide: event-driven capture mode, fence-checked read, or `ext-image-copy-capture` backend | - |
| 5 | Durability: keep tracing as scripts + docs; no eBPF dependency in the build | `scripts/`, this doc |

## Phase 1 result (2026-09-26)

Probe: `scripts/capture-handover/phase1-activity.bt`, run while continuously
repainting a terminal. 532 atomic commits over ~20 s.

| counter | value | meaning |
|---|---|---|
| `drm_mode_atomic_ioctl` | 532 | commits reach KMS (~26/s during motion) |
| `drm_atomic_set_fb_for_plane` | 532 | **exactly one per commit** - precise handover |
| `drm_vblank_event{,_queued,_delivered}` | 0 | no vblank tracepoint ever fires |
| `drm_crtc_arm_vblank_event` | 0 | no page-flip event is ever armed |
| `drm_crtc_send_vblank_event` | 0 | generic vblank event function never called |
| `drm_send_event` / `_locked` | 0 | **no event is delivered to userspace at all** |
| `drm_syncobj_timeline_wait_ioctl` | 59 | explicit sync: timeline syncobj waits |
| `drm_syncobj_wait_ioctl` | 0 | non-timeline syncobj wait unused |
| `drm_atomic_helper_wait_for_fences` | 0 | generic in-fence wait never called (nvidia has its own tail) |

### Verdict

- **Flip-event hypothesis: falsified.** The compositor arms no page-flip event
  and delivers no event to userspace, so the `drm_vblank_event*` tracepoints can
  never clock a capture on this stack. This is independent of the frozen counter.
- **The handover is precisely observable.** `drm_atomic_set_fb_for_plane` fires
  once per commit (with the fb id), giving an exact "compositor handed this fb
  to KMS" timestamp without polling.
- **Explicit sync confirmed** (`so_tl_wait > 0`, `so_wait == 0`). This is the
  mechanism behind the earlier stub `dma_resv` fence: the compositor syncs via
  `drm_syncobj` timeline points, not the dma-buf reservation.
- **`nv_drm_*` flip entry points are `notrace`** and cannot be probed by bpftrace
  or ftrace; only generic DRM functions are visible.

### Open question carried into Phase 2

No KMS "flip complete" event exists, but the frame may already be complete at
handover, depending on the composition mode:

- **direct scanout** (client buffer on the plane): the compositor waited on the
  producer's syncobj (`so_tl_wait`) before committing, so at handover the fb is
  complete - handover is a tear-free read point.
- **composited** (compositor's own swapchain on the plane): the compositor's
  render may still be in flight at handover, so completeness comes later, via a
  commit/out-fence we have not yet identified.

Phase 2 (`scripts/capture-handover/phase2-handover.bt`) measures: handover
cadence, whether `drm_mode_atomic_ioctl` blocks (its return would be a
completion signal), the producer-wait duration, commit-wait duration, and which
`dma_fence` contexts signal and when.

## Phase 2 result (2026-09-26)

`phase2-handover.bt`, 401 handovers over ~40 s of motion.

| observation | value | reading |
|---|---|---|
| `@atomic_ms` | 401/401 in `[0,1)` ms | atomic commits are **non-blocking**; the ioctl return is not a completion signal |
| `@handover_us` | 391/401 in 16-32 ms | ~30-60 Hz handover while active (a few long stalls = idle gaps) |
| `@signaled[6]` | **401 = handover count** | a **per-frame fence** exists: context 6 |
| `@signaled[433]` | 1598 (~4/frame) | another context, per-frame work (render passes?) |
| `@signaled[321/323/324/325/0]` | 4-7 each | occasional contexts |
| `@tl_wait` | 7, all 2-8 us | the compositor almost never blocks on a producer syncobj in-kernel |
| `@commit_wait` | 0 | no generic commit wait |

Sample timing: `handover` happens, then `ctx=6` signals **~6-11 ms later**, before
the next handover. `ctx=6`'s seqno increments by one per frame.

### Reading

- The compositor hands the fb to KMS **before its GPU work is complete**
  (non-blocking commit, nothing waits in-kernel). The frame becomes complete when
  **`dma_fence` context 6 signals**, a few ms after handover.
- `ctx=6` is therefore the **safe-read clock**: read the current fb after it
  signals and the frame is complete, with ~0 added latency - no one-refresh
  one-behind wait.
- `ctx=6` is **not** on the scanout dma-buf's reservation (syncprobe saw the stub
  fence), so it is not reachable via `EXPORT_SYNC_FILE`. It must be identified via
  `dma_fence_init` (driver/timeline names) - Phase 3.

### Practical consequence (no eBPF needed in the product)

Even without observing the fence from userspace, `signal6 - handover` gives the
**settle time**: capture the current fb at `handover + settle` (p99 of that
distribution) instead of waiting for the next handover (one-behind). That trades
a bounded, rare tear risk for roughly half the capture latency. Phase 3 measures
the distribution to size `settle`.

## Phase 3 result (2026-09-26)

Two runs of `phase3-fence-id.bt`: light (terminal scroll) and sustained (~60 Hz
handover, 2595 frames). Context 6 is 1:1 with handover in both (428/428,
2595/2595). **`dma_fence_emit` never fires for it** (`emit6=0`), and its
driver/timeline names render empty through bpftrace, so the tracepoint cannot
name its owner.

`signal6 - handover`:

| bucket (us) | light | sustained |
|---|---|---|
| 16-128 | 13 | 13 |
| 128-256 | - | 262 |
| 256-512 | 133 | 903 |
| 512-1K | 167 | 612 |
| 1K-2K | 104 | 621 |
| 2K-4K | 3 | 33 |
| 4K-8K | 10 | 80 |
| 8K-16K | 11 | 60 |
| 16K-64K | - | 11 |

p50 ~= **0.75 ms**, mean ~= **1.3 ms** in both runs; p90 ~1-2 ms; p99 ~8-16 ms.
A 10x heavier workload did not move the median - the delay is essentially
constant.

### Interpretation

- `emit6=0` means context 6 is **not** a GPU-submitted render fence (those call
  `dma_fence_emit`). Combined with the constant ~1 ms offset, it is most likely
  the **KMS commit fence** (`drm_crtc_commit`, `hw_done`), signalled once the
  atomic commit is programmed.
- That is *not automatically* a "frame complete" proof: it says the commit was
  programmed, which *should* imply the scanout buffer is ready (the driver must
  not scan out an unfinished buffer) but does not directly show the compositor's
  render finished. `syncprobe`'s stub fence says the render fence is not on the
  dma-buf reservation, so userspace cannot verify it.
- A settle read at `handover + a few ms` is therefore promising (16.5 ms
  one-behind -> ~2 ms) but must be validated for tearing, and context 6's owner
  should be confirmed. The GPU fences (context 433, ~4/frame) may signal later;
  they have not been timed yet.

## Phase 4 result (2026-09-26)

Kernel stacks from `phase4-ctx6-origin.bt`:

```
creator6   dma_fence_init
           __nv_drm_semsurf_fence_ctx_create_fence.isra.0
           nv_drm_semsurf_fence_create_ioctl          <- userspace ioctl
           drm_ioctl_kernel <- drm_ioctl <- __x64_sys_ioctl <- do_syscall_64

signaler6  dma_fence_signal_timestamp_locked
           dma_fence_signal
           __nv_drm_semsurf_ctx_process_completed
           __nv_drm_semsurf_ctx_callback
           SemaphoreSurfaceKapiCallback <- osEventNotification <- _semsurfNotifyCompleted
```

Context 6 is NVIDIA's **semsurf (semaphore-surface) fence**: created once per
frame by the compositor through the private `nv_drm_semsurf_fence_create_ioctl`,
and signalled from NVIDIA's semaphore-surface completion callback. It is a
**GPU-completion signal** for the frame, which is why it tracks handover 1:1
(~0.75 ms p50, p99 ~8-16 ms) and why `EXPORT_SYNC_FILE` shows a stub: the
compositor synchronizes via semsurf, not via the dma-buf reservation.

`emit=0` for both contexts is a red herring - NVIDIA's fence ops do not call
`dma_fence_emit`, so it does not distinguish GPU from software fences. Context
433 (~4/frame) is likewise not frame-aligned; its "vs handover" histogram mixes
four signals per frame and is not meaningful.

### Consequence

- The real per-frame **completion** primitive is the semsurf fence, and it is
  owned by the compositor's process/fd. It is **not reachable from our capture**
  (not on the dma-buf reservation, private NVIDIA mechanism).
- So there is no userspace wait for completion. The options are now:

| option | latency | tearing | deps |
|---|---|---|---|
| one-behind (current) | ~16.5 ms | none | none |
| settle read | settle ms | when render outlasts settle | none |
| eBPF-triggered read on the semsurf signal | ~1 ms | none | eBPF + root + runtime context discovery |
| `ext-image-copy-capture` | ~1 compositor frame | none (compositor release fence) | Wayland-client capture, forces composition + copy |

- A **fixed settle is only useful if tearing is acceptable**: to be ~tear-free you
  need to cover the p99 (~8-16 ms), which is essentially one-behind. At ~2 ms the
  median frame is covered but ~6-7% of frames tear.
- To beat one-behind *without* tearing you need either the **semsurf signal**
  (eBPF; discover the semsurf fence context at runtime) or **screencopy** (the
  compositor's own release fence, sync-correct by construction).

## Decision (open)

Given the KMS path cannot obtain a completion fence from userspace, the robust
options are screencopy (`ext-image-copy-capture`, already proposed) or an
opt-in eBPF-triggered read. The settle mode remains as a no-dependency
latency/tearing tradeoff for experimentation.

## Implementation (2026-09-26)

Two modes were added to the KMS capture (`crates/capture/src/drm.rs`,
`capture::fence`); `onbehind` stays the default.

- `--capture-sync settle[:ms]` (default 2 ms) - read the current fb `ms` after
  its commit was observed. A small bounded delay queue keeps commits in order
  and prevents falling arbitrarily behind. No dependencies. Tearing is possible
  whenever a frame's render outlasts the window (see the p99 above).
- `--capture-sync fence` (alias `bpf`) - read the current fb when the
  compositor's per-frame completion fence signals. `capture::fence` consumes the
  kernel `dma_fence:dma_fence_signaled` tracepoint through tracefs (no libbpf,
  same tracepoint an eBPF probe would use): it calibrates the completion context
  by matching `dma_fence_init` counts against the atomic-commit count, then
  filters `dma_fence_signaled` to it and reports each signal. Requires root
  (the server already runs under sudo) and an actively flipping compositor for
  the first ~400 ms calibration; on any failure it logs and falls back to
  one-behind.

Benchmark with the client overlay (`queue prep slot mat`) over `onbehind`,
`settle:2`, `settle:4`, and `fence`; the `fence` mode should show a queue near
its processing latency and no tearing.
