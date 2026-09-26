# Capture handover tracing

Diagnostics for how the compositor hands frames to DRM on the Hyprland/NVIDIA
server, and when a frame is complete enough to read. These are **measurement
tools**, not part of the server build.

## Why

The KMS capture (`crates/capture/src/drm.rs`) reads the scanout framebuffer by
polling `GETCRTC` and, because the scanout dma-buf exposes no producer fence
(`syncprobe`: stub fences only), uses the **one-behind** policy: capture the fb
that was just superseded. That is tear-free but costs up to one refresh
(`queue ~16.5 ms` at 60 Hz).

The vblank-synced mode that would avoid that cost is dead on this stack:
`CRTC_GET_SEQUENCE` returns `Ok` but the sequence is frozen at `1`, `ns 0`.

The open question this directory answers: **page-flip completion is a different
path from the vblank counter.** If its event fires, an event-driven capture can
key off the flip and drop the one-behind lag.

```
compositor (Hyprland)                 kernel (nvidia-drm)                    trace
  drmModeAtomicCommit  ->  drm_mode_atomic_ioctl
                             drm_atomic_set_fb_for_plane  (fb handed to KMS)  kprobe
                             ... plane update, flip programmed ...
  <-- flip complete <--      nv_drm_handle_flip_event
                               nv_drm_crtc_send_vblank_event
                                 drm_crtc_send_vblank_event
                                   tracepoint drm_vblank_event_delivered      tracepoint
  capture GETCRTC      ->    drm_mode_getcrtc  (our observation, "t0d")        kprobe
```

## Phase 1 - does the flip event fire?

> Result: **no** the DRM flip events never fire on this stack. The compositor's
> real per-frame completion primitive is an NVIDIA `semsurf` fence, reachable
> only through the `dma_fence` tracepoint. See `docs/capture-sync.md` and the
> `--capture-sync settle|fence` modes built from this.

Run as **root on the compositor host** and create visible motion during the run
(drag a window around, or play a video). A hardware cursor moving alone may not
repaint the primary plane.

```sh
# pure tracefs, no extra packages:
sudo scripts/capture-handover/phase1-flip-event.sh --duration 12
#   ... or, if hyprctl is on PATH and the session is reachable:
sudo scripts/capture-handover/phase1-flip-event.sh --duration 12 --motion-hypr

# bpftrace equivalent (if installed):
sudo bpftrace scripts/capture-handover/phase1-flip-event.bt
# the discriminator (generic DRM: did commits happen, was an event delivered?):
sudo bpftrace scripts/capture-handover/phase1-activity.bt
```

> **NVIDIA's flip entry points are `notrace`.** `nv_drm_handle_flip_event` cannot
> be attached by bpftrace or ftrace (the kernel rejects the kprobe). This is why
> the probes only use generic DRM functions and tracepoints; when NVIDIA crosses
> into generic DRM it becomes visible, and `drm_send_event*` tells us an event
> reached userspace even if the vblank tracepoint never fires.
> `phase1-nvidia-paths.bt` is kept only as a record of that dead end.

Nothing is installed and no service is changed; the script mounts tracefs if
needed, enables three tracepoints plus a few kprobes, traces for the duration,
then disables them and prints a summary.

### Verdict

| Observation | Meaning | Next |
|---|---|---|
| `drm_vblank_event_delivered` `> 0`, roughly at frame rate | Flip events fire. The frozen counter is isolated to `CRTC_GET_SEQUENCE`. | Phase 2: correlate `T_commit`/`T_flip`/`T_obs`; prototype an event-driven capture. |
| `> 0` but `seq` is a single frozen value | Events carry a stale sequence, but the timestamps are real. | Same; do not use `seq` for ordering. |
| `0`, but the nvidia kprobes are non-zero | NVIDIA completes flips without the generic event, or without an event at all. | Re-scope: screencopy backend stays the path. |
| all `0` | Motion did not repaint the primary plane, or flips are event-less. | Re-run with stronger motion; check `--motion-hypr`. |

### What "PASS" buys us

If the event fires, the current fb is known to be on screen and complete at
`T_flip`, so a capture triggered by that event is tear-free **without** the
one-refresh penalty - the intent of the dead `--capture-sync vblank` mode,
sourced from an event instead of the broken counter. Phase 2 then measures the
deltas before anything ships. Shipping eBPF in the product is a separate call;
Phase 1-3 can stay diagnostics.
