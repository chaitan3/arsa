#!/usr/bin/env bash
# Phase 1 - do page-flip-completion events actually fire on this nvidia-drm stack?
#
# The display's vblank *counter* is frozen (CRTC_GET_SEQUENCE -> seq 1, ns 0), so
# DRM_IOCTL_WAIT_VBLANK can never pace a capture. But flip completion is a
# separate path:
#
#     nv_drm_handle_flip_event -> nv_drm_crtc_send_vblank_event
#         -> drm_crtc_send_vblank_event -> tracepoint drm_vblank_event_delivered
#
# If that tracepoint fires once per presented frame, then the signal the broken
# `vblank` capture mode needed *is* available (as an event, not a counter), and
# an event-driven capture can drop the one-behind read (~one refresh of lag).
#
# This is pure tracefs: no bpftrace, no packages. Run as root on the compositor
# host for ~12 s and create visible motion during the run.
#
# usage:
#   sudo ./phase1-flip-event.sh [--duration SECONDS] [--motion-hypr|--motion-syncprobe]
#
# --motion-hypr     cycle Hyprland workspace transitions (full-screen animation)
# --motion-syncprobe run target/release/syncprobe to wiggle a synthetic mouse
# (default: you drag a window / play a video yourself)

set -u

DUR=12
MOTION=none
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--duration) DUR="${2:?--duration needs SECONDS}"; shift 2 ;;
    --motion-hypr) MOTION=hypr; shift ;;
    --motion-syncprobe) MOTION=syncprobe; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo $0 ...)" >&2; exit 1; }

# --- locate tracefs ------------------------------------------------------
TR=""
for c in /sys/kernel/tracing /sys/kernel/debug/tracing; do
  if mountpoint -q "$c" 2>/dev/null; then TR="$c"; break; fi
done
if [ -z "$TR" ]; then
  mkdir -p /sys/kernel/tracing
  if mount -t tracefs tracefs /sys/kernel/tracing 2>/dev/null; then TR=/sys/kernel/tracing; fi
fi
if [ -z "$TR" ] || [ ! -d "$TR/events" ]; then
  echo "tracefs not available; try: mount -t tracefs tracefs /sys/kernel/tracing" >&2
  exit 1
fi
cd "$TR"

# --- reset ---------------------------------------------------------------
echo 0 > tracing_on
: > trace 2>/dev/null || true
: > kprobe_events 2>/dev/null || true
echo 0 > events/enable 2>/dev/null || true

# --- tracepoints (the core probe) ---------------------------------------
ENABLED=""
for d in drm/drm_vblank_event drm/drm_vblank_event_queued drm/drm_vblank_event_delivered; do
  if [ -d "events/$d" ]; then echo 1 > "events/$d/enable"; ENABLED="$ENABLED $d"; fi
done
if [ -z "$ENABLED" ]; then echo "no drm vblank tracepoints found" >&2; exit 1; fi
echo "tracepoints enabled:$ENABLED"

# --- generic DRM kprobes (best effort) -----------------------------------
# NVIDIA marks its flip entry points `notrace`, so nv_drm_* cannot be probed;
# the generic DRM functions are what flip completion funnels through when an
# event is delivered.
add_kp() {
  if echo "p:$1 $2" >> kprobe_events 2>/dev/null; then
    echo "kprobe: $1 <- $2"
  else
    echo "kprobe: $1 <- $2  (unavailable, skipped)"
  fi
}
add_kp commit_atomic drm_mode_atomic_ioctl
add_kp commit_setfb  drm_atomic_set_fb_for_plane
add_kp flip_arm      drm_crtc_arm_vblank_event
add_kp flip_drm_send drm_crtc_send_vblank_event
add_kp evt_send      drm_send_event_locked
[ -d events/kprobes ] && echo 1 > events/kprobes/enable 2>/dev/null || true

# --- optional auto motion ------------------------------------------------
MOTION_PID=""
case "$MOTION" in
  hypr)
    if command -v hyprctl >/dev/null 2>&1; then
      ( i=0; while [ "$i" -lt $((DUR*2)) ]; do
          hyprctl dispatch workspace e+1 >/dev/null 2>&1; sleep 0.5; i=$((i+1))
        done ) &
      MOTION_PID=$!
    else
      echo "hyprctl not found - move a window manually" >&2
    fi ;;
  syncprobe)
    SP="$(command -v syncprobe 2>/dev/null || true)"
    [ -x "$SP" ] || SP=./target/release/syncprobe
    if [ -x "$SP" ]; then "$SP" --idle 1 --seconds "$DUR" >/dev/null 2>&1 & MOTION_PID=$!
    else echo "syncprobe binary not found - move a window manually" >&2; fi ;;
esac

# --- run -----------------------------------------------------------------
echo
echo "=== phase 1: tracing ${DUR}s ==="
if [ "$MOTION" = none ]; then
  echo ">>> GENERATE VISIBLE MOTION NOW: drag a window around / play a video <<<"
fi
echo 1 > tracing_on
sleep "$DUR"
echo 0 > tracing_on
[ -n "$MOTION_PID" ] && kill "$MOTION_PID" 2>/dev/null || true

# --- summary -------------------------------------------------------------
rate() { awk -v n="$1" -v d="$DUR" 'BEGIN { printf "%.1f", (d>0? n/d : 0) }'; }
echo
echo "=== results (${DUR}s) ==="
echo "trace lines: $(wc -l < trace 2>/dev/null | tr -d ' ')"
for ev in drm_vblank_event drm_vblank_event_queued drm_vblank_event_delivered; do
  n=$(grep -c " $ev:" trace 2>/dev/null || true); n=${n:-0}
  printf '%-34s %6s  (%s/s)\n' "$ev" "$n" "$(rate "$n")"
done
for k in commit_atomic commit_setfb flip_arm flip_drm_send evt_send; do
  n=$(grep -c "$k:" trace 2>/dev/null || true)
  printf '%-34s %6s\n' "kprobe:$k" "${n:-0}"
done

echo
echo "per-crtc drm_vblank_event_delivered:"
grep "drm_vblank_event_delivered:" trace 2>/dev/null \
  | sed -n 's/.*crtc=\([0-9-]*\).*/\1/p' | sort -n | uniq -c || echo "  (none)"

echo
echo "delivered seq values (one unique value => the counter is frozen, event still fires):"
grep "drm_vblank_event_delivered:" trace 2>/dev/null \
  | sed -n 's/.*seq=\([0-9]*\).*/\1/p' | sort -n | uniq -c | head -10 || echo "  (none)"

echo
echo "sample lines:"
grep -E "drm_vblank_event|flip_" trace 2>/dev/null | head -12 || true

# --- tidy (don't leave events on system-wide) ----------------------------
echo 0 > events/enable 2>/dev/null || true
: > kprobe_events 2>/dev/null || true

echo
echo "=== verdict ==="
D=$(grep -c "drm_vblank_event_delivered:" trace 2>/dev/null || true); D=${D:-0}
if [ "$D" -gt 0 ]; then
  echo "PASS: flip-completion events fire ($D in ${DUR}s)."
  echo "      the frozen counter is isolated to CRTC_GET_SEQUENCE;"
  echo "      an event-driven capture can key off the flip instead of one-behind."
else
  echo "FAIL: no drm_vblank_event_delivered during motion."
  echo "      first check commit_setfb/commit_atomic above: if they are 0, the"
  echo "      motion never repainted the primary plane (a hardware cursor alone"
  echo "      may not). If they are non-zero but evt_send is 0, flip completion"
  echo "      is not delivering a page-flip event at all; see docs/capture-sync.md."
fi
