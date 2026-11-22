#!/usr/bin/env bash
# Build the server (and client) in release and give the server binary
# CAP_SYS_ADMIN so it can capture the host's KMS scanout (GETFB2) without
# running the whole process as root.
#
# Run this as your NORMAL user, not root: the build must not leave root-owned
# artifacts in target/, and setcap is invoked through sudo internally.
#
# Note: file capabilities live on the file inode, so cargo/rustc rebuilding the
# binary (new inode) drops them. The build is therefore skipped when the
# binaries are already newer than the sources, so you can `cargo build` in one
# place (e.g. a container) and run this script on another (the host) just to
# re-apply the capability.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# CAP_SYS_ADMIN is the one that matters for GETFB2 ("master or CAP_SYS_ADMIN").
# It does NOT cover the other privileged bits:
#   --input uinput        -> /dev/uinput access (input group, or root)
#   --capture-sync fence  -> mounts tracefs (CAP_SYS_ADMIN) and writes it (CAP_DAC_OVERRIDE, or mount tracefs with uid=<you>)
#   --gpu-power           -> nvidia-smi clock floor (root)
# Set CAPS to widen the set if you need those, e.g.
#   CAPS=cap_sys_admin,cap_dac_override,cap_dac_read_search ./install.sh
CAPS="${CAPS:-cap_sys_admin,cap_dac_override,cap_ipc_lock}"

SERVER=target/release/arsa-server
CLIENT=target/release/arsa-client
LIBS=target/release/libpyrowave-shared.so
DEST=${1:-/usr/local/bin}
STAMP=target/release/.arsa-build-features
# SKIP_BUILD=1 forces the cargo build to be skipped (e.g. the binaries were
# built elsewhere, like a container, and this host only needs setcap/copy).
#
# The server's only default feature is the XDG portal source, which needs the
# PipeWire development files. Build with it when available; otherwise install a
# KMS/QEMU-only server. NO_PORTAL=1 forces the lean build.
FEATURE_KEY=portal
FEATURE_ARGS=()
if [ "${NO_PORTAL:-0}" = "1" ]; then
    FEATURE_KEY=no-portal
    FEATURE_ARGS+=(--no-default-features)
elif ! command -v pkg-config >/dev/null 2>&1 \
    || ! pkg-config --exists libpipewire-0.3 2>/dev/null; then
    FEATURE_KEY=no-portal
    FEATURE_ARGS+=(--no-default-features)
    echo "warning: libpipewire-0.3 development files not found;" >&2
    echo "         building the server without --source portal" >&2
fi

# setcap live in /usr/sbin (Debian) or /sbin, which are not on a normal
# user's PATH; resolve them explicitly rather than trusting PATH.
find_tool() {
    local name=$1 path
    if path=$(command -v "$name" 2>/dev/null); then
        printf '%s\n' "$path"
        return 0
    fi
    for path in /usr/sbin/$name /sbin/$name /usr/bin/$name /bin/$name; do
        if [ -x "$path" ]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    return 1
}

SETCAP=$(find_tool setcap) || {
    echo "error: setcap not found (install libcap2-bin / libcap)" >&2
    exit 1
}

# Newest mtime among the given paths.
newest_mtime() {
    find "$@" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1
}
# The server and client binaries do not include src/bin diagnostics; keep a
# diagnostic source change from making them look perpetually stale.
newest_mtime_no_bins() {
    find "$@" -type f -not -path '*/src/bin/*' -printf '%T@\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d. -f1
}

# Cargo relinks a binary only when a crate it depends on changes, so each binary
# is compared against *its own* sources: a change under crates/capture (say)
# relinks the server but not the client, and must not force a client rebuild.
# Keep these in sync with crates/{server,client}/Cargo.toml.
ROOT_FILES=(Cargo.toml Cargo.lock .cargo/config.toml)
SERVER_CRATES=(crates/proto crates/capture crates/rdma crates/audio \
    crates/auth crates/codec crates/vkcore crates/vkcodec crates/lifecycle \
    crates/server)
CLIENT_CRATES=(crates/proto crates/rdma crates/audio crates/auth \
    crates/codec crates/vkcore crates/lifecycle crates/client)

build_needed=1
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "==> SKIP_BUILD=1; skipping cargo build"
    build_needed=0
elif [ -x "$SERVER" ] && [ -x "$CLIENT" ]; then
    srv_src=$(newest_mtime_no_bins "${SERVER_CRATES[@]}" "${ROOT_FILES[@]}")
    cli_src=$(newest_mtime_no_bins "${CLIENT_CRATES[@]}" "${ROOT_FILES[@]}")
    srv=$(stat -c %Y "$SERVER" 2>/dev/null || echo 0)
    cli=$(stat -c %Y "$CLIENT" 2>/dev/null || echo 0)
    stamp_ok=0
    if [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$FEATURE_KEY" ]; then
        stamp_ok=1
    fi
    if [ -n "$srv_src" ] && [ -n "$cli_src" ] && [ "$stamp_ok" = "1" ] \
        && [ "$srv" -ge "$srv_src" ] && [ "$cli" -ge "$cli_src" ]; then
        echo "==> binaries are newer than the sources; skipping cargo build"
        build_needed=0
    fi
fi

if [ "$build_needed" = "1" ]; then
    echo "==> cargo build --release ${FEATURE_ARGS[*]} (server + client)"
    cargo build --release -p rdma-display-server "${FEATURE_ARGS[@]}"
    cargo build --release -p rdma-display-client
    # Cargo may decide a Cargo.toml-only change needs no relink; refresh the
    # binary mtimes so the next run's source check does not build forever.
    touch "$SERVER" "$CLIENT"
    mkdir -p target/release
    printf '%s\n' "$FEATURE_KEY" > "$STAMP"
fi

[ -x "$SERVER" ] || { echo "error: $SERVER not built" >&2; exit 1; }
[ -x "$CLIENT" ] || { echo "error: $CLIENT not built" >&2; exit 1; }

# File capabilities are ignored on a filesystem mounted nosuid.
if command -v findmnt >/dev/null 2>&1 \
    && findmnt -no OPTIONS --target "$SERVER" 2>/dev/null | grep -qw nosuid; then
    echo "warning: $(pwd) is mounted nosuid; file capabilities will be ignored" >&2
fi

echo "Copying release to $DEST"
if [ -d "$DEST" ] && [ -w "$DEST" ]; then
    COPY=(cp)
elif [ ! -e "$DEST" ] && [ -w "$(dirname "$DEST")" ]; then
    mkdir -p "$DEST"
    COPY=(cp)
else
    echo "==> $DEST is not writable; using sudo"
    sudo install -d "$DEST"
    COPY=(sudo cp)
fi
"${COPY[@]}" "$SERVER" "$DEST/"
"${COPY[@]}" "$CLIENT" "$DEST/"
if [ -f "$LIBS" ]; then
    # Follow the target/release symlink; install both the plain name and the
    # library SONAME so dlopen finds it before any system copy.
    "${COPY[@]}" -L "$LIBS" "$DEST/libpyrowave-shared.so"
    "${COPY[@]}" -L "$LIBS" "$DEST/libpyrowave-shared.so.0"
else
    echo "warning: $LIBS not found — the lossy codec will be unavailable" >&2
    echo "         (build it with scripts/build-pyrowave-docker.sh)" >&2
fi
if [ "$FEATURE_KEY" = "no-portal" ]; then
    echo "note: installed server has no --source portal (PipeWire dev files missing)" >&2
fi

echo "==> setcap $CAPS+ep $DEST/arsa-server"
sudo "$SETCAP" "$CAPS+ep" "$DEST"/arsa-server
