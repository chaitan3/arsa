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
CAPS="${CAPS:-cap_sys_admin}"

SERVER=target/release/rdma-display-server
CLIENT=target/release/rdma-display-client
LIBS=target/release/libpyrowave-shared.so
DEST=${1:-/usr/local/bin}
# SKIP_BUILD=1 forces the cargo build to be skipped (e.g. the binaries were
# built elsewhere, like a container, and this host only needs setcap/copy).

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

# Newest mtime among the files that feed the binaries. If both binaries are at
# least this new, they were built from the current sources (wherever that
# happened) and there is nothing to rebuild.
newest_source_mtime() {
    find crates Cargo.toml Cargo.lock .cargo/config.toml -type f \
        -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1
}

build_needed=1
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "==> SKIP_BUILD=1; skipping cargo build"
    build_needed=0
elif [ -x "$SERVER" ] && [ -x "$CLIENT" ]; then
    src=$(newest_source_mtime)
    srv=$(stat -c %Y "$SERVER" 2>/dev/null || echo 0)
    cli=$(stat -c %Y "$CLIENT" 2>/dev/null || echo 0)
    if [ -n "$src" ] && [ "$srv" -ge "$src" ] && [ "$cli" -ge "$src" ]; then
        echo "==> binaries are newer than the sources; skipping cargo build"
        build_needed=0
    fi
fi

if [ "$build_needed" = "1" ]; then
    echo "==> cargo build --release (server + client)"
    cargo build --release -p rdma-display-server -p rdma-display-client
fi

[ -x "$SERVER" ] || { echo "error: $SERVER not built" >&2; exit 1; }

# File capabilities are ignored on a filesystem mounted nosuid.
if command -v findmnt >/dev/null 2>&1 \
    && findmnt -no OPTIONS --target "$SERVER" 2>/dev/null | grep -qw nosuid; then
    echo "warning: $(pwd) is mounted nosuid; file capabilities will be ignored" >&2
fi

echo "Copying release to $DEST"
cp $SERVER $DEST
cp $CLIENT $DEST
if [ -f "$LIBS" ]; then
    cp $LIBS $DEST
else
    echo "warning: $LIBS not found — the lossy codec will be unavailable" >&2
    echo "         (build it with scripts/build-pyrowave-docker.sh)" >&2
fi

echo "==> setcap $CAPS+ep $SERVER"
sudo "$SETCAP" "$CAPS+ep" "$DEST"/rdma-display-server
