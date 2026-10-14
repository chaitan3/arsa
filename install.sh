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
# binaries were already built from the current git state (see the stamp below),
# so re-running this script just re-applies the capability.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# CAP_SYS_ADMIN is the one that matters for GETFB2 ("master or CAP_SYS_ADMIN").
# It does NOT cover the other privileged bits:
#   --input uinput        -> /dev/uinput access (input group, or root)
#   --capture-sync fence  -> tracefs writes (root or CAP_DAC_OVERRIDE)
#   --gpu-power           -> nvidia-smi clock floor (root)
# Set CAPS to widen the set if you need those, e.g.
#   CAPS=cap_sys_admin,cap_dac_override,cap_dac_read_search ./install.sh
CAPS="${CAPS:-cap_sys_admin}"

SERVER=target/release/rdma-display-server
CLIENT=target/release/rdma-display-client
LIBS=target/release/libpyrowave-shared.so
DEST=${1:-/usr/local/bin}
# Records the source state the binaries were built from. A repeat run at the
# same commit (and same tracked changes) skips the rebuild.
STAMP=target/release/.rdma-display-build-stamp

# setcap/getcap live in /usr/sbin (Debian) or /sbin, which are not on a normal
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
GETCAP=$(find_tool getcap) || {
    echo "error: getcap not found (install libcap2-bin / libcap)" >&2
    exit 1
}

# Source marker: HEAD plus a digest of tracked working-tree changes. Untracked
# files (logs, pyrowave/, ...) are ignored. Non-zero when not a git checkout,
# so the build always runs there.
source_stamp() {
    local head dirty
    head=$(git rev-parse HEAD 2>/dev/null) || return 1
    dirty=$(git status --porcelain --untracked-files=no 2>/dev/null | cksum)
    printf '%s %s' "$head" "$dirty"
}

if want=$(source_stamp) && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ] \
    && [ -x "$SERVER" ] && [ -x "$CLIENT" ]; then
    echo "==> up to date at $(git rev-parse --short HEAD); skipping cargo build"
else
    echo "==> cargo build --release (server + client)"
    cargo build --release -p rdma-display-server -p rdma-display-client
    # record the source state only after a successful build
    if want=$(source_stamp); then
        printf '%s\n' "$want" >"$STAMP"
    fi
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
sudo "$SETCAP" "$CAPS+ep" "$SERVER"

echo "==> installed:"
"$GETCAP" "$SERVER" "$CLIENT"
echo
echo "run it directly (no sudo): ./$SERVER"
