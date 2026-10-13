#!/usr/bin/env bash
# Build the server (and client) in release and give the server binary
# CAP_SYS_ADMIN so it can capture the host's KMS scanout (GETFB2) without
# running the whole process as root.
#
# Run this as your NORMAL user, not root: the build must not leave root-owned
# artifacts in target/, and setcap is invoked through sudo internally.
#
# Note: file capabilities live on the file inode, so cargo/rustc rebuilding the
# binary (new inode) drops them. Re-run this script after every rebuild.
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

for tool in setcap getcap; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found (install libcap2-bin / libcap)" >&2
        exit 1
    fi
done

echo "==> cargo build --release (server + client)"
cargo build --release -p rdma-display-server -p rdma-display-client

[ -x "$SERVER" ] || { echo "error: $SERVER not built" >&2; exit 1; }

# File capabilities are ignored on a filesystem mounted nosuid.
if command -v findmnt >/dev/null 2>&1 \
    && findmnt -no OPTIONS --target "$SERVER" 2>/dev/null | grep -qw nosuid; then
    echo "warning: $(pwd) is mounted nosuid; file capabilities will be ignored" >&2
fi

echo "==> setcap $CAPS+ep $SERVER"
sudo setcap "$CAPS+ep" "$SERVER"

echo "==> installed:"
getcap "$SERVER" "$CLIENT"
echo
echo "run it directly (no sudo): ./$SERVER"
