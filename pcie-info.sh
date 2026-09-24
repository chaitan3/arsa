#!/usr/bin/env bash
# pcie-info.sh - report the GPU's PCIe link (generation + width) so we can tell
# whether the measured ~2.5 ms real-scanout read is bus-limited.
#
# Run on the server. nvidia-smi needs no privileges; `lspci -vv` may need sudo
# for the LnkCap/LnkSta lines. Paste the whole output.

set -u

echo "== nvidia-smi: PCIe link (per GPU) =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max \
        --format=csv
else
    echo "nvidia-smi not found"
fi

echo
echo "== lspci: NVIDIA GPU link capability/status =="
if command -v lspci >/dev/null 2>&1; then
    # all NVIDIA PCI functions (vendor 10de), one BDF per line
    bdfs=$(lspci -Dn 2>/dev/null | grep -i '10de:' | cut -d' ' -f1)
    if [ -z "$bdfs" ]; then
        bdfs=$(lspci -D 2>/dev/null \
            | grep -iE 'vga|3d controller|display controller' \
            | grep -i nvidia | cut -d' ' -f1)
    fi
    if [ -z "$bdfs" ]; then
        echo "no NVIDIA PCI device found by lspci"
    else
        for b in $bdfs; do
            echo "-- $b"
            lspci -vv -s "$b" 2>/dev/null \
                | grep -E 'Link Speed|LnkCap|LnkSta' \
                || echo "   (run with sudo for LnkCap/LnkSta)"
            echo
        done
    fi
else
    echo "lspci not found"
fi

echo
echo "== interpretation =="
cat <<'EOF'
PCIe one-direction real throughput:
  Gen3 x16 ~ 13 GB/s      Gen3 x8 ~ 6.5 GB/s
A 4K RGBA frame is 31.6 MB, so a full-frame read over PCIe is:
  ~2.4 ms at x16          ~4.9 ms at x8

Measured: real compositor scanout read = 2.53 ms (~13 GB/s); a GBM tiled
VRAM buffer = 0.14 ms (~235 GB/s).

  * Link x16 -> 2.5 ms ~= PCIe => compositor's scanout is in system memory,
    every capture pays a full-frame PCIe read (bus floor).
  * Link x8  -> 2.5 ms is FASTER than the bus could carry => the read is not
    bus-limited; the compositor's buffer is in a fast VRAM path but read
    slowly (compressed / scanned-out state), which is a different root cause.
EOF
