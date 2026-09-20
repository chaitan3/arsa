#!/bin/bash
# C0 client-host viability checks for zero-copy slots (NIC -> GPU dmabuf).
# Run ON THE CLIENT HOST (192.168.4.3). Output is small; paste it all back.
set -u
hr() { printf '\n== %s ==\n' "$1"; }

hr "nvidia kernel module flavor (GPUDirect-over-dmabuf needs 'open')"
modinfo nvidia 2>/dev/null | grep -E '^(license|version)'
cat /proc/driver/nvidia/version 2>/dev/null | head -2

hr "session / compositor"
echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-?}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-?}"
kwin_wayland --version 2>/dev/null || qdbus6 org.kde.KWin /KWin org.kde.KWin.version 2>/dev/null || true

hr "GBM stack (how do we allocate scanout buffers?)"
ldconfig -p | grep -E 'libgbm|eglgbm|gbm_nvidia' || echo "no libgbm/egl-gbm found?!"
ls -l /dev/dri/

hr "GPU PCI + BAR size (peer writes land in BAR1; ReBAR strongly preferred)"
GPU_PCI=$(lspci -D | grep -i -E 'vga|3d|display' | awk '{print $1}' | head -1)
echo "GPU: $GPU_PCI"
sudo lspci -vvv -s "$GPU_PCI" 2>/dev/null | grep -E 'Region 1|Memory at|Kernel driver|LnkCap:|LnkSta:' | head -10

hr "NIC PCI + topology (P2P wants NIC and GPU under the same root complex)"
IBDEV=$(ls /sys/class/infiniband 2>/dev/null | head -1)
sudo lspci -Dnn | grep -i -E 'mellanox|connectx'
NIC_PCI=$(basename "$(readlink -f /sys/class/infiniband/${IBDEV:-none}/device 2>/dev/null)" 2>/dev/null)
echo "IB dev: ${IBDEV:-none}  NIC PCI: $NIC_PCI"
# shared sysfs realpath prefix = shared upstream bridge (deeper = same switch);
# I'll read the topology from the pasted output
echo "GPU path: $(readlink -f /sys/bus/pci/devices/${GPU_PCI} 2>/dev/null)"
echo "NIC path: $(readlink -f /sys/bus/pci/devices/${NIC_PCI} 2>/dev/null)"
echo "--- tree ---"
sudo lspci -t 2>/dev/null | grep -B3 -E 'VGA|3D|Mellanox|ConnectX' | head -20

hr "IOMMU state (works either way; matters for peer mapping perf)"
cat /proc/cmdline
sudo dmesg 2>/dev/null | grep -i -E 'DMAR|IOMMU|AMD-Vi' | head -6 || true

hr "memlock + RDMA device"
ulimit -l
ibv_devinfo 2>/dev/null | grep -E 'hca_id|link_layer|state|active_mtu' | head -9

hr "peermem / prior GDR registrations in dmesg"
sudo dmesg | grep -i -E 'peer|p2p|nvidia.?fs|gdr' | tail -10

hr "kernel version"
uname -a
