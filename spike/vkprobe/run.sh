#!/usr/bin/env bash
# Build and run the Vulkan/RDMA spike for the PyroWave integration.
#
# IMPORTANT: VK_EXT_external_memory_dma_buf is gated on being able to open the
# DRM render node. In a restricted container it is NOT advertised, which makes
# the dma-buf test inconclusive. Run this on the lab host (server side under
# sudo, or as a user in the `render` group) for a valid answer.
set -e
cd "$(dirname "$0")"

# Find Vulkan headers: prefer the system location, else a local checkout.
VKHDR="${VKHDR:-}"
if [ -z "$VKHDR" ]; then
    if [ -f /usr/include/vulkan/vulkan.h ]; then
        VKHDR=/usr/include
    else
        for d in "$HOME"/kytyps5/3rdparty/Vulkan-Headers/include \
                 "$HOME"/Vulkan-Headers/include; do
            [ -f "$d/vulkan/vulkan.h" ] && VKHDR="$d" && break
        done
    fi
fi
[ -n "$VKHDR" ] || { echo "set VKHDR to a dir containing vulkan/vulkan.h"; exit 1; }

echo "[build] VKHDR=$VKHDR"
gcc -O2 -Wall -o vkprobe vkprobe.c -I"$VKHDR" -lvulkan -lgbm -libverbs
exec ./vkprobe "$@"
