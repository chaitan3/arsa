# vkprobe — PyroWave integration spike

Standalone probe for the two unknowns before wiring a minimal Vulkan backend
for the lossy (PyroWave) path:

1. **Can Vulkan import a scanout-style dma-buf as a sampleable image?**
   NVIDIA gates `VK_EXT_external_memory_dma_buf` on being able to open the DRM
   render node, so this is only meaningful on a host with DRM access (the lab
   server, or a user in the `render` group).
2. **Can a Vulkan host-visible buffer be used as the RDMA staging slot?**
   (Plus a fallback for (1): `VK_EXT_external_memory_host`.)

It also reports whether the device meets PyroWave's shader requirements
(subgroups, subgroup size control, shaderInt16, storageBuffer8BitAccess).

## Run

```sh
./run.sh                 # builds and runs
VKHDR=/usr/include ./run.sh   # if headers are in the system location
```

Run the server-side copy under sudo (or as a user in `render`/`video`) so
`/dev/dri/renderD128` is openable.

## Reading the output

- `[gate] open /dev/dri/renderD128` — if `denied`, `[import]` is skipped and
  the dma-buf answer is inconclusive regardless of the driver version.
- `[import]` — the decisive dma-buf test: GBM makes a scanout-like tiled
  dma-buf, Vulkan imports it with its DRM modifier and does a GPU round trip.
- `[host-import]` — fallback: import a host pointer as a Vulkan buffer and a
  linear image (how PyroWave would be fed if dma-buf is unavailable).
- `[rdma]` — register a Vulkan host-visible buffer with `ibv_reg_mr`.

## Sandbox findings (RTX 2080 Ti, NVIDIA 610.57.04, no DRM access)

- PyroWave features: **all present** (subgroup ops 0x7ff, size control,
  shaderInt16, storageBuffer8BitAccess, subgroupSize 32).
- `VK_EXT_external_memory_dma_buf`: **not advertised** — the process cannot
  open `/dev/dri/renderD128` (not in `render`/`video`, no sudo). The DRM
  modifier list is still visible (six block-linear `0x…0606xx` + linear, all
  `sampled=1`), but the DMA_BUF external-handle query fails.
- `VK_EXT_external_memory_host`: **works** — 4 KiB alignment; buffer import and
  a linear R8 image both round-trip correctly through the GPU.
- `ibv_reg_mr`: **not testable here** (`/dev/infiniband` absent); host-visible
  Vulkan memory is the same system RAM the existing PBO staging registers, so
  this is low-risk.
