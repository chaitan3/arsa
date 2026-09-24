# Frame compression (codec selection)

Goal: cut the bytes that cross PCIe (readback + client upload) and the wire,
without giving up the option of a bit-exact path. Compression is **opt-in**;
the default is `raw`, byte-for-byte the historic behaviour.

## Selecting a codec

Server flag (the client negotiates automatically and sizes its slots to match):

```
--codec raw|lossless|lossy      (default raw)
```

- `raw` — no coding, no header. The RDMA slot holds the raw frame
  (`stride × height`), exactly as before.
- `lossless` — tile-delta against the previously sent frame (see below).
- `lossy` — PyroWave intra-only GPU codec (see below); needs
  `libpyrowave-shared.so` at runtime.

The id space is owned by the `rdma-display-codec` crate:

| id | codec    | implemented |
|----|----------|-------------|
| 0  | raw      | yes         |
| 1  | lossless | yes         |
| 2  | lossy    | yes         |

## Lossy (PyroWave, id 2)

Intra-only, GPU codec. The server imports the KMS scanout dma-buf into a
Vulkan image and runs PyroWave's `encode_gpu_scaled_synchronous` (RGB → YCbCr
+ scale) on it, then packetizes the bitstream into the staging slot — no GL
and no full-frame readback. The client pushes the packets into a PyroWave
decoder, decodes on the GPU into Y/U/V plane images, and converts to BGRA.

Flags on the server:

```
--lossy-budget BYTES   max bitstream bytes per frame (default 1000000)
--lossy-chroma 420|444 chroma subsampling (default 420)
```

The slot is `budget + 1 MiB` (packet framing overhead). The negotiated
`Codec` message carries the slot size and the chroma, so the client sizes its
planes to match. Because it is intra-only there is no reference-chain
invariant and no CRC: every frame is independently decodable.

### Building `libpyrowave-shared.so`

The `.so` is `dlopen`ed from beside the binaries (see
`.cargo/config.toml`'s `$ORIGIN` rpath), so it is dropped into
`target/release`. Build it against the lab's glibc with
`scripts/build-pyrowave-docker.sh` (Ubuntu 24.04 / glibc 2.39). Building on a
newer host bakes in symbol versions the lab lacks (e.g. `GLIBC_2.43` from
`sqrtf`/`acosf`) and the `dlopen` fails.

## Lossless format (v1)

Full specification and the CPU reference implementation live in
`crates/codec/src/lib.rs`. Summary:

- The frame is cut into `tile × tile` tiles (`tile = 1 << tile_log2`, default
  64 px).
- Every frame is **full** (all tiles) or **delta** against the previously
  decoded frame (only changed tiles).
- Because the codec is lossless, the server's reference and the client's
  reference are the same pixels, so they cannot drift.
- The payload carries a CRC32 so a desync or a transport fault is caught by
  the decoder instead of silently showing wrong pixels.

```text
offset  size            field
  0     1               magic 0xC0
  1     1               codec (1 = lossless)
  2     1               version
  3     1               flags: bit0 = delta / full
  4     1               tile_log2
  5     3               reserved
  8     4  LE           width
 12     4  LE           height
 16     4  LE           stride
 20     4  LE           changed tile count
 24     4  LE           index capacity (tiles_x * tiles_y)
 28     4  LE           crc32 of the payload
 32     cap*4 LE        changed-tile ids, in payload order
 32+cap*4  count*tile   tile payloads (fixed tile²·4 bytes each)
```

The index array is fixed capacity so the future GPU encoder can write payloads
at static offsets; the decoder reads the first `count` entries. Row padding
(`stride − width·4`) is never coded and never displayed.

## Slots and the wire

The staging/receive slot must hold the codec's worst case:

- raw: `stride × height`
- lossless: `32 + tiles·4 + tiles·tile²·4`
- lossy: `budget + 1 MiB` (the PyroWave bit budget plus packet framing)

The server sends the codec id right after `Meta`, before the client allocates
its slots, so both sides agree on the slot size. Each frame posts exactly its
coded length (not the whole slot), so a small delta costs a small RDMA WRITE.

## Reference-chain invariant

A lossless delta references the previous frame, so **the client must decode
every received frame in order**, even ones it will not present (the presenter
throws away all but the newest). The client does this in the present and
headless loops; skipping a frame would desync the chain and trip the CRC.

The server resets the reference on every new session, so the first frame a
client sees is always full.

## Where the cost sits

`lossless` encodes on the **GPU** (`rdma-display-codec-gpu`, GLES 3.1 compute).
The scanout dma-buf is imported into the codec's own GLES context
(`eglCreateImageKHR`) and the encode compute reads it directly — the frame never
leaves the GPU and only the coded delta is copied into the pinned staging slot.

### The import path dominates

| stage (4K, RTX 2080 Ti)                        | measured    |
|------------------------------------------------|-------------|
| encode delta (SSBO / image / external / fetch) | 0.17–0.33 ms |
| `eglCreateImageKHR` + target bind (uncached)   | ~3 ms/frame |

The import, not the encode, was the cost — so it is **cached per buffer**: a
`SrcImport` keyed by `(fd, offset, stride, fourcc, modifier, w, h)` is reused and
only evicted (with its EGLImage/texture destroyed) when the cache exceeds 4
entries. A compositor swaps between a handful of buffers, so steady-state frames
hit the cache and pay only the encode.

### Source binding (linear vs tiled)

- **Linear scanout** (`modifier == 0`): the EGLImage binds to a normal
  `GL_TEXTURE_2D`; the encode reads it with `texelFetch` (`ENC_FETCH_SRC`).
- **Tiled scanout**: it binds to `GL_TEXTURE_EXTERNAL_OES`; the encode samples it
  in compute (`ENC_EXT_SRC`). The external sampler is *not* the bottleneck — both
  are ~0.2 ms once the import is cached.
- If external-in-compute is unsupported by the driver, the fallback is a GPU→GPU
  blit to an RGBA8 image plus the image-input encode (`ENC_TEX_SRC`), ~0.33 ms.

The blit and image swizzle `.bgra`, so the source image's `R,G,B,A` equals the
wire bytes `B,G,R,A` — exactly what `glReadPixels(GL_BGRA)` of the materializer's
linear target yields — so the coded byte format is identical on every path.

### Fallbacks

- **Readback fallback** (no dmabuf, import unsupported, or no GPU encoder): the
  materializer reads the scanout back into the pinned staging slot, the encoder
  uploads it and tile-diffs as before. The CPU reference encoder is the last
  fallback if the GPU codec can't be created at all.

The per-frame *compute* is `cpu 6.6 / gpu 0.2 ms` for a delta and `cpu 9.3 /
gpu 8.1 ms` for a full frame. On high-motion content that compresses poorly
`raw` is still the cheaper choice.

All variants are benchmarked with `codecbench` (`[texture]`, `[external]`,
`[fetch]`, `[import]`) and required to be byte-identical to the SSBO encode and
CPU-decodable.

### Client display path (GPU lossless decode)

With `--codec lossless --output egl`, the client decodes deltas into a
persistent `GL_RGBA8` texture and the presenter draws *that texture* — the CPU
never touches full-frame pixels, and there is no per-frame `glTexImage2D` of the
frame. Headless (`--output none`) keeps the CPU decoder + full-frame upload.

**Setup (once, at session start).** `GpuVideoDecoder::attach_current(geom)`
attaches to `eglwin`'s *current* GLES 3.1 context (GL entry points come from
`eglGetProcAddress`), builds the decode compute program (`DEC_TEX_SRC`), and
allocates a persistently-mapped coded-input SSBO on binding 4
(`codec-gpu/src/lib.rs:1022`). `create_texture(w,h)` makes the immutable
`GL_RGBA8` target with `glTexStorage2D` (immutable storage is required for
`imageStore` in ES 3.1); its contents are defined by the first decode — the
server resets its reference on a new session/geometry, so the first frame is
always full and paints every tile (`codec-gpu/src/lib.rs:1040`). If attach or
texture creation fails, the client logs and falls back to the CPU decoder.

**Per frame** (`client/src/main.rs:273`):

1. `drain`: poll `try_next_frame()` until empty; stamp `t3` and `rearm_slot`.
2. **Decode in order.** Lossless deltas reference the previous frame, so every
   received frame must be decoded even if it will not be shown. For each:
   `Header::parse` gives the tile count and coded length
   `n = HEADER_LEN + tile_count*4 + count*tile_bytes`; `dec.upload(&coded[..n])`
   memcpys only the *coded* bytes into the decoder SSBO;
   `dec.decode_to_image(tex, count)` binds `tex` as a write-only image and
   dispatches one workgroup per changed tile, each `imageStore`-ing its payload
   at `(x0+lx, y0+ly)`. Unchanged tiles are not touched, so the texture *is* the
   reference (`codec-gpu/src/lib.rs:1115`). On a `Meta` geometry change it
   decodes that frame with the CPU decoder instead and stops using the stale
   texture.
3. **Present only the newest.** Superseded frames were decoded (to keep the
   chain) but are credited without presenting (`newest.replace` +
   `credit_slot`), so the display shows the latest content rather than replaying
   a backlog.
4. **Draw + swap.** `present_texture(tex, …)` (`eglwin.rs:610`) runs
   `draw_with`: a full-screen letterboxed triangle sampling `tex`, then the
   metrics overlay quad, then `eglSwapBuffers` (`swap_interval = 1`, i.e. FIFO
   pacing). No upload is issued. It returns `(t4, t5)`.
5. **Credit** the presented slot and `metrics.record(seq, t3, t4, t5, fb)`.

**Metrics mapping.** `upload = t4 − t3` (coded memcpy + decode dispatch +
the decode's `glFinish`), `present = t5 − t4` (draw + swap submission),
`display = tp − t5` (compositor queue + scanout, from `wp_presentation`). The
overlay is rasterized at `REFRESH_MS`, uploaded into its own texture, and drawn
every frame so it never flickers.

Compared with the CPU path (`link.frame` + `present`), the only per-frame host
traffic is the coded bytes; the frame's pixels stay in the texture between
frames. Note `decode_to_image` currently ends in `glFinish` — a per-frame CPU
sync that could become a fence, since the presenter's draw follows on the same
command stream.

**Presentation latency.** `display` (= compositor on-screen time − swap return)
dominates the client span and is the compositor's queue-to-scanout, not our
work. To let KWin bypass composition, the client marks the surface fully opaque
(`wl_surface.set_opaque_region`, updated on every size change) and uses an
alpha-free (XRGB) EGL window config, and runs `--fullscreen`. That makes the
buffer eligible for **direct scanout** to the primary plane; otherwise KWin
composites and adds a buffered stage. VRR and a refresh rate matching the
source further bound `display`.
