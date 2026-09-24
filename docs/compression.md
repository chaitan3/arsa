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
There are two input paths:

- **Texture-direct (default when the scanout is a dmabuf):** the encoder imports
  the KMS scanout dma-buf into its own GLES context (`eglCreateImageKHR` →
  `GL_TEXTURE_EXTERNAL_OES`), blits it once to an RGBA8 image (GPU→GPU — the
  channel normalize the readback used to do), and the compute shader reads that
  image directly. The frame never leaves the GPU: no full-frame `glReadPixels`
  to host and no re-upload. Only the coded delta is copied into the pinned
  staging slot, so a static frame costs about the changed tiles.
- **Readback fallback** (no dmabuf, import unsupported, or no GPU encoder): the
  materializer reads the scanout back into the pinned staging slot, the encoder
  uploads it and tile-diffs as before. The CPU reference encoder is the last
  fallback if the GPU codec can't be created at all.

The dmabuf blit swizzles `.bgra`, so the source image's `R,G,B,A` equals the
wire bytes `B,G,R,A` — exactly what `glReadPixels(GL_BGRA)` of the materializer's
linear target yields. The coded byte format is therefore identical on both
paths and the client cannot tell them apart.

The per-frame *compute* is roughly `cpu 6.6 ms / gpu 0.2 ms` for a delta and
`cpu 9.3 ms / gpu 8.1 ms` for a full frame, but those numbers exclude the
input: on the readback path a delta also pays ~5.5 ms `glReadPixels` plus the
re-upload, which the texture-direct path removes. On high-motion content that
compresses poorly `raw` is still the cheaper choice (a texture-direct delta then
costs about the same as the coded payload's readback).

On the client, `--output egl` decodes the tiles on the GPU straight into the
texture the presenter draws (`GpuVideoDecoder`, attached to the presenter's
GLES context), so a delta patches only the changed region: no full-frame CPU
decode and no full-frame `glTexImage2D` upload. Headless (`--output none`)
keeps the CPU decoder.
