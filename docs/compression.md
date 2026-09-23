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
- `lossy` — **reserved selector**, not implemented yet. It parses and has a
  stable id so a lossy codec can be added without a protocol change; the
  server refuses to start with it.

The id space is owned by the `rdma-display-codec` crate:

| id | codec    | implemented |
|----|----------|-------------|
| 0  | raw      | yes         |
| 1  | lossless | yes         |
| 2  | lossy    | reserved    |

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

## Where the cost sits today (and the next step)

The encode currently runs on the **CPU** in the server loop, after the GPU
readback has landed the raw frame in the pinned slot (`FrameCodec` in
`crates/server/main.rs`). That already removes the wire bytes, but the
readback still moves a full frame.

The next step is to move the encode into the capture materializer as a
**GLES 3.1 compute pass**: read the linear RGBA8 target (the same texture the
readback already blits to), tile-diff against a reference texture, and write
only the coded bytes into the pinned staging PBO. The byte format is
unchanged, so the client decoder is a drop-in. This is the point of keeping
the format in `rdma-display-codec` with a CPU reference — the GPU encoder is
verified against the same tests.

A lossy codec (id 2) slots in as a new encoder/decoder pair behind the same
selector and the same `Meta`/`Codec` negotiation.
