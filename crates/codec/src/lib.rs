//! Lossless frame codecs for arsa.
//!
//! The server turns the materialized frame (a linear BGRA/XRGB scanout) into a
//! self-describing coded frame before it is DMA-read into the RDMA staging
//! slot. The client decodes it back to a full frame. The default is `Raw`
//! (the historic path, zero overhead); `Lossless` is opt-in.
//!
//! ## Why a crate
//!
//! The production encoder is Vulkan compute (`crates/vkcodec`); the format,
//! header, geometry and CPU reference encoder/decoder live here. Keeping them
//! in one crate makes the format the single source of truth and lets the round
//! trip be tested without a GPU. The Vulkan encoder is required to produce
//! exactly this byte layout; the CPU encoder here is the reference
//! implementation and the arbiter of correctness.
//!
//! ## Lossless format (`CODEC_LOSSLESS`, version 2)
//!
//! The frame is cut into `tile × tile` tiles (`tile = 1 << tile_log2`). Each
//! frame is either **full** (every tile carried) or **delta** against the
//! previously decoded frame (only changed tiles carried). Because the codec
//! is lossless, the encoder's reference and the decoder's reference are the
//! same pixels, so they never drift.
//!
//! Changed tiles are no longer raw: each tile picks the cheapest of
//!
//! - `RAW` — the tile's valid pixels, row-major, `vw·vh` words;
//! - `REF` — runs of `cur ^ ref` (rows independent), `(run_len, residual)`;
//! - `DPCM` — runs of the horizontal predictor `cur[x] - cur[x-1]`
//!   (predictor 0 at the row start), rows independent.
//!
//! Runs are maximal within a row. `REF` is never used on a full frame (the
//! decoder has no reference yet). Modes are chosen by coded length (strictly
//! smaller wins, ties prefer `RAW`, then `REF`, then `DPCM`), so the coded
//! total is deterministic even though the GPU packs payloads in an arbitrary
//! (atomic) order.
//!
//! ```text
//! offset  size                field
//!   0     1                   magic = 0xC0
//!   1     1                   codec = 1
//!   2     1                   version = 2
//!   3     1                   flags: bit0 = delta (1) / full (0)
//!   4     1                   tile_log2
//!   5     3                   reserved (0)
//!   8     4  LE               width
//!  12     4  LE               height
//!  16     4  LE               stride
//!  20     4  LE               changed tile count
//!  24     4  LE               index capacity (tiles_x * tiles_y)
//!  28     4  LE               crc32 of the payload (see `finish_crc`)
//!  32     4  LE               total payload words
//!  36    12                   reserved (0)
//!  48     cap*12 LE           changed-tile entries {tile_id, off_words,
//!                             (mode << 16) | len_words}
//!  48+cap*12  payload_words*4  tile payloads, packed in write order
//! ```
//!
//! The index array is **fixed capacity** so a GPU encoder can write entries at
//! static offsets; the payload is variable length and packed densely by an
//! atomic bump allocator, with each entry carrying its own offset. A tile's
//! `REF`/`DPCM` payload starts with `vh` per-row run counts, followed by the
//! `(run_len, residual)` tokens in row order.
//!
//! `crc32` is filled by [`finish_crc`] after the coded bytes are complete. The
//! CPU encoder calls it; the GPU path calls it on the host after the readback
//! fence, before posting the frame (the payload is small, so this is cheap and
//! gives the client a hard integrity check on the coded stream).
//!
//! ## Adding lossy later
//!
//! [`Codec`] is the selector: `Raw`, `Lossless` today; a lossy codec is a new
//! id + a new encoder/decoder. The frame is self-describing via the header
//! `codec` byte, so the client dispatches without a protocol change.

use anyhow::{ensure, Result};

pub const MAGIC: u8 = 0xC0;
pub const VERSION: u8 = 2;
pub const HEADER_LEN: usize = 48;

/// Words per changed-tile index entry (v2): `{tile_id, payload_off, mode|len}`.
pub const INDEX_WORDS: usize = 3;
/// Payload word offset (relative to the payload base) of the first tile.
pub const PAYLOAD_BASE_WORDS: usize = HEADER_LEN / 4;

/// Per-tile payload modes (v2).
pub const MODE_RAW: u32 = 0;
/// `(run_len, residual)` runs of `cur ^ ref`, rows independent.
pub const MODE_REF: u32 = 1;
/// `(run_len, residual)` runs of horizontal DPCM, rows independent.
pub const MODE_DPCM: u32 = 2;

pub const CODEC_RAW: u8 = 0;
pub const CODEC_LOSSLESS: u8 = 1;
/// Reserved for a future lossy codec (hardware/vendor bitstream). Recognized
/// so ids are stable, but not implemented yet.
pub const CODEC_LOSSY: u8 = 2;

/// Bytes per pixel the materializer always produces (BGRA / XRGB, 4 bpp).
pub const BPP: u32 = 4;

/// The codec selected for a session. `Raw` is the default: no header, no
/// coding, byte-for-byte the historic path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Raw,
    Lossless,
    /// Reserved selector for a future lossy codec. Parses and round-trips as
    /// an id so a lossy codec can be added without a protocol change; nothing
    /// implements it yet (`implemented()` is false).
    Lossy,
}

impl Codec {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "raw" | "off" | "none" => Some(Codec::Raw),
            "lossless" | "ll" => Some(Codec::Lossless),
            "lossy" => Some(Codec::Lossy),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Codec::Raw => "raw",
            Codec::Lossless => "lossless",
            Codec::Lossy => "lossy",
        }
    }

    pub fn id(self) -> u8 {
        match self {
            Codec::Raw => CODEC_RAW,
            Codec::Lossless => CODEC_LOSSLESS,
            Codec::Lossy => CODEC_LOSSY,
        }
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            CODEC_RAW => Some(Codec::Raw),
            CODEC_LOSSLESS => Some(Codec::Lossless),
            CODEC_LOSSY => Some(Codec::Lossy),
            _ => None,
        }
    }

    /// Whether this build can encode/decode this codec. All current codecs
    /// are implemented; the lossy path requires `libpyrowave-shared` at
    /// runtime (checked when the encoder is constructed).
    pub fn implemented(self) -> bool {
        true
    }

    /// Bytes the RDMA staging/receive slot must hold for this codec: the raw
    /// frame, or the codec's worst case. `None` = context-dependent (lossy
    /// depends on the configured bit budget).
    pub fn slot_len(self, geom: &Geometry) -> Option<usize> {
        match self {
            Codec::Raw => Some(geom.frame_bytes()),
            Codec::Lossless => Some(geom.max_coded_len()),
            Codec::Lossy => None,
        }
    }
}


/// Geometry of the frames a codec instance works on. `stride` is the wire row
/// pitch in bytes (>= width * 4); tiles address pixels, rows address stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub tile_log2: u8,
}

impl Geometry {
    pub const DEFAULT_TILE_LOG2: u8 = 6; // 64 px

    pub fn new(width: u32, height: u32, stride: u32) -> Self {
        Self {
            width,
            height,
            stride,
            tile_log2: Self::DEFAULT_TILE_LOG2,
        }
    }

    pub fn with_tile_log2(mut self, tile_log2: u8) -> Self {
        self.tile_log2 = tile_log2;
        self
    }

    pub fn tile(&self) -> u32 {
        1u32 << self.tile_log2
    }

    pub fn tiles_x(&self) -> usize {
        self.width.div_ceil(self.tile()) as usize
    }

    pub fn tiles_y(&self) -> usize {
        self.height.div_ceil(self.tile()) as usize
    }

    /// Number of tiles covering the frame (the index capacity).
    pub fn tile_count(&self) -> usize {
        self.tiles_x() * self.tiles_y()
    }

    /// Fixed bytes per payload tile (padded; valid region is smaller at edges).
    pub fn tile_bytes(&self) -> usize {
        (self.tile() * self.tile() * BPP) as usize
    }

    /// Full frame bytes on the wire (stride * height).
    pub fn frame_bytes(&self) -> usize {
        self.stride as usize * self.height as usize
    }

    /// Worst-case coded length: full frame with every tile carried raw.
    pub fn max_coded_len(&self) -> usize {
        HEADER_LEN + self.tile_count() * INDEX_WORDS * 4 + self.tile_count() * self.tile_bytes()
    }

    /// Byte offset of the payload region (after the fixed index array).
    pub fn payload_off(&self) -> usize {
        HEADER_LEN + self.tile_count() * INDEX_WORDS * 4
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.width > 0 && self.height > 0, "zero frame dimensions");
        ensure!(self.stride >= self.width * BPP, "stride {} < width*4 {}", self.stride, self.width * BPP);
        ensure!(self.stride % BPP == 0, "stride {} not a multiple of 4", self.stride);
        ensure!((4..=8).contains(&self.tile_log2), "tile_log2 {} out of range", self.tile_log2);
        Ok(())
    }
}

// --- CRC32 (IEEE, reflected), no external deps ------------------------------

fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

pub fn crc32(bytes: &[u8]) -> u32 {
    let t = crc32_table();
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn wr_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

// --- header -----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub codec: u8,
    pub version: u8,
    pub delta: bool,
    /// payload CRC present (bit 1 of `flags`). When false the decoder skips
    /// verification; the encoder omits the (expensive) CRC pass.
    pub has_crc: bool,
    pub tile_log2: u8,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub count: u32,
    pub index_capacity: u32,
    pub crc32: u32,
    /// Total coded payload words (v2). The coded length is
    /// `payload_off() + payload_words·4`.
    pub payload_words: u32,
}

impl Header {
    pub fn geometry(&self) -> Geometry {
        Geometry {
            width: self.width,
            height: self.height,
            stride: self.stride,
            tile_log2: self.tile_log2,
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Header> {
        ensure!(bytes.len() >= HEADER_LEN, "coded frame shorter than header");
        ensure!(bytes[0] == MAGIC, "bad codec magic {:#x}", bytes[0]);
        let codec = bytes[1];
        let version = bytes[2];
        ensure!(version == VERSION, "unsupported codec version {version}");
        ensure!(Codec::from_id(codec).is_some(), "unknown codec id {codec}");
        ensure!(codec == CODEC_LOSSLESS, "header is not a lossless frame");
        Ok(Header {
            codec,
            version,
            delta: bytes[3] & 1 != 0,
            has_crc: bytes[3] & 2 != 0,
            tile_log2: bytes[4],
            width: rd_u32(bytes, 8),
            height: rd_u32(bytes, 12),
            stride: rd_u32(bytes, 16),
            count: rd_u32(bytes, 20),
            index_capacity: rd_u32(bytes, 24),
            crc32: rd_u32(bytes, 28),
            payload_words: rd_u32(bytes, 32),
        })
    }

    pub fn write(&self, out: &mut [u8]) {
        debug_assert!(out.len() >= HEADER_LEN);
        out[..HEADER_LEN].fill(0);
        out[0] = MAGIC;
        out[1] = self.codec;
        out[2] = self.version;
        out[3] = self.delta as u8 | ((self.has_crc as u8) << 1);
        out[4] = self.tile_log2;
        wr_u32(out, 8, self.width);
        wr_u32(out, 12, self.height);
        wr_u32(out, 16, self.stride);
        wr_u32(out, 20, self.count);
        wr_u32(out, 24, self.index_capacity);
        wr_u32(out, 28, self.crc32);
        wr_u32(out, 32, self.payload_words);
    }
}

/// Compute the CRC32 of the payload region and stamp it into a completed
/// coded frame's header. Must be called after every payload byte is written
/// (GPU fence waited / CPU encode finished) and before the frame is posted.
pub fn finish_crc(coded: &mut [u8]) -> Result<u32> {
    let h = Header::parse(coded)?;
    let cap = h.index_capacity as usize;
    ensure!(h.count as usize <= cap, "tile count {} > capacity {}", h.count, cap);
    let payload_off = HEADER_LEN + cap * INDEX_WORDS * 4;
    let payload_len = h.payload_words as usize * 4;
    ensure!(
        coded.len() >= payload_off + payload_len,
        "coded frame truncated ({} < {})",
        coded.len(),
        payload_off + payload_len
    );
    let crc = crc32(&coded[payload_off..payload_off + payload_len]);
    wr_u32(coded, 28, crc);
    Ok(crc)
}

// --- tile helpers -----------------------------------------------------------

#[inline]
fn valid_dims(g: &Geometry, tx: usize, ty: usize) -> (usize, usize) {
    let tile = g.tile() as usize;
    let w = (g.width as usize - tx * tile).min(tile);
    let h = (g.height as usize - ty * tile).min(tile);
    (w, h)
}

#[inline]
fn tile_row_off(g: &Geometry, tx: usize, ty: usize, row: usize) -> usize {
    let tile = g.tile() as usize;
    (ty * tile + row) * g.stride as usize + tx * tile * BPP as usize
}

/// Byte offset of pixel `(x, y)` within tile `(tx, ty)` in a frame with the
/// geometry's stride. `x`/`y` are tile-local.
#[inline]
fn gw(g: &Geometry, tx: usize, ty: usize, x: usize, y: usize) -> usize {
    (ty * g.tile() as usize + y) * g.stride as usize + (tx * g.tile() as usize + x) * BPP as usize
}

#[inline]
fn rd_word(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

#[inline]
fn wr_word(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// Choose the cheapest mode for one tile and build its payload words into
/// `out_words`; returns `(mode, len_words)`. This is the arbiter of the wire
/// format: the GPU encode shaders must choose the same mode and emit the same
/// tokens so the coded total matches.
fn tile_plan(
    g: &Geometry,
    cur: &[u8],
    reference: &[u8],
    tx: usize,
    ty: usize,
    allow_ref: bool,
    counts: &mut Vec<u32>,
    out_words: &mut Vec<u32>,
) -> (u32, usize) {
    let (vw, vh) = valid_dims(g, tx, ty);
    let n = vw * vh;

    // Count maximal runs per row for both residuals in one pass. Rows are
    // independent (a run never crosses a row boundary), which is what keeps
    // DPCM decode a per-row prefix sum.
    let (mut runs_ref, mut runs_dpcm) = (0usize, 0usize);
    let (mut prev_ref, mut prev_dpcm) = (0u32, 0u32);
    let mut prev_cur = 0u32;
    for y in 0..vh {
        for x in 0..vw {
            let o = gw(g, tx, ty, x, y);
            let w = rd_word(cur, o);
            let r = w ^ rd_word(reference, o);
            let d = w.wrapping_sub(if x == 0 { 0 } else { prev_cur });
            if x == 0 || r != prev_ref {
                runs_ref += 1;
                prev_ref = r;
            }
            if x == 0 || d != prev_dpcm {
                runs_dpcm += 1;
                prev_dpcm = d;
            }
            prev_cur = w;
        }
    }

    let mut mode = MODE_RAW;
    let mut best = n;
    if allow_ref && (vh + 2 * runs_ref) < best {
        best = vh + 2 * runs_ref;
        mode = MODE_REF;
    }
    if (vh + 2 * runs_dpcm) < best {
        mode = MODE_DPCM;
    }

    out_words.clear();
    if mode == MODE_RAW {
        for y in 0..vh {
            for x in 0..vw {
                out_words.push(rd_word(cur, gw(g, tx, ty, x, y)));
            }
        }
        return (mode, n);
    }

    // per-row run counts, then the tokens in row order
    counts.clear();
    counts.resize(vh, 0);
    for y in 0..vh {
        let (mut runs, mut prev, mut pcur) = (0u32, 0u32, 0u32);
        for x in 0..vw {
            let o = gw(g, tx, ty, x, y);
            let w = rd_word(cur, o);
            let res = if mode == MODE_REF {
                w ^ rd_word(reference, o)
            } else {
                w.wrapping_sub(if x == 0 { 0 } else { pcur })
            };
            if x == 0 || res != prev {
                runs += 1;
                prev = res;
            }
            pcur = w;
        }
        counts[y] = runs;
    }
    out_words.extend_from_slice(counts);
    for y in 0..vh {
        let (mut prev, mut runlen, mut pcur) = (0u32, 0u32, 0u32);
        for x in 0..vw {
            let o = gw(g, tx, ty, x, y);
            let w = rd_word(cur, o);
            let res = if mode == MODE_REF {
                w ^ rd_word(reference, o)
            } else {
                w.wrapping_sub(if x == 0 { 0 } else { pcur })
            };
            if x == 0 {
                prev = res;
                runlen = 1;
            } else if res == prev {
                runlen += 1;
            } else {
                out_words.push(runlen);
                out_words.push(prev);
                prev = res;
                runlen = 1;
            }
            pcur = w;
        }
        out_words.push(runlen);
        out_words.push(prev);
    }
    (mode, out_words.len())
}

fn tile_equal(g: &Geometry, a: &[u8], b: &[u8], tx: usize, ty: usize) -> bool {
    let (w, h) = valid_dims(g, tx, ty);
    let row_bytes = w * BPP as usize;
    for row in 0..h {
        let o = tile_row_off(g, tx, ty, row);
        if a[o..o + row_bytes] != b[o..o + row_bytes] {
            return false;
        }
    }
    true
}

// --- CPU reference encoder / decoder ---------------------------------------

/// Reference encoder. Holds the previous frame so delta frames can skip
/// unchanged tiles. The GPU encoder is its accelerated twin; the bytes it
/// produces must match this layout exactly.
pub struct Encoder {
    geom: Geometry,
    reference: Vec<u8>,
    have_reference: bool,
    /// compute the payload CRC (default on; the streaming path turns it off
    /// because a full-frame CRC costs tens of ms and RC RDMA is reliable)
    with_crc: bool,
}

impl Encoder {
    pub fn new(geom: Geometry) -> Result<Self> {
        geom.validate()?;
        Ok(Self {
            geom,
            reference: vec![0u8; geom.frame_bytes()],
            have_reference: false,
            with_crc: true,
        })
    }

    /// Enable/disable the payload CRC (see [`Header::has_crc`]).
    pub fn set_crc(&mut self, on: bool) {
        self.with_crc = on;
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    /// Forget the reference: the next frame is encoded full. Called on each
    /// new session (and after a geometry change) so the client can rebuild.
    pub fn reset(&mut self) {
        self.have_reference = false;
    }

    /// Re-point at new geometry (drops the reference).
    pub fn set_geometry(&mut self, geom: Geometry) -> Result<()> {
        geom.validate()?;
        self.geom = geom;
        self.reference.resize(geom.frame_bytes(), 0);
        self.have_reference = false;
        Ok(())
    }

    /// Encode `cur` (stride·height bytes) into `out`; returns the coded length.
    pub fn encode(&mut self, cur: &[u8], out: &mut [u8]) -> Result<usize> {
        let g = self.geom;
        ensure!(cur.len() >= g.frame_bytes(), "input shorter than a frame");
        ensure!(out.len() >= g.max_coded_len(), "output buffer too small");
        let cap = g.tile_count();
        let delta = self.have_reference;
        let payload_off = g.payload_off();

        let mut counts: Vec<u32> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        let mut count = 0usize;
        let mut cursor = 0usize; // payload words written so far

        for ty in 0..g.tiles_y() {
            for tx in 0..g.tiles_x() {
                if delta && tile_equal(&g, cur, &self.reference, tx, ty) {
                    continue;
                }
                let id = (ty * g.tiles_x() + tx) as u32;
                let (mode, len) = tile_plan(
                    &g, cur, &self.reference, tx, ty, delta, &mut counts, &mut words,
                );
                let e = HEADER_LEN + count * INDEX_WORDS * 4;
                wr_u32(out, e, id);
                wr_u32(out, e + 4, cursor as u32);
                wr_u32(out, e + 8, (mode << 16) | (len as u32));
                let po = payload_off + cursor * 4;
                for (i, &w) in words.iter().enumerate() {
                    wr_word(out, po + i * 4, w);
                }
                cursor += len;
                count += 1;

                // advance the reference so the next delta can compare
                let (vw, vh) = valid_dims(&g, tx, ty);
                for y in 0..vh {
                    for x in 0..vw {
                        let o = gw(&g, tx, ty, x, y);
                        let v = rd_word(cur, o);
                        wr_word(&mut self.reference, o, v);
                    }
                }
            }
        }

        let h = Header {
            codec: CODEC_LOSSLESS,
            version: VERSION,
            delta,
            has_crc: self.with_crc,
            tile_log2: g.tile_log2,
            width: g.width,
            height: g.height,
            stride: g.stride,
            count: count as u32,
            index_capacity: cap as u32,
            crc32: 0,
            payload_words: cursor as u32,
        };
        h.write(out);
        let len = payload_off + cursor * 4;
        if self.with_crc {
            finish_crc(&mut out[..len])?;
        }

        self.have_reference = true;
        Ok(len)
    }
}

/// Reference decoder. Keeps the reconstructed frame so delta frames apply.
pub struct Decoder {
    geom: Geometry,
    frame: Vec<u8>,
    have_frame: bool,
}

impl Decoder {
    pub fn new(geom: Geometry) -> Result<Self> {
        geom.validate()?;
        Ok(Self {
            geom,
            frame: vec![0u8; geom.frame_bytes()],
            have_frame: false,
        })
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    /// The most recently decoded frame, without decoding again. The CPU
    /// lossless client decodes every received frame (to advance the delta
    /// chain), so the presenter must borrow this rather than call `decode` on
    /// the newest frame a second time — re-applying a delta against an
    /// already-advanced reference corrupts the chain.
    pub fn frame(&self) -> &[u8] {
        &self.frame
    }

    /// Decode `coded` and return the full reconstructed frame (stride·height).
    /// The returned slice is valid until the next decode.
    pub fn decode(&mut self, coded: &[u8]) -> Result<&[u8]> {
        let h = Header::parse(coded)?;
        let g = h.geometry();
        g.validate()?;
        if g != self.geom {
            self.geom = g;
            self.frame.resize(g.frame_bytes(), 0);
            self.have_frame = false;
        }
        ensure!(
            h.index_capacity as usize == g.tile_count(),
            "index capacity {} != {}",
            h.index_capacity,
            g.tile_count()
        );
        ensure!(h.count as usize <= h.index_capacity as usize, "tile count over capacity");
        if h.delta {
            ensure!(self.have_frame, "delta frame before any full frame");
        }

        let cap = h.index_capacity as usize;
        let payload_off = HEADER_LEN + cap * INDEX_WORDS * 4;
        let payload_len = h.payload_words as usize * 4;
        ensure!(
            coded.len() >= payload_off + payload_len,
            "coded frame truncated"
        );
        // compute the CRC only when the producer asked for it (the streaming
        // path omits it: a full-frame CRC is tens of ms)
        if h.has_crc {
            let want = crc32(&coded[payload_off..payload_off + payload_len]);
            ensure!(want == h.crc32, "payload crc mismatch ({:08x} != {:08x})", want, h.crc32);
        }

        let frame_bytes = g.frame_bytes();
        for k in 0..h.count as usize {
            let e = HEADER_LEN + k * INDEX_WORDS * 4;
            let id = rd_u32(coded, e) as usize;
            let off = rd_u32(coded, e + 4) as usize;
            let w2 = rd_u32(coded, e + 8);
            let mode = w2 >> 16;
            let len = (w2 & 0xffff) as usize;
            ensure!(id < cap, "tile id {id} out of range");
            let tx = id % g.tiles_x();
            let ty = id / g.tiles_x();
            let (vw, vh) = valid_dims(&g, tx, ty);
            let base = payload_off + off * 4;
            ensure!(base + len * 4 <= coded.len(), "tile payload out of range");
            match mode {
                MODE_RAW => {
                    ensure!(len >= vw * vh, "raw tile too short");
                    for i in 0..vw * vh {
                        let (x, y) = (i % vw, i / vw);
                        let o = gw(&g, tx, ty, x, y);
                        wr_word(&mut self.frame, o, rd_word(coded, base + i * 4));
                    }
                }
                MODE_REF | MODE_DPCM => {
                    ensure!(len >= vh, "rle tile too short");
                    let mut tok = base + vh * 4;
                    for y in 0..vh {
                        let runs = rd_u32(coded, base + y * 4) as usize;
                        let (mut x, mut acc) = (0usize, 0u32);
                        for _ in 0..runs {
                            ensure!(tok + 8 <= coded.len(), "rle token out of range");
                            let rl = rd_u32(coded, tok) as usize;
                            let val = rd_u32(coded, tok + 4);
                            tok += 8;
                            ensure!(x + rl <= vw, "rle run overflows row");
                            for _ in 0..rl {
                                let o = gw(&g, tx, ty, x, y);
                                let out = if mode == MODE_REF {
                                    rd_word(&self.frame, o) ^ val
                                } else {
                                    acc = acc.wrapping_add(val);
                                    acc
                                };
                                wr_word(&mut self.frame, o, out);
                                x += 1;
                            }
                        }
                        ensure!(x == vw, "rle row underfull");
                    }
                }
                _ => anyhow::bail!("unknown lossless tile mode {mode}"),
            }
        }
        self.have_frame = true;
        Ok(&self.frame[..frame_bytes])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so tests don't need a rand dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn byte(&mut self) -> u8 {
            self.next() as u8
        }
    }

    fn geom() -> Geometry {
        // awkward on purpose: not a tile multiple in either axis
        Geometry::new(323, 197, 323 * 4)
    }

    fn random_frame(g: &Geometry, rng: &mut Rng) -> Vec<u8> {
        (0..g.frame_bytes()).map(|_| rng.byte()).collect()
    }

    /// Compare only the visible pixel region: row padding (stride - width*4)
    /// is never displayed and is left unspecified by the raw paths and this
    /// codec.
    fn assert_pixels_eq(g: &Geometry, got: &[u8], want: &[u8], what: &str) {
        let n = g.width as usize * 4;
        for row in 0..g.height as usize {
            let off = row * g.stride as usize;
            assert_eq!(
                &got[off..off + n],
                &want[off..off + n],
                "{what}: pixel mismatch at row {row}"
            );
        }
    }

    fn encode_decode(enc: &mut Encoder, dec: &mut Decoder, frame: &[u8]) -> Vec<u8> {
        let g = enc.geometry();
        let mut buf = vec![0u8; g.max_coded_len()];
        let n = enc.encode(frame, &mut buf).expect("encode");
        let decoded = dec.decode(&buf[..n]).expect("decode").to_vec();
        assert_pixels_eq(&g, &decoded, frame, "round trip");
        buf.truncate(n);
        buf
    }

    #[test]
    fn codec_parse() {
        assert_eq!(Codec::parse("raw"), Some(Codec::Raw));
        assert_eq!(Codec::parse("off"), Some(Codec::Raw));
        assert_eq!(Codec::parse("lossless"), Some(Codec::Lossless));
        assert_eq!(Codec::parse("lossy"), Some(Codec::Lossy));
        assert_eq!(Codec::parse("nope"), None);
        for c in [Codec::Raw, Codec::Lossless, Codec::Lossy] {
            assert_eq!(Codec::from_id(c.id()), Some(c));
        }
        assert!(Codec::Raw.implemented());
        assert!(Codec::Lossless.implemented());
        assert!(Codec::Lossy.implemented());
    }

    #[test]
    fn geometry_tiles() {
        let g = Geometry::new(323, 197, 323 * 4);
        assert_eq!(g.tiles_x(), 6); // ceil(323/64)
        assert_eq!(g.tiles_y(), 4); // ceil(197/64)
        assert_eq!(g.tile_count(), 24);
        assert!(g.max_coded_len() > g.frame_bytes());
        g.validate().unwrap();
        // raw keeps today's exact slot size; lossless reserves its worst case
        assert_eq!(Codec::Raw.slot_len(&g), Some(g.frame_bytes()));
        assert_eq!(Codec::Lossless.slot_len(&g), Some(g.max_coded_len()));
        assert_eq!(Codec::Lossy.slot_len(&g), None);
    }

    /// A desktop-ish frame (flat background, a small moving box): most tiles
    /// are unchanged between frames, so deltas must be a small fraction of
    /// the raw frame. This is the workload the lossless codec is for.
    #[test]
    fn delta_savings_on_desktop_like_content() {
        let g = Geometry::new(1920, 1080, 1920 * 4);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let mut buf = vec![0u8; g.max_coded_len()];

        let mut frame = vec![0x20u8; g.frame_bytes()];
        // full first frame
        let n0 = enc.encode(&frame, &mut buf).unwrap();
        dec.decode(&buf[..n0]).unwrap();
        let h0 = Header::parse(&buf[..n0]).unwrap();
        assert!(!h0.delta);
        assert_eq!(h0.count as usize, g.tile_count(), "first frame carries all tiles");

        let mut last_delta = 0usize;
        for step in 0..30u32 {
            // move a 64x64 box one tile at a time
            let bx = 64 * (step % 20) as usize;
            let by = 64 * ((step / 20) % 8) as usize;
            // paint the box
            for y in by..(by + 64) {
                let row = y * g.stride as usize;
                for x in bx..(bx + 64) {
                    let o = row + x * 4;
                    frame[o] = 0xF0;
                    frame[o + 1] = 0x80;
                    frame[o + 2] = 0x10;
                }
            }
            let n = enc.encode(&frame, &mut buf).unwrap();
            let decoded = dec.decode(&buf[..n]).unwrap();
            assert_pixels_eq(&g, decoded, &frame, "desktop delta");
            last_delta = n;
            // clear the box (a couple of tiles change next frame)
            for y in by..(by + 64) {
                let row = y * g.stride as usize;
                for x in bx..(bx + 64) {
                    let o = row + x * 4;
                    frame[o] = 0x20;
                    frame[o + 1] = 0x20;
                    frame[o + 2] = 0x20;
                }
            }
        }
        assert!(
            last_delta < g.frame_bytes() / 4,
            "desktop delta {} should be well under a quarter frame {}",
            last_delta,
            g.frame_bytes()
        );
    }

    #[test]
    fn first_frame_is_full_then_deltas_are_empty() {
        let g = geom();
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();

        let f = random_frame(&g, &mut rng);
        let first = encode_decode(&mut enc, &mut dec, &f);
        let h = Header::parse(&first).unwrap();
        assert!(!h.delta, "first frame must be full");
        assert_eq!(h.count as usize, g.tile_count(), "full frame carries all tiles");
        assert_eq!(Header::parse(&first).unwrap().crc32, crc32_payload(&first));

        // Same pixels again: delta with zero changed tiles => tiny frame.
        let second = encode_decode(&mut enc, &mut dec, &f);
        let h2 = Header::parse(&second).unwrap();
        assert!(h2.delta);
        assert_eq!(h2.count, 0, "identical frame must carry no tiles");
        assert_eq!(
            second.len(),
            HEADER_LEN + g.tile_count() * INDEX_WORDS * 4,
            "empty delta is header + fixed index array"
        );
        assert_eq!(h2.payload_words, 0);
    }

    #[test]
    fn single_tile_change_carries_one_tile() {
        let g = geom();
        let mut rng = Rng(42);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();

        let mut f = random_frame(&g, &mut rng);
        encode_decode(&mut enc, &mut dec, &f.clone());

        // Change one pixel inside tile (2,1).
        let (tx, ty) = (2usize, 1usize);
        let off = (ty * g.tile() as usize + 3) * g.stride as usize
            + (tx * g.tile() as usize + 5) * 4;
        f[off] ^= 0xFF;
        let coded = encode_decode(&mut enc, &mut dec, &f);
        let h = Header::parse(&coded).unwrap();
        assert!(h.delta);
        assert_eq!(h.count, 1, "exactly one changed tile");
    }

    #[test]
    fn sequence_with_drift_stays_exact() {
        let g = geom();
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let mut f = random_frame(&g, &mut rng);
        for step in 0..40 {
            // mutate a growing number of pixels
            for _ in 0..(step * 37) {
                let p = (rng.next() as usize) % f.len();
                f[p] = rng.byte();
            }
            encode_decode(&mut enc, &mut dec, &f);
        }
    }

    #[test]
    fn dims_that_are_tile_multiples() {
        let g = Geometry::new(256, 128, 256 * 4);
        let mut rng = Rng(7);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let f = random_frame(&g, &mut rng);
        encode_decode(&mut enc, &mut dec, &f);
        assert_eq!(g.tiles_x(), 4);
        assert_eq!(g.tiles_y(), 2);
    }

    #[test]
    fn stride_larger_than_width() {
        // padded rows: stride has slack the tiles must not read as pixels
        let g = Geometry::new(100, 40, 100 * 4 + 64);
        let mut rng = Rng(99);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let f = random_frame(&g, &mut rng);
        encode_decode(&mut enc, &mut dec, &f);
    }

    #[test]
    fn reset_forces_full_frame() {
        let g = geom();
        let mut rng = Rng(5);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let f = random_frame(&g, &mut rng);
        encode_decode(&mut enc, &mut dec, &f);
        enc.reset();
        let coded = encode_decode(&mut enc, &mut dec, &f);
        assert!(!Header::parse(&coded).unwrap().delta, "reset => full frame");
    }

    #[test]
    fn decoder_rejects_delta_without_reference() {
        let g = geom();
        let mut rng = Rng(11);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let mut f = random_frame(&g, &mut rng);
        encode_decode(&mut enc, &mut dec, &f);
        // Fresh decoder cannot apply the delta.
        let mut dec2 = Decoder::new(g).unwrap();
        f[0] ^= 1;
        let mut buf = vec![0u8; g.max_coded_len()];
        let n = enc.encode(&f, &mut buf).unwrap();
        assert!(dec2.decode(&buf[..n]).is_err());
    }

    #[test]
    fn corrupt_payload_is_caught() {
        let g = geom();
        let mut rng = Rng(3);
        let mut enc = Encoder::new(g).unwrap();
        let mut dec = Decoder::new(g).unwrap();
        let f = random_frame(&g, &mut rng);
        let mut buf = vec![0u8; g.max_coded_len()];
        let n = enc.encode(&f, &mut buf).unwrap();
        let h = Header::parse(&buf[..n]).unwrap();
        let payload_off = HEADER_LEN + h.index_capacity as usize * INDEX_WORDS * 4;
        buf[payload_off + 1] ^= 0x80;
        assert!(dec.decode(&buf[..n]).is_err(), "crc must reject corruption");
    }

    #[test]
    fn crc_is_optional() {
        let g = geom();
        let mut rng = Rng(0x5eed);
        let mut enc = Encoder::new(g).unwrap();
        enc.set_crc(false);
        let mut dec = Decoder::new(g).unwrap();
        let f = random_frame(&g, &mut rng);
        let mut buf = vec![0u8; g.max_coded_len()];
        let n = enc.encode(&f, &mut buf).unwrap();
        let h = Header::parse(&buf[..n]).unwrap();
        assert!(!h.has_crc);
        assert_eq!(h.crc32, 0);
        // no CRC to recompute; a corrupted payload is NOT caught (by design)
        let decoded = dec.decode(&buf[..n]).unwrap();
        assert_pixels_eq(&g, decoded, &f, "no-crc round trip");
    }

    fn crc32_payload(coded: &[u8]) -> u32 {
        let h = Header::parse(coded).unwrap();
        let payload_off = HEADER_LEN + h.index_capacity as usize * INDEX_WORDS * 4;
        let len = h.payload_words as usize * 4;
        crc32(&coded[payload_off..payload_off + len])
    }
}
