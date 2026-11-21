layout(std430, set = 0, binding = 1) buffer Ref { uint ref[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Out { uint outb[]; };
layout(std430, set = 0, binding = 3) buffer Cnt { uint count; uint alloc; };
const uint TILE = 64u;
const uint TW = TILE * TILE;
shared uint s_runs_r[256];
shared uint s_runs_d[256];
shared uint s_roff[256];
shared uint s_mode;
shared uint s_off;
shared uint s_changed;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tile_id = gl_WorkGroupID.x;
    uint tx = tile_id % u_tiles_x;
    uint ty = tile_id / u_tiles_x;
    uint x0 = tx * TILE, y0 = ty * TILE;
    uint vw = min(TILE, u_width - x0);
    uint vh = min(TILE, u_height - y0);

    bool changed = (u_full != 0u);
    if (!changed) {
        uint acc = 0u;
        for (uint i = tid; i < TW; i += 256u) {
            uint lx = i % TILE, ly = i / TILE;
            if (lx >= vw || ly >= vh) continue;
            uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
            acc |= (word(x0 + lx, y0 + ly) ^ ref[gi]);
        }
        changed = (acc != 0u);
    }
    if (tid == 0u) s_changed = 0u;
    barrier();
    if (changed) atomicOr(s_changed, 1u);
    barrier();
    if (s_changed == 0u) return;

    if (tid < vh) {
        uint y = tid;
        uint runs_r = 0u, runs_d = 0u;
        uint prev_r = 0u, prev_d = 0u, pcur = 0u;
        for (uint px = 0u; px < vw; ++px) {
            uint w = word(x0 + px, y0 + y);
            uint rf = ref[(y0 + y) * u_stride_px + (x0 + px)];
            uint r = w ^ rf;
            uint d = w - ((px == 0u) ? 0u : pcur);
            if (px == 0u || r != prev_r) { runs_r++; prev_r = r; }
            if (px == 0u || d != prev_d) { runs_d++; prev_d = d; }
            pcur = w;
        }
        s_runs_r[y] = runs_r;
        s_runs_d[y] = runs_d;
    }
    barrier();

    if (tid == 0u) {
        uint tr = 0u, td = 0u;
        for (uint y = 0u; y < vh; ++y) { tr += s_runs_r[y]; td += s_runs_d[y]; }
        uint mode = 0u;
        uint best = vw * vh;
        if (u_full == 0u && (vh + 2u * tr) < best) { best = vh + 2u * tr; mode = 1u; }
        if ((vh + 2u * td) < best) { best = vh + 2u * td; mode = 2u; }
        s_mode = mode;
        uint off = atomicAdd(alloc, (mode == 0u) ? (vw * vh) : best);
        s_off = off;
        if (mode != 0u) {
            uint acc = 0u;
            for (uint y = 0u; y < vh; ++y) {
                s_roff[y] = acc;
                acc += (mode == 1u) ? s_runs_r[y] : s_runs_d[y];
            }
        }
        uint k = atomicAdd(count, 1u);
        uint e = 12u + k * 3u;
        outb[e] = tile_id;
        outb[e + 1u] = off;
        outb[e + 2u] = (best & 0xffffu) | (mode << 16u);
    }
    barrier();

    uint mode = s_mode;
    uint base = u_pbase_words + s_off;
    if (mode == 0u) {
        for (uint i = tid; i < vw * vh; i += 256u) {
            uint lx = i % vw, ly = i / vw;
            outb[base + i] = word(x0 + lx, y0 + ly);
        }
    } else if (tid < vh) {
        uint y = tid;
        outb[base + y] = (mode == 1u) ? s_runs_r[y] : s_runs_d[y];
        uint tok = base + vh + 2u * s_roff[y];
        uint lr = 0u, lv = 0u, pcur = 0u;
        for (uint i = 0u; i < vw; ++i) {
            uint w = word(x0 + i, y0 + y);
            uint rf = ref[(y0 + y) * u_stride_px + (x0 + i)];
            uint res = (mode == 1u) ? (w ^ rf) : (w - ((i == 0u) ? 0u : pcur));
            if (i == 0u) { lv = res; lr = 1u; }
            else if (res == lv) { lr++; }
            else { outb[tok] = lr; outb[tok + 1u] = lv; tok += 2u; lv = res; lr = 1u; }
            pcur = w;
        }
        outb[tok] = lr;
        outb[tok + 1u] = lv;
    }
    barrier();

    if (u_update_ref != 0u) {
        for (uint i = tid; i < TW; i += 256u) {
            uint lx = i % TILE, ly = i / TILE;
            if (lx < vw && ly < vh) {
                uint gi = (y0 + ly) * u_stride_px + (x0 + lx);
                ref[gi] = word(x0 + lx, y0 + ly);
            }
        }
    }
}
