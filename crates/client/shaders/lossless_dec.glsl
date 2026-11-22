#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer In { uint inb[]; } coded;
layout(set = 0, binding = 1, rgba8) uniform image2D frame;

layout(push_constant) uniform PC {
    uint width;
    uint height;
    uint stride_px;
    uint tiles_x;
    uint count;
    uint pbase_words;
    uint slot_base_words;
} pc;

const uint TILE = 64u;
shared uint s_cnt[256];

uint read_coded(uint i) { return coded.inb[pc.slot_base_words + i]; }
uint load_px(uint x, uint y) {
    return packUnorm4x8(imageLoad(frame, ivec2(int(x), int(y))));
}
void store_px(uint x, uint y, uint v) {
    imageStore(frame, ivec2(int(x), int(y)), unpackUnorm4x8(v));
}

void main() {
    uint k = gl_WorkGroupID.x;
    if (k >= pc.count) return;
    uint tid = gl_LocalInvocationID.x;
    uint i0 = 12u + k * 3u;
    uint tile_id = read_coded(i0);
    uint off = read_coded(i0 + 1u);
    uint mode = read_coded(i0 + 2u) >> 16u;
    uint tx = tile_id % pc.tiles_x;
    uint ty = tile_id / pc.tiles_x;
    uint x0 = tx * TILE;
    uint y0 = ty * TILE;
    uint vw = min(TILE, pc.width - x0);
    uint vh = min(TILE, pc.height - y0);
    uint base = pc.pbase_words + off;
    if (mode == 0u) {
        for (uint i = tid; i < vw * vh; i += 256u) {
            uint lx = i % vw;
            uint ly = i / vw;
            store_px(x0 + lx, y0 + ly, read_coded(base + i));
        }
        return;
    }
    if (tid < vh) s_cnt[tid] = read_coded(base + tid);
    barrier();
    if (tid < vh) {
        uint y = tid;
        uint roff = 0u;
        for (uint yy = 0u; yy < y; ++yy) roff += s_cnt[yy];
        uint tok = base + vh + 2u * roff;
        uint runs = s_cnt[y];
        uint x = 0u;
        uint acc = 0u;
        for (uint r = 0u; r < runs; ++r) {
            uint len = read_coded(tok);
            uint val = read_coded(tok + 1u);
            tok += 2u;
            for (uint j = 0u; j < len; ++j) {
                uint nv;
                if (mode == 1u) {
                    nv = load_px(x0 + x, y0 + y) ^ val;
                } else {
                    acc += val;
                    nv = acc;
                }
                store_px(x0 + x, y0 + y, nv);
                x++;
            }
        }
    }
}
