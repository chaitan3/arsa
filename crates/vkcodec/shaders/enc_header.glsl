#version 450
layout(local_size_x = 1) in;
layout(push_constant) uniform PC {
    uint width;
    uint height;
    uint stride;
    uint tile_log2;
    uint index_capacity;
    uint delta;
} pc;
layout(std430, set = 0, binding = 0) writeonly buffer Out { uint outb[]; };
layout(std430, set = 0, binding = 1) readonly buffer Cnt { uint count; uint alloc; } cnt;

void main() {
    // Header::write layout, words 0..11:
    //   0 magic|codec|version|flags, 1 tile_log2, 2 width, 3 height,
    //   4 stride, 5 count, 6 index_capacity, 7 crc, 8 payload_words.
    outb[0] = 0xC0u | (1u << 8) | (2u << 16) | (pc.delta << 24);
    outb[1] = pc.tile_log2;
    outb[2] = pc.width;
    outb[3] = pc.height;
    outb[4] = pc.stride;
    outb[5] = cnt.count;
    outb[6] = pc.index_capacity;
    outb[7] = 0u; // CRC is intentionally omitted on the zero-copy path
    outb[8] = cnt.alloc;
    outb[9] = 0u;
    outb[10] = 0u;
    outb[11] = 0u;
}
