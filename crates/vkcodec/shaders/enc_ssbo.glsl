#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PC {
    uint u_width;
    uint u_height;
    uint u_stride_px;
    uint u_tiles_x;
    uint u_full;
    uint u_pbase_words;
    uint u_update_ref;
} pc;
#define u_width pc.u_width
#define u_height pc.u_height
#define u_stride_px pc.u_stride_px
#define u_tiles_x pc.u_tiles_x
#define u_full pc.u_full
#define u_pbase_words pc.u_pbase_words
#define u_update_ref pc.u_update_ref

layout(std430, set = 0, binding = 0) readonly buffer Cur { uint cur[]; };
uint word(uint x, uint y) { return cur[y * u_stride_px + x]; }

//__COMMON__
