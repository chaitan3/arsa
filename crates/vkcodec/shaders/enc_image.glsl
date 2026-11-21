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

layout(set = 0, binding = 0) uniform sampler2D u_src;
uint word(uint x, uint y) {
    vec2 uv = (vec2(float(x), float(y)) + 0.5) / vec2(float(u_width), float(u_height));
    return packUnorm4x8(texture(u_src, uv));
}

//__COMMON__
