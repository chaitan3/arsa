#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0) uniform sampler2D src;
layout(set = 0, binding = 1, rgba8) uniform image2D dst;
layout(push_constant) uniform PC { uint width; uint height; uint stride_px; } pc;
void main() {
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= pc.width || p.y >= pc.height) return;
    vec2 uv = (vec2(p) + 0.5) / vec2(pc.width, pc.height);
    imageStore(dst, ivec2(p), texture(src, uv));
}
