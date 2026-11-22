#version 450
layout(location = 0) out vec4 frag;
layout(set = 0, binding = 0) uniform sampler2D overlay_tex;
layout(push_constant) uniform PC {
    vec2 origin;
    vec2 size;
    vec2 win;
} pc;

void main() {
    vec2 p = gl_FragCoord.xy;
    if (p.x < pc.origin.x || p.y < pc.origin.y ||
        p.x >= pc.origin.x + pc.size.x || p.y >= pc.origin.y + pc.size.y) {
        discard;
    }
    vec2 uv = (p - pc.origin) / pc.size;
    frag = texture(overlay_tex, uv);
}
