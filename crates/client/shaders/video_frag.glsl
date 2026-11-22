#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag;
layout(set = 0, binding = 0) uniform sampler2D video_tex;
layout(push_constant) uniform PC {
    vec2 win;
    vec2 frame;
    int scale_mode;
} pc;

float bicubic_w(float x) {
    x = abs(x);
    if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}
vec4 sf(vec2 uv, vec2 texel, float fp) {
    if (pc.scale_mode == 2) {
        vec2 s = uv / texel - 0.5;
        vec2 ip = floor(s);
        vec2 f = s - ip;
        vec4 c = vec4(0.0);
        for (int j = -1; j <= 2; j++) {
            for (int i = -1; i <= 2; i++) {
                float w = bicubic_w(float(i) - f.x) * bicubic_w(float(j) - f.y);
                c += w * texture(video_tex, (ip + vec2(float(i) + 0.5, float(j) + 0.5)) * texel);
            }
        }
        return c;
    }
    if (pc.scale_mode == 3) {
        fp = clamp(fp, 1.0, 4.0);
        if (fp <= 1.25) return texture(video_tex, uv);
        vec2 src = uv / texel;
        vec4 acc = vec4(0.0);
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                vec2 off = (vec2(float(i), float(j)) + 0.5) * 0.25 - 0.5;
                acc += texture(video_tex, (src + off * fp) * texel);
            }
        }
        return acc * 0.0625;
    }
    return texture(video_tex, uv);
}

void main() {
    vec2 px = v_uv * pc.win;
    float scl = min(pc.win.x / max(pc.frame.x, 1.0), pc.win.y / max(pc.frame.y, 1.0));
    vec2 pic = pc.frame * scl;
    vec2 org = (pc.win - pic) * 0.5;
    if (all(greaterThanEqual(px, org)) && all(lessThan(px, org + pic))) {
        vec2 uv = (px - org) / pic;
        frag = sf(uv, 1.0 / pc.frame, 1.0 / max(scl, 1e-3)).bgra;
    } else {
        frag = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
