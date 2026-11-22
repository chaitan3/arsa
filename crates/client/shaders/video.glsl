#version 450
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    // Vulkan's NDC y = -1 is the top of the viewport (framebuffer y grows
    // downward), so p.y = 0 must map to image v = 0 (top row). The old
    // GL-style 1.0 - p.y here flipped the whole video vertically.
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
