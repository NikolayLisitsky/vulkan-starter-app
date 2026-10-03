#version 450 core

layout(location = 0) in vec3 in_color;
layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;
    vec4 tint;
} global_uniforms;

void main() {
    out_color = vec4(in_color * global_uniforms.tint.rgb, global_uniforms.tint.a);
}
