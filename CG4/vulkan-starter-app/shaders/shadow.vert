#version 450

layout(location = 0) in vec3 in_position;

layout(push_constant) uniform ShadowPushConstants {
    mat4 light_view_projection;
} pc;

layout(std140, set = 0, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float alpha;
    vec3 specular;
    float emissive_strength;
} model;

void main() {
    gl_Position = pc.light_view_projection * model.model * vec4(in_position, 1.0);
}