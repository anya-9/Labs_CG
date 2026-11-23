#version 450

layout(location = 0) out vec4 out_color;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float alpha;
    vec3 specular;
    float emissive_strength;
} model;

void main() {
    out_color = vec4(model.albedo, 1.0);
}