#version 450

layout(location = 0) in vec3 in_position;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    mat4 shadow_projection;
    vec3 camera_position;
    uint point_light_count;
    vec3 directional_light_direction;
    uint spot_light_count;
    vec3 directional_light_color;
    uint sun_shadow_enabled;
    vec3 ambient_light;
    uint spot_shadow_mask;
    uint point_shadow_mask;
    uint pad5;
    uint pad6;
    uint pad7;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float alpha;
    vec3 specular;
    float emissive_strength;
} model;

void main() {
    gl_Position = scene.view_projection * model.model * vec4(in_position, 1.0);
}