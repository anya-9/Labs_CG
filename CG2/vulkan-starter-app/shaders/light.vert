#version 450

layout(location = 0) in vec3 in_position;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    uint point_light_count;
    vec3 directional_light_direction;
    uint spot_light_count;
    vec3 directional_light_color;
    float pad3;
    vec3 ambient_light;
    float pad4;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float alpha;
    vec3 specular;
    float pad;
} model;

void main() {
    gl_Position = scene.view_projection * model.model * vec4(in_position, 1.0);
}