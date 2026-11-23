#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 frag_world_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_uv;

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
    float emissive_strength;
} model;

void main() {
    vec4 world_pos = model.model * vec4(in_position, 1.0);
    frag_world_pos = world_pos.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(model.model)));
    frag_normal = normal_matrix * in_normal;

    frag_uv = vec2(1.0 - in_uv.x, in_uv.y); 

    gl_Position = scene.view_projection * world_pos;
}