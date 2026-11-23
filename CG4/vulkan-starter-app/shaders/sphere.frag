#version 450

layout(location = 0) in vec3 frag_world_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec4 frag_sun_shadow_pos;

layout(location = 0) out vec4 out_color;

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

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float quadratic;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

struct SpotLight {
    vec3 position;
    float intensity;
    vec3 color;
    float quadratic;
    vec3 direction;
    float inner_cos;
    float outer_cos;
    float pad1;
    float pad2;
    float pad3;
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

const uint MAX_SPOT_SHADOWS  = 8u;
const uint MAX_POINT_SHADOWS = 4u;

layout(std430, set = 0, binding = 3) readonly buffer ShadowMatrices {
    mat4 spot_shadow_matrices[8];
    mat4 point_shadow_matrices[24];
};

layout(set = 0, binding = 4) uniform sampler2DShadow sun_shadow_map;
layout(set = 0, binding = 5) uniform sampler2DArrayShadow spot_shadow_maps;
layout(set = 0, binding = 6) uniform sampler2DArrayShadow point_shadow_maps;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float alpha;
    vec3 specular;
    float emissive_strength;
} model;

layout(set = 2, binding = 0) uniform sampler2D albedo_map;
layout(set = 2, binding = 1) uniform sampler2D specular_map;
layout(set = 2, binding = 2) uniform sampler2D emissive_map;

layout(push_constant) uniform PushConstants {
    uint flags;
    float time;
    float wave_amplitude;
    float pad;
} pc;

const uint FLAG_UV_WAVE = 1u;
const uint FLAG_PIXEL = 2u;

float sunShadow() {
    if (scene.sun_shadow_enabled == 0u) return 1.0;

    vec3 p = frag_sun_shadow_pos.xyz / frag_sun_shadow_pos.w;
    p.xy = p.xy * 0.5 + 0.5;

    if (p.z > 1.0 || p.z < 0.0) return 1.0;

    return texture(sun_shadow_map, p);
}

float spotShadow(uint i, vec3 world_pos) {
    if (i >= MAX_SPOT_SHADOWS || (scene.spot_shadow_mask & (1u << i)) == 0u) return 1.0;

    vec4 clip = spot_shadow_matrices[i] * vec4(world_pos, 1.0);
    if (clip.w <= 0.0) return 1.0;

    vec3 p = clip.xyz / clip.w;
    p.xy = p.xy * 0.5 + 0.5;
    if (p.z > 1.0 || p.z < 0.0) return 1.0;

    return texture(spot_shadow_maps, vec4(p.xy, float(i), p.z));
}

float pointShadow(uint i, vec3 world_pos, vec3 light_pos) {
    if (i >= MAX_POINT_SHADOWS || (scene.point_shadow_mask & (1u << i)) == 0u) return 1.0;

    vec3 d = world_pos - light_pos;
    vec3 a = abs(d);

    uint face;
    if (a.x >= a.y && a.x >= a.z) face = (d.x > 0.0) ? 0u : 1u;
    else if (a.y >= a.z)          face = (d.y > 0.0) ? 2u : 3u;
    else                          face = (d.z > 0.0) ? 4u : 5u;

    uint layer = i * 6u + face;

    vec4 clip = point_shadow_matrices[layer] * vec4(world_pos, 1.0);
    if (clip.w <= 0.0) return 1.0;

    vec3 p = clip.xyz / clip.w;
    p.xy = p.xy * 0.5 + 0.5;
    if (p.z > 1.0 || p.z < 0.0) return 1.0;

    return texture(point_shadow_maps, vec4(p.xy, float(layer), p.z));
}

void main() {
    vec2 uv = frag_uv;
    float t = pc.time;

    if ((pc.flags & FLAG_UV_WAVE) != 0u) {
        uv += pc.wave_amplitude * vec2(sin(uv.y * 18.0 + t * 2.0),
                                       cos(uv.x * 15.0 + t * 1.7));
    }

    vec3 tex_albedo = texture(albedo_map, uv).rgb;

    if ((pc.flags & FLAG_PIXEL) != 0u) {
        float pixel_size = 32.0;
        vec2 pixel_uv = floor(frag_uv * pixel_size) / pixel_size;
        tex_albedo = texture(albedo_map, pixel_uv).rgb;
    }

    vec3 albedo = tex_albedo * model.albedo;
    vec3 spec_color = texture(specular_map, uv).rgb * model.specular;
    vec3 emissive = texture(emissive_map, uv).rgb * model.emissive_strength;

    vec3 N = normalize(frag_normal);
    vec3 V = normalize(scene.camera_position - frag_world_pos);

    vec3 ambient = scene.ambient_light * albedo;

    vec3 L_dir = normalize(-scene.directional_light_direction);
    float diff_dir = max(dot(N, L_dir), 0.0);
    float sun_shadow = sunShadow();
    vec3 diffuse_dir = scene.directional_light_color * diff_dir * albedo * sun_shadow;

    vec3 H_dir = normalize(L_dir + V);
    float spec_dir = pow(max(dot(N, H_dir), 0.0), model.alpha);
    vec3 specular_dir = scene.directional_light_color * spec_dir * spec_color * sun_shadow;

    vec3 point_sum = vec3(0.0);
    for (uint i = 0; i < scene.point_light_count; ++i) {
        PointLight pl = point_lights[i];

        vec3 light_vec = pl.position - frag_world_pos;
        float dist = length(light_vec);
        vec3 L = light_vec / dist;

        float diff = max(dot(N, L), 0.0);

        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), model.alpha);

        float atten = 1.0 / (1.0 + pl.quadratic * dist * dist);

        vec3 light_color = pl.color * pl.intensity;
        float shadow = pointShadow(i, frag_world_pos, pl.position);
        point_sum += light_color * atten * shadow * (diff * albedo + spec * spec_color);
    }

    vec3 spot_sum = vec3(0.0);
    for (uint i = 0; i < scene.spot_light_count; ++i) {
        SpotLight sl = spot_lights[i];

        vec3 light_vec = sl.position - frag_world_pos;
        float dist = length(light_vec);
        vec3 L = light_vec / dist;

        float theta = dot(L, normalize(-sl.direction));

        float epsilon = sl.inner_cos - sl.outer_cos;
        float spot_factor = clamp((theta - sl.outer_cos) / epsilon, 0.0, 1.0);

        if (spot_factor <= 0.0) continue;

        float diff = max(dot(N, L), 0.0);

        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), model.alpha);

        float atten = 1.0 / (1.0 + sl.quadratic * dist * dist);

        vec3 light_color = sl.color * sl.intensity;
        float shadow = spotShadow(i, frag_world_pos);
        spot_sum += light_color * atten * spot_factor * shadow * (diff * albedo + spec * spec_color);
    }

    vec3 result = ambient + diffuse_dir + specular_dir + point_sum + spot_sum + emissive;
    out_color = vec4(result, 1.0);
}