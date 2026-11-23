#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <imgui.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_PERLIN_IMPLEMENTATION
#include "stb_perlin.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

namespace {

constexpr float PI = 3.14159265358979323846f;

constexpr uint32_t MAX_POINT_LIGHTS = 16;
constexpr uint32_t MAX_SPOT_LIGHTS  = 8;
constexpr uint32_t MAX_LIGHT_VIS = MAX_POINT_LIGHTS + 3 * MAX_SPOT_LIGHTS;

constexpr uint32_t MAX_OBJECTS   = 8;
constexpr uint32_t MAX_MATERIALS = 8;

constexpr VkFormat SHADOW_FORMAT = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t SUN_SHADOW_SIZE   = 2048;
constexpr uint32_t SPOT_SHADOW_SIZE  = 1024;
constexpr uint32_t POINT_SHADOW_SIZE = 512;
constexpr uint32_t MAX_POINT_SHADOWS = 4;
constexpr uint32_t POINT_FACES       = 6;
constexpr float LOCAL_SHADOW_NEAR = 0.1f;
constexpr float LOCAL_SHADOW_FAR  = 50.0f;
constexpr float DEPTH_BIAS_CONSTANT_SCALE = 100.0f;

constexpr uint32_t DYN_STRIDE = 256;
constexpr uint32_t MAX_DRAW_SLOTS = MAX_OBJECTS + MAX_LIGHT_VIS;

constexpr uint32_t pointSlot(uint32_t i)      { return MAX_OBJECTS + i; }
constexpr uint32_t spotSlot(uint32_t i)       { return MAX_OBJECTS + MAX_POINT_LIGHTS + i; }
constexpr uint32_t innerConeSlot(uint32_t i)  { return MAX_OBJECTS + MAX_POINT_LIGHTS + MAX_SPOT_LIGHTS + i; }
constexpr uint32_t outerConeSlot(uint32_t i)  { return MAX_OBJECTS + MAX_POINT_LIGHTS + 2 * MAX_SPOT_LIGHTS + i; }

struct Mesh {
    VkBuffer vbuf = VK_NULL_HANDLE;
    VmaAllocation vbuf_alloc = VK_NULL_HANDLE;
    VkBuffer ibuf = VK_NULL_HANDLE;
    VmaAllocation ibuf_alloc = VK_NULL_HANDLE;
    uint32_t index_count = 0;
    uint32_t vertex_count = 0;
};

enum MeshId { MESH_SPHERE = 0, MESH_CUBE, MESH_PLANE, MESH_CONE_LINES, MESH_COUNT };

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t mip_levels = 1;
};

struct ShadowMap {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView array_view = VK_NULL_HANDLE;
    std::vector<VkImageView> layer_views;
    uint32_t size = 0;
    uint32_t layers = 0;
};

constexpr VkFormat COLOR_TEXTURE_FORMAT = VK_FORMAT_R8G8B8A8_SRGB;
constexpr VkFormat DATA_TEXTURE_FORMAT  = VK_FORMAT_R8G8B8A8_UNORM;

constexpr const char* TEX_PATTERN = "textures/pattern.jpg";
constexpr const char* TEX_RINGS = "textures/rings.jpg";

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

constexpr uint32_t FLAG_UV_WAVE = 1u << 0;
constexpr uint32_t FLAG_PIXEL = 1u << 1;

struct PushConstants {
    uint32_t flags;
    float time;
    float wave_amplitude;
    float pad;
};
static_assert(sizeof(PushConstants) == 16, "push constants layout mismatch");

struct Material {
    std::string name;
    int albedo = 0;
    int specular = 0;
    int emissive = 0;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct Resources {
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    VkDescriptorSetLayout desc_layout_scene = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_model = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_material = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    Mesh meshes[MESH_COUNT];

    VkBuffer scene_ubuf = VK_NULL_HANDLE;
    VmaAllocation scene_ubuf_alloc = VK_NULL_HANDLE;
    void* scene_ubuf_mapped = nullptr;
    VkDescriptorSet desc_set_scene = VK_NULL_HANDLE;

    VkBuffer model_ubuf = VK_NULL_HANDLE;
    VmaAllocation model_ubuf_alloc = VK_NULL_HANDLE;
    uint8_t* model_ubuf_mapped = nullptr;
    VkDescriptorSet desc_set_model = VK_NULL_HANDLE;

    VkBuffer point_light_buf = VK_NULL_HANDLE;
    VmaAllocation point_light_alloc = VK_NULL_HANDLE;
    void* point_light_mapped = nullptr;

    VkBuffer spot_light_buf = VK_NULL_HANDLE;
    VmaAllocation spot_light_alloc = VK_NULL_HANDLE;
    void* spot_light_mapped = nullptr;

    VkPipelineLayout light_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline light_pipeline = VK_NULL_HANDLE;

    VkPipelineLayout line_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline line_pipeline = VK_NULL_HANDLE;

    ShadowMap sun_shadow;
    ShadowMap spot_shadow;
    ShadowMap point_shadow;
    VkSampler shadow_sampler = VK_NULL_HANDLE;
    VkPipelineLayout shadow_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline = VK_NULL_HANDLE;
    VkBuffer shadow_matrix_buf = VK_NULL_HANDLE;
    VmaAllocation shadow_matrix_alloc = VK_NULL_HANDLE;
    void* shadow_matrix_mapped = nullptr;

    VkSampler sampler = VK_NULL_HANDLE;
    std::vector<Texture> textures;
    int black_texture = -1;
    std::vector<Material> materials;

    VkCommandPool upload_pool = VK_NULL_HANDLE;
};
Resources res;

struct SceneUniformData {
    glm::mat4 view_projection;
    glm::mat4 shadow_projection;
    glm::vec3 camera_position;
    uint32_t point_light_count;
    glm::vec3 directional_light_direction;
    uint32_t spot_light_count;
    glm::vec3 directional_light_color;
    uint32_t sun_shadow_enabled;
    glm::vec3 ambient_light;
    uint32_t spot_shadow_mask;
    uint32_t point_shadow_mask;
    uint32_t pad5;
    uint32_t pad6;
    uint32_t pad7;
};
static_assert(sizeof(SceneUniformData) == 208, "scene UBO layout mismatch");

struct ShadowMatricesData {
    glm::mat4 spot[MAX_SPOT_LIGHTS];
    glm::mat4 point[MAX_POINT_SHADOWS * POINT_FACES];
};
static_assert(MAX_SPOT_LIGHTS == 8 && MAX_POINT_SHADOWS == 4, "update array sizes in sphere.frag");
static_assert(sizeof(ShadowMatricesData) == 32 * 64, "shadow matrices layout mismatch");

struct ModelUniformData {
    glm::mat4 model;
    glm::vec3 albedo;
    float alpha;
    glm::vec3 specular;
    float emissive_strength;
};
static_assert(sizeof(ModelUniformData) == 96, "model UBO layout mismatch");
static_assert(sizeof(ModelUniformData) <= DYN_STRIDE, "model UBO must fit into dynamic stride");

struct PointLight {
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
    float quadratic;
};
static_assert(sizeof(PointLight) == 32, "PointLight layout mismatch");

struct SpotLight {
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
    float quadratic;
    glm::vec3 direction;
    float inner_cos;
    float outer_cos;
    float pad1;
    float pad2;
    float pad3;
};
static_assert(sizeof(SpotLight) == 64, "SpotLight layout mismatch");

float g_fov = 60.0f;
float g_time = 0.0f;

glm::vec3 g_camera_position = {0.0f, 0.0f, 5.0f};
float g_camera_yaw = -90.0f;
float g_camera_pitch = -5.0f;

glm::vec3 g_light_direction = {-0.5f, -1.0f, -0.3f};
glm::vec3 g_light_color = {1.0f, 1.0f, 1.0f};
float g_light_intensity = 0.6f;

glm::vec3 g_ambient_color = {1.0f, 1.0f, 1.0f};
float g_ambient_intensity = 0.08f;

int g_point_light_count = 0;

struct PointLightParams {
    glm::vec3 position = {2.0f, 1.0f, 0.0f};
    glm::vec3 color = {1.0f, 0.7f, 0.3f};
    float intensity = 1.0f;
    float quadratic = 0.032f;
    bool cast_shadow = true;
};
PointLightParams g_point_lights[MAX_POINT_LIGHTS];

int g_spot_light_count = 0;

struct SpotLightParams {
    glm::vec3 position = {3.0f, 2.0f, 3.0f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    glm::vec3 direction = {-1.0f, -1.0f, -1.0f};
    float inner_angle = 15.0f;
    float outer_angle = 25.0f;
    float quadratic = 0.032f;
    bool cast_shadow = true;
};
SpotLightParams g_spot_lights[MAX_SPOT_LIGHTS];

bool g_show_light_visualization = true;
float g_cone_length = 2.0f;

bool  g_sun_shadow = true;
float g_shadow_area = 24.0f;
float g_sun_bias_constant = 2.0f;
float g_sun_bias_slope = 2.0f;
float g_local_bias_constant = 2.0f;
float g_local_bias_slope = 2.0f;

struct ShadowFrame {
    glm::mat4 sun = glm::mat4(1.0f);
    glm::mat4 spot[MAX_SPOT_LIGHTS];
    glm::mat4 point[MAX_POINT_SHADOWS * POINT_FACES];
    uint32_t spot_mask = 0;
    uint32_t point_mask = 0;
};
ShadowFrame g_shadow;

struct SceneObject {
    std::string name;
    int mesh = MESH_SPHERE;
    glm::vec3 position = {0, 0, 0};
    glm::vec3 rotation = {0, 0, 0};
    glm::vec3 scale = {1, 1, 1};
    int material = 0;

    glm::vec3 tint = {1, 1, 1};
    glm::vec3 specular = {1, 1, 1};
    float shininess = 32.0f;
    float emissive_strength = 1.0f;

    bool fx_wave = false;
    bool fx_dual = false;
    float wave_amplitude = 0.03f;
};
std::vector<SceneObject> g_objects;
PushConstants g_push[MAX_OBJECTS];

void generateSphere(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, int stacks, int slices, float radius) {
    verts.clear();
    indices.clear();
    verts.reserve((stacks + 1) * (slices + 1));
    indices.reserve(stacks * slices * 6);

    for (int i = 0; i <= stacks; ++i) {
        const float v = float(i) / float(stacks);
        const float phi = v * PI;
        for (int j = 0; j <= slices; ++j) {
            const float u = float(j) / float(slices);
            const float theta = u * 2.0f * PI;
            Vertex vert;
            vert.pos = { std::sin(phi) * std::cos(theta) * radius,
                         std::cos(phi) * radius,
                         std::sin(phi) * std::sin(theta) * radius };
            vert.normal = glm::normalize(vert.pos);
            vert.uv = { u, v };
            verts.push_back(vert);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const uint32_t a = uint32_t(i * (slices + 1) + j);
            const uint32_t b = a + uint32_t(slices + 1);

            indices.push_back(a);
            indices.push_back(a + 1);
            indices.push_back(b);

            indices.push_back(a + 1);
            indices.push_back(b + 1);
            indices.push_back(b);
        }
    }
}

void generateCube(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, float size) {
    verts.clear();
    indices.clear();
    const float s = size;
    verts = {
        {{-s, -s, -s}, {0,0,1}, {0,0}},
        {{ s, -s, -s}, {0,0,1}, {0,0}},
        {{ s,  s, -s}, {0,0,1}, {0,0}},
        {{-s,  s, -s}, {0,0,1}, {0,0}},
        {{-s, -s,  s}, {0,0,1}, {0,0}},
        {{ s, -s,  s}, {0,0,1}, {0,0}},
        {{ s,  s,  s}, {0,0,1}, {0,0}},
        {{-s,  s,  s}, {0,0,1}, {0,0}},
    };

    indices = {
        0, 1, 2,  2, 3, 0,
        4, 6, 5,  6, 7, 4,
        0, 3, 7,  7, 4, 0,
        1, 5, 6,  6, 2, 1,
        0, 4, 5,  5, 1, 0,
        3, 2, 6,  6, 7, 3,
    };
}

void generatePlane(std::vector<Vertex>& verts, std::vector<uint32_t>& indices, float half_size, float uv_repeat) {
    const float s = half_size;
    const float t = uv_repeat;
    verts = {
        {{-s, 0.0f,  s}, {0, 1, 0}, {0, 0}},
        {{ s, 0.0f,  s}, {0, 1, 0}, {t, 0}},
        {{ s, 0.0f, -s}, {0, 1, 0}, {t, t}},
        {{-s, 0.0f, -s}, {0, 1, 0}, {0, t}},
    };

    indices = { 0, 1, 2,  0, 2, 3 };
}

void generateConeLines(std::vector<Vertex>& verts, int segments, float angle_rad) {
    verts.clear();

    const float height = 1.0f;
    const float radius = std::tan(angle_rad) * height;
    const glm::vec3 apex = { 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < segments; ++i) {
        const float theta = 2.0f * PI * float(i) / float(segments);
        const glm::vec3 p = { radius * std::cos(theta), height, radius * std::sin(theta) };
        verts.push_back({ apex, {0,0,0}, {0,0} });
        verts.push_back({ p,    {0,0,0}, {0,0} });
    }

    for (int i = 0; i < segments; ++i) {
        const float theta1 = 2.0f * PI * float(i) / float(segments);
        const float theta2 = 2.0f * PI * float(i + 1) / float(segments);
        const glm::vec3 p1 = { radius * std::cos(theta1), height, radius * std::sin(theta1) };
        const glm::vec3 p2 = { radius * std::cos(theta2), height, radius * std::sin(theta2) };
        verts.push_back({ p1, {0,0,0}, {0,0} });
        verts.push_back({ p2, {0,0,0}, {0,0} });
    }
}

VkQueue graphicsQueue() { return graphics::internal::context.graphics_queue; }
uint32_t graphicsQueueFamily() { return graphics::internal::context.graphics_queue_index; }

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage,
                  VkBuffer& out_buf, VmaAllocation& out_alloc, void** out_mapped = nullptr) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = memory_usage;
    if (out_mapped) {
        ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(graphics::internal::context.allocator,
                        &bi, &ai, &out_buf, &out_alloc, &info) != VK_SUCCESS) {
        return false;
    }
    if (out_mapped) *out_mapped = info.pMappedData;
    return true;
}

bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkBuffer& out_buf, VmaAllocation& out_alloc, void** out_mapped) {
    return createBuffer(size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU, out_buf, out_alloc, out_mapped);
}

VkCommandBuffer beginSingleTimeCommands() {
    auto& ctx = graphics::internal::context;

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool = res.upload_pool;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void endSingleTimeCommands(VkCommandBuffer cmd) {
    auto& ctx = graphics::internal::context;

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue());

    vkFreeCommandBuffers(ctx.device, res.upload_pool, 1, &cmd);
}

bool createUploadPool() {
    auto& ctx = graphics::internal::context;

    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pi.queueFamilyIndex = graphicsQueueFamily();
    return vkCreateCommandPool(ctx.device, &pi, nullptr, &res.upload_pool) == VK_SUCCESS;
}

bool createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                             VkBuffer& out_buf, VmaAllocation& out_alloc) {
    auto& ctx = graphics::internal::context;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                      staging, staging_alloc, &mapped)) return false;
    std::memcpy(mapped, data, size_t(size));

    if (!createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                      out_buf, out_alloc)) {
        vmaDestroyBuffer(ctx.allocator, staging, staging_alloc);
        return false;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cmd, staging, out_buf, 1, &region);
    endSingleTimeCommands(cmd);

    vmaDestroyBuffer(ctx.allocator, staging, staging_alloc);
    return true;
}

bool createMesh(Mesh& mesh, const std::vector<Vertex>& verts, const std::vector<uint32_t>& indices) {
    mesh.vertex_count = uint32_t(verts.size());
    mesh.index_count = uint32_t(indices.size());

    if (!createDeviceLocalBuffer(verts.data(), verts.size() * sizeof(Vertex),
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vbuf, mesh.vbuf_alloc))
        return false;

    if (!indices.empty()) {
        if (!createDeviceLocalBuffer(indices.data(), indices.size() * sizeof(uint32_t),
                                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.ibuf, mesh.ibuf_alloc))
            return false;
    }
    return true;
}

void imageBarrier(VkCommandBuffer cmd, VkImage image,
                  VkImageLayout old_layout, VkImageLayout new_layout,
                  VkAccessFlags src_access, VkAccessFlags dst_access,
                  VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                  uint32_t base_mip, uint32_t mip_count) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, base_mip, mip_count, 0, 1 };
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

int createTexture(const uint8_t* rgba, uint32_t width, uint32_t height, VkFormat format, bool gen_mips) {
    auto& ctx = graphics::internal::context;

    Texture tex;
    tex.mip_levels = gen_mips
        ? uint32_t(std::floor(std::log2(float(std::max(width, height))))) + 1
        : 1;

    const VkDeviceSize size = VkDeviceSize(width) * height * 4;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                      staging, staging_alloc, &mapped)) return -1;
    std::memcpy(mapped, rgba, size_t(size));

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = { width, height, 1 };
    ii.mipLevels = tex.mip_levels;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(ctx.allocator, &ii, &ai, &tex.image, &tex.alloc, nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(ctx.allocator, staging, staging_alloc);
        return -1;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();

    imageBarrier(cmd, tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 0, tex.mip_levels);

    VkBufferImageCopy copy{};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(cmd, staging, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    int32_t mip_w = int32_t(width);
    int32_t mip_h = int32_t(height);
    for (uint32_t i = 1; i < tex.mip_levels; ++i) {
        imageBarrier(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, i - 1, 1);

        const int32_t next_w = std::max(1, mip_w / 2);
        const int32_t next_h = std::max(1, mip_h / 2);

        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
        blit.srcOffsets[1] = { mip_w, mip_h, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
        blit.dstOffsets[1] = { next_w, next_h, 1 };
        vkCmdBlitImage(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        imageBarrier(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, i - 1, 1);

        mip_w = next_w;
        mip_h = next_h;
    }

    imageBarrier(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 tex.mip_levels - 1, 1);

    endSingleTimeCommands(cmd);
    vmaDestroyBuffer(ctx.allocator, staging, staging_alloc);

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = tex.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, tex.mip_levels, 0, 1 };
    if (vkCreateImageView(ctx.device, &vi, nullptr, &tex.view) != VK_SUCCESS) {
        vmaDestroyImage(ctx.allocator, tex.image, tex.alloc);
        return -1;
    }

    res.textures.push_back(tex);
    return int(res.textures.size()) - 1;
}

bool createSampler() {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.minLod = 0.0f;
    si.maxLod = VK_LOD_CLAMP_NONE;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    return vkCreateSampler(graphics::internal::context.device, &si, nullptr, &res.sampler) == VK_SUCCESS;
}

bool createShadowSampler() {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    return vkCreateSampler(graphics::internal::context.device, &si, nullptr, &res.shadow_sampler) == VK_SUCCESS;
}

bool createShadowMap(ShadowMap& sm, uint32_t size, uint32_t layers, bool as_array) {
    auto& ctx = graphics::internal::context;
    sm.size = size;
    sm.layers = layers;

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = SHADOW_FORMAT;
    ii.extent = { size, size, 1 };
    ii.mipLevels = 1;
    ii.arrayLayers = layers;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;

    ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(ctx.allocator, &ii, &ai, &sm.image, &sm.alloc, nullptr) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = sm.image;
    vi.viewType = as_array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = SHADOW_FORMAT;
    vi.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers };
    if (vkCreateImageView(ctx.device, &vi, nullptr, &sm.array_view) != VK_SUCCESS) return false;

    sm.layer_views.assign(layers, VK_NULL_HANDLE);
    for (uint32_t l = 0; l < layers; ++l) {
        VkImageViewCreateInfo lv = vi;
        lv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        lv.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, l, 1 };
        if (vkCreateImageView(ctx.device, &lv, nullptr, &sm.layer_views[l]) != VK_SUCCESS) return false;
    }
    return true;
}

void destroyShadowMap(ShadowMap& sm) {
    auto& ctx = graphics::internal::context;
    for (VkImageView v : sm.layer_views) {
        if (v) vkDestroyImageView(ctx.device, v, nullptr);
    }
    if (sm.array_view) vkDestroyImageView(ctx.device, sm.array_view, nullptr);
    if (sm.image) vmaDestroyImage(ctx.allocator, sm.image, sm.alloc);
    sm = ShadowMap{};
}

bool createShadowResources() {
    if (!createShadowSampler()) return false;
    if (!createShadowMap(res.sun_shadow, SUN_SHADOW_SIZE, 1, false)) return false;
    if (!createShadowMap(res.spot_shadow, SPOT_SHADOW_SIZE, MAX_SPOT_LIGHTS, true)) return false;
    if (!createShadowMap(res.point_shadow, POINT_SHADOW_SIZE, MAX_POINT_SHADOWS * POINT_FACES, true)) return false;
    return true;
}

void depthImageBarrier(VkCommandBuffer cmd, const ShadowMap& sm,
                       VkImageLayout old_layout, VkImageLayout new_layout,
                       VkAccessFlags src_access, VkAccessFlags dst_access,
                       VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = sm.image;
    b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, sm.layers };
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

std::vector<uint8_t> makeEmissiveMap(uint32_t w, uint32_t h, glm::vec3 glow, uint32_t seed) {
    std::vector<uint8_t> out(size_t(w) * h * 4);

    const float frequency = 6.0f;

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float fx = (float(x) / float(w)) * frequency;
            float fy = (float(y) / float(h)) * frequency;

            float shift = float(seed % 100) * 10.0f;
            float n = stb_perlin_noise3(fx + shift, fy + shift, 0.0f, 0, 0, 0);

            n = n * n;
            glm::vec3 c = glm::clamp(glm::vec3(n) * glow, 0.0f, 1.0f);

            const size_t i = (size_t(y) * w + x) * 4;
            out[i + 0] = uint8_t(c.r * 255.0f);
            out[i + 1] = uint8_t(c.g * 255.0f);
            out[i + 2] = uint8_t(c.b * 255.0f);
            out[i + 3] = 255;
        }
    }
    return out;
}

std::vector<uint8_t> makeCheckerboard(uint32_t size, uint32_t cell, glm::vec3 c0, glm::vec3 c1) {
    std::vector<uint8_t> out(size_t(size) * size * 4);
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const bool odd = ((x / cell) + (y / cell)) & 1;
            const glm::vec3 c = odd ? c1 : c0;
            const size_t i = (size_t(y) * size + x) * 4;
            out[i + 0] = uint8_t(c.r * 255.0f);
            out[i + 1] = uint8_t(c.g * 255.0f);
            out[i + 2] = uint8_t(c.b * 255.0f);
            out[i + 3] = 255;
        }
    }
    return out;
}

struct Image {
    std::vector<uint8_t> pixels;
    uint32_t width = 0, height = 0;
};

Image loadImage(const char* path) {
    Image img;
    int w = 0, h = 0, channels = 0;
    stbi_uc* data = stbi_load(path, &w, &h, &channels, STBI_rgb_alpha);
    if (!data) {
        std::cerr << "Cannot load image: " << path << " (" << stbi_failure_reason() << ")\n";
        img.width = img.height = 64;
        img.pixels = makeCheckerboard(64, 8, {1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
        return img;
    }
    img.width = uint32_t(w);
    img.height = uint32_t(h);
    img.pixels.assign(data, data + size_t(w) * h * 4);
    stbi_image_free(data);
    return img;
}

bool addFileMaterial(const char* name, const char* albedo_path, const char* spec_path, glm::vec3 glow, uint32_t seed) {
    const Image imgA = loadImage(albedo_path);
    const int albedo = createTexture(imgA.pixels.data(), imgA.width, imgA.height, COLOR_TEXTURE_FORMAT, true);

    const Image imgS = loadImage(spec_path);
    const int spec = createTexture(imgS.pixels.data(), imgS.width, imgS.height, DATA_TEXTURE_FORMAT, true);

    const auto emis_px = makeEmissiveMap(imgA.width, imgA.height, glow, seed);
    const int emis = createTexture(emis_px.data(), imgA.width, imgA.height, COLOR_TEXTURE_FORMAT, true);

    if (albedo < 0 || spec < 0 || emis < 0) return false;

    Material m;
    m.name = name;
    m.albedo = albedo;
    m.specular = spec;
    m.emissive = emis;
    res.materials.push_back(m);
    return true;
}

bool createMaterials() {
    const uint8_t black[4] = { 0, 0, 0, 255 };
    res.black_texture = createTexture(black, 1, 1, DATA_TEXTURE_FORMAT, false);
    if (res.black_texture < 0) return false;

    if (!addFileMaterial("Pattern", TEX_PATTERN, "textures/pattern_spec.png", {1.0f, 0.45f, 0.1f}, 1337)) return false;
    if (!addFileMaterial("Saturn rings", TEX_RINGS, "textures/rings_spec.png", {0.1f, 0.8f, 1.0f}, 4242)) return false;
    {
        const auto px = makeCheckerboard(256, 32, {0.85f, 0.85f, 0.85f}, {0.25f, 0.28f, 0.35f});
        const int albedo = createTexture(px.data(), 256, 256, COLOR_TEXTURE_FORMAT, true);
        if (albedo < 0) return false;

        Material m;
        m.name = "Checker (no spec/emissive)";
        m.albedo = albedo;
        m.specular = res.black_texture;
        m.emissive = res.black_texture;
        res.materials.push_back(m);
    }

    auto& ctx = graphics::internal::context;
    for (Material& m : res.materials) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &res.desc_layout_material;
        if (vkAllocateDescriptorSets(ctx.device, &ai, &m.set) != VK_SUCCESS) return false;

        const int tex_ids[3] = { m.albedo, m.specular, m.emissive };
        VkDescriptorImageInfo infos[3]{};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t b = 0; b < 3; ++b) {
            infos[b].sampler = res.sampler;
            infos[b].imageView = res.textures[tex_ids[b]].view;
            infos[b].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = m.set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[b].pImageInfo = &infos[b];
        }
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
    }
    return true;
}

bool createDescriptors() {
    auto& ctx = graphics::internal::context;

    {
        VkDescriptorSetLayoutBinding bindings[7]{};

        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        for (uint32_t b = 4; b <= 6; ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 7;
        li.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_scene) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &binding;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_model) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 3;
        li.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_material) != VK_SUCCESS)
            return false;
    }

    VkDescriptorPoolSize pool_sizes[4]{};
    pool_sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    pool_sizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1 };
    pool_sizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    pool_sizes[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * MAX_MATERIALS + 3 };

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = 2 + MAX_MATERIALS;
    pi.poolSizeCount = 4;
    pi.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(ctx.device, &pi, nullptr, &res.desc_pool) != VK_SUCCESS)
        return false;

    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &res.desc_layout_scene;
        if (vkAllocateDescriptorSets(ctx.device, &ai, &res.desc_set_scene) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &res.desc_layout_model;
        if (vkAllocateDescriptorSets(ctx.device, &ai, &res.desc_set_model) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorBufferInfo ubo_bi{};
        ubo_bi.buffer = res.scene_ubuf;
        ubo_bi.range = sizeof(SceneUniformData);

        VkDescriptorBufferInfo point_bi{};
        point_bi.buffer = res.point_light_buf;
        point_bi.range = sizeof(PointLight) * MAX_POINT_LIGHTS;

        VkDescriptorBufferInfo spot_bi{};
        spot_bi.buffer = res.spot_light_buf;
        spot_bi.range = sizeof(SpotLight) * MAX_SPOT_LIGHTS;

        VkDescriptorBufferInfo shadow_mat_bi{};
        shadow_mat_bi.buffer = res.shadow_matrix_buf;
        shadow_mat_bi.range = sizeof(ShadowMatricesData);

        VkDescriptorImageInfo shadow_ii[3]{};
        const ShadowMap* shadow_maps[3] = { &res.sun_shadow, &res.spot_shadow, &res.point_shadow };
        for (int k = 0; k < 3; ++k) {
            shadow_ii[k].sampler = res.shadow_sampler;
            shadow_ii[k].imageView = shadow_maps[k]->array_view;
            shadow_ii[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkWriteDescriptorSet writes[7]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 0, 0, 1,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &ubo_bi, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 1, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &point_bi, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 2, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spot_bi, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 3, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadow_mat_bi, nullptr };
        for (uint32_t k = 0; k < 3; ++k) {
            writes[4 + k] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                              res.desc_set_scene, 4 + k, 0, 1,
                              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadow_ii[k], nullptr, nullptr };
        }
        vkUpdateDescriptorSets(ctx.device, 7, writes, 0, nullptr);
    }

    {
        VkDescriptorBufferInfo bi{};
        bi.buffer = res.model_ubuf;
        bi.offset = 0;
        bi.range = sizeof(ModelUniformData);

        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                res.desc_set_model, 0, 0, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &bi, nullptr };
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    return true;
}

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open shader: " << path << "\n";
        return VK_NULL_HANDLE;
    }
    const size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };

    VkShaderModule result;
    if (vkCreateShaderModule(graphics::internal::context.device,
                             &info, nullptr, &result) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << "\n";
        return VK_NULL_HANDLE;
    }
    return result;
}

struct PipelineDesc {
    const char* vert_path;
    const char* frag_path;
    uint32_t attribute_count;
    VkPrimitiveTopology topology;
    VkCullModeFlags cull_mode;
    uint32_t set_layout_count;
    bool use_push_constants;
};

bool createPipelineImpl(const PipelineDesc& d, VkPipelineLayout& out_layout, VkPipeline& out_pipeline) {
    auto& ctx = graphics::internal::context;

    VkShaderModule vs = loadShaderModule(d.vert_path);
    VkShaderModule fs = loadShaderModule(d.frag_path);
    if (!vs || !fs) return false;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr };

    VkVertexInputBindingDescription vib{};
    vib.binding = 0;
    vib.stride = sizeof(Vertex);
    vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription via[3]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
    via[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };
    via[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = d.attribute_count;
    vi.pVertexAttributeDescriptions = via;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = d.topology;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = d.cull_mode;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    cba.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyn_states;

    VkDescriptorSetLayout set_layouts[] = { res.desc_layout_scene, res.desc_layout_model, res.desc_layout_material };

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = d.set_layout_count;
    pli.pSetLayouts = set_layouts;
    if (d.use_push_constants) {
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
    }
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &out_layout) != VK_SUCCESS)
        return false;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dyn;
    gp.layout = out_layout;
    gp.renderPass = ctx.render_pass;
    gp.subpass = 0;

    const VkResult r = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gp, nullptr, &out_pipeline);
    vkDestroyShaderModule(ctx.device, vs, nullptr);
    vkDestroyShaderModule(ctx.device, fs, nullptr);
    return r == VK_SUCCESS;
}

bool createPipeline() {
    return createPipelineImpl({ "shaders/sphere.vert.spv", "shaders/sphere.frag.spv",
                                3, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT,
                                3, true },
                              res.pipeline_layout, res.pipeline);
}

bool createLightPipeline() {
    return createPipelineImpl({ "shaders/light.vert.spv", "shaders/light.frag.spv",
                                1, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_NONE,
                                2, false },
                              res.light_pipeline_layout, res.light_pipeline);
}

bool createLinePipeline() {
    return createPipelineImpl({ "shaders/light.vert.spv", "shaders/light.frag.spv",
                                1, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE,
                                2, false },
                              res.line_pipeline_layout, res.line_pipeline);
}

bool createShadowPipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vs = loadShaderModule("shaders/shadow.vert.spv");
    if (!vs) return false;

    VkPipelineShaderStageCreateInfo stage{};
    stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
              VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr };

    VkVertexInputBindingDescription vib{};
    vib.binding = 0;
    vib.stride = sizeof(Vertex);
    vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription via{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &via;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;

    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthBiasEnable = VK_TRUE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 0;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                    VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 3;
    dyn.pDynamicStates = dyn_states;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &res.desc_layout_model;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &res.shadow_pipeline_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(ctx.device, vs, nullptr);
        return false;
    }

    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 0;
    rendering_info.depthAttachmentFormat = SHADOW_FORMAT;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.pNext = &rendering_info;
    gp.stageCount = 1;
    gp.pStages = &stage;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dyn;
    gp.layout = res.shadow_pipeline_layout;
    gp.renderPass = VK_NULL_HANDLE;

    const VkResult r = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gp, nullptr, &res.shadow_pipeline);
    vkDestroyShaderModule(ctx.device, vs, nullptr);
    return r == VK_SUCCESS;
}

bool setupGeometry() {
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;

    generateSphere(verts, idx, 20, 20, 1.0f);
    if (!createMesh(res.meshes[MESH_SPHERE], verts, idx)) return false;

    generateCube(verts, idx, 0.5f);
    if (!createMesh(res.meshes[MESH_CUBE], verts, idx)) return false;

    generatePlane(verts, idx, 10.0f, 8.0f);
    if (!createMesh(res.meshes[MESH_PLANE], verts, idx)) return false;

    generateConeLines(verts, 32, PI / 4.0f);
    if (!createMesh(res.meshes[MESH_CONE_LINES], verts, {})) return false;

    if (!createHostBuffer(sizeof(SceneUniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          res.scene_ubuf, res.scene_ubuf_alloc, &res.scene_ubuf_mapped)) return false;

    void* model_mapped = nullptr;
    if (!createHostBuffer(VkDeviceSize(DYN_STRIDE) * MAX_DRAW_SLOTS, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          res.model_ubuf, res.model_ubuf_alloc, &model_mapped)) return false;
    res.model_ubuf_mapped = static_cast<uint8_t*>(model_mapped);
    std::memset(res.model_ubuf_mapped, 0, size_t(DYN_STRIDE) * MAX_DRAW_SLOTS);

    if (!createHostBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          res.point_light_buf, res.point_light_alloc, &res.point_light_mapped)) return false;

    if (!createHostBuffer(sizeof(SpotLight) * MAX_SPOT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          res.spot_light_buf, res.spot_light_alloc, &res.spot_light_mapped)) return false;

    if (!createHostBuffer(sizeof(ShadowMatricesData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          res.shadow_matrix_buf, res.shadow_matrix_alloc, &res.shadow_matrix_mapped)) return false;
    std::memset(res.shadow_matrix_mapped, 0, sizeof(ShadowMatricesData));

    return true;
}

void setupScene() {
    g_objects.clear();

    {
        SceneObject o;
        o.name = "Sphere";
        o.mesh = MESH_SPHERE;
        o.position = { -1.5f, 0.0f, 0.0f };
        o.material = 0;
        g_objects.push_back(o);
    }

    {
        SceneObject o;
        o.name = "Animated sphere";
        o.mesh = MESH_SPHERE;
        o.position = { 1.5f, 0.0f, 0.0f };
        o.material = 1;
        o.fx_wave = true;
        g_objects.push_back(o);
    }

    {

        SceneObject o;
        o.name = "Floor";
        o.mesh = MESH_PLANE;
        o.position = { 0.0f, -1.5f, 0.0f };
        o.material = 2;
        g_objects.push_back(o);
    }
}

void destroyMesh(Mesh& m) {
    auto& ctx = graphics::internal::context;
    if (m.vbuf) vmaDestroyBuffer(ctx.allocator, m.vbuf, m.vbuf_alloc);
    if (m.ibuf) vmaDestroyBuffer(ctx.allocator, m.ibuf, m.ibuf_alloc);
    m = Mesh{};
}

void writeModelSlot(uint32_t slot, const ModelUniformData& data) {
    std::memcpy(res.model_ubuf_mapped + size_t(slot) * DYN_STRIDE, &data, sizeof(data));
}

}

bool initialize() {
    if (!createUploadPool()) { std::cerr << "Failed to create upload command pool\n"; return false; }
    if (!createSampler()) { std::cerr << "Failed to create sampler\n"; return false; }
    if (!setupGeometry()) { std::cerr << "Failed to setup geometry\n"; return false; }
    if (!createShadowResources()) { std::cerr << "Failed to create shadow resources\n"; return false; }
    if (!createDescriptors()) { std::cerr << "Failed to create descriptors\n"; return false; }
    if (!createMaterials()) { std::cerr << "Failed to create materials\n"; return false; }
    if (!createPipeline()) { std::cerr << "Failed to create pipeline\n"; return false; }
    if (!createLightPipeline()) { std::cerr << "Failed to create light pipeline\n"; return false; }
    if (!createLinePipeline()) { std::cerr << "Failed to create line pipeline\n"; return false; }
    if (!createShadowPipeline()) { std::cerr << "Failed to create shadow pipeline\n"; return false; }
    setupScene();
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);

    if (res.pipeline) vkDestroyPipeline(ctx.device, res.pipeline, nullptr);
    if (res.pipeline_layout) vkDestroyPipelineLayout(ctx.device, res.pipeline_layout, nullptr);
    if (res.light_pipeline) vkDestroyPipeline(ctx.device, res.light_pipeline, nullptr);
    if (res.light_pipeline_layout) vkDestroyPipelineLayout(ctx.device, res.light_pipeline_layout, nullptr);
    if (res.line_pipeline) vkDestroyPipeline(ctx.device, res.line_pipeline, nullptr);
    if (res.line_pipeline_layout) vkDestroyPipelineLayout(ctx.device, res.line_pipeline_layout, nullptr);
    if (res.shadow_pipeline) vkDestroyPipeline(ctx.device, res.shadow_pipeline, nullptr);
    if (res.shadow_pipeline_layout) vkDestroyPipelineLayout(ctx.device, res.shadow_pipeline_layout, nullptr);

    destroyShadowMap(res.sun_shadow);
    destroyShadowMap(res.spot_shadow);
    destroyShadowMap(res.point_shadow);
    if (res.shadow_sampler) vkDestroySampler(ctx.device, res.shadow_sampler, nullptr);
    if (res.shadow_matrix_buf) vmaDestroyBuffer(ctx.allocator, res.shadow_matrix_buf, res.shadow_matrix_alloc);

    if (res.desc_pool) vkDestroyDescriptorPool(ctx.device, res.desc_pool, nullptr);
    if (res.desc_layout_scene) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_scene, nullptr);
    if (res.desc_layout_model) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_model, nullptr);
    if (res.desc_layout_material) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_material, nullptr);

    if (res.sampler) vkDestroySampler(ctx.device, res.sampler, nullptr);
    for (Texture& t : res.textures) {
        if (t.view) vkDestroyImageView(ctx.device, t.view, nullptr);
        if (t.image) vmaDestroyImage(ctx.allocator, t.image, t.alloc);
    }
    res.textures.clear();
    res.materials.clear();

    if (res.scene_ubuf) vmaDestroyBuffer(ctx.allocator, res.scene_ubuf, res.scene_ubuf_alloc);
    if (res.model_ubuf) vmaDestroyBuffer(ctx.allocator, res.model_ubuf, res.model_ubuf_alloc);
    if (res.point_light_buf) vmaDestroyBuffer(ctx.allocator, res.point_light_buf, res.point_light_alloc);
    if (res.spot_light_buf) vmaDestroyBuffer(ctx.allocator, res.spot_light_buf, res.spot_light_alloc);

    for (Mesh& m : res.meshes) destroyMesh(m);

    if (res.upload_pool) vkDestroyCommandPool(ctx.device, res.upload_pool, nullptr);
}

static glm::vec3 shadowUp(const glm::vec3& dir) {
    return std::abs(dir.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
}

void update(double time) {
    g_time = float(time);

    vkDeviceWaitIdle(graphics::internal::context.device);

    ImGui::Begin("Shadow Mapping Lab");

    ImGui::Text("Projection");
    ImGui::SliderFloat("FOV", &g_fov, 20.0f, 120.0f);
    ImGui::Separator();

    ImGui::Text("Camera");
    ImGui::SliderFloat3("Position", &g_camera_position.x, -10.0f, 10.0f);
    ImGui::SliderFloat("Yaw", &g_camera_yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &g_camera_pitch, -89.0f, 89.0f);
    ImGui::Separator();

    std::vector<const char*> material_names;
    for (const Material& m : res.materials) material_names.push_back(m.name.c_str());

    ImGui::Text("Objects");
    for (size_t i = 0; i < g_objects.size(); ++i) {
        SceneObject& o = g_objects[i];
        ImGui::PushID(int(i));
        if (ImGui::CollapsingHeader(o.name.c_str())) {
            ImGui::SliderFloat3("Position", &o.position.x, -10.0f, 10.0f);
            ImGui::SliderFloat3("Rotation", &o.rotation.x, -180.0f, 180.0f);
            ImGui::SliderFloat3("Scale", &o.scale.x, 0.1f, 8.0f);

            ImGui::Combo("Material", &o.material, material_names.data(), int(material_names.size()));

            ImGui::ColorEdit3("Tint", &o.tint.x);
            ImGui::ColorEdit3("Specular multiplier", &o.specular.x);
            ImGui::SliderFloat("Shininess (alpha)", &o.shininess, 1.0f, 256.0f);
            ImGui::SliderFloat("Emissive strength", &o.emissive_strength, 0.0f, 4.0f);

            ImGui::Text("Non-trivial sampling (push constants)");
            ImGui::Checkbox("UV wave", &o.fx_wave);
            ImGui::Checkbox("Pixelation", &o.fx_dual);
            ImGui::SliderFloat("Wave amplitude", &o.wave_amplitude, 0.0f, 0.1f);
        }
        ImGui::PopID();
    }
    ImGui::Separator();

    ImGui::Text("Directional Light");
    ImGui::SliderFloat3("Direction", &g_light_direction.x, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light Color", &g_light_color.x);
    ImGui::SliderFloat("Intensity", &g_light_intensity, 0.0f, 5.0f);
    ImGui::Separator();

    ImGui::Text("Shadows");
    ImGui::Checkbox("Sun casts shadow", &g_sun_shadow);
    ImGui::SliderFloat("Shadow area (ortho size)", &g_shadow_area, 4.0f, 60.0f);
    ImGui::SliderFloat("Sun bias: constant (x100)", &g_sun_bias_constant, 0.0f, 50.0f);
    ImGui::SliderFloat("Sun bias: slope", &g_sun_bias_slope, 0.0f, 10.0f);
    ImGui::SliderFloat("Spot/Point bias: constant (x100)", &g_local_bias_constant, 0.0f, 50.0f);
    ImGui::SliderFloat("Spot/Point bias: slope", &g_local_bias_slope, 0.0f, 10.0f);
    ImGui::Separator();

    ImGui::Text("Point Lights (%d / %d)", g_point_light_count, MAX_POINT_LIGHTS);
    if (ImGui::Button("Add Point Light")) {
        if (g_point_light_count < (int)MAX_POINT_LIGHTS) {
            g_point_lights[g_point_light_count] = PointLightParams{};
            g_point_light_count++;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Point Light")) {
        if (g_point_light_count > 0) g_point_light_count--;
    }
    ImGui::Separator();

    for (int i = 0; i < g_point_light_count; ++i) {
        ImGui::PushID(i);
        std::string label = "Light " + std::to_string(i);
        if (ImGui::CollapsingHeader(label.c_str())) {
            ImGui::SliderFloat3("Position", &g_point_lights[i].position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &g_point_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &g_point_lights[i].intensity, 0.0f, 10.0f);
            ImGui::SliderFloat("Quadratic", &g_point_lights[i].quadratic, 0.0f, 1.0f);
            if (uint32_t(i) < MAX_POINT_SHADOWS) {
                ImGui::Checkbox("Cast shadow", &g_point_lights[i].cast_shadow);
            } else {
                ImGui::TextDisabled("Shadow: only first %u point lights", MAX_POINT_SHADOWS);
            }
        }
        ImGui::PopID();
    }
    ImGui::Separator();

    ImGui::Text("Spot Lights (%d / %d)", g_spot_light_count, MAX_SPOT_LIGHTS);
    if (ImGui::Button("Add Spot Light")) {
        if (g_spot_light_count < (int)MAX_SPOT_LIGHTS) {
            g_spot_lights[g_spot_light_count] = SpotLightParams{};
            g_spot_light_count++;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Spot Light")) {
        if (g_spot_light_count > 0) g_spot_light_count--;
    }
    ImGui::Separator();

    for (int i = 0; i < g_spot_light_count; ++i) {
        ImGui::PushID(1000 + i);
        std::string label = "Spot " + std::to_string(i);
        if (ImGui::CollapsingHeader(label.c_str())) {
            ImGui::SliderFloat3("Position", &g_spot_lights[i].position.x, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &g_spot_lights[i].color.x);
            ImGui::SliderFloat("Intensity", &g_spot_lights[i].intensity, 0.0f, 10.0f);
            ImGui::SliderFloat3("Direction", &g_spot_lights[i].direction.x, -1.0f, 1.0f);
            ImGui::SliderFloat("Inner Angle", &g_spot_lights[i].inner_angle, 1.0f, 89.0f);
            ImGui::SliderFloat("Outer Angle", &g_spot_lights[i].outer_angle, 1.0f, 89.0f);
            ImGui::SliderFloat("Quadratic", &g_spot_lights[i].quadratic, 0.0f, 1.0f);
            ImGui::Checkbox("Cast shadow", &g_spot_lights[i].cast_shadow);
        }
        ImGui::PopID();
    }
    ImGui::Separator();

    ImGui::Checkbox("Show Light Visualization", &g_show_light_visualization);
    ImGui::SliderFloat("Cone Length", &g_cone_length, 1.0f, 15.0f);
    ImGui::Separator();

    ImGui::Text("Ambient Light");
    ImGui::ColorEdit3("Ambient Color", &g_ambient_color.x);
    ImGui::SliderFloat("Ambient Intensity", &g_ambient_intensity, 0.0f, 1.0f);

    ImGui::Separator();
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::End();

    auto& ctx = graphics::internal::context;
    const float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);

    const float yaw_rad = glm::radians(g_camera_yaw);
    const float pitch_rad = glm::radians(g_camera_pitch);
    glm::vec3 forward;
    forward.x = std::cos(yaw_rad) * std::cos(pitch_rad);
    forward.y = std::sin(pitch_rad);
    forward.z = std::sin(yaw_rad) * std::cos(pitch_rad);
    forward = glm::normalize(forward);

    const glm::mat4 view = glm::lookAt(g_camera_position, g_camera_position + forward,
                                       glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
    proj[1][1] *= -1.0f;

    {

        const glm::vec3 sun_dir = glm::normalize(g_light_direction);
        const float half_size = g_shadow_area * 0.5f;
        const glm::vec3 target = g_camera_position;
        const glm::vec3 eye = target - sun_dir * g_shadow_area;

        const glm::mat4 sun_view = glm::lookAt(eye, target, shadowUp(sun_dir));
        glm::mat4 sun_proj = glm::ortho(-half_size, half_size, -half_size, half_size,
                                        0.0f, 2.0f * g_shadow_area);
        sun_proj[1][1] *= -1.0f;
        glm::mat4 sun_vp = sun_proj * sun_view;

        glm::vec4 origin = sun_vp * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        origin *= float(SUN_SHADOW_SIZE) * 0.5f;
        glm::vec4 offset = (glm::round(origin) - origin) * (2.0f / float(SUN_SHADOW_SIZE));
        offset.z = 0.0f;
        offset.w = 0.0f;
        sun_vp[3] += offset;
        g_shadow.sun = sun_vp;

        g_shadow.spot_mask = 0;
        for (int i = 0; i < g_spot_light_count && i < int(MAX_SPOT_LIGHTS); ++i) {
            const SpotLightParams& sl = g_spot_lights[i];
            const glm::vec3 dir = glm::normalize(sl.direction);
            const float fov = std::min(2.0f * sl.outer_angle + 4.0f, 170.0f);

            const glm::mat4 v = glm::lookAt(sl.position, sl.position + dir, shadowUp(dir));
            glm::mat4 p = glm::perspective(glm::radians(fov), 1.0f, LOCAL_SHADOW_NEAR, LOCAL_SHADOW_FAR);
            p[1][1] *= -1.0f;
            g_shadow.spot[i] = p * v;
            if (sl.cast_shadow) g_shadow.spot_mask |= (1u << i);
        }

        static const glm::vec3 face_dirs[POINT_FACES] = {
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1} };
        static const glm::vec3 face_ups[POINT_FACES] = {
            {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, 1, 0} };

        g_shadow.point_mask = 0;
        for (int i = 0; i < g_point_light_count && i < int(MAX_POINT_SHADOWS); ++i) {
            const PointLightParams& pl = g_point_lights[i];
            glm::mat4 p = glm::perspective(glm::radians(92.0f), 1.0f, LOCAL_SHADOW_NEAR, LOCAL_SHADOW_FAR);
            p[1][1] *= -1.0f;
            for (uint32_t f = 0; f < POINT_FACES; ++f) {
                const glm::mat4 v = glm::lookAt(pl.position, pl.position + face_dirs[f], face_ups[f]);
                g_shadow.point[uint32_t(i) * POINT_FACES + f] = p * v;
            }
            if (pl.cast_shadow) g_shadow.point_mask |= (1u << i);
        }

        ShadowMatricesData smd{};
        for (uint32_t i = 0; i < MAX_SPOT_LIGHTS; ++i) smd.spot[i] = g_shadow.spot[i];
        for (uint32_t i = 0; i < MAX_POINT_SHADOWS * POINT_FACES; ++i) smd.point[i] = g_shadow.point[i];
        std::memcpy(res.shadow_matrix_mapped, &smd, sizeof(smd));
    }

    SceneUniformData su{};
    su.view_projection = proj * view;
    su.shadow_projection = g_shadow.sun;
    su.sun_shadow_enabled = g_sun_shadow ? 1u : 0u;
    su.spot_shadow_mask = g_shadow.spot_mask;
    su.point_shadow_mask = g_shadow.point_mask;
    su.camera_position = g_camera_position;
    su.directional_light_direction = glm::normalize(g_light_direction);
    su.directional_light_color = g_light_color * g_light_intensity;
    su.ambient_light = g_ambient_color * g_ambient_intensity;
    su.point_light_count = uint32_t(g_point_light_count);
    su.spot_light_count = uint32_t(g_spot_light_count);
    std::memcpy(res.scene_ubuf_mapped, &su, sizeof(su));

    for (size_t i = 0; i < g_objects.size() && i < MAX_OBJECTS; ++i) {
        const SceneObject& o = g_objects[i];

        glm::mat4 model = glm::translate(glm::mat4(1.0f), o.position);
        model = glm::rotate(model, glm::radians(o.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(o.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(o.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, o.scale);

        ModelUniformData mu{};
        mu.model = model;
        mu.albedo = o.tint;
        mu.specular = o.specular;
        mu.alpha = o.shininess;
        mu.emissive_strength = o.emissive_strength;
        writeModelSlot(uint32_t(i), mu);

        PushConstants pc{};
        pc.flags = (o.fx_wave ? FLAG_UV_WAVE : 0u)
                 | (o.fx_dual ? FLAG_PIXEL : 0u);
        pc.time = g_time;
        pc.wave_amplitude = o.wave_amplitude;
        g_push[i] = pc;
    }

    PointLight lights[MAX_POINT_LIGHTS]{};
    for (int i = 0; i < g_point_light_count; ++i) {
        lights[i].position = g_point_lights[i].position;
        lights[i].intensity = g_point_lights[i].intensity;
        lights[i].color = g_point_lights[i].color;
        lights[i].quadratic = g_point_lights[i].quadratic;
    }
    std::memcpy(res.point_light_mapped, lights, sizeof(PointLight) * g_point_light_count);

    SpotLight spots[MAX_SPOT_LIGHTS]{};
    for (int i = 0; i < g_spot_light_count; ++i) {
        spots[i].position = g_spot_lights[i].position;
        spots[i].intensity = g_spot_lights[i].intensity;
        spots[i].color = g_spot_lights[i].color;
        spots[i].quadratic = g_spot_lights[i].quadratic;
        spots[i].direction = glm::normalize(g_spot_lights[i].direction);
        spots[i].inner_cos = std::cos(glm::radians(g_spot_lights[i].inner_angle));
        spots[i].outer_cos = std::cos(glm::radians(g_spot_lights[i].outer_angle));
    }
    std::memcpy(res.spot_light_mapped, spots, sizeof(SpotLight) * g_spot_light_count);

    for (int i = 0; i < g_point_light_count; ++i) {
        ModelUniformData mu{};
        mu.model = glm::translate(glm::mat4(1.0f), g_point_lights[i].position);
        mu.model = glm::scale(mu.model, glm::vec3(0.15f));
        mu.albedo = g_point_lights[i].color;
        writeModelSlot(pointSlot(uint32_t(i)), mu);
    }

    for (int i = 0; i < g_spot_light_count; ++i) {
        ModelUniformData mu{};
        mu.model = glm::translate(glm::mat4(1.0f), g_spot_lights[i].position);
        mu.model = glm::scale(mu.model, glm::vec3(0.1f));
        mu.albedo = g_spot_lights[i].color;
        writeModelSlot(spotSlot(uint32_t(i)), mu);

        const glm::vec3 dir = glm::normalize(g_spot_lights[i].direction);
        const glm::vec3 from = glm::vec3(0, 1, 0);
        const float cos_a = glm::clamp(glm::dot(from, dir), -1.0f, 1.0f);
        glm::mat4 rot = glm::mat4(1.0f);
        if (cos_a < 0.9999f) {
            if (cos_a > -0.9999f) {
                const glm::vec3 axis = glm::normalize(glm::cross(from, dir));
                rot = glm::rotate(glm::mat4(1.0f), std::acos(cos_a), axis);
            } else {
                rot = glm::rotate(glm::mat4(1.0f), PI, glm::vec3(1, 0, 0));
            }
        }

        struct ConeDraw { float angle_deg; uint32_t slot; glm::vec3 color; };
        const ConeDraw cones[2] = {
            { g_spot_lights[i].inner_angle, innerConeSlot(uint32_t(i)), glm::vec3(1.0f, 0.85f, 0.2f) },
            { g_spot_lights[i].outer_angle, outerConeSlot(uint32_t(i)), glm::vec3(1.0f, 1.0f, 1.0f)  },
        };

        for (const ConeDraw& c : cones) {
            const float radius = std::tan(glm::radians(c.angle_deg)) * g_cone_length;
            const glm::mat4 scale_m = glm::scale(glm::mat4(1.0f),
                                                 glm::vec3(radius, g_cone_length, radius));
            ModelUniformData cm{};
            cm.model = glm::translate(glm::mat4(1.0f), g_spot_lights[i].position) * rot * scale_m;
            cm.albedo = c.color;
            writeModelSlot(c.slot, cm);
        }
    }
}

static void drawSceneDepth(VkCommandBuffer cmd, const glm::mat4& light_vp) {
    vkCmdPushConstants(cmd, res.shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(glm::mat4), &light_vp);

    for (size_t i = 0; i < g_objects.size() && i < MAX_OBJECTS; ++i) {
        const Mesh& mesh = res.meshes[g_objects[i].mesh];

        const VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vbuf, &zero);
        vkCmdBindIndexBuffer(cmd, mesh.ibuf, 0, VK_INDEX_TYPE_UINT32);

        const uint32_t dynamic_offset = uint32_t(i) * DYN_STRIDE;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.shadow_pipeline_layout,
                                0, 1, &res.desc_set_model, 1, &dynamic_offset);
        vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
    }
}

static void renderShadowLayer(VkCommandBuffer cmd, const ShadowMap& sm, uint32_t layer,
                              const glm::mat4& light_vp, float bias_constant, float bias_slope) {
    VkRenderingAttachmentInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth.imageView = sm.layer_views[layer];
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { {0, 0}, {sm.size, sm.size} };
    ri.layerCount = 1;
    ri.pDepthAttachment = &depth;

    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{};
    vp.width = float(sm.size);
    vp.height = float(sm.size);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D sc{ {0, 0}, {sm.size, sm.size} };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdSetDepthBias(cmd, bias_constant * DEPTH_BIAS_CONSTANT_SCALE, 0.0f, bias_slope);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.shadow_pipeline);
    drawSceneDepth(cmd, light_vp);

    vkCmdEndRendering(cmd);
}

static void renderShadowMaps(VkCommandBuffer cmd) {
    ShadowMap* maps[3] = { &res.sun_shadow, &res.spot_shadow, &res.point_shadow };

    for (ShadowMap* m : maps) {
        depthImageBarrier(cmd, *m,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                          0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
    }

    if (g_sun_shadow) {
        renderShadowLayer(cmd, res.sun_shadow, 0, g_shadow.sun, g_sun_bias_constant, g_sun_bias_slope);
    }

    for (uint32_t i = 0; i < MAX_SPOT_LIGHTS; ++i) {
        if (g_shadow.spot_mask & (1u << i)) {
            renderShadowLayer(cmd, res.spot_shadow, i, g_shadow.spot[i],
                              g_local_bias_constant, g_local_bias_slope);
        }
    }

    for (uint32_t i = 0; i < MAX_POINT_SHADOWS; ++i) {
        if (g_shadow.point_mask & (1u << i)) {
            for (uint32_t f = 0; f < POINT_FACES; ++f) {
                renderShadowLayer(cmd, res.point_shadow, i * POINT_FACES + f,
                                  g_shadow.point[i * POINT_FACES + f],
                                  g_local_bias_constant, g_local_bias_slope);
            }
        }
    }

    for (ShadowMap* m : maps) {
        depthImageBarrier(cmd, *m,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;
    VkCommandBuffer cmd = fd.command_buffer;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    renderShadowMaps(cmd);

    VkClearValue clears[2]{};
    clears[0].color = {{ 0.05f, 0.05f, 0.10f, 1.0f }};
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = ctx.render_pass;
    rp.framebuffer = fd.framebuffer;
    rp.renderArea = { {0, 0}, ctx.swapchain_extent };
    rp.clearValueCount = 2;
    rp.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width = float(ctx.swapchain_extent.width);
    vp.height = float(ctx.swapchain_extent.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D sc{}; sc.extent = ctx.swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.pipeline_layout,
                            0, 1, &res.desc_set_scene, 0, nullptr);

    for (size_t i = 0; i < g_objects.size() && i < MAX_OBJECTS; ++i) {
        const SceneObject& o = g_objects[i];
        const Mesh& mesh = res.meshes[o.mesh];
        const Material& mat = res.materials[o.material];

        const VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vbuf, &zero);
        vkCmdBindIndexBuffer(cmd, mesh.ibuf, 0, VK_INDEX_TYPE_UINT32);

        const VkDescriptorSet sets[] = { res.desc_set_model, mat.set };
        const uint32_t dynamic_offset = uint32_t(i) * DYN_STRIDE;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.pipeline_layout,
                                1, 2, sets, 1, &dynamic_offset);

        vkCmdPushConstants(cmd, res.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &g_push[i]);

        vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
    }

    if (g_show_light_visualization) {
        const Mesh& cube = res.meshes[MESH_CUBE];

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.light_pipeline);
        const VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &cube.vbuf, &zero);
        vkCmdBindIndexBuffer(cmd, cube.ibuf, 0, VK_INDEX_TYPE_UINT32);

        auto drawCube = [&](uint32_t slot) {
            const VkDescriptorSet sets[] = { res.desc_set_scene, res.desc_set_model };
            const uint32_t dynamic_offset = slot * DYN_STRIDE;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.light_pipeline_layout,
                                    0, 2, sets, 1, &dynamic_offset);
            vkCmdDrawIndexed(cmd, cube.index_count, 1, 0, 0, 0);
        };

        for (int i = 0; i < g_point_light_count; ++i) drawCube(pointSlot(uint32_t(i)));
        for (int i = 0; i < g_spot_light_count; ++i)  drawCube(spotSlot(uint32_t(i)));

        if (g_spot_light_count > 0) {
            const Mesh& cone = res.meshes[MESH_CONE_LINES];
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.line_pipeline);
            vkCmdBindVertexBuffers(cmd, 0, 1, &cone.vbuf, &zero);

            for (int i = 0; i < g_spot_light_count; ++i) {
                const uint32_t slots[2] = { innerConeSlot(uint32_t(i)), outerConeSlot(uint32_t(i)) };
                for (uint32_t slot : slots) {
                    const VkDescriptorSet sets[] = { res.desc_set_scene, res.desc_set_model };
                    const uint32_t dynamic_offset = slot * DYN_STRIDE;
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, res.line_pipeline_layout,
                                            0, 2, sets, 1, &dynamic_offset);
                    vkCmdDraw(cmd, cone.vertex_count, 1, 0, 0);
                }
            }
        }
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

}