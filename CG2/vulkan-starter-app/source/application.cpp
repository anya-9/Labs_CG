#include "application.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>
#include <string>

#include <imgui.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

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
// слоты UBO: точечные кубики + кубики прожекторов + внутренние конусы + внешние конусы
constexpr uint32_t MAX_LIGHT_VIS = MAX_POINT_LIGHTS + 3 * MAX_SPOT_LIGHTS;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
};


struct Resources {
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    VkDescriptorSetLayout desc_layout_scene = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_model = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    VkBuffer vbuf = VK_NULL_HANDLE;
    VmaAllocation vbuf_alloc = VK_NULL_HANDLE;
    VkBuffer ibuf = VK_NULL_HANDLE;
    VmaAllocation ibuf_alloc = VK_NULL_HANDLE;
    uint32_t index_count = 0;

    // Scene UBO
    VkBuffer scene_ubuf = VK_NULL_HANDLE;
    VmaAllocation scene_ubuf_alloc = VK_NULL_HANDLE;
    void* scene_ubuf_mapped = nullptr;
    VkDescriptorSet desc_set_scene = VK_NULL_HANDLE;

    // Model UBO (сфера)
    VkBuffer model_ubuf = VK_NULL_HANDLE;
    VmaAllocation model_ubuf_alloc = VK_NULL_HANDLE;
    void* model_ubuf_mapped = nullptr;
    VkDescriptorSet desc_set_model = VK_NULL_HANDLE;

    // Point lights SSBO
    VkBuffer point_light_buf = VK_NULL_HANDLE;
    VmaAllocation point_light_alloc = VK_NULL_HANDLE;
    void* point_light_mapped = nullptr;

    // Spot lights SSBO
    VkBuffer spot_light_buf = VK_NULL_HANDLE;
    VmaAllocation spot_light_alloc = VK_NULL_HANDLE;
    void* spot_light_mapped = nullptr;

    // Пайплайн визуализации (кубы)
    VkPipelineLayout light_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline light_pipeline = VK_NULL_HANDLE;

    // Пайплайн линий (конусы)
    VkPipelineLayout line_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline line_pipeline = VK_NULL_HANDLE;

    // Геометрия куба
    VkBuffer light_cube_vbuf = VK_NULL_HANDLE;
    VmaAllocation light_cube_vbuf_alloc = VK_NULL_HANDLE;
    VkBuffer light_cube_ibuf = VK_NULL_HANDLE;
    VmaAllocation light_cube_ibuf_alloc = VK_NULL_HANDLE;
    uint32_t light_cube_index_count = 0;

    // Геометрия конуса (линии)
    VkBuffer cone_vbuf = VK_NULL_HANDLE;
    VmaAllocation cone_vbuf_alloc = VK_NULL_HANDLE;
    uint32_t cone_vertex_count = 0;

    // Model UBO для визуализации - массив
    VkBuffer light_model_ubuf[MAX_LIGHT_VIS] = {};
    VmaAllocation light_model_ubuf_alloc[MAX_LIGHT_VIS] = {};
    void* light_model_ubuf_mapped[MAX_LIGHT_VIS] = {};
    VkDescriptorSet light_desc_set_model[MAX_LIGHT_VIS] = {};
};
Resources res;


struct SceneUniformData {
    glm::mat4 view_projection;
    glm::vec3 camera_position;
    uint32_t point_light_count;
    glm::vec3 directional_light_direction;
    uint32_t spot_light_count;
    glm::vec3 directional_light_color;
    float pad3;
    glm::vec3 ambient_light;
    float pad4;
};
static_assert(sizeof(SceneUniformData) == 128, "scene UBO layout mismatch");

struct ModelUniformData {
    glm::mat4 model;
    glm::vec3 albedo;
    float alpha;
    glm::vec3 specular;
    float pad;
};
static_assert(sizeof(ModelUniformData) == 96, "model UBO layout mismatch");


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

glm::vec3 g_camera_position = {0.0f, 0.0f, 6.0f};
float g_camera_yaw = -90.0f;
float g_camera_pitch = 0.0f;

glm::vec3 g_light_direction = {-0.5f, -1.0f, -0.3f};
glm::vec3 g_light_color = {1.0f, 1.0f, 1.0f};
float g_light_intensity = 1.0f;

glm::vec3 g_ambient_color = {1.0f, 1.0f, 1.0f};
float g_ambient_intensity = 0.1f;

glm::vec3 g_material_albedo = {1.0f, 0.3f, 0.3f};
glm::vec3 g_material_specular = {1.0f, 1.0f, 1.0f};
float g_material_alpha = 32.0f;

glm::vec3 g_sphere_rotation = {0.0f, 0.0f, 0.0f};
glm::vec3 g_sphere_scale = {1.0f, 1.0f, 1.0f};

int g_point_light_count = 0;

struct PointLightParams {
    glm::vec3 position = {2.0f, 1.0f, 0.0f};
    glm::vec3 color = {1.0f, 0.7f, 0.3f};
    float intensity = 1.0f;
    float quadratic = 0.032f;
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
};
SpotLightParams g_spot_lights[MAX_SPOT_LIGHTS];

bool g_show_light_visualization = true;

float g_cone_length = 2.0f;


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
            vert.color = { 1.0f, 1.0f, 1.0f };
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
        {{-s, -s, -s}, {1,1,1}, {0,0,0}},
        {{ s, -s, -s}, {1,1,1}, {0,0,0}},
        {{ s,  s, -s}, {1,1,1}, {0,0,0}},
        {{-s,  s, -s}, {1,1,1}, {0,0,0}},
        {{-s, -s,  s}, {1,1,1}, {0,0,0}},
        {{ s, -s,  s}, {1,1,1}, {0,0,0}},
        {{ s,  s,  s}, {1,1,1}, {0,0,0}},
        {{-s,  s,  s}, {1,1,1}, {0,0,0}},
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


void generateConeLines(std::vector<Vertex>& verts,
                       int segments,
                       float angle_rad) {
    verts.clear();

    const float height = 1.0f;
    const float radius = std::tan(angle_rad) * height;

    const glm::vec3 apex = { 0.0f, 0.0f, 0.0f };
    const glm::vec3 color = { 1.0f, 1.0f, 1.0f };

    for (int i = 0; i < segments; ++i) {
        const float theta = 2.0f * PI * float(i) / float(segments);
        const glm::vec3 p = { radius * std::cos(theta),
                              height,
                              radius * std::sin(theta) };

        verts.push_back({ apex, color, {0,0,0} });
        verts.push_back({ p, color, {0,0,0} });
    }

    for (int i = 0; i < segments; ++i) {
        const float theta1 = 2.0f * PI * float(i) / float(segments);
        const float theta2 = 2.0f * PI * float(i + 1) / float(segments);

        glm::vec3 p1 = { radius * std::cos(theta1), 
                         height, 
                         radius * std::sin(theta1) };
        glm::vec3 p2 = { radius * std::cos(theta2), 
                         height, 
                         radius * std::sin(theta2) };

        verts.push_back({ p1, color, {0,0,0} });
        verts.push_back({ p2, color, {0,0,0} });
    }
}


bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& out_buf, VmaAllocation& out_alloc,
                  void** out_mapped = nullptr) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
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


bool createDescriptors() {
    auto& ctx = graphics::internal::context;

    // set 0: scene layout - UBO + SSBO
    {
        VkDescriptorSetLayoutBinding bindings[3]{};

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

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 3;
        li.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_scene) != VK_SUCCESS)
            return false;
    }

    // set 1: model layout
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &binding;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_model) != VK_SUCCESS)
            return false;
    }

    const uint32_t total_sets = 2 + MAX_LIGHT_VIS;

    VkDescriptorPoolSize pool_sizes[2]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = 2 + MAX_LIGHT_VIS;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = total_sets;
    pi.poolSizeCount = 2;
    pi.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(ctx.device, &pi, nullptr, &res.desc_pool) != VK_SUCCESS)
        return false;

    // scene
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &res.desc_layout_scene;
        if (vkAllocateDescriptorSets(ctx.device, &ai, &res.desc_set_scene) != VK_SUCCESS)
            return false;
    }

    // model
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &res.desc_layout_model;
        if (vkAllocateDescriptorSets(ctx.device, &ai, &res.desc_set_model) != VK_SUCCESS)
            return false;
    }

    // light model sets
    {
        std::array<VkDescriptorSetLayout, MAX_LIGHT_VIS> layouts;
        layouts.fill(res.desc_layout_model);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = MAX_LIGHT_VIS;
        ai.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(ctx.device, &ai, res.light_desc_set_model) != VK_SUCCESS)
            return false;
    }

    // Update scene
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

        VkWriteDescriptorSet writes[3]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 0, 0, 1,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &ubo_bi, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 1, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &point_bi, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                      res.desc_set_scene, 2, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spot_bi, nullptr };
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
    }

    // Update model
    {
        VkDescriptorBufferInfo bi{};
        bi.buffer = res.model_ubuf;
        bi.range = sizeof(ModelUniformData);

        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                res.desc_set_model, 0, 0, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi, nullptr };
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    // Update light model sets
    for (uint32_t i = 0; i < MAX_LIGHT_VIS; ++i) {
        VkDescriptorBufferInfo bi{};
        bi.buffer = res.light_model_ubuf[i];
        bi.range = sizeof(ModelUniformData);

        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                                res.light_desc_set_model[i], 0, 0, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi, nullptr };
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


bool createPipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vs = loadShaderModule("shaders/sphere.vert.spv");
    VkShaderModule fs = loadShaderModule("shaders/sphere.frag.spv");
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
    via[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };
    via[2] = { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = via;

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
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
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

    VkDescriptorSetLayout set_layouts[] = { res.desc_layout_scene, res.desc_layout_model };

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 2;
    pli.pSetLayouts = set_layouts;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &res.pipeline_layout) != VK_SUCCESS)
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
    gp.layout = res.pipeline_layout;
    gp.renderPass = ctx.render_pass;
    gp.subpass = 0;

    const VkResult r = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gp, nullptr, &res.pipeline);
    vkDestroyShaderModule(ctx.device, vs, nullptr);
    vkDestroyShaderModule(ctx.device, fs, nullptr);
    return r == VK_SUCCESS;
}


bool createLightPipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vs = loadShaderModule("shaders/light.vert.spv");
    VkShaderModule fs = loadShaderModule("shaders/light.frag.spv");
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

    VkVertexInputAttributeDescription via[1]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = via;

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

    VkDescriptorSetLayout set_layouts[] = { res.desc_layout_scene, res.desc_layout_model };

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 2;
    pli.pSetLayouts = set_layouts;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &res.light_pipeline_layout) != VK_SUCCESS)
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
    gp.layout = res.light_pipeline_layout;
    gp.renderPass = ctx.render_pass;
    gp.subpass = 0;

    const VkResult r = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gp, nullptr, &res.light_pipeline);
    vkDestroyShaderModule(ctx.device, vs, nullptr);
    vkDestroyShaderModule(ctx.device, fs, nullptr);
    return r == VK_SUCCESS;
}


bool createLinePipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vs = loadShaderModule("shaders/light.vert.spv");
    VkShaderModule fs = loadShaderModule("shaders/light.frag.spv");
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

    VkVertexInputAttributeDescription via[1]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = via;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
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

    VkDescriptorSetLayout set_layouts[] = { res.desc_layout_scene, res.desc_layout_model };

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 2;
    pli.pSetLayouts = set_layouts;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &res.line_pipeline_layout) != VK_SUCCESS)
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
    gp.layout = res.line_pipeline_layout;
    gp.renderPass = ctx.render_pass;
    gp.subpass = 0;

    const VkResult r = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gp, nullptr, &res.line_pipeline);
    vkDestroyShaderModule(ctx.device, vs, nullptr);
    vkDestroyShaderModule(ctx.device, fs, nullptr);
    return r == VK_SUCCESS;
}


bool setupGeometry() {
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    generateSphere(verts, idx, 20, 20, 1.0f);
    res.index_count = uint32_t(idx.size());

    if (!createBuffer(verts.size() * sizeof(Vertex),
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      res.vbuf, res.vbuf_alloc)) return false;
    if (!createBuffer(idx.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      res.ibuf, res.ibuf_alloc)) return false;

    void* mapped = nullptr;
    vmaMapMemory(graphics::internal::context.allocator, res.vbuf_alloc, &mapped);
    std::memcpy(mapped, verts.data(), verts.size() * sizeof(Vertex));
    vmaUnmapMemory(graphics::internal::context.allocator, res.vbuf_alloc);

    vmaMapMemory(graphics::internal::context.allocator, res.ibuf_alloc, &mapped);
    std::memcpy(mapped, idx.data(), idx.size() * sizeof(uint32_t));
    vmaUnmapMemory(graphics::internal::context.allocator, res.ibuf_alloc);

    if (!createBuffer(sizeof(SceneUniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      res.scene_ubuf, res.scene_ubuf_alloc, &res.scene_ubuf_mapped)) return false;

    if (!createBuffer(sizeof(ModelUniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      res.model_ubuf, res.model_ubuf_alloc, &res.model_ubuf_mapped)) return false;

    if (!createBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      res.point_light_buf, res.point_light_alloc, &res.point_light_mapped)) return false;

    if (!createBuffer(sizeof(SpotLight) * MAX_SPOT_LIGHTS,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      res.spot_light_buf, res.spot_light_alloc, &res.spot_light_mapped)) return false;


    std::vector<Vertex> cube_verts;
    std::vector<uint32_t> cube_idx;
    generateCube(cube_verts, cube_idx, 0.5f);
    res.light_cube_index_count = uint32_t(cube_idx.size());

    if (!createBuffer(cube_verts.size() * sizeof(Vertex),
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      res.light_cube_vbuf, res.light_cube_vbuf_alloc)) return false;
    if (!createBuffer(cube_idx.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      res.light_cube_ibuf, res.light_cube_ibuf_alloc)) return false;

    void* cube_mapped = nullptr;
    vmaMapMemory(graphics::internal::context.allocator, res.light_cube_vbuf_alloc, &cube_mapped);
    std::memcpy(cube_mapped, cube_verts.data(), cube_verts.size() * sizeof(Vertex));
    vmaUnmapMemory(graphics::internal::context.allocator, res.light_cube_vbuf_alloc);

    vmaMapMemory(graphics::internal::context.allocator, res.light_cube_ibuf_alloc, &cube_mapped);
    std::memcpy(cube_mapped, cube_idx.data(), cube_idx.size() * sizeof(uint32_t));
    vmaUnmapMemory(graphics::internal::context.allocator, res.light_cube_ibuf_alloc);

    
    
    std::vector<Vertex> cone_verts;
    generateConeLines(cone_verts, 32, PI / 4.0f); // единичный конус: высота 1, радиус 1; реальные размеры задаются масштабом
    res.cone_vertex_count = uint32_t(cone_verts.size());

    if (!createBuffer(cone_verts.size() * sizeof(Vertex),
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      res.cone_vbuf, res.cone_vbuf_alloc)) return false;

    void* cone_mapped = nullptr;
    vmaMapMemory(graphics::internal::context.allocator, res.cone_vbuf_alloc, &cone_mapped);
    std::memcpy(cone_mapped, cone_verts.data(), cone_verts.size() * sizeof(Vertex));
    vmaUnmapMemory(graphics::internal::context.allocator, res.cone_vbuf_alloc);

    
    for (uint32_t i = 0; i < MAX_LIGHT_VIS; ++i) {
        if (!createBuffer(sizeof(ModelUniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          res.light_model_ubuf[i], res.light_model_ubuf_alloc[i],
                          &res.light_model_ubuf_mapped[i])) return false;
    }

    return true;
}

} // namespace


bool initialize() {
    if (!setupGeometry()) { std::cerr << "Failed to setup geometry\n"; return false; }
    if (!createDescriptors()) { std::cerr << "Failed to create descriptors\n"; return false; }
    if (!createPipeline()) { std::cerr << "Failed to create pipeline\n"; return false; }
    if (!createLightPipeline()) { std::cerr << "Failed to create light pipeline\n"; return false; }
    if (!createLinePipeline()) { std::cerr << "Failed to create line pipeline\n"; return false; }
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
    if (res.desc_pool) vkDestroyDescriptorPool(ctx.device, res.desc_pool, nullptr);
    if (res.desc_layout_scene) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_scene, nullptr);
    if (res.desc_layout_model) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_model, nullptr);

    if (res.scene_ubuf) {
        if (res.scene_ubuf_mapped) vmaUnmapMemory(ctx.allocator, res.scene_ubuf_alloc);
        vmaDestroyBuffer(ctx.allocator, res.scene_ubuf, res.scene_ubuf_alloc);
    }
    if (res.model_ubuf) {
        if (res.model_ubuf_mapped) vmaUnmapMemory(ctx.allocator, res.model_ubuf_alloc);
        vmaDestroyBuffer(ctx.allocator, res.model_ubuf, res.model_ubuf_alloc);
    }
    if (res.point_light_buf) {
        if (res.point_light_mapped) vmaUnmapMemory(ctx.allocator, res.point_light_alloc);
        vmaDestroyBuffer(ctx.allocator, res.point_light_buf, res.point_light_alloc);
    }
    if (res.spot_light_buf) {
        if (res.spot_light_mapped) vmaUnmapMemory(ctx.allocator, res.spot_light_alloc);
        vmaDestroyBuffer(ctx.allocator, res.spot_light_buf, res.spot_light_alloc);
    }
    if (res.vbuf) vmaDestroyBuffer(ctx.allocator, res.vbuf, res.vbuf_alloc);
    if (res.ibuf) vmaDestroyBuffer(ctx.allocator, res.ibuf, res.ibuf_alloc);

    for (uint32_t i = 0; i < MAX_LIGHT_VIS; ++i) {
        if (res.light_model_ubuf[i]) {
            if (res.light_model_ubuf_mapped[i])
                vmaUnmapMemory(ctx.allocator, res.light_model_ubuf_alloc[i]);
            vmaDestroyBuffer(ctx.allocator, res.light_model_ubuf[i], res.light_model_ubuf_alloc[i]);
        }
    }
    if (res.light_cube_vbuf) vmaDestroyBuffer(ctx.allocator, res.light_cube_vbuf, res.light_cube_vbuf_alloc);
    if (res.light_cube_ibuf) vmaDestroyBuffer(ctx.allocator, res.light_cube_ibuf, res.light_cube_ibuf_alloc);
    if (res.cone_vbuf) vmaDestroyBuffer(ctx.allocator, res.cone_vbuf, res.cone_vbuf_alloc);
}

void update(double time) {
    (void)time;

    ImGui::Begin("Sphere Lab");

    ImGui::Text("Projection");
    ImGui::SliderFloat("FOV", &g_fov, 20.0f, 120.0f);
    ImGui::Separator();

    ImGui::Text("Camera");
    ImGui::SliderFloat3("Position", &g_camera_position.x, -10.0f, 10.0f);
    ImGui::Text("Sphere Rotation");
    ImGui::SliderFloat3("Rotation", &g_sphere_rotation.x, -180.0f, 180.0f);
    ImGui::Separator();
    ImGui::SliderFloat("Yaw", &g_camera_yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &g_camera_pitch, -89.0f, 89.0f);
    ImGui::Separator();

    ImGui::Text("Sphere Scale");
    ImGui::SliderFloat3("Scale", &g_sphere_scale.x, 0.1f, 3.0f);
    ImGui::Separator();

    ImGui::Text("Directional Light");
    ImGui::SliderFloat3("Direction", &g_light_direction.x, -1.0f, 1.0f);
    ImGui::ColorEdit3("Light Color", &g_light_color.x);
    ImGui::SliderFloat("Intensity", &g_light_intensity, 0.0f, 5.0f);
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

    ImGui::Text("Material");
    ImGui::ColorEdit3("Albedo", &g_material_albedo.x);
    ImGui::ColorEdit3("Specular Color", &g_material_specular.x);
    ImGui::SliderFloat("Shininess (alpha)", &g_material_alpha, 1.0f, 256.0f);

    ImGui::Separator();
    ImGui::Text("Vertices: 441   FPS: %.1f", ImGui::GetIO().Framerate);
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


    SceneUniformData su{};
    su.view_projection = proj * view;
    su.camera_position = g_camera_position;
    su.directional_light_direction = glm::normalize(g_light_direction);
    su.directional_light_color = g_light_color * g_light_intensity;
    su.ambient_light = g_ambient_color * g_ambient_intensity;
    su.point_light_count = uint32_t(g_point_light_count);
    su.spot_light_count = uint32_t(g_spot_light_count);
    std::memcpy(res.scene_ubuf_mapped, &su, sizeof(su));

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::rotate(model, glm::radians(g_sphere_rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(g_sphere_rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(g_sphere_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    model = glm::scale(model, g_sphere_scale);

    ModelUniformData mu{};
    mu.model = model;
    mu.albedo = g_material_albedo;
    mu.specular = g_material_specular;
    mu.alpha = g_material_alpha;
    std::memcpy(res.model_ubuf_mapped, &mu, sizeof(mu));

  
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

        // if (g_spot_lights[i].inner_angle > g_spot_lights[i].outer_angle) {
        //     g_spot_lights[i].inner_angle = g_spot_lights[i].outer_angle - 1.0f;
        // } 

        spots[i].inner_cos = std::cos(glm::radians(g_spot_lights[i].inner_angle));
        spots[i].outer_cos = std::cos(glm::radians(g_spot_lights[i].outer_angle));
    }
    std::memcpy(res.spot_light_mapped, spots, sizeof(SpotLight) * g_spot_light_count);
}

void render(const graphics::internal::FrameData& fd) {
    
    auto& ctx = graphics::internal::context;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fd.command_buffer, &bi);

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
    vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width = float(ctx.swapchain_extent.width);
    vp.height = float(ctx.swapchain_extent.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);

    VkRect2D sc{}; sc.extent = ctx.swapchain_extent;
    vkCmdSetScissor(fd.command_buffer, 0, 1, &sc);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, res.pipeline);

    VkBuffer vb[] = { res.vbuf };
    VkDeviceSize offs[] = { 0 };
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vb, offs);
    vkCmdBindIndexBuffer(fd.command_buffer, res.ibuf, 0, VK_INDEX_TYPE_UINT32);

    VkDescriptorSet sets_to_bind[] = { res.desc_set_scene, res.desc_set_model };
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            res.pipeline_layout, 0, 2, sets_to_bind, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, res.index_count, 1, 0, 0, 0);


    if (g_show_light_visualization) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, res.light_pipeline);

        VkBuffer cube_vb[] = { res.light_cube_vbuf };
        VkDeviceSize cube_offs[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, cube_vb, cube_offs);
        vkCmdBindIndexBuffer(fd.command_buffer, res.light_cube_ibuf, 0, VK_INDEX_TYPE_UINT32);

        for (int i = 0; i < g_point_light_count; ++i) {
            ModelUniformData mu{};
            mu.model = glm::translate(glm::mat4(1.0f), g_point_lights[i].position);
            mu.model = glm::scale(mu.model, glm::vec3(0.15f));
            mu.albedo = g_point_lights[i].color;
            std::memcpy(res.light_model_ubuf_mapped[i], &mu, sizeof(mu));

            VkDescriptorSet sets[] = { res.desc_set_scene, res.light_desc_set_model[i] };
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    res.light_pipeline_layout, 0, 2, sets, 0, nullptr);
            vkCmdDrawIndexed(fd.command_buffer, res.light_cube_index_count, 1, 0, 0, 0);
        }

        for (int i = 0; i < g_spot_light_count; ++i) {
            const int idx = MAX_POINT_LIGHTS + i;
            ModelUniformData mu{};
            mu.model = glm::translate(glm::mat4(1.0f), g_spot_lights[i].position);
            mu.model = glm::scale(mu.model, glm::vec3(0.1f));
            mu.albedo = g_spot_lights[i].color;
            std::memcpy(res.light_model_ubuf_mapped[idx], &mu, sizeof(mu));

            VkDescriptorSet sets[] = { res.desc_set_scene, res.light_desc_set_model[idx] };
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    res.light_pipeline_layout, 0, 2, sets, 0, nullptr);
            vkCmdDrawIndexed(fd.command_buffer, res.light_cube_index_count, 1, 0, 0, 0);
        }

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, res.line_pipeline);
        VkBuffer cone_vb[] = { res.cone_vbuf };
        VkDeviceSize cone_offs[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, cone_vb, cone_offs);

        for (int i = 0; i < g_spot_light_count; ++i) {
            glm::vec3 dir = glm::normalize(g_spot_lights[i].direction);

            glm::vec3 from = glm::vec3(0, 1, 0);
            float cos_a = glm::clamp(glm::dot(from, dir), -1.0f, 1.0f);
            glm::mat4 rot = glm::mat4(1.0f);
            if (cos_a < 0.9999f) {
                if (cos_a > -0.9999f) {
                    glm::vec3 axis = glm::normalize(glm::cross(from, dir));
                    rot = glm::rotate(glm::mat4(1.0f), std::acos(cos_a), axis);
                } else {
                    rot = glm::rotate(glm::mat4(1.0f), PI, glm::vec3(1, 0, 0));
                }
            }

            float inner_deg = g_spot_lights[i].inner_angle;
            float outer_deg = g_spot_lights[i].outer_angle;
            // if (inner_deg > outer_deg) inner_deg = outer_deg - 1.0f;

            struct ConeDraw { float angle_deg; int idx; glm::vec3 color; };
            const ConeDraw cones[2] = {
                { inner_deg, int(MAX_POINT_LIGHTS + MAX_SPOT_LIGHTS + i),     glm::vec3(1.0f, 0.85f, 0.2f) },
                { outer_deg, int(MAX_POINT_LIGHTS + 2 * MAX_SPOT_LIGHTS + i), glm::vec3(1.0f, 1.0f, 1.0f)  },
            };

            for (const ConeDraw& c : cones) {
                const float radius = std::tan(glm::radians(c.angle_deg)) * g_cone_length;
                glm::mat4 scale_m = glm::scale(glm::mat4(1.0f),
                                               glm::vec3(radius, g_cone_length, radius));

                ModelUniformData mu{};
                mu.model = glm::translate(glm::mat4(1.0f), g_spot_lights[i].position) * rot * scale_m;
                mu.albedo = c.color;
                std::memcpy(res.light_model_ubuf_mapped[c.idx], &mu, sizeof(mu));

                VkDescriptorSet sets[] = { res.desc_set_scene, res.light_desc_set_model[c.idx] };
                vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        res.line_pipeline_layout, 0, 2, sets, 0, nullptr);
                vkCmdDraw(fd.command_buffer, res.cone_vertex_count, 1, 0, 0);
            }
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application