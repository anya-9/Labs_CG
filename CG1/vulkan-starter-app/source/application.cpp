#include "application.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>

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


struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};


struct ObjectState {
    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};
    glm::vec4 tint = {1.0f, 1.0f, 1.0f, 1.0f};
    bool animated = true;
    float anim_speed = 1.0f;
    float orbit_radius = 3.0f;
    float orbit_y_speed = 1.0f;
};


struct Resources {
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Два layout'а: scene (set 0) и model (set 1)
    VkDescriptorSetLayout desc_layout_scene = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_model = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    VkBuffer vbuf = VK_NULL_HANDLE;
    VmaAllocation vbuf_alloc = VK_NULL_HANDLE;
    VkBuffer ibuf = VK_NULL_HANDLE;
    VmaAllocation ibuf_alloc = VK_NULL_HANDLE;
    uint32_t index_count = 0;

    static constexpr uint32_t MAX_OBJECTS = 2;

    // Scene UBO - один на всю сцену
    VkBuffer scene_ubuf = VK_NULL_HANDLE;
    VmaAllocation scene_ubuf_alloc = VK_NULL_HANDLE;
    void* scene_ubuf_mapped = nullptr;
    VkDescriptorSet desc_set_scene = VK_NULL_HANDLE;

    // Model UBO - по одному на каждый объект
    VkBuffer model_ubuf[MAX_OBJECTS] = {};
    VmaAllocation model_ubuf_alloc[MAX_OBJECTS] = {};
    void* model_ubuf_mapped[MAX_OBJECTS] = {};
    VkDescriptorSet desc_sets_model[MAX_OBJECTS] = {};
};

Resources res;
ObjectState objects[Resources::MAX_OBJECTS];

// UBO структуры
// Scene: view * projection (общее для всех объектов)
struct SceneUniformData {
    glm::mat4 view_projection;
};
static_assert(sizeof(SceneUniformData) == 64, "scene UBO layout mismatch");

// Model: матрица объекта + цвет (своё для каждого объекта)
struct ModelUniformData {
    glm::mat4 model;
    glm::vec4 tint;
};
static_assert(sizeof(ModelUniformData) == 80, "model UBO layout mismatch");

// UI-состояние
int g_proj_mode = 1;    // 0 = ortho, 1 = persp
float g_fov = 60.0f;
float g_ortho_size = 3.0f;
int g_current_obj = 0;
float g_anim_time = 0.0f;
float g_last_time = 0.0f;

// Генерация сферы
void generateSphere(std::vector<Vertex>& verts,
                    std::vector<uint32_t>& indices,
                    int stacks, int slices, float radius) {
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

            vert.color = { 1.0f, 0.0f, 0.0f };
            verts.push_back(vert);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const uint32_t a = uint32_t(i * (slices + 1) + j);
            const uint32_t b = a + uint32_t(slices + 1);

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);

            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
        }
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

    // set 0: scene layout
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &binding;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_scene) != VK_SUCCESS)
            return false;
    }

    // set 1: model layout
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &binding;

        if (vkCreateDescriptorSetLayout(ctx.device, &li, nullptr, &res.desc_layout_model) != VK_SUCCESS)
            return false;
    }

    // Pool
    // Всего нужно: 1 scene set + MAX_OBJECTS model sets
    const uint32_t total_sets = 1 + Resources::MAX_OBJECTS;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = total_sets;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.maxSets = total_sets;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &pool_size;

    if (vkCreateDescriptorPool(ctx.device, &pi, nullptr, &res.desc_pool) != VK_SUCCESS)
        return false;

    // Аллокация scene set
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &res.desc_layout_scene;

        if (vkAllocateDescriptorSets(ctx.device, &ai, &res.desc_set_scene) != VK_SUCCESS)
            return false;
    }

    // Аллокация model sets
    {
        std::array<VkDescriptorSetLayout, Resources::MAX_OBJECTS> layouts;
        layouts.fill(res.desc_layout_model);

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = res.desc_pool;
        ai.descriptorSetCount = Resources::MAX_OBJECTS;
        ai.pSetLayouts = layouts.data();

        if (vkAllocateDescriptorSets(ctx.device, &ai, res.desc_sets_model) != VK_SUCCESS)
            return false;
    }

    // Update scene set
    {
        VkDescriptorBufferInfo bi{};
        bi.buffer = res.scene_ubuf;
        bi.offset = 0;
        bi.range = sizeof(SceneUniformData);

        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = res.desc_set_scene;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &bi;

        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    // Update model sets
    for (uint32_t i = 0; i < Resources::MAX_OBJECTS; ++i) {
        VkDescriptorBufferInfo bi{};
        bi.buffer = res.model_ubuf[i];
        bi.offset = 0;
        bi.range = sizeof(ModelUniformData);

        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = res.desc_sets_model[i];
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &bi;

        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    return true;
}

// Загрузка шейдера
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

// Пайплайн
bool createPipeline() {
    auto& ctx = graphics::internal::context;

    VkShaderModule vk_vertex_shader = loadShaderModule("shaders/sphere.vert.spv");
    VkShaderModule vk_fragment_shader = loadShaderModule("shaders/sphere.frag.spv");
    if (!vk_vertex_shader || !vk_fragment_shader) return false;

    VkPipelineShaderStageCreateInfo stage_infos[2]{};
    stage_infos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_infos[0].stage= VK_SHADER_STAGE_VERTEX_BIT;
    stage_infos[0].module = vk_vertex_shader;
    stage_infos[0].pName = "main";
    stage_infos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_infos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stage_infos[1].module = vk_fragment_shader;
    stage_infos[1].pName = "main";

    VkVertexInputBindingDescription vib{};
    vib.binding = 0;
    vib.stride = sizeof(Vertex);
    vib.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription via[2]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
    via[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 2;
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

    VkDynamicState dyn_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyn_states;

    // Pipeline Layout: два set layout'а,  индекс 0 = set 0 (scene), индекс 1 = set 1 (model)
    VkDescriptorSetLayout set_layouts[] = {
        res.desc_layout_scene,   // set 0
        res.desc_layout_model,   // set 1
    };

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 2;
    pli.pSetLayouts = set_layouts;
    if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &res.pipeline_layout) != VK_SUCCESS)
        return false;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2;
    gp.pStages = stage_infos;
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
    vkDestroyShaderModule(ctx.device, vk_vertex_shader, nullptr);
    vkDestroyShaderModule(ctx.device, vk_fragment_shader, nullptr);
    return r == VK_SUCCESS;
}


bool setupGeometry() {
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    generateSphere(verts, idx, 10, 10, 1.0f);
    res.index_count = uint32_t(idx.size());

    if (!createBuffer(verts.size() * sizeof(Vertex),
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      res.vbuf, res.vbuf_alloc))
        return false;
    if (!createBuffer(idx.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      res.ibuf, res.ibuf_alloc))
        return false;

    void* mapped = nullptr;
    vmaMapMemory(graphics::internal::context.allocator, res.vbuf_alloc, &mapped);
    std::memcpy(mapped, verts.data(), verts.size() * sizeof(Vertex));
    vmaUnmapMemory(graphics::internal::context.allocator, res.vbuf_alloc);

    vmaMapMemory(graphics::internal::context.allocator, res.ibuf_alloc, &mapped);
    std::memcpy(mapped, idx.data(), idx.size() * sizeof(uint32_t));
    vmaUnmapMemory(graphics::internal::context.allocator, res.ibuf_alloc);

    if (!createBuffer(sizeof(SceneUniformData),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      res.scene_ubuf, res.scene_ubuf_alloc,
                      &res.scene_ubuf_mapped))
        return false;

    for (uint32_t i = 0; i < Resources::MAX_OBJECTS; ++i) {
        if (!createBuffer(sizeof(ModelUniformData),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          res.model_ubuf[i], res.model_ubuf_alloc[i],
                          &res.model_ubuf_mapped[i]))
            return false;
    }
    return true;
}
} // namespace


bool initialize() {
    if (!setupGeometry()) {
        std::cerr << "Failed to setup geometry\n";
        return false;
    }
    if (!createDescriptors()) {
        std::cerr << "Failed to create descriptors\n";
        return false;
    }
    if (!createPipeline()) {
        std::cerr << "Failed to create pipeline\n";
        return false;
    }

    objects[0].position = { -1.5f, 0.0f, 0.0f };
    objects[0].tint = { 1.0f, 1.0f, 1.0f, 1.0f };

    objects[1].position = {  1.5f, 0.0f, 0.0f };
    objects[1].tint = { 1.0f, 0.5f, 0.5f, 1.0f };
    objects[1].anim_speed = 1.5f;
    objects[1].orbit_radius = 2.0f;
    objects[1].orbit_y_speed = 1.7f;

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);

    if (res.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, res.pipeline, nullptr);
    if (res.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, res.pipeline_layout, nullptr);
    if (res.desc_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx.device, res.desc_pool, nullptr);
    if (res.desc_layout_scene != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_scene, nullptr);
    if (res.desc_layout_model != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, res.desc_layout_model, nullptr);

    // Scene UBO
    if (res.scene_ubuf != VK_NULL_HANDLE) {
        if (res.scene_ubuf_mapped) vmaUnmapMemory(ctx.allocator, res.scene_ubuf_alloc);
        vmaDestroyBuffer(ctx.allocator, res.scene_ubuf, res.scene_ubuf_alloc);
    }

    // Model UBO
    for (uint32_t i = 0; i < Resources::MAX_OBJECTS; ++i) {
        if (res.model_ubuf[i] != VK_NULL_HANDLE) {
            if (res.model_ubuf_mapped[i]) vmaUnmapMemory(ctx.allocator, res.model_ubuf_alloc[i]);
            vmaDestroyBuffer(ctx.allocator, res.model_ubuf[i], res.model_ubuf_alloc[i]);
        }
    }

    if (res.vbuf != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, res.vbuf, res.vbuf_alloc);
    if (res.ibuf != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, res.ibuf, res.ibuf_alloc);
}


void update(double time) {
    const float t = float(time);
    const float dt = (g_last_time == 0.0f) ? 0.0f : (t - g_last_time);
    g_last_time = t;

    // ImGui
    ImGui::Begin("Sphere Lab");

    ImGui::Text("Projection");
    ImGui::RadioButton("Orthographic", &g_proj_mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Perspective",  &g_proj_mode, 1);

    if (g_proj_mode == 1) ImGui::SliderFloat("FOV", &g_fov, 20.0f, 120.0f);
    else ImGui::SliderFloat("Ortho size", &g_ortho_size, 0.5f, 10.0f);

    ImGui::Separator();

    ImGui::Combo("Object", &g_current_obj, "Sphere 0\0Sphere 1\0");
    ObjectState& o = objects[g_current_obj];

    ImGui::Text("Transform");
    ImGui::SliderFloat3("Position", &o.position.x, -10.0f, 10.0f);
    ImGui::SliderFloat3("Rotation", &o.rotation.x, -PI, PI);
    ImGui::SliderFloat3("Scale", &o.scale.x, 0.05f, 10.0f);

    ImGui::Separator();
    ImGui::Checkbox("Animate", &o.animated);
    ImGui::SliderFloat("Anim speed", &o.anim_speed, 0.0f, 5.0f);
    ImGui::SliderFloat("Orbit radius", &o.orbit_radius, 0.0f, 8.0f);
    ImGui::SliderFloat("Orbit Y speed", &o.orbit_y_speed, 0.0f, 4.0f);

    ImGui::Separator();
    ImGui::ColorEdit4("Tint", &o.tint.x);

    ImGui::Separator();
    ImGui::Text("Vertices: 121   FPS: %.1f", ImGui::GetIO().Framerate);

    ImGui::End();

    // Анимация
    g_anim_time += dt;
    for (auto& obj : objects) {
        if (!obj.animated) continue;

        const float at = g_anim_time * obj.anim_speed;

        obj.position.x = std::cos(at) * obj.orbit_radius;
        obj.position.z = std::sin(at) * obj.orbit_radius;
        obj.position.y = std::sin(at * obj.orbit_y_speed) * 1.0f;

        obj.rotation.x += dt * obj.anim_speed * 1.2f;
        obj.rotation.y += dt * obj.anim_speed * 0.8f;
        obj.rotation.z += dt * obj.anim_speed * 0.5f;
    }

    // Обновление UBO
    auto& ctx = graphics::internal::context;
    const float aspect = float(ctx.swapchain_extent.width) /
                         float(ctx.swapchain_extent.height);

    const glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 6.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 proj;
    if (g_proj_mode == 1) {
        proj = glm::perspective(glm::radians(g_fov), aspect, 0.1f, 100.0f);
    } else {
        proj = glm::ortho(-g_ortho_size * aspect,  g_ortho_size * aspect,
                          -g_ortho_size,           g_ortho_size,
                           0.1f,                   100.0f               );
    }
    proj[1][1] *= -1.0f;

    // Scene UBO - обновление один раз на кадр
    SceneUniformData su{};
    su.view_projection = proj * view;
    std::memcpy(res.scene_ubuf_mapped, &su, sizeof(su));

    // Model UBO - для каждого объекта
    for (uint32_t i = 0; i < Resources::MAX_OBJECTS; ++i) {
        const ObjectState& obj = objects[i];

        glm::mat4 model = glm::translate(glm::mat4(1.0f), obj.position);
        model = glm::rotate(model, obj.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, obj.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, obj.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, obj.scale);

        ModelUniformData mu{};
        mu.model = model;
        mu.tint = obj.tint;

        std::memcpy(res.model_ubuf_mapped[i], &mu, sizeof(mu));
    }
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
    vp.x = 0.0f;
    vp.y = 0.0f;
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

    for (uint32_t i = 0; i < Resources::MAX_OBJECTS; ++i) {
        VkDescriptorSet sets_to_bind[] = {
            res.desc_set_scene,
            res.desc_sets_model[i],
        };

        vkCmdBindDescriptorSets(fd.command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                res.pipeline_layout,
                                0,        // firstSet = 0
                                2,        // setCount = 2
                                sets_to_bind,
                                0, nullptr);

        vkCmdDrawIndexed(fd.command_buffer, res.index_count, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application