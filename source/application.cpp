#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <imgui.h>
#include <vulkan/vulkan.h>

namespace application {
namespace {

constexpr uint32_t CYLINDER_SEGMENTS = 50;
constexpr uint32_t OBJECT_COUNT = 2;
constexpr float PI = 3.14159265358979323846f;

struct Vertex {
    float position[3];
    float color[3];
};

struct Mat4 {
    // Column-major matrix, directly compatible with GLSL mat4.
    float elements[16]{};
};

struct alignas(16) GlobalUniforms {
    float mvp[16];
    float tint[4];
};

struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    bool manually_mapped = false;
};

VkPipeline vk_pipeline = VK_NULL_HANDLE;
VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
std::array<VkDescriptorSet, OBJECT_COUNT> vk_descriptor_sets{};

BufferResource vertex_buffer;
BufferResource index_buffer;
std::array<BufferResource, OBJECT_COUNT> uniform_buffers{};
uint32_t index_count = 0;
uint32_t vertex_count = 0;

int projection_mode = 0; // 0 - perspective, 1 - orthographic
float position[3] = { 0.0f, 0.0f, 0.0f };
float rotation_degrees[3] = { -15.0f, 25.0f, 0.0f };
float object_scale[3] = { 1.0f, 1.0f, 1.0f };

bool animation_playing = true;
float animation_speed = 0.8f;
float trajectory_radius = 1.25f;
float trajectory_height = 0.65f;
float animation_phase = 0.0f;
double previous_time = -1.0;

float object_color[3] = { 1.0f, 1.0f, 1.0f };
float second_object_color[3] = { 0.55f, 0.80f, 1.0f };
bool draw_second_object = true;
bool show_demo_window = false;

float camera_distance = 7.0f;
float perspective_fov_degrees = 60.0f;
float orthographic_half_height = 3.5f;

float radians(float degrees) {
    return degrees * PI / 180.0f;
}

Mat4 identity() {
    Mat4 result{};
    result.elements[0] = 1.0f;
    result.elements[5] = 1.0f;
    result.elements[10] = 1.0f;
    result.elements[15] = 1.0f;
    return result;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) {
                value += a.elements[k * 4 + row] * b.elements[column * 4 + k];
            }
            result.elements[column * 4 + row] = value;
        }
    }

    return result;
}

Mat4 translation(float x, float y, float z) {
    Mat4 result = identity();
    result.elements[12] = x;
    result.elements[13] = y;
    result.elements[14] = z;
    return result;
}

Mat4 scale(float x, float y, float z) {
    Mat4 result{};
    result.elements[0] = x;
    result.elements[5] = y;
    result.elements[10] = z;
    result.elements[15] = 1.0f;
    return result;
}

Mat4 rotationX(float angle) {
    Mat4 result = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.elements[5] = c;
    result.elements[6] = s;
    result.elements[9] = -s;
    result.elements[10] = c;
    return result;
}

Mat4 rotationY(float angle) {
    Mat4 result = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.elements[0] = c;
    result.elements[2] = -s;
    result.elements[8] = s;
    result.elements[10] = c;
    return result;
}

Mat4 rotationZ(float angle) {
    Mat4 result = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    result.elements[0] = c;
    result.elements[1] = s;
    result.elements[4] = -s;
    result.elements[5] = c;
    return result;
}

Mat4 perspective(float vertical_fov, float aspect, float near_plane, float far_plane) {
    Mat4 result{};
    const float f = 1.0f / std::tan(vertical_fov * 0.5f);

    result.elements[0] = f / aspect;
    // Vulkan framebuffer coordinates have the opposite Y convention to the
    // one used by the matrix derivation in the lecture, so Y is flipped here.
    result.elements[5] = -f;
    result.elements[10] = far_plane / (near_plane - far_plane);
    result.elements[11] = -1.0f;
    result.elements[14] = (far_plane * near_plane) / (near_plane - far_plane);
    return result;
}

Mat4 orthographic(float half_width, float half_height, float near_plane, float far_plane) {
    Mat4 result = identity();

    result.elements[0] = 1.0f / half_width;
    result.elements[5] = -1.0f / half_height;
    result.elements[10] = 1.0f / (near_plane - far_plane);
    result.elements[14] = near_plane / (near_plane - far_plane);
    return result;
}

Mat4 composeModel(float px, float py, float pz,
                  float rx, float ry, float rz,
                  float sx, float sy, float sz) {
    const Mat4 t = translation(px, py, pz);
    const Mat4 r = multiply(rotationZ(rz), multiply(rotationY(ry), rotationX(rx)));
    const Mat4 s = scale(sx, sy, sz);
    return multiply(t, multiply(r, s));
}

float proceduralColor(float value) {
    return std::clamp(0.20f + 0.80f * value, 0.0f, 1.0f);
}

void buildCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    constexpr float radius = 1.0f;
    constexpr float half_height = 1.0f;

    vertices.clear();
    indices.clear();
    vertices.reserve(CYLINDER_SEGMENTS * 2 + 2);
    indices.reserve(CYLINDER_SEGMENTS * 12);

    // Top ring: exactly 50 perimeter vertices.
    for (uint32_t i = 0; i < CYLINDER_SEGMENTS; ++i) {
        const float angle = 2.0f * PI * static_cast<float>(i) /
                            static_cast<float>(CYLINDER_SEGMENTS);
        const float x = radius * std::cos(angle);
        const float z = radius * std::sin(angle);

        vertices.push_back({
            { x, half_height, z },
            {
                proceduralColor((x / radius + 1.0f) * 0.5f),
                proceduralColor(1.0f),
                proceduralColor((z / radius + 1.0f) * 0.5f),
            }
        });
    }

    // Bottom ring: exactly 50 perimeter vertices.
    for (uint32_t i = 0; i < CYLINDER_SEGMENTS; ++i) {
        const float angle = 2.0f * PI * static_cast<float>(i) /
                            static_cast<float>(CYLINDER_SEGMENTS);
        const float x = radius * std::cos(angle);
        const float z = radius * std::sin(angle);

        vertices.push_back({
            { x, -half_height, z },
            {
                proceduralColor((x / radius + 1.0f) * 0.5f),
                proceduralColor(0.0f),
                proceduralColor((z / radius + 1.0f) * 0.5f),
            }
        });
    }

    const uint32_t top_center = static_cast<uint32_t>(vertices.size());
    vertices.push_back({ { 0.0f, half_height, 0.0f }, { 0.60f, 1.0f, 0.60f } });

    const uint32_t bottom_center = static_cast<uint32_t>(vertices.size());
    vertices.push_back({ { 0.0f, -half_height, 0.0f }, { 0.60f, 0.25f, 0.60f } });

    const uint32_t top_start = 0;
    const uint32_t bottom_start = CYLINDER_SEGMENTS;

    for (uint32_t i = 0; i < CYLINDER_SEGMENTS; ++i) {
        const uint32_t next = (i + 1) % CYLINDER_SEGMENTS;

        const uint32_t top_i = top_start + i;
        const uint32_t top_next = top_start + next;
        const uint32_t bottom_i = bottom_start + i;
        const uint32_t bottom_next = bottom_start + next;

        // Side surface: two triangles per segment.
        indices.push_back(top_i);
        indices.push_back(top_next);
        indices.push_back(bottom_next);

        indices.push_back(top_i);
        indices.push_back(bottom_next);
        indices.push_back(bottom_i);

        // Top cap (+Y normal).
        indices.push_back(top_center);
        indices.push_back(top_next);
        indices.push_back(top_i);

        // Bottom cap (-Y normal).
        indices.push_back(bottom_center);
        indices.push_back(bottom_i);
        indices.push_back(bottom_next);
    }
}

bool createMappedBuffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        BufferResource& result) {
    auto& context = graphics::internal::context;

    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    VmaAllocationInfo allocation_result{};
    if (vmaCreateBuffer(context.allocator,
                        &buffer_info,
                        &allocation_info,
                        &result.buffer,
                        &result.allocation,
                        &allocation_result) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    result.size = size;
    result.mapped = allocation_result.pMappedData;

    if (result.mapped == nullptr) {
        if (vmaMapMemory(context.allocator, result.allocation, &result.mapped) != VK_SUCCESS) {
            std::cerr << "Failed to map Vulkan buffer memory\n";
            vmaDestroyBuffer(context.allocator, result.buffer, result.allocation);
            result = {};
            return false;
        }
        result.manually_mapped = true;
    }

    return true;
}

bool uploadBuffer(BufferResource& buffer, const void* data, VkDeviceSize size) {
    auto& context = graphics::internal::context;

    if (buffer.mapped == nullptr || size > buffer.size) {
        return false;
    }

    std::memcpy(buffer.mapped, data, static_cast<size_t>(size));
    return vmaFlushAllocation(context.allocator, buffer.allocation, 0, size) == VK_SUCCESS;
}

void destroyBuffer(BufferResource& buffer) {
    auto& context = graphics::internal::context;

    if (buffer.buffer == VK_NULL_HANDLE) {
        return;
    }

    if (buffer.manually_mapped) {
        vmaUnmapMemory(context.allocator, buffer.allocation);
    }

    vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
    buffer = {};
}

VkShaderModule loadShaderModule(const char* path) {
    auto& context = graphics::internal::context;

    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open shader: " << path << '\n';
        return VK_NULL_HANDLE;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        std::cerr << "Invalid SPIR-V file: " << path << '\n';
        return VK_NULL_HANDLE;
    }

    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);

    const VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<size_t>(size),
        .pCode = code.data(),
    };

    VkShaderModule shader = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.device, &shader_info, nullptr, &shader) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << '\n';
        return VK_NULL_HANDLE;
    }

    return shader;
}

bool createDescriptorResources() {
    auto& context = graphics::internal::context;

    const VkDescriptorSetLayoutBinding uniform_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &uniform_binding,
    };

    if (vkCreateDescriptorSetLayout(context.device,
                                    &layout_info,
                                    nullptr,
                                    &vk_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout\n";
        return false;
    }

    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk_descriptor_set_layout,
    };

    if (vkCreatePipelineLayout(context.device,
                               &pipeline_layout_info,
                               nullptr,
                               &vk_pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = OBJECT_COUNT,
    };

    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = OBJECT_COUNT,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };

    if (vkCreateDescriptorPool(context.device,
                               &pool_info,
                               nullptr,
                               &vk_descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    std::array<VkDescriptorSetLayout, OBJECT_COUNT> layouts{};
    layouts.fill(vk_descriptor_set_layout);

    const VkDescriptorSetAllocateInfo allocation_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk_descriptor_pool,
        .descriptorSetCount = OBJECT_COUNT,
        .pSetLayouts = layouts.data(),
    };

    if (vkAllocateDescriptorSets(context.device,
                                 &allocation_info,
                                 vk_descriptor_sets.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets\n";
        return false;
    }

    for (uint32_t i = 0; i < OBJECT_COUNT; ++i) {
        const VkDescriptorBufferInfo buffer_info = {
            .buffer = uniform_buffers[i].buffer,
            .offset = 0,
            .range = sizeof(GlobalUniforms),
        };

        const VkWriteDescriptorSet descriptor_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_descriptor_sets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_info,
        };

        vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);
    }

    return true;
}

bool createGraphicsPipeline() {
    auto& context = graphics::internal::context;

    const VkShaderModule vertex_shader = loadShaderModule("shaders/cylinder.vert.spv");
    if (vertex_shader == VK_NULL_HANDLE) {
        return false;
    }

    const VkShaderModule fragment_shader = loadShaderModule("shaders/cylinder.frag.spv");
    if (fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device, vertex_shader, nullptr);
        return false;
    }

    const VkPipelineShaderStageCreateInfo shader_stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader,
            .pName = "main",
        },
    };

    const VkVertexInputBindingDescription binding_description = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription attribute_descriptions[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_description,
        .vertexAttributeDescriptionCount =
            static_cast<uint32_t>(std::size(attribute_descriptions)),
        .pVertexAttributeDescriptions = attribute_descriptions,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkPipelineViewportStateCreateInfo viewport_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo rasterization_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        // For the laboratory geometry we keep both sides visible. Depth testing
        // still resolves which fragments are in front.
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    const VkPipelineColorBlendAttachmentState color_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic_state_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(std::size(dynamic_states)),
        .pDynamicStates = dynamic_states,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = static_cast<uint32_t>(std::size(shader_stages)),
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly_info,
        .pViewportState = &viewport_info,
        .pRasterizationState = &rasterization_info,
        .pMultisampleState = &multisample_info,
        .pDepthStencilState = &depth_info,
        .pColorBlendState = &color_blend_info,
        .pDynamicState = &dynamic_state_info,
        .layout = vk_pipeline_layout,
        .renderPass = context.render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    const VkResult pipeline_result = vkCreateGraphicsPipelines(
        context.device,
        VK_NULL_HANDLE,
        1,
        &pipeline_info,
        nullptr,
        &vk_pipeline);

    vkDestroyShaderModule(context.device, fragment_shader, nullptr);
    vkDestroyShaderModule(context.device, vertex_shader, nullptr);

    if (pipeline_result != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }

    return true;
}

void writeUniform(uint32_t object_index, const Mat4& mvp, const float tint[3]) {
    GlobalUniforms uniforms{};
    std::memcpy(uniforms.mvp, mvp.elements, sizeof(uniforms.mvp));
    uniforms.tint[0] = tint[0];
    uniforms.tint[1] = tint[1];
    uniforms.tint[2] = tint[2];
    uniforms.tint[3] = 1.0f;

    uploadBuffer(uniform_buffers[object_index], &uniforms, sizeof(uniforms));
}

void updateUniforms() {
    const auto& context = graphics::internal::context;

    const float width = static_cast<float>(std::max(context.swapchain_extent.width, 1u));
    const float height = static_cast<float>(std::max(context.swapchain_extent.height, 1u));
    const float aspect = width / height;

    const Mat4 projection = projection_mode == 0
        ? perspective(radians(perspective_fov_degrees), aspect, 0.1f, 100.0f)
        : orthographic(orthographic_half_height * aspect,
                       orthographic_half_height,
                       0.1f,
                       100.0f);

    const Mat4 view = translation(0.0f, 0.0f, -camera_distance);

    const float animated_x = trajectory_radius * std::cos(animation_phase);
    const float animated_y = trajectory_height * std::sin(animation_phase * 2.0f);
    const float animated_z = trajectory_radius * 0.45f * std::sin(animation_phase);

    const Mat4 first_model = composeModel(
        position[0] + animated_x,
        position[1] + animated_y,
        position[2] + animated_z,
        radians(rotation_degrees[0]) + animation_phase * 0.35f,
        radians(rotation_degrees[1]) + animation_phase * 0.75f,
        radians(rotation_degrees[2]) + animation_phase * 0.20f,
        object_scale[0],
        object_scale[1],
        object_scale[2]);

    const Mat4 first_mvp = multiply(projection, multiply(view, first_model));
    writeUniform(0, first_mvp, object_color);

    // The second cylinder uses a separate uniform buffer and a separate
    // VkDescriptorSet. It deliberately shares geometry and the pipeline.
    const Mat4 second_model = composeModel(
        -2.25f,
        -0.55f,
        -0.65f,
        radians(20.0f),
        -animation_phase * 0.35f,
        radians(-18.0f),
        0.55f,
        0.75f,
        0.55f);

    const Mat4 second_mvp = multiply(projection, multiply(view, second_model));
    writeUniform(1, second_mvp, second_object_color);
}

void resetControls() {
    projection_mode = 0;
    position[0] = 0.0f;
    position[1] = 0.0f;
    position[2] = 0.0f;
    rotation_degrees[0] = -15.0f;
    rotation_degrees[1] = 25.0f;
    rotation_degrees[2] = 0.0f;
    object_scale[0] = 1.0f;
    object_scale[1] = 1.0f;
    object_scale[2] = 1.0f;

    animation_playing = true;
    animation_speed = 0.8f;
    trajectory_radius = 1.25f;
    trajectory_height = 0.65f;
    animation_phase = 0.0f;

    object_color[0] = 1.0f;
    object_color[1] = 1.0f;
    object_color[2] = 1.0f;
    second_object_color[0] = 0.55f;
    second_object_color[1] = 0.80f;
    second_object_color[2] = 1.0f;
    draw_second_object = true;

    camera_distance = 7.0f;
    perspective_fov_degrees = 60.0f;
    orthographic_half_height = 3.5f;
}

void destroyApplicationResources() {
    auto& context = graphics::internal::context;

    if (vk_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device, vk_pipeline, nullptr);
        vk_pipeline = VK_NULL_HANDLE;
    }

    if (vk_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
        vk_descriptor_pool = VK_NULL_HANDLE;
        vk_descriptor_sets.fill(VK_NULL_HANDLE);
    }

    if (vk_pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
        vk_pipeline_layout = VK_NULL_HANDLE;
    }

    if (vk_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);
        vk_descriptor_set_layout = VK_NULL_HANDLE;
    }

    for (auto& uniform_buffer : uniform_buffers) {
        destroyBuffer(uniform_buffer);
    }

    destroyBuffer(index_buffer);
    destroyBuffer(vertex_buffer);
}

} // namespace

bool initialize() {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    buildCylinder(vertices, indices);

    vertex_count = static_cast<uint32_t>(vertices.size());
    index_count = static_cast<uint32_t>(indices.size());

    if (!createMappedBuffer(sizeof(Vertex) * vertices.size(),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            vertex_buffer) ||
        !uploadBuffer(vertex_buffer,
                      vertices.data(),
                      sizeof(Vertex) * vertices.size())) {
        std::cerr << "Failed to initialize vertex buffer\n";
        destroyApplicationResources();
        return false;
    }

    if (!createMappedBuffer(sizeof(uint32_t) * indices.size(),
                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            index_buffer) ||
        !uploadBuffer(index_buffer,
                      indices.data(),
                      sizeof(uint32_t) * indices.size())) {
        std::cerr << "Failed to initialize index buffer\n";
        destroyApplicationResources();
        return false;
    }

    for (auto& uniform_buffer : uniform_buffers) {
        if (!createMappedBuffer(sizeof(GlobalUniforms),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                uniform_buffer)) {
            std::cerr << "Failed to initialize uniform buffer\n";
            destroyApplicationResources();
            return false;
        }
    }

    if (!createDescriptorResources()) {
        destroyApplicationResources();
        return false;
    }

    if (!createGraphicsPipeline()) {
        destroyApplicationResources();
        return false;
    }

    resetControls();
    updateUniforms();
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);
    destroyApplicationResources();
}

void update(double time) {
    if (previous_time < 0.0) {
        previous_time = time;
    }

    const double raw_delta = time - previous_time;
    previous_time = time;
    const float delta = static_cast<float>(std::clamp(raw_delta, 0.0, 0.1));

    if (animation_playing) {
        animation_phase += delta * animation_speed;
    }

    ImGui::Begin("Lab 1 - Variant 8: Cylinder");
    ImGui::Text("Vulkan | 50 perimeter vertices per base");
    ImGui::Text("Total vertices: %u | indices: %u", vertex_count, index_count);
    ImGui::Separator();

    ImGui::Text("Projection");
    ImGui::RadioButton("Perspective", &projection_mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Orthographic", &projection_mode, 1);

    if (projection_mode == 0) {
        ImGui::SliderFloat("Field of view", &perspective_fov_degrees, 25.0f, 100.0f, "%.1f deg");
    } else {
        ImGui::SliderFloat("Ortho half-height", &orthographic_half_height, 1.5f, 8.0f, "%.2f");
    }
    ImGui::SliderFloat("Camera distance", &camera_distance, 3.5f, 15.0f, "%.2f");

    ImGui::Separator();
    ImGui::Text("Affine transformations");
    ImGui::DragFloat3("Position", position, 0.02f, -5.0f, 5.0f);
    ImGui::DragFloat3("Rotation", rotation_degrees, 0.5f, -360.0f, 360.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", object_scale, 0.01f, 0.10f, 3.0f);

    ImGui::Separator();
    ImGui::Text("Complex trajectory animation");
    if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
        animation_playing = !animation_playing;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart animation")) {
        animation_phase = 0.0f;
    }
    ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Vertical amplitude", &trajectory_height, 0.0f, 2.0f, "%.2f");

    ImGui::Separator();
    ImGui::Text("Colors");
    ImGui::ColorEdit3("Cylinder tint", object_color);
    ImGui::TextWrapped("The tint is multiplied by procedural per-vertex colors generated from local coordinates.");

    ImGui::Separator();
    ImGui::Text("Second object / descriptor set #2");
    ImGui::Checkbox("Draw second cylinder", &draw_second_object);
    ImGui::ColorEdit3("Second cylinder tint", second_object_color);

    ImGui::Separator();
    if (ImGui::Button("Reset all controls")) {
        resetControls();
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show ImGui demo", &show_demo_window);

    ImGui::End();

    if (show_demo_window) {
        ImGui::ShowDemoWindow(&show_demo_window);
    }

    updateUniforms();
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;

    vkResetCommandBuffer(fd.command_buffer, 0);

    const VkCommandBufferBeginInfo command_buffer_begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    if (vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin) != VK_SUCCESS) {
        std::cerr << "Failed to begin application command buffer\n";
        return;
    }

    VkClearValue clear_values[2]{};
    clear_values[0].color.float32[0] = 0.025f;
    clear_values[0].color.float32[1] = 0.030f;
    clear_values[0].color.float32[2] = 0.045f;
    clear_values[0].color.float32[3] = 1.0f;
    clear_values[1].depthStencil.depth = 1.0f;
    clear_values[1].depthStencil.stencil = 0;

    const VkRenderPassBeginInfo render_pass_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = static_cast<uint32_t>(std::size(clear_values)),
        .pClearValues = clear_values,
    };

    vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {
        .offset = { 0, 0 },
        .extent = context.swapchain_extent,
    };

    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

    const VkDeviceSize vertex_offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.buffer, &vertex_offset);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(fd.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk_pipeline_layout,
                            0,
                            1,
                            &vk_descriptor_sets[0],
                            0,
                            nullptr);
    vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

    if (draw_second_object) {
        vkCmdBindDescriptorSets(fd.command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                vk_pipeline_layout,
                                0,
                                1,
                                &vk_descriptor_sets[1],
                                0,
                                nullptr);
        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(fd.command_buffer);

    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to finish application command buffer\n";
    }
}

} // namespace application
