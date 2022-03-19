#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MATRIX_SIZE 4096
#define WORKGROUP_SIZE 32

#define ARRAY_LEN(x) (sizeof(x)/sizeof(x[0]))

typedef struct VKDeviceAndComputeQueue
{
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
} VKDeviceAndComputeQueue;

typedef struct VKBufferAndMemory
{
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    uint32_t bufferSize;
} VKBufferAndMemory;

typedef struct VKPipelineDefinition
{
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
} VKPipelineDefinition;

typedef struct VKState
{
    VkInstance instance;
    VkPhysicalDevice phyDevice;

    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;

    uint32_t matrixASize;
    VKBufferAndMemory matrixABufferAndMemory;
    VKBufferAndMemory matrixBBufferAndMemory;
    VKBufferAndMemory matrixCBufferAndMemory;

    VKBufferAndMemory matrixADevice;
    VKBufferAndMemory matrixBDevice;
    VKBufferAndMemory matrixCDevice;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkQueryPool queryPool;
    VKPipelineDefinition pipelineDefinition;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
} VKState;

#define VK_CALL(f) 																				        \
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__);                  \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

#ifdef _WIN32
#include <Windows.h>
double getWallTime()
{
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){ assert(false && "QueryPerformanceFrequency() error"); }
    if (!QueryPerformanceCounter(&time)){ assert(false && "QueryPerformanceCounter() error"); }
    return (double)time.QuadPart / freq.QuadPart;
}
#endif

static uint8_t *
readEntireFile(uint32_t *fileLength, const char *fileName)
{
    FILE *fp = fopen(fileName, "rb");
    if (fp == NULL)
    {
        printf("Could not find or open file: %s\n", fileName);
    }

    fseek(fp, 0, SEEK_END);
    *fileLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *str = malloc(sizeof(char) * (*fileLength + 1));
    fread(str, *fileLength, sizeof(char), fp);
    fclose(fp);
    str[*fileLength] = 0x0;

    return (uint8_t *)str;
}

static VkInstance
createInstance()
{
    VkApplicationInfo appInfo = { 0 };
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Compute";
    appInfo.applicationVersion = 0;
    appInfo.pEngineName = "notanengine";
    appInfo.engineVersion = 0;
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = { 0 };
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &appInfo;

    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = NULL;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = NULL;

    VkInstance instance;
    VK_CALL(vkCreateInstance(&createInfo, NULL, &instance));

    return instance;
}

static VkPhysicalDevice
findPhysicalDevice(VkInstance instance)
{
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0)
    {
        assert(false && "No physical devices found!");
    }

    VkPhysicalDevice *devicesArray = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    VkResult ss = vkEnumeratePhysicalDevices(instance, &deviceCount, devicesArray);
    
    // TODO(radomski): Choose the most powerfull GPU
    VkPhysicalDevice result = devicesArray[1];
    free(devicesArray);
    return result;
}

static uint32_t
getComputeQueueFamilyIndex(VkPhysicalDevice phyDevice)
{
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, NULL);

    VkQueueFamilyProperties *queueFamiliesArray = malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, queueFamiliesArray);

    uint32_t queueFamilyIndex = 0;
    for(; queueFamilyIndex < queueFamilyCount; queueFamilyIndex++)
    {
        VkQueueFamilyProperties queueProperty = queueFamiliesArray[queueFamilyIndex];

        if(queueProperty.queueCount > 0 && (queueProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            break;
        }
    }

    if(queueFamilyIndex == queueFamilyCount)
    {
        assert(false && "Did not found queue that supports compute operations!");
    }

    free(queueFamiliesArray);
    return queueFamilyIndex;
}

static VKDeviceAndComputeQueue
createDevice(VkPhysicalDevice phyDevice)
{
    VkDeviceQueueCreateInfo queueCreateInfo = { 0 };
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    uint32_t queueFamilyIndex = getComputeQueueFamilyIndex(phyDevice);
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriorities = 1.0;
    queueCreateInfo.pQueuePriorities = &queuePriorities;

    VkDeviceCreateInfo deviceCreateInfo = { 0 };
    VkPhysicalDeviceFeatures deviceFeatures = { 0 };

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = 0; // TODO(radomski): add validation layers
    deviceCreateInfo.ppEnabledLayerNames = NULL;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VkDevice device = { 0 };
    VK_CALL(vkCreateDevice(phyDevice, &deviceCreateInfo, NULL, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VKDeviceAndComputeQueue result = { 0 };
    result.device = device;
    result.computeQueue = queue;
    result.computeQueueFamilyIndex = queueFamilyIndex;
    return result;
}

static uint32_t
findMemoryType(VkPhysicalDevice phyDevice, uint32_t memoryTypeBits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memoryProps;
    vkGetPhysicalDeviceMemoryProperties(phyDevice, &memoryProps);

    // TODO(radomski): Read about it
    // Source:
    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
    for (uint32_t memoryIndex = 0; memoryIndex < memoryProps.memoryTypeCount; ++memoryIndex)
    {
        if ((memoryTypeBits & (1 << memoryIndex)) &&
            ((memoryProps.memoryTypes[memoryIndex].propertyFlags & props) == props))
            return memoryIndex;
    }

    assert(false && "Failed to find the memory type!");
    return 0; // Just to wave away the warning
}

static VKBufferAndMemory
createBuffer(VKState *state, uint32_t bufferSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlagBits memoryFlags)
{
    VkBufferCreateInfo bufferCreateInfo = { 0 };
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = usageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = { 0 };
    VK_CALL(vkCreateBuffer(state->device, &bufferCreateInfo, NULL, &buffer));

    VkMemoryRequirements memoryReqs;
    vkGetBufferMemoryRequirements(state->device, buffer, &memoryReqs);

    VkDeviceMemory bufferMemory = { 0 };
    VkMemoryAllocateInfo allocateInfo = { 0 };
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryReqs.size;
    allocateInfo.memoryTypeIndex = findMemoryType(state->phyDevice, memoryReqs.memoryTypeBits, memoryFlags);

    VK_CALL(vkAllocateMemory(state->device, &allocateInfo, NULL, &bufferMemory)); // allocate memory on device.
    VK_CALL(vkBindBufferMemory(state->device, buffer, bufferMemory, 0));

    VKBufferAndMemory result = { 0 };
    result.buffer = buffer;
    result.bufferMemory = bufferMemory;
    result.bufferSize = bufferSize;
    return result;
}

static void
bindDescriptorSetWithBuffers(VKState *state,
                             VKBufferAndMemory matrixA,
                             VKBufferAndMemory matrixB,
                             VKBufferAndMemory matrixC)
{
    // Bind buffer with descriptor set
    VkDescriptorBufferInfo descriptorBufferInfoArray[3] = { 0 };
    descriptorBufferInfoArray[0].buffer = matrixA.buffer;
    descriptorBufferInfoArray[0].offset = 0;
    descriptorBufferInfoArray[0].range = matrixA.bufferSize;

    descriptorBufferInfoArray[1].buffer = matrixB.buffer;
    descriptorBufferInfoArray[1].offset = 0;
    descriptorBufferInfoArray[1].range = matrixB.bufferSize;

    descriptorBufferInfoArray[2].buffer = matrixC.buffer;
    descriptorBufferInfoArray[2].offset = 0;
    descriptorBufferInfoArray[2].range = matrixC.bufferSize;

    VkWriteDescriptorSet writeDescriptorSetsArray[3] = { 0 };
    writeDescriptorSetsArray[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[0].dstSet = state->descriptorSet;
    writeDescriptorSetsArray[0].dstBinding = 0;
    writeDescriptorSetsArray[0].descriptorCount = 1;
    writeDescriptorSetsArray[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[0].pBufferInfo = &descriptorBufferInfoArray[0];

    writeDescriptorSetsArray[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[1].dstSet = state->descriptorSet;
    writeDescriptorSetsArray[1].dstBinding = 1;
    writeDescriptorSetsArray[1].descriptorCount = 1;
    writeDescriptorSetsArray[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[1].pBufferInfo = &descriptorBufferInfoArray[1];

    writeDescriptorSetsArray[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[2].dstSet = state->descriptorSet;
    writeDescriptorSetsArray[2].dstBinding = 2;
    writeDescriptorSetsArray[2].descriptorCount = 1;
    writeDescriptorSetsArray[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[2].pBufferInfo = &descriptorBufferInfoArray[2];

    vkUpdateDescriptorSets(state->device, ARRAY_LEN(writeDescriptorSetsArray), writeDescriptorSetsArray, 0, NULL);
}

static void
createMatrixBuffers(VKState *state)
{
    uint32_t matrixASize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    uint32_t matrixBSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    uint32_t matrixCSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    state->matrixABufferAndMemory = createBuffer(state, matrixASize, usageFlags, memoryFlags);
    state->matrixBBufferAndMemory = createBuffer(state, matrixBSize, usageFlags, memoryFlags);
    state->matrixCBufferAndMemory = createBuffer(state, matrixCSize, usageFlags, memoryFlags);

    // On device memory buffers 
    state->matrixADevice = createBuffer(state, matrixASize, usageFlags, deviceMemoryFlags);
    state->matrixBDevice = createBuffer(state, matrixBSize, usageFlags, deviceMemoryFlags);
    state->matrixCDevice = createBuffer(state, matrixCSize, usageFlags, deviceMemoryFlags);
}

static void
copyStagingBufferToDevice(VKState *state, VKBufferAndMemory staging, VKBufferAndMemory device)
{
    assert(staging.bufferSize == device.bufferSize);
    VkCommandBufferAllocateInfo allocInfo = { 0 };
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = state->commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer = { 0 };
    vkAllocateCommandBuffers(state->device, &allocInfo, &cmdBuffer);

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    VkBufferCopy copyRegion = { .size = staging.bufferSize };
    vkCmdCopyBuffer(cmdBuffer, staging.buffer, device.buffer, 1, &copyRegion);

    vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer,
    };

    vkQueueSubmit(state->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(state->computeQueue);

    vkFreeCommandBuffers(state->device, state->commandPool, 1, &cmdBuffer);
}

static VkDescriptorSetLayout
createDescriptorSetLayout(VkDevice device)
{
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindingArray[3] = { 0 };
    descriptorSetLayoutBindingArray[0].binding = 0;
    descriptorSetLayoutBindingArray[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindingArray[0].descriptorCount = 1;
    descriptorSetLayoutBindingArray[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindingArray[1].binding = 1;
    descriptorSetLayoutBindingArray[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindingArray[1].descriptorCount = 1;
    descriptorSetLayoutBindingArray[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindingArray[2].binding = 2;
    descriptorSetLayoutBindingArray[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindingArray[2].descriptorCount = 1;
    descriptorSetLayoutBindingArray[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { 0 };
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = ARRAY_LEN(descriptorSetLayoutBindingArray);
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindingArray;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CALL(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));

    return descriptorSetLayout;
}

static VkDescriptorPool
createDescriptorPool(VkDevice device)
{
    VkDescriptorPoolSize descriptorPoolSize = { 0 };
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { 0 };
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    VkDescriptorPool descriptorPool;
    VK_CALL(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

    return descriptorPool;
}

static VkDescriptorSet
createDescriptorSet(VkDevice device,
                    VkDescriptorSetLayout descriptorSetLayout,
                    VkDescriptorPool descriptorPool)
{
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { 0 };
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    VK_CALL(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));
    return descriptorSet;
}

static VkQueryPool
createQueryPool(VkDevice device)
{
    VkQueryPoolCreateInfo queryPoolCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = NULL,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2
    };

    VkQueryPool queryPool = { 0 };
    VK_CALL(vkCreateQueryPool(device, &queryPoolCreateInfo, NULL, &queryPool));
    return queryPool;
}

static VKPipelineDefinition
createComputePipeline(VkDevice device, VkDescriptorSetLayout descriptorSetLayout)
{
    uint32_t filelength;
    uint8_t *spirvBinary = readEntireFile(&filelength, "shaders/shader_faster.spirv");
    VkShaderModuleCreateInfo createInfo = { 0 };
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = (uint32_t *)spirvBinary;
    createInfo.codeSize = filelength;

    VkShaderModule computeShaderModule;
    VK_CALL(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
    free(spirvBinary);

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = { 0 };
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = computeShaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { 0 };
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CALL(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

    VkPipeline pipeline;
    VkComputePipelineCreateInfo pipelineCreateInfo = { 0 };
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    VK_CALL(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline));

    VKPipelineDefinition result = { 0 };
    result.pipeline = pipeline;
    result.pipelineLayout = pipelineLayout;
    return result;
}

static void
createCommandBuffer(VKState *state)
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { 0 };
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = state->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VK_CALL(vkAllocateCommandBuffers(state->device, &commandBufferAllocateInfo, &state->commandBuffer));

    VkCommandBufferBeginInfo beginInfo = { 0 };
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(state->commandBuffer, &beginInfo));

    VkCommandBufferBeginInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL,
    };

    vkCmdResetQueryPool(state->commandBuffer, state->queryPool, 0, 1);
    vkCmdWriteTimestamp(state->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state->queryPool, 0);

    vkCmdBindPipeline(state->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipelineDefinition.pipeline);
    vkCmdBindDescriptorSets(state->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipelineDefinition.pipelineLayout, 0, 1, &state->descriptorSet, 0, NULL);

    vkCmdDispatch(state->commandBuffer,
                  MATRIX_SIZE / WORKGROUP_SIZE, // how many workgroups to dispatch in X
                  MATRIX_SIZE / WORKGROUP_SIZE, // how many workgroups to dispatch in Y
                  1);

    vkCmdWriteTimestamp(state->commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state->queryPool, 1);

    VK_CALL(vkEndCommandBuffer(state->commandBuffer));
}

static VKState
initalizeVulkan()
{
    VKState result = { 0 };
    VkInstance instance = createInstance();
    VkPhysicalDevice phyDevice = findPhysicalDevice(instance);
    VKDeviceAndComputeQueue deviceAndQueue = createDevice(phyDevice);

    VkDescriptorSetLayout descriptorSetLayout = createDescriptorSetLayout(deviceAndQueue.device);
    VkDescriptorPool descriptorPool = createDescriptorPool(deviceAndQueue.device);
    VkDescriptorSet descriptorSet = createDescriptorSet(deviceAndQueue.device, descriptorSetLayout, descriptorPool);

    result.instance = instance;
    result.phyDevice = phyDevice;
    result.device = deviceAndQueue.device;
    result.computeQueue = deviceAndQueue.computeQueue;
    result.computeQueueFamilyIndex = deviceAndQueue.computeQueueFamilyIndex;
    result.descriptorSetLayout = descriptorSetLayout;
    result.descriptorPool = descriptorPool;
    result.descriptorSet = descriptorSet;

    VkCommandPoolCreateInfo commandPoolCreateInfo = { 0 };
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = result.computeQueueFamilyIndex;
    VK_CALL(vkCreateCommandPool(result.device, &commandPoolCreateInfo, NULL, &result.commandPool));

    VkQueryPool queryPool = createQueryPool(deviceAndQueue.device);
    VKPipelineDefinition pipelineDefinition = createComputePipeline(deviceAndQueue.device, descriptorSetLayout);

    result.queryPool = queryPool;
    result.pipelineDefinition = pipelineDefinition;

    return result;
}

static void
runCommandBuffer(VKState instance)
{
    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &instance.commandBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {0};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;

    for(int i = 0; i < 8; i++)
    {
        VK_CALL(vkCreateFence(instance.device, &fenceCreateInfo, NULL, &fence));
        VK_CALL(vkQueueSubmit(instance.computeQueue, 1, &submitInfo, fence));
        VK_CALL(vkWaitForFences(instance.device, 1, &fence, VK_TRUE, 100000000000));
        uint64_t ts[2];
        VK_CALL(vkGetQueryPoolResults(instance.device, instance.queryPool,
                                               0, 2, sizeof(uint64_t) * 2, ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
        double execTime = (ts[1] - ts[0]) / 1e9;
        double gflops = ((2 * pow(MATRIX_SIZE, 3)) / execTime) / 1e9;
        printf("It took %fs [%f GFLOPS]\n", execTime, gflops);

        vkDestroyFence(instance.device, fence, NULL);
    }
}

static void
fillMatrixWithData(VKState *state, VKBufferAndMemory buffer)
{
    void *mappedMemory = NULL;
    vkMapMemory(state->device, buffer.bufferMemory, 0, buffer.bufferSize, 0, &mappedMemory);
    float *floatMappedMemory = mappedMemory;

    for (int row = 0; row < MATRIX_SIZE; row++)
    {
        for (int col = 0; col < MATRIX_SIZE; col++)
        {
            floatMappedMemory[row * MATRIX_SIZE + col] = 2.0;
        }
    }

    vkUnmapMemory(state->device, buffer.bufferMemory);
}

int main()
{
    VKState state = initalizeVulkan();

    createMatrixBuffers(&state);

    // Verify working sgemm
    fillMatrixWithData(&state, state.matrixABufferAndMemory);
    fillMatrixWithData(&state, state.matrixBBufferAndMemory);

    copyStagingBufferToDevice(&state, state.matrixABufferAndMemory, state.matrixADevice);
    copyStagingBufferToDevice(&state, state.matrixBBufferAndMemory, state.matrixBDevice);
    copyStagingBufferToDevice(&state, state.matrixCBufferAndMemory, state.matrixCDevice);

    bindDescriptorSetWithBuffers(&state, state.matrixADevice, state.matrixBDevice, state.matrixCDevice);
    createCommandBuffer(&state);

    runCommandBuffer(state);

    {
        void *mappedMemory = NULL;
        vkMapMemory(state.device,
                    state.matrixCBufferAndMemory.bufferMemory, 0,
                    state.matrixCBufferAndMemory.bufferSize, 0,
                    &mappedMemory);
        float *floatMappedMemory = mappedMemory;

        printf("floatMappedMemory = %f\n", floatMappedMemory[0]);
    }

    return 0;
}