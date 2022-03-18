#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 64
#define WORKGROUP_SIZE 32

typedef struct VulkanDeviceAndComputeQueue
{
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
} VulkanDeviceAndComputeQueue;

typedef struct VulkanBufferAndMemory
{
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    uint32_t bufferSize;
} VulkanBufferAndMemory;

typedef struct VulkanPipelineDefinition
{
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
} VulkanPipelineDefinition;

typedef struct VulkanCommandBufferAndPool
{
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
} VulkanCommandBufferAndPool;

typedef struct VulkanInstance
{
    VkInstance instance;
    VkPhysicalDevice phyDevice;
    VulkanDeviceAndComputeQueue deviceAndQueue;

    uint32_t matrixASize;
    VulkanBufferAndMemory bufferAndMemory;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VulkanPipelineDefinition pipelineDefinition;
    VulkanCommandBufferAndPool commandBufferAndPool;
} VulkanInstance;

#define VK_ASSERT_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

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
    VK_ASSERT_RESULT(vkCreateInstance(&createInfo, NULL, &instance));

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
    VkPhysicalDevice result = devicesArray[0];
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

static VulkanDeviceAndComputeQueue
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
    VK_ASSERT_RESULT(vkCreateDevice(phyDevice, &deviceCreateInfo, NULL, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VulkanDeviceAndComputeQueue result = { 0 };
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

static VulkanBufferAndMemory
createBuffer(VkPhysicalDevice phyDevice, VkDevice device, uint32_t bufferSize)
{
    VkBufferCreateInfo bufferCreateInfo = { 0 };
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = { 0 };
    VK_ASSERT_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer));

    VkMemoryRequirements memoryReqs;
    vkGetBufferMemoryRequirements(device, buffer, &memoryReqs);

    VkDeviceMemory bufferMemory = { 0 };
    VkMemoryAllocateInfo allocateInfo = { 0 };
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryReqs.size;
    allocateInfo.memoryTypeIndex = findMemoryType(
        phyDevice,
        memoryReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VK_ASSERT_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory)); // allocate memory on device.
    VK_ASSERT_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));

    VulkanBufferAndMemory result = { 0 };
    result.buffer = buffer;
    result.bufferMemory = bufferMemory;
    result.bufferSize = bufferSize;
    return result;
}

static VkDescriptorSetLayout
createDescriptorSetLayout(VkDevice device)
{
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = { 0 };
    descriptorSetLayoutBinding.binding = 0;
    descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { 0 };
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 1;
    descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_ASSERT_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));

    return descriptorSetLayout;
}

static VkDescriptorPool
createDescriptorPool(VkDevice device)
{
    VkDescriptorPoolSize descriptorPoolSize = { 0 };
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = { 0 };
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    VkDescriptorPool descriptorPool;
    VK_ASSERT_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

    return descriptorPool;
}

static VkDescriptorSet
createDescriptorSet(VkDevice device, VulkanBufferAndMemory buffer,
                    VkDescriptorSetLayout descriptorSetLayout,
                    VkDescriptorPool descriptorPool)
{
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { 0 };
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    VK_ASSERT_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    // Bind buffer with descriptor set
    VkDescriptorBufferInfo descriptorBufferInfo = { 0 };
    descriptorBufferInfo.buffer = buffer.buffer;
    descriptorBufferInfo.offset = 0;
    descriptorBufferInfo.range = buffer.bufferSize;

    VkWriteDescriptorSet writeDescriptorSet = { 0 };
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet;
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;

    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);

    return descriptorSet;
}

static VulkanPipelineDefinition
createComputePipeline(VkDevice device, VkDescriptorSetLayout descriptorSetLayout)
{
    uint32_t filelength;
    uint8_t *spirvBinary = readEntireFile(&filelength, "shaders/comp.spv");
    VkShaderModuleCreateInfo createInfo = { 0 };
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = (uint32_t *)spirvBinary;
    createInfo.codeSize = filelength;

    VkShaderModule computeShaderModule;
    VK_ASSERT_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
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
    VK_ASSERT_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

    VkPipeline pipeline;
    VkComputePipelineCreateInfo pipelineCreateInfo = { 0 };
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    VK_ASSERT_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline));

    VulkanPipelineDefinition result = { 0 };
    result.pipeline = pipeline;
    result.pipelineLayout = pipelineLayout;
    return result;
}

static VulkanCommandBufferAndPool
createCommandBuffer(VulkanDeviceAndComputeQueue deviceAndQueue,
                    VkDescriptorSet descriptorSet,
                    VulkanPipelineDefinition pipeline)
{
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkCommandPoolCreateInfo commandPoolCreateInfo = { 0 };
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = deviceAndQueue.computeQueueFamilyIndex;
    VK_ASSERT_RESULT(vkCreateCommandPool(deviceAndQueue.device, &commandPoolCreateInfo, NULL, &commandPool));

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { 0 };
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VK_ASSERT_RESULT(vkAllocateCommandBuffers(deviceAndQueue.device, &commandBufferAllocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = { 0 };
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_ASSERT_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

    vkCmdDispatch(commandBuffer,
                  MATRIX_SIZE / WORKGROUP_SIZE, // how many workgroups to dispatch in X
                  MATRIX_SIZE / WORKGROUP_SIZE, // how many workgroups to dispatch in Y
                  1);

    VK_ASSERT_RESULT(vkEndCommandBuffer(commandBuffer));

    VulkanCommandBufferAndPool result = { 0 };
    result.commandBuffer = commandBuffer;
    result.commandPool = commandPool;
    return result;
}

static VulkanInstance
initalizeVulkan()
{
    VulkanInstance result = { 0 };
    VkInstance instance = createInstance();
    VkPhysicalDevice phyDevice = findPhysicalDevice(instance);
    VulkanDeviceAndComputeQueue deviceAndQueue = createDevice(phyDevice);

    uint32_t matrixASize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    VulkanBufferAndMemory bufferAndMemory = createBuffer(phyDevice, deviceAndQueue.device, matrixASize);
    VkDescriptorSetLayout descriptorSetLayout = createDescriptorSetLayout(deviceAndQueue.device);
    VkDescriptorPool descriptorPool = createDescriptorPool(deviceAndQueue.device);
    VkDescriptorSet descriptorSet = createDescriptorSet(deviceAndQueue.device, bufferAndMemory, descriptorSetLayout, descriptorPool);
    VulkanPipelineDefinition pipelineDefinition = createComputePipeline(deviceAndQueue.device, descriptorSetLayout);
    VulkanCommandBufferAndPool commandBufferAndPool = createCommandBuffer(deviceAndQueue, descriptorSet, pipelineDefinition);

    result.instance = instance;
    result.phyDevice = phyDevice;
    result.deviceAndQueue = deviceAndQueue;
    result.matrixASize = matrixASize;
    result.bufferAndMemory = bufferAndMemory;
    result.descriptorSetLayout = descriptorSetLayout;
    result.descriptorPool = descriptorPool;
    result.descriptorSet = descriptorSet;
    result.pipelineDefinition = pipelineDefinition;
    result.commandBufferAndPool = commandBufferAndPool;

    return result;
}

static void
runCommandBuffer(VulkanInstance instance)
{
    VkSubmitInfo submitInfo = { 0 };
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &instance.commandBufferAndPool.commandBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = { 0 };
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;

    VK_ASSERT_RESULT(vkCreateFence(instance.deviceAndQueue.device, &fenceCreateInfo, NULL, &fence));
    VK_ASSERT_RESULT(vkQueueSubmit(instance.deviceAndQueue.computeQueue, 1, &submitInfo, fence));
    VK_ASSERT_RESULT(vkWaitForFences(instance.deviceAndQueue.device, 1, &fence, VK_TRUE, 100000000000));

    vkDestroyFence(instance.deviceAndQueue.device, fence, NULL);
}

int main()
{
    VulkanInstance instance = initalizeVulkan();
    runCommandBuffer(instance);

    // Verify working matrix
    void *mappedMemory = NULL;
    vkMapMemory(instance.deviceAndQueue.device,
                instance.bufferAndMemory.bufferMemory, 0,
                instance.bufferAndMemory.bufferSize, 0,
                &mappedMemory);
    float *floatMappedMemory = mappedMemory;

    printf("floatMappedMemory = %f\n", floatMappedMemory[0]);

    return 0;
}