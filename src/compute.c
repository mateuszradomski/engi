#define _CRT_SECURE_NO_WARNINGS

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "compute.h"
#include "expVector.h"

#define WORKGROUP_SIZE 32

#define ARRAY_LEN(x) (sizeof(x)/sizeof(x[0]))

#define RUNS_PER_VERSION 5

typedef struct VKDeviceAndComputeQueue
{
    VkDevice device;
    VkQueue computeQueue;
    u32 computeQueueFamilyIndex;
} VKDeviceAndComputeQueue;

typedef struct VKBufferAndMemory
{
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    u32 bufferSize;
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
    u32 computeQueueFamilyIndex;

    VkQueryPool queryPool;
    VkCommandPool commandPool;
} VKState;

typedef struct ScenarioELLSimple
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioELLSimple;

typedef struct ScenarioELL2Buffer
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matHost;
    VKBufferAndMemory matFloatHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matDevice;
    VKBufferAndMemory matFloatDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioELL2Buffer;

typedef struct ScenarioSELL
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matHeaderAndColIndexHost;
    VKBufferAndMemory matRowOffsetsHost;
    VKBufferAndMemory matFloatHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matHeaderAndColIndexDevice;
    VKBufferAndMemory matRowOffsetsDevice;
    VKBufferAndMemory matFloatDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioSELL;

#define VK_CALL(f) 																				        \
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__);                  \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

u32 glob_rand_state = 0x34fae2;

static u32
xorshift32()
{
	u32 *state = &glob_rand_state;
	u32 x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return *state = x;
}

static double
randomUnilateral()
{
    u32 rand = xorshift32();
    float result = ((float)(rand) / (float)(0xffffffff));
    return result;
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

static bool
StringsMatchRaw(char *str1, u32 str1Length, char *str2, u32 str2Length) {
    if(str1Length == str2Length) {
        for(int i = 0; i < str1Length; i++) {
            if(str1[i] != str2[i]) {
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}

static bool
StringsMatch(Str str1, Str str2)
{
    return StringsMatchRaw(str1.bytes, str1.length, str2.bytes, str2.length);
}


static StrSplitIter
StringSplit(Str string, char *delim)
{
  StrSplitIter it = { 0 };

  it.str = string.bytes;
  it.strLength = string.length;
  it.head = string.bytes;
  it.delim = delim;
  it.delimLength = strlen(delim);

  return it;
}

static Str
NextInSplit(StrSplitIter *it)
{
  Str result = { 0 };

  u32 alreadyRead = (u32)(it->head - it->str);

  if(alreadyRead < it->strLength) {
    result.bytes = it->head;
    u32 bytesLeft = it->strLength - alreadyRead;

    for (u32 i = 0;
         (i < bytesLeft) && (it->delimLength < bytesLeft) &&
          !StringsMatchRaw(it->head, it->delimLength, it->delim, it->delimLength);
         i++)
    {
        it->head++;
        result.length += 1;
    }

    if(StringsMatchRaw(it->head, it->delimLength, it->delim, it->delimLength)) {
      it->head += it->delimLength;
    }
  }

  return result;
}

static Data
readEntireFile(const char *fileName)
{
    Data result = { 0 };

    FILE *fp = fopen(fileName, "rb");
    if (fp == NULL)
    {
        printf("Could not find or open file: %s\n", fileName);
    }

    fseek(fp, 0, SEEK_END);
    result.length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    u8 *str = malloc(sizeof(u8) * (result.length + 1));
    fread(str, result.length, sizeof(char), fp);
    fclose(fp);
    str[result.length] = 0x0;
    result.bytes = str;

    return result;
}

static Str
readEntireFileStr(const char *filename)
{
    Str result = { 0 };
    Data data = readEntireFile(filename);

    result.bytes = (char *)data.bytes;
    result.length = data.length - 1;

    return result;
}

static void
printRunInfo(RunInformation *runInfo, u32 len)
{
    const char *col1 = "Exec time [s]";
    const char *col2 = "GFLOPs";
    printf("| %15s | %15s |\n", col1, col2);
    for(u32 i = 0; i < len; i++)
    {
        printf("| %15f | %15f |\n", runInfo[i].time, runInfo[i].gflops);
    }
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
    appInfo.apiVersion = VK_API_VERSION_1_2;

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
    u32 deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0)
    {
        assert(false && "No physical devices found!");
    }

    VkPhysicalDevice *devicesArray = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    VkResult ss = vkEnumeratePhysicalDevices(instance, &deviceCount, devicesArray);
    
    // TODO(radomski): Choose the most powerfull GPU
    printf("[Vulkan Init]: deviceCount = %u\n", deviceCount);
    VkPhysicalDevice result = devicesArray[0];
    free(devicesArray);

    VkPhysicalDeviceProperties props = { 0 };
    vkGetPhysicalDeviceProperties(result, &props);
    printf("[Vulkan Init]: Chosen device (%s)\n", props.deviceName);

    return result;
}

static u32
getComputeQueueFamilyIndex(VkPhysicalDevice phyDevice)
{
    u32 queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, NULL);

    VkQueueFamilyProperties *queueFamiliesArray = malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, queueFamiliesArray);

    u32 queueFamilyIndex = 0;
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
    u32 queueFamilyIndex = getComputeQueueFamilyIndex(phyDevice);
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

static u32
findMemoryType(VkPhysicalDevice phyDevice, u32 memoryTypeBits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memoryProps;
    vkGetPhysicalDeviceMemoryProperties(phyDevice, &memoryProps);

    // TODO(radomski): Read about it
    // Source:
    // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
    for (u32 memoryIndex = 0; memoryIndex < memoryProps.memoryTypeCount; ++memoryIndex)
    {
        if ((memoryTypeBits & (1 << memoryIndex)) &&
            ((memoryProps.memoryTypes[memoryIndex].propertyFlags & props) == props))
            return memoryIndex;
    }

    assert(false && "Failed to find the memory type!");
    return 0; // Just to wave away the warning
}

static VKBufferAndMemory
createBuffer(VKState *state, u32 bufferSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlagBits memoryFlags)
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
bindScenarioELLSimpleDescriptorSetWithBuffers(VKState *state, ScenarioELLSimple *scnELLSimple)
{
    // Bind buffer with descriptor set
    VkDescriptorBufferInfo descriptorBufferInfoArray[3] = { 0 };
    descriptorBufferInfoArray[0].buffer = scnELLSimple->matDevice.buffer;
    descriptorBufferInfoArray[0].offset = 0;
    descriptorBufferInfoArray[0].range = scnELLSimple->matDevice.bufferSize;

    descriptorBufferInfoArray[1].buffer = scnELLSimple->inVecDevice.buffer;
    descriptorBufferInfoArray[1].offset = 0;
    descriptorBufferInfoArray[1].range = scnELLSimple->inVecDevice.bufferSize;

    descriptorBufferInfoArray[2].buffer = scnELLSimple->outVecDevice.buffer;
    descriptorBufferInfoArray[2].offset = 0;
    descriptorBufferInfoArray[2].range = scnELLSimple->outVecDevice.bufferSize;

    VkWriteDescriptorSet writeDescriptorSetsArray[3] = { 0 };
    writeDescriptorSetsArray[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[0].dstSet = scnELLSimple->descriptorSet;
    writeDescriptorSetsArray[0].dstBinding = 0;
    writeDescriptorSetsArray[0].descriptorCount = 1;
    writeDescriptorSetsArray[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[0].pBufferInfo = &descriptorBufferInfoArray[0];

    writeDescriptorSetsArray[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[1].dstSet = scnELLSimple->descriptorSet;
    writeDescriptorSetsArray[1].dstBinding = 1;
    writeDescriptorSetsArray[1].descriptorCount = 1;
    writeDescriptorSetsArray[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[1].pBufferInfo = &descriptorBufferInfoArray[1];

    writeDescriptorSetsArray[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[2].dstSet = scnELLSimple->descriptorSet;
    writeDescriptorSetsArray[2].dstBinding = 2;
    writeDescriptorSetsArray[2].descriptorCount = 1;
    writeDescriptorSetsArray[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[2].pBufferInfo = &descriptorBufferInfoArray[2];

    vkUpdateDescriptorSets(state->device, ARRAY_LEN(writeDescriptorSetsArray), writeDescriptorSetsArray, 0, NULL);
}

static void
bindScenarioELL2BufferDescriptorSetWithBuffers(VKState *state, ScenarioELL2Buffer *scnELL2Buffer)
{
    // Bind buffer with descriptor set
    VkDescriptorBufferInfo descriptorBufferInfoArray[4] = { 0 };
    descriptorBufferInfoArray[0].buffer = scnELL2Buffer->matDevice.buffer;
    descriptorBufferInfoArray[0].offset = 0;
    descriptorBufferInfoArray[0].range = scnELL2Buffer->matDevice.bufferSize;

    descriptorBufferInfoArray[1].buffer = scnELL2Buffer->matFloatDevice.buffer;
    descriptorBufferInfoArray[1].offset = 0;
    descriptorBufferInfoArray[1].range = scnELL2Buffer->matFloatDevice.bufferSize;

    descriptorBufferInfoArray[2].buffer = scnELL2Buffer->inVecDevice.buffer;
    descriptorBufferInfoArray[2].offset = 0;
    descriptorBufferInfoArray[2].range = scnELL2Buffer->inVecDevice.bufferSize;

    descriptorBufferInfoArray[3].buffer = scnELL2Buffer->outVecDevice.buffer;
    descriptorBufferInfoArray[3].offset = 0;
    descriptorBufferInfoArray[3].range = scnELL2Buffer->outVecDevice.bufferSize;

    VkWriteDescriptorSet writeDescriptorSetsArray[4] = { 0 };
    writeDescriptorSetsArray[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[0].dstSet = scnELL2Buffer->descriptorSet;
    writeDescriptorSetsArray[0].dstBinding = 0;
    writeDescriptorSetsArray[0].descriptorCount = 1;
    writeDescriptorSetsArray[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[0].pBufferInfo = &descriptorBufferInfoArray[0];

    writeDescriptorSetsArray[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[1].dstSet = scnELL2Buffer->descriptorSet;
    writeDescriptorSetsArray[1].dstBinding = 1;
    writeDescriptorSetsArray[1].descriptorCount = 1;
    writeDescriptorSetsArray[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[1].pBufferInfo = &descriptorBufferInfoArray[1];

    writeDescriptorSetsArray[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[2].dstSet = scnELL2Buffer->descriptorSet;
    writeDescriptorSetsArray[2].dstBinding = 2;
    writeDescriptorSetsArray[2].descriptorCount = 1;
    writeDescriptorSetsArray[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[2].pBufferInfo = &descriptorBufferInfoArray[2];

    writeDescriptorSetsArray[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[3].dstSet = scnELL2Buffer->descriptorSet;
    writeDescriptorSetsArray[3].dstBinding = 3;
    writeDescriptorSetsArray[3].descriptorCount = 1;
    writeDescriptorSetsArray[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[3].pBufferInfo = &descriptorBufferInfoArray[3];

    vkUpdateDescriptorSets(state->device, ARRAY_LEN(writeDescriptorSetsArray), writeDescriptorSetsArray, 0, NULL);
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
createConsecutiveDescriptorSetLayout(VkDevice device, u32 num)
{
    u32 size = num * sizeof(VkDescriptorSetLayoutBinding);
    VkDescriptorSetLayoutBinding *descriptorSetLayoutBindingArray = malloc(size);
    memset(descriptorSetLayoutBindingArray, 0, size);

    for(u32 i = 0; i < num; i++)
    {
        descriptorSetLayoutBindingArray[i].binding = i;
        descriptorSetLayoutBindingArray[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindingArray[i].descriptorCount = 1;
        descriptorSetLayoutBindingArray[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { 0 };
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = num;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindingArray;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CALL(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));

    free(descriptorSetLayoutBindingArray);

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
createComputePipeline(VkDevice device, const char *shaderPath, VkDescriptorSetLayout descriptorSetLayout)
{
    Data spirvData = readEntireFile(shaderPath);
    VkShaderModuleCreateInfo createInfo = { 0 };
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = (u32 *)spirvData.bytes;
    createInfo.codeSize = spirvData.length;

    VkShaderModule computeShaderModule;
    VK_CALL(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
    free(spirvData.bytes);

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

static VkCommandBuffer
createCommandBuffer(VKState *state, VKPipelineDefinition *pipelineDefinition, VkDescriptorSet *descriptorSet,
                    u32 dispatchX, u32 dispatchY, u32 dispatchZ)
{
    VkCommandBuffer result = { 0 };

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { 0 };
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = state->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VK_CALL(vkAllocateCommandBuffers(state->device, &commandBufferAllocateInfo, &result));

    VkCommandBufferBeginInfo beginInfo = { 0 };
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(result, &beginInfo));

    VkCommandBufferBeginInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL,
    };

    vkCmdBindPipeline(result, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineDefinition->pipeline);
    vkCmdBindDescriptorSets(result, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineDefinition->pipelineLayout, 0, 1, descriptorSet, 0, NULL);

    vkCmdResetQueryPool(result, state->queryPool, 0, 1);
    vkCmdWriteTimestamp(result, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state->queryPool, 0);

    vkCmdDispatch(result, dispatchX, dispatchY, dispatchZ);

    vkCmdWriteTimestamp(result, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state->queryPool, 1);

    VK_CALL(vkEndCommandBuffer(result));

    return result;
}

static VKState
initalizeVulkan()
{
    VKState result = { 0 };
    VkInstance instance = createInstance();
    VkPhysicalDevice phyDevice = findPhysicalDevice(instance);
    VKDeviceAndComputeQueue deviceAndQueue = createDevice(phyDevice);

    result.instance = instance;
    result.phyDevice = phyDevice;
    result.device = deviceAndQueue.device;
    result.computeQueue = deviceAndQueue.computeQueue;
    result.computeQueueFamilyIndex = deviceAndQueue.computeQueueFamilyIndex;

    VkCommandPoolCreateInfo commandPoolCreateInfo = { 0 };
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = result.computeQueueFamilyIndex;
    VK_CALL(vkCreateCommandPool(result.device, &commandPoolCreateInfo, NULL, &result.commandPool));

    result.queryPool = createQueryPool(deviceAndQueue.device);

    return result;
}

static double
runCommandBuffer(VKState *instance, VkCommandBuffer *commandBuffer)
{
    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = commandBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {0};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;

    VK_CALL(vkCreateFence(instance->device, &fenceCreateInfo, NULL, &fence));
    VK_CALL(vkQueueSubmit(instance->computeQueue, 1, &submitInfo, fence));
    VK_CALL(vkWaitForFences(instance->device, 1, &fence, VK_TRUE, 100000000000));
    u64 ts[2];
    VK_CALL(vkGetQueryPoolResults(instance->device, instance->queryPool,
                                           0, 2, sizeof(u64) * 2, ts, sizeof(u64), VK_QUERY_RESULT_64_BIT));
    vkDestroyFence(instance->device, fence, NULL);

    double execTime = (ts[1] - ts[0]) / 1e9;
    return execTime;
}

static void
printMatrix(float *data, u32 matrixSize)
{
    for (u32 row = 0; row < matrixSize; row++)
    {
        for (u32 col = 0; col < matrixSize; col++)
        {
            printf("%f ", data[row * matrixSize + col]);
        }

        printf("\n");
    }
}

static void
printBufferedVector(VKState *state, VKBufferAndMemory buffer)
{
    void *mappedMemory = NULL;
    vkMapMemory(state->device, buffer.bufferMemory, 0, buffer.bufferSize, 0, &mappedMemory);
    float *mappedMemoryFloat = (float *)mappedMemory;
    for(int i = 0; i < 10; i++) {
        printf("[%f]", mappedMemoryFloat[i]);
    }
    printf("\n");
    vkUnmapMemory(state->device, buffer.bufferMemory);
}

static COOMatrix
ReadMatrixFormatToCOO(const char *filename)
{
    Str str = readEntireFileStr(filename);
    COOMatrix result = { 0 };

    bool isSymmetric = false;

    StrSplitIter lineIter = StringSplit(str, "\r\n");
    Str line = NextInSplit(&lineIter);

    {
        // First line of the header contains info about encoding
        StrSplitIter partIter = StringSplit(line, " ");
        Str part = { 0 };
        while((part = NextInSplit(&partIter)).bytes != NULL)
        {
            Str sym = LIT_STR_TO_STRING_SLICE("symmetric");
            if(StringsMatch(part, sym)) {
                isSymmetric = true;
            }
        }
    }

    while((line = NextInSplit(&lineIter)).bytes != NULL)
    {
        if(line.bytes[0] != '%') {
            break;
        }
    }

    u32 totalDataAllocated = 0;

    {
        // This line has M x N [element count]
        StrSplitIter partIter = StringSplit(line, " ");
        Str MStr = NextInSplit(&partIter);
        Str NStr = NextInSplit(&partIter);
        Str ElementNumStr = NextInSplit(&partIter);
        assert(MStr.bytes && NStr.bytes && ElementNumStr.bytes);

        u32 factor = isSymmetric ? 2 : 1;
        result.elementNum = atoi(ElementNumStr.bytes) * factor;
        u32 toAllocate = result.elementNum * sizeof(result.data[0]);
        result.data = malloc(toAllocate);
        result.row = malloc(toAllocate);
        result.col = malloc(toAllocate);
        totalDataAllocated += 3 * toAllocate;

        printf("[COOMatrix Parse]: MStr = %.*s\n", MStr.length, MStr.bytes);
        printf("[COOMatrix Parse]: NStr = %.*s\n", NStr.length, NStr.bytes);
        printf("[COOMatrix Parse]: ElementNum = %d\n", result.elementNum);
        printf("[COOMatrix Parse]: totalDataAllocated = %uMB\n", TO_MEGABYTES(totalDataAllocated));
    }

    u32 elementIndex = 0;
    while((line = NextInSplit(&lineIter)).bytes != NULL)
    {
        StrSplitIter partIter = StringSplit(line, " ");
        Str RowStr = NextInSplit(&partIter);
        Str ColStr = NextInSplit(&partIter);
        Str ValueStr = NextInSplit(&partIter);
        assert(RowStr.bytes && ColStr.bytes && ValueStr.length == 0);

        u32 row = atoi(RowStr.bytes);
        u32 col = atoi(ColStr.bytes);
        result.row[elementIndex] = row;
        result.col[elementIndex] = col;
        result.data[elementIndex] = 1.0f;
        elementIndex += 1;

        if(isSymmetric) {
            if(col == row) {
                result.elementNum -= 1;
            } else {
                result.row[elementIndex] = col;
                result.col[elementIndex] = row;
                result.data[elementIndex] = 1.0f;
                elementIndex += 1;
            }
        }
    }
    assert(elementIndex == result.elementNum);

    return result;
}

static ELLMatrix
COOToELLMatrix(COOMatrix matrix)
{
    ELLMatrix result = { 0 };

    u32 minRow = INT_MAX;
    u32 minCol = INT_MAX;
    u32 maxRow = 0;
    u32 maxCol = 0;

    for(int i = 0; i < matrix.elementNum; i++)
    {
        minRow = MIN(matrix.row[i], minRow);
        minCol = MIN(matrix.col[i], minCol);
        maxRow = MAX(matrix.row[i], maxRow);
        maxCol = MAX(matrix.col[i], maxCol);
    }

    u32 M = maxRow-minRow+1;
    u32 *PArray = malloc(M*sizeof(u32));
    memset(PArray, 0, M*sizeof(u32));

    for (int i = 0; i < matrix.elementNum; i++)
    {
        PArray[matrix.row[i] - minRow] += 1;
    }

    u32 P = 0;
    for(u32 rowIndex = 0; rowIndex < M; rowIndex++)
    {
        P = MAX(P, PArray[rowIndex]);
    }

    free(PArray);

    u32 totalDataAllocated = 0;

    result.P = P;
    result.M = M;
    result.N = maxCol-minCol+1;
    result.data = malloc(M*P*sizeof(result.data[0]));
    result.columnIndex = malloc(M * P * sizeof(result.columnIndex[0]));
    result.elementNum = matrix.elementNum;

    totalDataAllocated += M*P*sizeof(result.data[0]);
    totalDataAllocated += M * P * sizeof(result.columnIndex[0]);

    printf("[ELLMatrix Parse]: Pmax = %u\n", result.P);
    printf("[ELLMatrix Parse]: M = %u\n", result.M);
    printf("[ELLMatrix Parse]: N = %u\n", result.N);
    printf("[ELLMatrix Parse]: totalDataAllocated = %uMB\n", TO_MEGABYTES(totalDataAllocated));
    
    memset(result.data, 0, M*P*sizeof(result.data[0]));
    memset(result.columnIndex, 0xff, M*P*sizeof(result.columnIndex[0]));

    for (int i = 0; i < matrix.elementNum; i++)
    {
        u32 startIndex = (matrix.row[i] - minRow) * result.P;
        u32 endIndex = (matrix.row[i] - minRow + 1) * result.P;
        for(u32 k = startIndex; k < endIndex; k++)
        {
            if(result.columnIndex[k] == INVALID_COLUMN) {
                result.columnIndex[k] = matrix.col[i] - minCol;
                result.data[k] = matrix.data[i];
                break;
            }
        }
    }

    return result;
}

static SELLMatrix
ELLToSELLMatrix(ELLMatrix matrix)
{
    SELLMatrix result = { 0 };

    result.C = 2;
    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum; 

    printf("[SELLMatrix Parse]: M = %u\n", result.M);
    printf("[SELLMatrix Parse]: N = %u\n", result.N);
    printf("[SELLMatrix Parse]: C = %u\n", result.C);

    u32 totalDataAllocated = 0;

    u32 sliceCount = DIV_CEIL(result.M, result.C);
    u32 rowOffsetsSize = (sliceCount + 1) * sizeof(result.rowOffsets[0]);
    result.rowOffsets = malloc(rowOffsetsSize);
    memset(result.rowOffsets, 0, rowOffsetsSize);
    totalDataAllocated += rowOffsetsSize;

    u32 *P = malloc(result.C * sizeof(P[0]));
    for(u32 i = 0; i < sliceCount; i++)
    {
        memset(P, 0, result.C * sizeof(P[0]));
        u32 offset = i * result.C * matrix.P;
        for(u32 sliceIdx = 0; sliceIdx < result.C; sliceIdx++)
        {
            for(u32 Pi = 0; Pi < matrix.P; Pi++)
            {
                P[sliceIdx] += matrix.columnIndex[Pi + sliceIdx*matrix.P + offset] != INVALID_COLUMN;
            }
        }

        u32 lastOffset = result.rowOffsets[i];
        u32 currentOffset = 0;
        for(u32 i = 0; i < result.C; i++)
        {
            currentOffset = MAX(P[i], currentOffset);
        }
        currentOffset *= result.C;
        result.rowOffsets[i+1] = currentOffset + lastOffset;
    }

    free(P);

    u32 elementsToAllocate = result.rowOffsets[sliceCount];
    u32 rawDataSize = elementsToAllocate * sizeof(result.columnIndex[0]);
    result.columnIndex = malloc(rawDataSize);
    result.data = malloc(rawDataSize);
    totalDataAllocated += 2 * rawDataSize;

    for(u32 i = 0; i < sliceCount; i++)
    {
        u32 sliceP = (result.rowOffsets[i+1] - result.rowOffsets[i]) / result.C;

        for(u32 sliceIdx = 0; sliceIdx < result.C; sliceIdx++)
        {
            u32 ELLOffset = (sliceIdx * matrix.P) + (i * result.C * matrix.P);
            u32 SELLOffset = result.rowOffsets[i] + sliceIdx * sliceP;
            u32 size = sliceP * sizeof(result.columnIndex[0]);

            void *colDst  = result.columnIndex + SELLOffset;
            void *colSrc  = matrix.columnIndex + ELLOffset;
            void *dataDst = result.data + SELLOffset;
            void *dataSrc = matrix.data + ELLOffset;
            memcpy(colDst, colSrc, size);
            memcpy(dataDst, dataSrc, size);
        }
    }

    printf("[SELLMatrix Parse]: totalDataAllocated = %uMB\n", TO_MEGABYTES(totalDataAllocated));

    return result;
}

static Vector
getSetVector(float v, u32 len)
{
    Vector res = { 0 };

    res.len = len;
    res.data = malloc(res.len * sizeof(res.data[0]));
    for(int i = 0; i < res.len; i++) {
        res.data[i] = v;
    }

    return res;
}

static void
InVecToSSBO(VKState *state, Vector vec, VKBufferAndMemory ssbo)
{
    void *mappedMemory = NULL;
    vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
    memcpy(mappedMemory, vec.data, vec.len * sizeof(vec.data[0]));
    vkUnmapMemory(state->device, ssbo.bufferMemory);
}

static void
checkIfVectorIsSame(VKState *state, VKBufferAndMemory ssbo, const float *expected, u32 len)
{
    void *mappedMemory = NULL;
    vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
    float *mappedMemoryFloat = (float *)mappedMemory;
    for(u32 i = 0; i < len; i++)
    {
        if(mappedMemoryFloat[i] != expected[i]) {
            printf("i, lhs == rhs | %d, %f == %f\n", i, mappedMemoryFloat[i], expected[i]);
            assert(mappedMemoryFloat[i] == expected[i]);
        }
    }
    vkUnmapMemory(state->device, ssbo.bufferMemory);

    printf("[Vector match check]: Pass!\n");
}

static ScenarioELLSimple
createScenarioELLSimple(VKState *state, ELLMatrix *matrix)
{
    ScenarioELLSimple result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 3);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 matrixSize = 2*matrix->M*matrix->P*sizeof(matrix->data[0])+3*sizeof(u32);
    u32 vectorSize = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost = createBuffer(state, matrixSize, usageFlags, memoryFlags);
    result.inVecHost  = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice = createBuffer(state, matrixSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice  = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->P;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);
        u32 MP = matrix->M * matrix->P;

        memcpy(data, matrix->columnIndex, MP * sizeof(matrix->columnIndex[0]));
        data += MP * sizeof(matrix->columnIndex[0]);
        memcpy(data, matrix->data, MP * sizeof(matrix->data[0]));

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, getSetVector(1.0, matrix->N), result.inVecHost);

    copyStagingBufferToDevice(state, result.matHost, result.matDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    bindScenarioELLSimpleDescriptorSetWithBuffers(state, &result);
    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v1.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELLSimple(VKState *state, ScenarioELLSimple *scn, ELLMatrix *matrix)
{
    u32 dispatchX = DIV_CEIL(matrix->M, WORKGROUP_SIZE);
    u32 dispatchY = 1;
    u32 dispatchZ = 1;

    scn->commandBuffer = createCommandBuffer(state, &scn->pipelineDefinition, &scn->descriptorSet,
                                             dispatchX, dispatchY, dispatchZ);

    RunInformation runInfo[RUNS_PER_VERSION] = { 0 };
    for(u32 i = 0; i < RUNS_PER_VERSION; i++)
    {
        u32 nonZeroCount = matrix->elementNum;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e9;
    }

    printRunInfo(runInfo, ARRAY_LEN(runInfo));

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    checkIfVectorIsSame(state, scn->outVecHost, expectedVector, matrix->N);
}

static ScenarioELL2Buffer
createScenarioELL2Buffer(VKState *state, ELLMatrix *matrix)
{
    ScenarioELL2Buffer result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 4);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 matrixSizeIntData = matrix->M*matrix->P*sizeof(matrix->data[0])+3*sizeof(u32);
    u32 matrixSizeFloatData = matrix->M*matrix->P*sizeof(matrix->data[0]);
    u32 vectorSize = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost      = createBuffer(state, matrixSizeIntData, usageFlags, memoryFlags);
    result.matFloatHost = createBuffer(state, matrixSizeFloatData, usageFlags, memoryFlags);
    result.inVecHost       = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost      = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice      = createBuffer(state, matrixSizeIntData, usageFlags, deviceMemoryFlags);
    result.matFloatDevice = createBuffer(state, matrixSizeFloatData, usageFlags, deviceMemoryFlags);
    result.inVecDevice       = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice      = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->P;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);

        u32 MP = matrix->M * matrix->P;
        memcpy(data, matrix->columnIndex, MP * sizeof(matrix->columnIndex[0]));

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);

        u32 MP = matrix->M * matrix->P;
        memcpy(mappedMemory, matrix->data, MP * sizeof(matrix->data[0]));

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, getSetVector(1.0, matrix->N), result.inVecHost);

    copyStagingBufferToDevice(state, result.matHost, result.matDevice);
    copyStagingBufferToDevice(state, result.matFloatHost, result.matFloatDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    bindScenarioELL2BufferDescriptorSetWithBuffers(state, &result);
    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v2.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELL2Buffer(VKState *state, ScenarioELL2Buffer *scn, ELLMatrix *matrix)
{
    u32 dispatchX = DIV_CEIL(matrix->M, WORKGROUP_SIZE);
    u32 dispatchY = 1;
    u32 dispatchZ = 1;

    scn->commandBuffer = createCommandBuffer(state, &scn->pipelineDefinition, &scn->descriptorSet,
                                             dispatchX, dispatchY, dispatchZ);

    RunInformation runInfo[RUNS_PER_VERSION] = { 0 };
    for(u32 i = 0; i < RUNS_PER_VERSION; i++)
    {
        u32 nonZeroCount = matrix->elementNum;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e9;
    }

    printRunInfo(runInfo, ARRAY_LEN(runInfo));

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    checkIfVectorIsSame(state, scn->outVecHost, expectedVector, matrix->N);
}

static void
bindScenarioSELLDescriptorSetWithBuffers(VKState *state, ScenarioSELL *scn)
{
    // Bind buffer with descriptor set
    VkDescriptorBufferInfo descriptorBufferInfoArray[5] = { 0 };
    descriptorBufferInfoArray[0].buffer = scn->matHeaderAndColIndexDevice.buffer;
    descriptorBufferInfoArray[0].offset = 0;
    descriptorBufferInfoArray[0].range = scn->matHeaderAndColIndexDevice.bufferSize;

    descriptorBufferInfoArray[1].buffer = scn->matRowOffsetsDevice.buffer;
    descriptorBufferInfoArray[1].offset = 0;
    descriptorBufferInfoArray[1].range = scn->matRowOffsetsDevice.bufferSize;

    descriptorBufferInfoArray[2].buffer = scn->matFloatDevice.buffer;
    descriptorBufferInfoArray[2].offset = 0;
    descriptorBufferInfoArray[2].range = scn->matFloatDevice.bufferSize;

    descriptorBufferInfoArray[3].buffer = scn->inVecDevice.buffer;
    descriptorBufferInfoArray[3].offset = 0;
    descriptorBufferInfoArray[3].range = scn->inVecDevice.bufferSize;

    descriptorBufferInfoArray[4].buffer = scn->outVecDevice.buffer;
    descriptorBufferInfoArray[4].offset = 0;
    descriptorBufferInfoArray[4].range = scn->outVecDevice.bufferSize;

    VkWriteDescriptorSet writeDescriptorSetsArray[5] = { 0 };
    writeDescriptorSetsArray[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[0].dstSet = scn->descriptorSet;
    writeDescriptorSetsArray[0].dstBinding = 0;
    writeDescriptorSetsArray[0].descriptorCount = 1;
    writeDescriptorSetsArray[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[0].pBufferInfo = &descriptorBufferInfoArray[0];

    writeDescriptorSetsArray[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[1].dstSet = scn->descriptorSet;
    writeDescriptorSetsArray[1].dstBinding = 1;
    writeDescriptorSetsArray[1].descriptorCount = 1;
    writeDescriptorSetsArray[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[1].pBufferInfo = &descriptorBufferInfoArray[1];

    writeDescriptorSetsArray[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[2].dstSet = scn->descriptorSet;
    writeDescriptorSetsArray[2].dstBinding = 2;
    writeDescriptorSetsArray[2].descriptorCount = 1;
    writeDescriptorSetsArray[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[2].pBufferInfo = &descriptorBufferInfoArray[2];

    writeDescriptorSetsArray[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[3].dstSet = scn->descriptorSet;
    writeDescriptorSetsArray[3].dstBinding = 3;
    writeDescriptorSetsArray[3].descriptorCount = 1;
    writeDescriptorSetsArray[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[3].pBufferInfo = &descriptorBufferInfoArray[3];

    writeDescriptorSetsArray[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSetsArray[4].dstSet = scn->descriptorSet;
    writeDescriptorSetsArray[4].dstBinding = 4;
    writeDescriptorSetsArray[4].descriptorCount = 1;
    writeDescriptorSetsArray[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSetsArray[4].pBufferInfo = &descriptorBufferInfoArray[4];

    vkUpdateDescriptorSets(state->device, ARRAY_LEN(writeDescriptorSetsArray), writeDescriptorSetsArray, 0, NULL);
}

static ScenarioSELL
createScenarioSELL(VKState *state, SELLMatrix *matrix)
{
    ScenarioSELL result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 5);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 sliceCount        = DIV_CEIL(matrix->M, matrix->C);
    u32 elementsAllocated = matrix->rowOffsets[sliceCount];

    u32 headerSize      = 3*sizeof(u32);
    u32 columnIndexSize = elementsAllocated * sizeof(matrix->columnIndex[0]);
    u32 rowOffsetsSize  = (sliceCount+1) * sizeof(matrix->rowOffsets[0]);
    u32 floatDataSize   = elementsAllocated * sizeof(matrix->data[0]);
    u32 vectorSize      = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHeaderAndColIndexHost = createBuffer(state, headerSize + columnIndexSize, usageFlags, memoryFlags);
    result.matRowOffsetsHost        = createBuffer(state, rowOffsetsSize, usageFlags, memoryFlags);
    result.matFloatHost             = createBuffer(state, floatDataSize, usageFlags, memoryFlags);
    result.inVecHost                = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost               = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matHeaderAndColIndexDevice = createBuffer(state, headerSize + columnIndexSize, usageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice        = createBuffer(state, rowOffsetsSize, usageFlags, deviceMemoryFlags);
    result.matFloatDevice             = createBuffer(state, floatDataSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice                = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice               = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHeaderAndColIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->C;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);

        memcpy(data, matrix->columnIndex, columnIndexSize);

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matRowOffsetsHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->rowOffsets, rowOffsetsSize);

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->data, floatDataSize);

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, getSetVector(1.0, matrix->N), result.inVecHost);

    copyStagingBufferToDevice(state, result.matHeaderAndColIndexHost, result.matHeaderAndColIndexDevice);
    copyStagingBufferToDevice(state, result.matRowOffsetsHost, result.matRowOffsetsDevice);
    copyStagingBufferToDevice(state, result.matFloatHost, result.matFloatDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    bindScenarioSELLDescriptorSetWithBuffers(state, &result);
    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v3.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioSELL(VKState *state, ScenarioSELL *scn, SELLMatrix *matrix)
{
    u32 dispatchX = DIV_CEIL(matrix->M, WORKGROUP_SIZE);
    u32 dispatchY = 1;
    u32 dispatchZ = 1;

    scn->commandBuffer = createCommandBuffer(state, &scn->pipelineDefinition, &scn->descriptorSet,
                                             dispatchX, dispatchY, dispatchZ);

    RunInformation runInfo[RUNS_PER_VERSION] = { 0 };
    for(u32 i = 0; i < RUNS_PER_VERSION; i++)
    {
        u32 nonZeroCount = matrix->elementNum;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e9;
    }

    printRunInfo(runInfo, ARRAY_LEN(runInfo));

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    checkIfVectorIsSame(state, scn->outVecHost, expectedVector, matrix->N);
}

int main()
{
    COOMatrix bcsstk30COO = ReadMatrixFormatToCOO("data/bcsstk30.mtx");
    ELLMatrix bcsstk30ELL = COOToELLMatrix(bcsstk30COO);
    SELLMatrix bcsstk30SELL = ELLToSELLMatrix(bcsstk30ELL);

    VKState state = initalizeVulkan();

    printf("========================================\n");

    ScenarioELLSimple scnELLSimple = createScenarioELLSimple(&state, &bcsstk30ELL);
    runScenarioELLSimple(&state, &scnELLSimple, &bcsstk30ELL);

    printf("========================================\n");

    ScenarioELL2Buffer scnELL2Buffer = createScenarioELL2Buffer(&state, &bcsstk30ELL);
    runScenarioELL2Buffer(&state, &scnELL2Buffer, &bcsstk30ELL);

    printf("========================================\n");

    ScenarioSELL scnSELL = createScenarioSELL(&state, &bcsstk30SELL);
    runScenarioSELL(&state, &scnSELL, &bcsstk30SELL);

    printf("========================================\n");

    return 0;
}
