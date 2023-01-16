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
#include <emmintrin.h>

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

#define WORKGROUP_SIZE 32

#define ARRAY_LEN(x) (sizeof(x)/sizeof(x[0]))

#define RUNS_PER_VERSION 10
 
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
RunInfoList runInfos;

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

static u32
isZeroes(void *buffer, u32 bufferSize)
{
#if 1 
    if(bufferSize % 8 == 0) {
        assert(bufferSize != 0);
        u64 mask = 0;
        u64 *qwords = (u64 *)buffer;
        for(u32 i = 0; i < bufferSize / 8; i++)
        {
            mask |= qwords[i];
        }
        return mask == 0;
    } else {
        u8 *bytes = (u8*)buffer;
        for(u32 i = 0; i < bufferSize; i++)
        {
            if(bytes[i] != 0) {
                return false;
            }
        }

        return true;
    }
#else
    u8 *bytes = (u8*)buffer;
    for(u32 i = 0; i < bufferSize; i++)
    {
        if(bytes[i] != 0) {
            return false;
        }
    }

    return true;
#endif
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
         (i < bytesLeft) && (it->delimLength <= bytesLeft) &&
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

static bool
isWhitespace(char c)
{
    return c == 0x20 || (c >= 0x09 && c <= 0x0d);
}

static Str
StringTrim(Str str)
{
    while(str.length > 0 && isWhitespace(str.bytes[0]))
    {
        str.bytes++;
        str.length--;
    }

    while(str.length > 0 && isWhitespace(str.bytes[str.length - 1]))
    {
        str.length--;
    }

    return str;
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
saveRunInfo(char *name, RunInformation *runInfo, u32 len, double maxEpsilon)
{
    RunInfoNode *node = malloc(sizeof(RunInfoNode));

    SLL_QUEUE_PUSH(runInfos.head, runInfos.tail, node);

    node->name = name;
    node->maxEpsilon = maxEpsilon;
    node->len = len;
    node->infos = malloc(sizeof(RunInformation) * len);
    memcpy(node->infos, runInfo, sizeof(RunInformation) * len);
}

static void
printRunStats()
{
    const char *col1 = "Name";
    const char *col2 = "Exec time [ms]";
    const char *col3 = "Exec time SD";
    const char *col4 = "GFLOPs";
    const char *col5 = "GFLOPs SD";
    const char *col6 = "Max epsilon";
    printf("| %15s | %15s | %15s | %15s | %15s | %15s |\n",
           col1, col2, col3, col4, col5, col6);

    RunInfoNode *node = runInfos.head;
    while(node)
    {
        double gflopSum = 0.0;
        double timeSum = 0.0;
        for(u32 i = 0; i < node->len; i++) {
            gflopSum += node->infos[i].gflops;
            timeSum += node->infos[i].time;
        }
        double gflopAvg = gflopSum / node->len;
        double timeAvg = timeSum / node->len;

        double gflopVar = 0.0;
        double timeVar = 0.0;
        for(u32 i = 0; i < node->len; i++) {
            gflopVar += powf(gflopAvg - node->infos[i].gflops, 2);
            timeVar += powf(timeAvg - node->infos[i].time, 2);
        }
        double gflopSD = sqrtf(gflopVar / (node->len - 1));
        double timeSD = sqrtf(timeVar / (node->len - 1));

        printf("| %15s | %15f | %15f | %15f | %15f | %15f |\n",
               node->name, timeAvg, timeSD, gflopAvg, gflopSD, node->maxEpsilon);

        SLL_QUEUE_POP(runInfos.head, runInfos.tail);

        RunInfoNode *tmp = node;
        node = node->next;
        free(tmp->infos);
        free(tmp);
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
destroyBuffer(VKState *state, VKBufferAndMemory *buff)
{
    vkFreeMemory(state->device, buff->bufferMemory, NULL);
    vkDestroyBuffer(state->device, buff->buffer, NULL);
}

static void
bindDescriptorSetWithBuffers(VKState *state, VkDescriptorSet descriptorSet,
                             VKBufferAndMemory *buffers, u32 *offsets, u32 len)
{
    u32 bufferInfoSize = len * sizeof(VkDescriptorBufferInfo);
    VkDescriptorBufferInfo *descriptorBufferInfo = malloc(bufferInfoSize);
    memset(descriptorBufferInfo, 0, bufferInfoSize);

    for(u32 i = 0; i < len; i++)
    {
        descriptorBufferInfo[i].buffer = buffers[i].buffer;
        descriptorBufferInfo[i].offset = offsets[i];
        descriptorBufferInfo[i].range = buffers[i].bufferSize;
    }

    u32 writeDescSize = len * sizeof(VkWriteDescriptorSet);
    VkWriteDescriptorSet *writeDescriptorSets = malloc(writeDescSize);
    memset(writeDescriptorSets, 0, writeDescSize);

    for(u32 i = 0; i < len; i++)
    {
        writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[i].dstSet = descriptorSet;
        writeDescriptorSets[i].dstBinding = i;
        writeDescriptorSets[i].descriptorCount = 1;
        writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[i].pBufferInfo = &descriptorBufferInfo[i];
    }

    vkUpdateDescriptorSets(state->device, len, writeDescriptorSets, 0, NULL);

    free(descriptorBufferInfo);
    free(writeDescriptorSets);
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

    double execTime = (ts[1] - ts[0]) / 1e6;
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

static Vector
createRandomUnilateralVector(u32 len)
{
    Vector res = getSetVector(0.0f, len);

    for(u32 i = 0; i < len; i++)
    {
        res.data[i] = randomUnilateral();
    }

    return res;
}

static void
destroyVector(Vector vec)
{
    free(vec.data);
}

static COOMatrix
ReadMatrixFormatToCOO(const char *filename)
{
    Str str = readEntireFileStr(filename);
    double start = getWallTime();
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
        StrSplitIter partIter = StringSplit(StringTrim(line), " ");
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

        printf("[COOMatrix Parse]: MStr = %.*s, NStr = %.*s, ElementNum = %u\n",
               MStr.length, MStr.bytes, NStr.length, NStr.bytes, result.elementNum);
    }

    u32 elementIndex = 0;
    while((line = NextInSplit(&lineIter)).bytes != NULL)
    {
        StrSplitIter partIter = StringSplit(StringTrim(line), " ");
        Str RowStr = NextInSplit(&partIter);
        Str ColStr = NextInSplit(&partIter);
        Str ValueStr = NextInSplit(&partIter);
        assert(RowStr.bytes && ColStr.bytes);

        float value = 1.0f;
        if(ValueStr.length != 0) {
            value = atof(ValueStr.bytes);
        }

        u32 row = atoi(RowStr.bytes);
        u32 col = atoi(ColStr.bytes);
        result.row[elementIndex] = row;
        result.col[elementIndex] = col;
        result.data[elementIndex] = value;
        elementIndex += 1;

        if(isSymmetric) {
            if(col == row) {
                result.elementNum -= 1;
            } else {
                result.row[elementIndex] = col;
                result.col[elementIndex] = row;
                result.data[elementIndex] = value;
                elementIndex += 1;
            }
        }
    }
    assert(elementIndex == result.elementNum);

    u32 minRow = INT_MAX;
    u32 minCol = INT_MAX;
    u32 maxRow = 0;
    u32 maxCol = 0;

    for(int i = 0; i < result.elementNum; i++)
    {
        minRow = MIN(result.row[i], minRow);
        minCol = MIN(result.col[i], minCol);
        maxRow = MAX(result.row[i], maxRow);
        maxCol = MAX(result.col[i], maxCol);
    }

    result.M = maxRow-minRow+1;
    result.N = maxCol-minCol+1;

    double end = getWallTime();
    printf("[COOMatrix Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static void
destroyCOOMatrix(COOMatrix mat)
{
    free(mat.data);
    free(mat.row);
    free(mat.col);
}

static ELLMatrix
COOToELLMatrix(COOMatrix matrix)
{
    double start = getWallTime();
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

    printf("[ELLMatrix Parse]: M = %u, N = %u, P = %u\n", result.M, result.N, result.P);
    
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

    double end = getWallTime();
    printf("[ELLMatrix Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static void
destroyELLMatrix(ELLMatrix mat)
{
    free(mat.data);
    free(mat.columnIndex);
}

static SELLMatrix
ELLToSELLMatrix(ELLMatrix matrix)
{
    double start = getWallTime();
    SELLMatrix result = { 0 };

    result.C = 2;
    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum; 

    printf("[SELLMatrix Parse]: M = %u, N = %u, C = %u\n", result.M, result.N, result.C);

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
        for(u32 sliceIdx = 0; sliceIdx < result.C && sliceIdx < (result.M - i * result.C); sliceIdx++)
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

        for(u32 sliceIdx = 0; sliceIdx < result.C && sliceIdx < (result.M - i * result.C); sliceIdx++)
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

    double end = getWallTime();
    printf("[SELLMatrix Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static void
destroySELLMatrix(SELLMatrix mat)
{
    free(mat.data);
    free(mat.columnIndex);
    free(mat.rowOffsets);
}

static CSRMatrix
ELLToCSRMatrix(ELLMatrix matrix)
{
    double start = getWallTime();
    CSRMatrix result = { 0 };

    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum; 

    printf("[CSRMatrix Parse]: M = %u, N = %u\n", result.M, result.N);

    u32 valuesSize         = result.elementNum * sizeof(result.data[0]);
    u32 columnIndexesSize  = result.elementNum * sizeof(u32);
    u32 rowOffsetsSize     = (result.M+2) * sizeof(u32);
    u32 totalDataAllocated = valuesSize + columnIndexesSize + rowOffsetsSize;

    result.data        = malloc(valuesSize);
    result.columnIndex = malloc(columnIndexesSize);
    result.rowOffsets  = malloc(rowOffsetsSize);
    result.rowOffsets[0] = 0;

    u32 head = 0;
    u32 rowHead = 1;
    for(u32 row = 0; row < matrix.M; row++)
    {
        u32 p = 0;
        for(; p < matrix.P; p++)
        {
            if(matrix.columnIndex[row * matrix.P + p] == INVALID_COLUMN) {
                break;
            }

            result.data[head]        = matrix.data[row * matrix.P + p];
            result.columnIndex[head] = matrix.columnIndex[row * matrix.P + p];
            head += 1;
        }

        result.rowOffsets[rowHead] = result.rowOffsets[rowHead - 1] + p;
        rowHead += 1;
    }

    double end = getWallTime();
    printf("[CSRMatrix Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static void
destroyCSRMatrix(CSRMatrix mat)
{
    free(mat.data);
    free(mat.columnIndex);
    free(mat.rowOffsets);
}

static CSCMatrix
ELLToCSCMatrix(ELLMatrix matrix)
{
    double start = getWallTime();
    CSCMatrix result = { 0 };

    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum; 

    printf("[CSCMatrix Parse]: M = %u, N = %u\n", result.M, result.N);

    u32 valuesSize         = result.elementNum * sizeof(result.data[0]);
    u32 rowIndexesSize     = result.elementNum * sizeof(u32);
    u32 columnOffsets      = (result.N+2) * sizeof(u32);
    u32 totalDataAllocated = valuesSize + rowIndexesSize + columnOffsets;

    result.data          = malloc(valuesSize);
    result.rowIndex      = malloc(rowIndexesSize);
    result.columnOffsets = malloc(columnOffsets);
    result.columnOffsets[0] = 0;

    u32 head = 0;
    u32 colHead = 1;
    u32 *rowFront = malloc(result.M * sizeof(u32));
    memset(rowFront, 0, result.M * sizeof(u32));
    for(u32 col = 0; col < matrix.N; col++)
    {
        u32 p = 0;
        for(u32 row = 0; row < matrix.M; row++)
        {
            if(matrix.columnIndex[row * matrix.P + rowFront[row]] == col) {
                result.data[head]     = matrix.data[row * matrix.P + rowFront[row]];
                result.rowIndex[head] = row;
                rowFront[row] += 1;
                head += 1;
                p += 1;
            }
        }

        result.columnOffsets[colHead] = result.columnOffsets[colHead - 1] + p;
        colHead += 1;
    }
    free(rowFront);

    double end = getWallTime();
    printf("[CSCMatrix Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static void
destroyCSCMatrix(CSCMatrix mat)
{
    free(mat.data);
    free(mat.rowIndex);
    free(mat.columnOffsets);
}

static BSRMatrix
ELLToBSRMatrix(ELLMatrix matrix, u32 blockSize)
{
    double start = getWallTime();
    BSRMatrix result = { 0 };

    result.blockSize = blockSize;
    result.MB = DIV_CEIL(matrix.M, result.blockSize);
    result.NB = DIV_CEIL(matrix.N, result.blockSize);

    printf("[BSRMatrix Parse]: MB = %u, NB = %u\n", result.MB, result.NB);

    u32 *rowFront = malloc(matrix.M * sizeof(u32));
    memset(rowFront, 0, matrix.M * sizeof(u32));

    for(u32 rowbi = 0; rowbi < result.MB; rowbi++) {
        u32 globalRow = rowbi * blockSize;
        bool hasColumns = true;
        while(hasColumns)
        {
            hasColumns = false;

            u32 smallestCol = U32_MAX;
            for(u32 rbi = 0; rbi < blockSize && (globalRow + rbi < matrix.M); rbi++) {
                for(u32 cbi = 0; cbi < blockSize; cbi++) {
                    if(rowFront[globalRow + rbi] + cbi < matrix.P) {
                        smallestCol = MIN(smallestCol, matrix.columnIndex[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi]);
                    }
                }
            }
            if(smallestCol == U32_MAX) 
            {
                break;
            }

            u32 colOffsetInBlock = smallestCol % blockSize;
            u32 smallestMultipleOfBlockSize = smallestCol - colOffsetInBlock;
            bool hasNonZeroBlock = false;
            for(u32 rbi = 0; rbi < blockSize && (globalRow + rbi < matrix.M); rbi++) {
                int cbi = MIN(blockSize - 1, (matrix.N - (rbi) * matrix.P) - 1);
                for(; cbi >= 0; cbi--) {
                    if(rowFront[globalRow + rbi] + cbi < matrix.P) {
                        if(matrix.columnIndex[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi] < smallestMultipleOfBlockSize + blockSize) {
                            hasNonZeroBlock = true;
                            break;
                        }
                    }
                }
                rowFront[globalRow + rbi] += cbi + 1;
                if(rowFront[globalRow + rbi] < matrix.P) {
                    hasColumns = true;
                }
            }
            if(hasNonZeroBlock) {
                result.nnzb += 1;
            }
        } 
    }
    memset(rowFront, 0, matrix.M * sizeof(u32));

    u32 floatDataSize   = result.nnzb * blockSize * blockSize * sizeof(float);
    u32 rowOffsetsSize  = (result.MB+1) * sizeof(u32);
    u32 colIndiciesSize = result.nnzb * sizeof(u32);
    u32 totalDataAllocated = floatDataSize + rowOffsetsSize + colIndiciesSize;

    result.data        = calloc(1, floatDataSize);
    result.rowOffsets  = calloc(1, rowOffsetsSize);
    result.colIndicies = calloc(1, colIndiciesSize);

    u32 scratchBlockSize = sizeof(float) * blockSize * blockSize;
    float *scratchBlock = malloc(scratchBlockSize);
    memset(scratchBlock, 0, scratchBlockSize);

    u8 *writeHead = (u8 *)result.data;
    u32 colIndexHead = 0;
    u32 rowOffsetsHead = 0; 
    result.rowOffsets[rowOffsetsHead++] = 0;
    for(u32 rowbi = 0; rowbi < result.MB; rowbi++) {
        u32 globalRow = rowbi * blockSize;
        bool hasColumns = true;
        while(hasColumns)
        {
            hasColumns = false;

            u32 smallestCol = U32_MAX;
            for(u32 rbi = 0; rbi < blockSize && (globalRow + rbi < matrix.M); rbi++) {
                for(u32 cbi = 0; cbi < blockSize; cbi++) {
                    if(rowFront[globalRow + rbi] + cbi < matrix.P) {
                        smallestCol = MIN(smallestCol, matrix.columnIndex[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi]);
                    }
                }
            }
            if(smallestCol == U32_MAX) 
            {
                break;
            }

            u32 colOffsetInBlock = smallestCol % blockSize;
            u32 smallestMultipleOfBlockSize = smallestCol - colOffsetInBlock;
            for(u32 rbi = 0; rbi < blockSize && (globalRow + rbi < matrix.M); rbi++) {
                int cbi = MIN(blockSize - 1, (matrix.N - (rbi) * matrix.P) - 1);
                for(; cbi >= 0; cbi--) {
                    if(rowFront[globalRow + rbi] + cbi < matrix.P) {
                        u32 col = matrix.columnIndex[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi];
                        if(col < smallestMultipleOfBlockSize + blockSize) {
                            for(u32 ei = 0; ei < cbi+1; ei++)
                            {
                                u32 col = matrix.columnIndex[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + ei];
                                scratchBlock[(col - smallestMultipleOfBlockSize) + rbi * blockSize] = matrix.data[rowFront[globalRow + rbi] + (globalRow + rbi) * matrix.P + ei];
                            }
                            break;
                        }
                    }
                }
                rowFront[globalRow + rbi] += cbi + 1;
                if(rowFront[globalRow + rbi] < matrix.P) {
                    hasColumns = true;
                }
            }

            memcpy(writeHead, scratchBlock, scratchBlockSize);
            writeHead += scratchBlockSize;
            result.colIndicies[colIndexHead++] = smallestMultipleOfBlockSize / blockSize;
            result.rowOffsets[rowOffsetsHead] += 1;
            memset(scratchBlock, 0, scratchBlockSize);
        } 
        if(rowOffsetsHead < result.MB) {
            result.rowOffsets[rowOffsetsHead+1] = result.rowOffsets[rowOffsetsHead];
        }
        rowOffsetsHead += 1;
    }

    double end = getWallTime();
    printf("[BSRMatrix Fast Parse]: Parsing took %.2lfs and allocated %uMB\n", end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static void
destroyBSRMatrix(BSRMatrix mat)
{
    free(mat.data);
    free(mat.rowOffsets);
    free(mat.colIndicies);
}

static Vector
ELLMatrixMulVec(ELLMatrix mat, Vector vec)
{
    Vector result = getSetVector(0.0f, vec.len);

    for(u32 row = 0; row < mat.M; row++)
    {
        for(u32 P = 0; P < mat.P; P++)
        {
            u32 cellOffset = row * mat.P + P;
            u32 col = mat.columnIndex[cellOffset];
            if(col == INVALID_COLUMN) { break; }
            result.data[row] += vec.data[col] * mat.data[cellOffset];
        }
    }

    return result;
}

static void
runTestsForCPUMatrixMul()
{
    COOMatrix matCOO = ReadMatrixFormatToCOO("data/bcsstk30.mtx");
    ELLMatrix matELL = COOToELLMatrix(matCOO);
    Vector vec = getSetVector(1.0, matELL.N);

    Vector res = ELLMatrixMulVec(matELL, vec);
    assert(res.len == vec.len);
    for(u32 i = 0; i < vec.len; i++)
    {
        if(res.data[i] != expectedVector[i]) {
            printf("i, lhs == rhs | %d, %f == %f\n", i, res.data[i], expectedVector[i]);
            assert(res.data[i] == expectedVector[i]);
        }
    }

    destroyCOOMatrix(matCOO);
    destroyELLMatrix(matELL);
    destroyVector(vec);
    destroyVector(res);
}

static void
InVecToSSBO(VKState *state, Vector vec, VKBufferAndMemory ssbo)
{
    void *mappedMemory = NULL;
    vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
    memcpy(mappedMemory, vec.data, vec.len * sizeof(vec.data[0]));
    vkUnmapMemory(state->device, ssbo.bufferMemory);
}

static double
checkIfVectorIsSame(VKState *state, VKBufferAndMemory ssbo, Vector expVec)
{
    bool success = true;
    float *floatData = NULL;
    vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, (void **)&floatData);

    float epsilonLimit = 1e-3;
    float maxEpsilon = 0.0;

    for(u32 i = 0; i < expVec.len; i++)
    {
        float epsilon = fabs(floatData[i] - expVec.data[i]);
        maxEpsilon = MAX(epsilon, maxEpsilon);

        if(epsilon > epsilonLimit) {
            printf("[Vector match check]: (i, lhs == rhs) => (%d, %f == %f)\n", i, floatData[i], expVec.data[i]);
            success = false;
            break;
        }
    }
    vkUnmapMemory(state->device, ssbo.bufferMemory);

    return maxEpsilon;
}

static ScenarioCOOSimple
createScenarioCOOSimple(VKState *state, COOMatrix *matrix, Vector vec)
{
    ScenarioCOOSimple result = { 0 };

    const u32 INPUT_MAT_DESC = 3;
    const u32 INPUT_VEC_DESC = 1;
    const u32 OUTPUT_VEC_DESC = 1;
    u32 descriptorCount = INPUT_MAT_DESC + INPUT_VEC_DESC + OUTPUT_VEC_DESC;
    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, descriptorCount);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    const u32 HEADER_SIZE = sizeof(matrix->elementNum) + sizeof(matrix->N) + sizeof(matrix->M);
    u32 matrixFloatSize           = matrix->elementNum*sizeof(matrix->data[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixRowSize             = matrix->elementNum*sizeof(matrix->row[0]);
    u32 matrixColSize             = matrix->elementNum*sizeof(matrix->col[0]);
    u32 vectorSize                = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, memoryFlags);
    result.matRowHost   = createBuffer(state, matrixRowSize, usageFlags, memoryFlags);
    result.matColHost   = createBuffer(state, matrixColSize, usageFlags, memoryFlags);
    result.inVecHost    = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost   = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, deviceMemoryFlags);
    result.matColDevice   = createBuffer(state, matrixColSize, usageFlags, deviceMemoryFlags);
    result.matRowDevice   = createBuffer(state, matrixRowSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice    = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice   = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->elementNum;
        u32MappedMemory[1] = matrix->M;
        u32MappedMemory[2] = matrix->N;
        u32MappedMemory += 3;
        mappedMemory = (void *)u32MappedMemory;
        memcpy(mappedMemory, matrix->data, matrixFloatSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matRowHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->row, matrixRowSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matColHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->col, matrixColSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, vec, result.inVecHost);

    {
        VKBufferAndMemory ssbo = result.outVecHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memset(mappedMemory, 0, vectorSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    copyStagingBufferToDevice(state, result.matFloatHost, result.matFloatDevice);
    copyStagingBufferToDevice(state, result.matRowHost,   result.matRowDevice);
    copyStagingBufferToDevice(state, result.matColHost,   result.matColDevice);
    copyStagingBufferToDevice(state, result.inVecHost,    result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost,   result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matFloatDevice,
        result.matRowDevice,
        result.matColDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0, 0, 0 };

    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_coo.spv", result.descriptorSetLayout);
    
    return result;
}

static void
runScenarioCOOSimple(VKState *state, ScenarioCOOSimple *scn, COOMatrix *matrix, Vector expVec)
{
    u32 dispatchX = DIV_CEIL(matrix->elementNum, WORKGROUP_SIZE);
    u32 dispatchY = 1;
    u32 dispatchZ = 1;

    scn->commandBuffer = createCommandBuffer(state, &scn->pipelineDefinition, &scn->descriptorSet,
                                             dispatchX, dispatchY, dispatchZ);

    RunInformation runInfo[RUNS_PER_VERSION] = { 0 };
    for(u32 i = 0; i < RUNS_PER_VERSION; i++)
    {
        // NOTE(radomski): Since we are always adding onto this output vector
        // we need to assume that it's zero. We are zeroing it right here since
        // the outVecHost holds zeroes for as long as we don't copy the last
        // result into it.
		copyStagingBufferToDevice(state, scn->outVecHost, scn->outVecDevice);
        u32 nonZeroCount = matrix->elementNum;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("COOSimple", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioCOOSimple(VKState *state, ScenarioCOOSimple *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matColDevice);
    destroyBuffer(state, &scn->matRowDevice);
    destroyBuffer(state, &scn->matFloatDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matColHost);
    destroyBuffer(state, &scn->matRowHost);
    destroyBuffer(state, &scn->matFloatHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioELLSimple
createScenarioELLSimple(VKState *state, ELLMatrix *matrix, Vector vec)
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

    InVecToSSBO(state, vec, result.inVecHost);

    copyStagingBufferToDevice(state, result.matHost, result.matDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0 };

    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v1.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELLSimple(VKState *state, ScenarioELLSimple *scn, ELLMatrix *matrix, Vector expVec)
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
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("ELLSimple", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioELLSimple(VKState *state, ScenarioELLSimple *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioELLBufferOffset
createScenarioELLBufferOffset(VKState *state, ELLMatrix *matrix, Vector vec)
{
    ScenarioELLBufferOffset result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 4);
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

    InVecToSSBO(state, vec, result.inVecHost);

    copyStagingBufferToDevice(state, result.matHost, result.matDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matDevice,
        result.matDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 floatOffset = 3 * sizeof(matrix->P) + (matrix->M * matrix->P * sizeof(matrix->columnIndex[0]));
    u32 offsets[] = { 0, floatOffset, 0, 0 };
    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v2.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELLBufferOffset(VKState *state, ScenarioELLBufferOffset *scn, ELLMatrix *matrix, Vector expVec)
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
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("ELLBufferOffset", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioELLBufferOffset(VKState *state, ScenarioELLBufferOffset *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioELL2Buffer
createScenarioELL2Buffer(VKState *state, ELLMatrix *matrix, Vector vec)
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

    InVecToSSBO(state, vec, result.inVecHost);

    copyStagingBufferToDevice(state, result.matHost, result.matDevice);
    copyStagingBufferToDevice(state, result.matFloatHost, result.matFloatDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matDevice,
        result.matFloatDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0, 0 };
    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v2.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELL2Buffer(VKState *state, ScenarioELL2Buffer *scn, ELLMatrix *matrix, Vector expVec)
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
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("ELL2Buffer", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioELL2Buffer(VKState *state, ScenarioELL2Buffer *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matFloatDevice);
    destroyBuffer(state, &scn->matDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matFloatHost);
    destroyBuffer(state, &scn->matHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioSELL
createScenarioSELL(VKState *state, SELLMatrix *matrix, Vector vec)
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

    InVecToSSBO(state, vec, result.inVecHost);

    copyStagingBufferToDevice(state, result.matHeaderAndColIndexHost, result.matHeaderAndColIndexDevice);
    copyStagingBufferToDevice(state, result.matRowOffsetsHost, result.matRowOffsetsDevice);
    copyStagingBufferToDevice(state, result.matFloatHost, result.matFloatDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matHeaderAndColIndexDevice,
        result.matRowOffsetsDevice,
        result.matFloatDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0, 0, 0 };
    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v3.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioSELL(VKState *state, ScenarioSELL *scn, SELLMatrix *matrix, Vector expVec)
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
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("SELL", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioSELL(VKState *state, ScenarioSELL *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matFloatDevice);
    destroyBuffer(state, &scn->matRowOffsetsDevice);
    destroyBuffer(state, &scn->matHeaderAndColIndexDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matFloatHost);
    destroyBuffer(state, &scn->matRowOffsetsHost);
    destroyBuffer(state, &scn->matHeaderAndColIndexHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioSELLOffsets
createScenarioSELLOffsets(VKState *state, SELLMatrix *matrix, Vector vec)
{
    ScenarioSELLOffsets result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 5);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 sliceCount        = DIV_CEIL(matrix->M, matrix->C);
    u32 elementsAllocated = matrix->rowOffsets[sliceCount];

    u32 headerSize      = 3*sizeof(u32);
    u32 columnIndexSize = elementsAllocated * sizeof(matrix->columnIndex[0]);
    u32 rowOffsetsSize  = (sliceCount+1) * sizeof(matrix->rowOffsets[0]);
    u32 floatDataSize   = elementsAllocated * sizeof(matrix->data[0]);
    u32 matSize         = headerSize + columnIndexSize + rowOffsetsSize + floatDataSize;
    u32 vectorSize      = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost    = createBuffer(state, matSize, usageFlags, memoryFlags);
    result.inVecHost  = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice    = createBuffer(state, matSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice  = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->C;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);

        memcpy(data, matrix->columnIndex, columnIndexSize);
        data += columnIndexSize;
        memcpy(data, matrix->rowOffsets, rowOffsetsSize);
        data += rowOffsetsSize;
        memcpy(data, matrix->data, floatDataSize);

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, vec, result.inVecHost);

    copyStagingBufferToDevice(state, result.matHost, result.matDevice);
    copyStagingBufferToDevice(state, result.inVecHost, result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost, result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matDevice,
        result.matDevice,
        result.matDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, columnIndexSize + headerSize, columnIndexSize + headerSize + rowOffsetsSize, 0, 0 };
    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_v3.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioSELLOffsets(VKState *state, ScenarioSELLOffsets *scn, SELLMatrix *matrix, Vector expVec)
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
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("SELLOffsets", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioSELLOffsets(VKState *state, ScenarioSELLOffsets *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioCSR
createScenarioCSR(VKState *state, CSRMatrix *matrix, Vector vec)
{
    ScenarioCSR result = { 0 };

    const u32 INPUT_MAT_DESC = 3;
    const u32 INPUT_VEC_DESC = 1;
    const u32 OUTPUT_VEC_DESC = 1;
    u32 descriptorCount = INPUT_MAT_DESC + INPUT_VEC_DESC + OUTPUT_VEC_DESC;
    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, descriptorCount);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    const u32 HEADER_SIZE = sizeof(matrix->elementNum) + sizeof(matrix->N) + sizeof(matrix->M);
    u32 matrixFloatSize           = matrix->elementNum*sizeof(matrix->data[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixColumnIndexSize     = matrix->elementNum*sizeof(matrix->columnIndex[0]);
    u32 matrixRowOffsetsSize      = (matrix->M+2)*sizeof(matrix->rowOffsets[0]);
    u32 vectorSize                = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost      = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, memoryFlags);
    result.matRowOffsetsHost = createBuffer(state, matrixRowOffsetsSize, usageFlags, memoryFlags);
    result.matColIndexHost   = createBuffer(state, matrixColumnIndexSize, usageFlags, memoryFlags);
    result.inVecHost         = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost        = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice      = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice = createBuffer(state, matrixRowOffsetsSize, usageFlags, deviceMemoryFlags);
    result.matColIndexDevice   = createBuffer(state, matrixColumnIndexSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice         = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice        = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->elementNum;
        u32MappedMemory[1] = matrix->M;
        u32MappedMemory[2] = matrix->N;
        u32MappedMemory += 3;
        mappedMemory = (void *)u32MappedMemory;
        memcpy(mappedMemory, matrix->data, matrixFloatSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matColIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->columnIndex, matrixColumnIndexSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matRowOffsetsHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->rowOffsets, matrixRowOffsetsSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, vec, result.inVecHost);

    {
        VKBufferAndMemory ssbo = result.outVecHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memset(mappedMemory, 0, vectorSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    copyStagingBufferToDevice(state, result.matFloatHost,      result.matFloatDevice);
    copyStagingBufferToDevice(state, result.matColIndexHost,   result.matColIndexDevice);
    copyStagingBufferToDevice(state, result.matRowOffsetsHost, result.matRowOffsetsDevice);
    copyStagingBufferToDevice(state, result.inVecHost,         result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost,        result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matFloatDevice,
        result.matColIndexDevice,
        result.matRowOffsetsDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0, 0, 0 };

    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_csr.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioCSR(VKState *state, ScenarioCSR *scn, CSRMatrix *matrix, Vector expVec)
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
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("CSR", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioCSR(VKState *state, ScenarioCSR *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matFloatDevice);
    destroyBuffer(state, &scn->matRowOffsetsDevice);
    destroyBuffer(state, &scn->matColIndexDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matFloatHost);
    destroyBuffer(state, &scn->matRowOffsetsHost);
    destroyBuffer(state, &scn->matColIndexHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioCSC
createScenarioCSC(VKState *state, CSCMatrix *matrix, Vector vec)
{
    ScenarioCSC result = { 0 };

    const u32 INPUT_MAT_DESC = 3;
    const u32 INPUT_VEC_DESC = 1;
    const u32 OUTPUT_VEC_DESC = 1;
    u32 descriptorCount = INPUT_MAT_DESC + INPUT_VEC_DESC + OUTPUT_VEC_DESC;
    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, descriptorCount);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    const u32 HEADER_SIZE = sizeof(matrix->elementNum) + sizeof(matrix->N) + sizeof(matrix->M);
    u32 matrixFloatSize           = matrix->elementNum*sizeof(matrix->data[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixRowIndexSize        = matrix->elementNum*sizeof(matrix->rowIndex[0]);
    u32 matrixColOffsetsSize      = (matrix->N+2)*sizeof(matrix->columnOffsets[0]);
    u32 vectorSize                = matrix->N*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost      = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, memoryFlags);
    result.matColOffsetsHost = createBuffer(state, matrixColOffsetsSize, usageFlags, memoryFlags);
    result.matRowIndexHost   = createBuffer(state, matrixRowIndexSize, usageFlags, memoryFlags);
    result.inVecHost         = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost        = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice      = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, deviceMemoryFlags);
    result.matColOffsetsDevice = createBuffer(state, matrixColOffsetsSize, usageFlags, deviceMemoryFlags);
    result.matRowIndexDevice   = createBuffer(state, matrixRowIndexSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice         = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice        = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->elementNum;
        u32MappedMemory[1] = matrix->M;
        u32MappedMemory[2] = matrix->N;
        u32MappedMemory += 3;
        mappedMemory = (void *)u32MappedMemory;
        memcpy(mappedMemory, matrix->data, matrixFloatSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matRowIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->rowIndex, matrixRowIndexSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matColOffsetsHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->columnOffsets, matrixColOffsetsSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, vec, result.inVecHost);

    {
        VKBufferAndMemory ssbo = result.outVecHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memset(mappedMemory, 0, vectorSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    copyStagingBufferToDevice(state, result.matFloatHost,      result.matFloatDevice);
    copyStagingBufferToDevice(state, result.matRowIndexHost,   result.matRowIndexDevice);
    copyStagingBufferToDevice(state, result.matColOffsetsHost, result.matColOffsetsDevice);
    copyStagingBufferToDevice(state, result.inVecHost,         result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost,        result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matFloatDevice,
        result.matRowIndexDevice,
        result.matColOffsetsDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0, 0, 0 };

    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_csc.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioCSC(VKState *state, ScenarioCSC *scn, CSCMatrix *matrix, Vector expVec)
{
    u32 dispatchX = DIV_CEIL(matrix->N, WORKGROUP_SIZE);
    u32 dispatchY = 1;
    u32 dispatchZ = 1;

    scn->commandBuffer = createCommandBuffer(state, &scn->pipelineDefinition, &scn->descriptorSet,
                                             dispatchX, dispatchY, dispatchZ);

    RunInformation runInfo[RUNS_PER_VERSION] = { 0 };
    for(u32 i = 0; i < RUNS_PER_VERSION; i++)
    {
        // NOTE(radomski): Since we are always adding onto this output vector
        // we need to assume that it's zero. We are zeroing it right here since
        // the outVecHost holds zeroes for as long as we don't copy the last
        // result into it.
		copyStagingBufferToDevice(state, scn->outVecHost, scn->outVecDevice);
        u32 nonZeroCount = matrix->elementNum;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("CSC", runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioCSC(VKState *state, ScenarioCSC *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matFloatDevice);
    destroyBuffer(state, &scn->matColOffsetsDevice);
    destroyBuffer(state, &scn->matRowIndexDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matFloatHost);
    destroyBuffer(state, &scn->matColOffsetsHost);
    destroyBuffer(state, &scn->matRowIndexHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static ScenarioBSR
createScenarioBSR(VKState *state, BSRMatrix *matrix, Vector vec)
{
    ScenarioBSR result = { 0 };

    const u32 INPUT_MAT_DESC = 3;
    const u32 INPUT_VEC_DESC = 1;
    const u32 OUTPUT_VEC_DESC = 1;
    u32 descriptorCount = INPUT_MAT_DESC + INPUT_VEC_DESC + OUTPUT_VEC_DESC;
    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, descriptorCount);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    const u32 HEADER_SIZE = sizeof(matrix->blockSize) + sizeof(matrix->nnzb) + sizeof(matrix->NB) + sizeof(matrix->MB);
    u32 matrixFloatSize           = matrix->nnzb*matrix->blockSize*matrix->blockSize*sizeof(matrix->data[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixRowOffsetsSize        = (matrix->MB+1)*sizeof(matrix->rowOffsets[0]);
    u32 matrixColIndiciesSize      = matrix->nnzb*sizeof(matrix->colIndicies[0]);
    u32 vectorSize                = vec.len*sizeof(matrix->data[0]);

    VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost      = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, memoryFlags);
    result.matColIndiciesHost = createBuffer(state, matrixColIndiciesSize, usageFlags, memoryFlags);
    result.matRowOffsetsHost   = createBuffer(state, matrixRowOffsetsSize, usageFlags, memoryFlags);
    result.inVecHost         = createBuffer(state, vectorSize, usageFlags, memoryFlags);
    result.outVecHost        = createBuffer(state, vectorSize, usageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice      = createBuffer(state, matrixFloatSizeWithHeader, usageFlags, deviceMemoryFlags);
    result.matColIndiciesDevice = createBuffer(state, matrixColIndiciesSize, usageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice   = createBuffer(state, matrixRowOffsetsSize, usageFlags, deviceMemoryFlags);
    result.inVecDevice         = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);
    result.outVecDevice        = createBuffer(state, vectorSize, usageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->blockSize;
        u32MappedMemory[1] = matrix->nnzb;
        u32MappedMemory[2] = matrix->MB;
        u32MappedMemory[3] = matrix->NB;
        u32MappedMemory += 4;
        mappedMemory = (void *)u32MappedMemory;
        memcpy(mappedMemory, matrix->data, matrixFloatSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matRowOffsetsHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->rowOffsets, matrixRowOffsetsSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matColIndiciesHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->colIndicies, matrixColIndiciesSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    InVecToSSBO(state, vec, result.inVecHost);

    {
        VKBufferAndMemory ssbo = result.outVecHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memset(mappedMemory, 0, vectorSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    copyStagingBufferToDevice(state, result.matFloatHost,       result.matFloatDevice);
    copyStagingBufferToDevice(state, result.matRowOffsetsHost,  result.matRowOffsetsDevice);
    copyStagingBufferToDevice(state, result.matColIndiciesHost, result.matColIndiciesDevice);
    copyStagingBufferToDevice(state, result.inVecHost,          result.inVecDevice);
    copyStagingBufferToDevice(state, result.outVecHost,         result.outVecDevice);

    VKBufferAndMemory buffers[] = { 
        result.matFloatDevice,
        result.matRowOffsetsDevice,
        result.matColIndiciesDevice,
        result.inVecDevice,
        result.outVecDevice
    };
    u32 offsets[] = { 0, 0, 0, 0, 0 };

    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_bsr.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioBSR(VKState *state, ScenarioBSR *scn, BSRMatrix *matrix, Vector expVec)
{
    u32 dispatchX = DIV_CEIL(matrix->MB, WORKGROUP_SIZE);
    u32 dispatchY = 1;
    u32 dispatchZ = 1;

    scn->commandBuffer = createCommandBuffer(state, &scn->pipelineDefinition, &scn->descriptorSet,
                                             dispatchX, dispatchY, dispatchZ);

    RunInformation runInfo[RUNS_PER_VERSION] = { 0 };
    for(u32 i = 0; i < RUNS_PER_VERSION; i++)
    {
        // NOTE(radomski): Since we are always adding onto this output vector
        // we need to assume that it's zero. We are zeroing it right here since
        // the outVecHost holds zeroes for as long as we don't copy the last
        // result into it.
		copyStagingBufferToDevice(state, scn->outVecHost, scn->outVecDevice);
        u32 nonZeroCount = matrix->nnzb * matrix->blockSize * matrix->blockSize;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxEpsilon = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    char *name = malloc(16);
    sprintf(name, "BSR %d", matrix->blockSize);
    saveRunInfo(name, runInfo, ARRAY_LEN(runInfo), maxEpsilon);
}

static void
destroyScenarioBSR(VKState *state, ScenarioBSR *scn)
{
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &scn->commandBuffer);
    vkDestroyPipeline(state->device, scn->pipelineDefinition.pipeline, NULL);
    vkDestroyPipelineLayout(state->device, scn->pipelineDefinition.pipelineLayout, NULL);

    destroyBuffer(state, &scn->outVecDevice);
    destroyBuffer(state, &scn->inVecDevice);
    destroyBuffer(state, &scn->matFloatDevice);
    destroyBuffer(state, &scn->matColIndiciesDevice);
    destroyBuffer(state, &scn->matRowOffsetsDevice);

    destroyBuffer(state, &scn->outVecHost);
    destroyBuffer(state, &scn->inVecHost);
    destroyBuffer(state, &scn->matFloatHost);
    destroyBuffer(state, &scn->matColIndiciesHost);
    destroyBuffer(state, &scn->matRowOffsetsHost);

    vkFreeDescriptorSets(state->device, scn->descriptorPool, 1, &scn->descriptorSet);
    vkDestroyDescriptorPool(state->device, scn->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device, scn->descriptorSetLayout, NULL);
}

static void
runTestsForMatrix(VKState *state, const char *filename)
{
    printf("=== [%31s] ===\n", filename);

    COOMatrix matCOO   = ReadMatrixFormatToCOO(filename);
    ELLMatrix matELL   = COOToELLMatrix(matCOO);
    SELLMatrix matSELL = ELLToSELLMatrix(matELL);
    CSRMatrix matCSR   = ELLToCSRMatrix(matELL);
    CSCMatrix matCSC   = ELLToCSCMatrix(matELL);
    BSRMatrix matBSR_2 = ELLToBSRMatrix(matELL, 2);
    BSRMatrix matBSR_4 = ELLToBSRMatrix(matELL, 4);
    BSRMatrix matBSR_8 = ELLToBSRMatrix(matELL, 8);

    Vector vec = createRandomUnilateralVector(matELL.N);
    Vector expVec = ELLMatrixMulVec(matELL, vec);

    ScenarioCOOSimple scnCOOSimple = createScenarioCOOSimple(state, &matCOO, vec);
    runScenarioCOOSimple(state, &scnCOOSimple, &matCOO, expVec);
    destroyScenarioCOOSimple(state, &scnCOOSimple);

    ScenarioELLSimple scnELLSimple = createScenarioELLSimple(state, &matELL, vec);
    runScenarioELLSimple(state, &scnELLSimple, &matELL, expVec);
    destroyScenarioELLSimple(state, &scnELLSimple);

    ScenarioELLBufferOffset scnELLBufferOffset = createScenarioELLBufferOffset(state, &matELL, vec);
    runScenarioELLBufferOffset(state, &scnELLBufferOffset, &matELL, expVec);
    destroyScenarioELLBufferOffset(state, &scnELLBufferOffset);

    ScenarioELL2Buffer scnELL2Buffer = createScenarioELL2Buffer(state, &matELL, vec);
    runScenarioELL2Buffer(state, &scnELL2Buffer, &matELL, expVec);
    destroyScenarioELL2Buffer(state, &scnELL2Buffer);

    ScenarioSELL scnSELL = createScenarioSELL(state, &matSELL, vec);
    runScenarioSELL(state, &scnSELL, &matSELL, expVec);
    destroyScenarioSELL(state, &scnSELL);

    ScenarioSELLOffsets scnSELLOffsets = createScenarioSELLOffsets(state, &matSELL, vec);
    runScenarioSELLOffsets(state, &scnSELLOffsets, &matSELL, expVec);
    destroyScenarioSELLOffsets(state, &scnSELLOffsets);

    ScenarioCSR scnCSR = createScenarioCSR(state, &matCSR, vec);
    runScenarioCSR(state, &scnCSR, &matCSR, expVec);
    destroyScenarioCSR(state, &scnCSR);

    ScenarioCSC scnCSC = createScenarioCSC(state, &matCSC, vec);
    runScenarioCSC(state, &scnCSC, &matCSC, expVec);
    destroyScenarioCSC(state, &scnCSC);

    ScenarioBSR scnBSR_2 = createScenarioBSR(state, &matBSR_2, vec);
    runScenarioBSR(state, &scnBSR_2, &matBSR_2, expVec);
    destroyScenarioBSR(state, &scnBSR_2);

    ScenarioBSR scnBSR_4 = createScenarioBSR(state, &matBSR_4, vec);
    runScenarioBSR(state, &scnBSR_4, &matBSR_4, expVec);
    destroyScenarioBSR(state, &scnBSR_4);

    ScenarioBSR scnBSR_8 = createScenarioBSR(state, &matBSR_8, vec);
    runScenarioBSR(state, &scnBSR_8, &matBSR_8, expVec);
    destroyScenarioBSR(state, &scnBSR_8);

    destroyCOOMatrix(matCOO);
    destroyELLMatrix(matELL);
    destroySELLMatrix(matSELL);

    destroyCSRMatrix(matCSR);
    destroyCSCMatrix(matCSC);
    destroyBSRMatrix(matBSR_2);
    destroyBSRMatrix(matBSR_4);
    destroyBSRMatrix(matBSR_8);
}

int main()
{
    VKState state = initalizeVulkan();

#ifdef TESTS
    runTestsForCPUMatrixMul();
#endif

#if 0
    runTestsForMatrix(&state, "data/test.mtx");
    printRunStats();
#endif

#if 1
    runTestsForMatrix(&state, "data/beaflw.mtx");
    printRunStats();
    runTestsForMatrix(&state, "data/bcsstk30.mtx");
    printRunStats();
    runTestsForMatrix(&state, "data/bcsstk32.mtx");
    printRunStats();
    runTestsForMatrix(&state, "data/s3dkt3m2.mtx");
    printRunStats();
#endif

    return 0;
}
