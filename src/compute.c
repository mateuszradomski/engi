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

#define WORKGROUP_SIZE 32

#define ARRAY_LEN(x) (sizeof(x)/sizeof(x[0]))

#define RUNS_PER_VERSION 1000

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

static size_t
alignTo(size_t value, size_t alignment)
{
    return value + (alignment - (value & (alignment - 1)));
}

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
saveRunInfo(char *name, RunInformation *runInfo, u32 len, double maxError, u32 matrixSize, char *filename)
{
    RunInfoNode *node = malloc(sizeof(RunInfoNode));

    SLL_QUEUE_PUSH(runInfos.head, runInfos.tail, node);

    node->name = name;
    node->maxError = maxError;
    node->len = len;
    node->infos = malloc(sizeof(RunInformation) * len);
    node->matrixSize = matrixSize;
    node->filename = filename;
    memcpy(node->infos, runInfo, sizeof(RunInformation) * len);
}

static int
compRunInfoSummaries(void *va, void *vb)
{
    RunInfoSummary *a = (RunInfoSummary *)(va);
    RunInfoSummary *b = (RunInfoSummary *)(vb);
    return a->timeAvg > b->timeAvg;
}

void sort(void * arr, size_t data_size, size_t elem_size, int(*compare)(void * x, void * y)){
    size_t length = data_size/elem_size;
    for(int i = 0; i<length; i++){
        for(int j = 0; j<length-1; j++){
            if(compare(arr + elem_size*j, arr + elem_size * (j+1)) > 0){
                char temp[elem_size];
                memcpy(temp,arr + elem_size*j,elem_size);
                memcpy(arr + elem_size*j,arr + elem_size * (j+1), elem_size);
                memcpy(arr + elem_size * (j+1),temp, elem_size);
            };
        }

    }
}

static void
saveResultsToFile(RunInfoNode *node)
{
#if defined(WIN32)
    char *basename = strstr(node->filename, "/") + 1;

    char filename[MAX_PATH] = { 0 };
    sprintf(filename, "result_%s_%s.json", basename, node->name);
    HANDLE fileHandle = CreateFileA(filename, GENERIC_READ | GENERIC_WRITE, 0x0, 0x0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0x0);

    char *scratchBuffer = malloc(8 * 1024 * 1024);
    char *writeHead = scratchBuffer;

    writeHead += sprintf(writeHead, "{ \"matrix_format\": \"%s\", \"matrix_name\": \"%s\", \"matrix_size\": %d, \"max_error\": %f,\n\"data\":[\n",
                         node->name, node->filename, node->matrixSize, node->maxError);
    for(u32 i = 0; i < node->len; i++)
    {
        writeHead += sprintf(writeHead, "{ \"gflop\": %f, \"time_ms\": %f },\n", node->infos[i].gflops, node->infos[i].time);
    }
    writeHead -= 2; // remove the last comma
    writeHead += sprintf(writeHead, "]}");
    unsigned long writtenByteCount = 0;
    u64 bytesToWriteCount = ((size_t)writeHead - (size_t)scratchBuffer);
    WriteFile(fileHandle, scratchBuffer, bytesToWriteCount, &writtenByteCount, 0x0);
    assert(writtenByteCount == bytesToWriteCount);
    free(scratchBuffer);

    CloseHandle(fileHandle);
#endif
}

static void
printRunStats()
{
    const char *col1 = "Name";
    const char *col2 = "Exec time [ms]";
    const char *col3 = "Exec time SD";
    const char *col4 = "GFLOPs";
    const char *col5 = "GFLOPs SD";
    const char *col6 = "Max error [%]";
    printf("| %15s | %15s | %15s | %15s | %15s | %15s |\n",
           col1, col2, col3, col4, col5, col6);

    char *dirName = "results";
    CreateDirectory(dirName, 0x0);
    SetCurrentDirectory(dirName);

    static int matrixNum = 0;
    matrixNum++;
    RunInfoNode *node = runInfos.head;
    char *basename = strstr(node->filename, "/") + 1;
    char buffer[128] = { 0 };
    sprintf(buffer, "%d_%.*s", matrixNum, (size_t)strstr(basename, ".") - (size_t)basename, basename);
    CreateDirectory(buffer, 0x0);
    SetCurrentDirectory(buffer);

    RunInfoSummary *summaries = malloc(128 * sizeof(RunInfoSummary));
    u32 summaryCount = 0;
    while(node)
    {
        saveResultsToFile(node);

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

        RunInfoSummary summary = {
            node->name,
            timeAvg, timeSD,
            gflopAvg, gflopSD,
            node->maxError
        };
        summaries[summaryCount++] = summary;

        SLL_QUEUE_POP(runInfos.head, runInfos.tail);

        RunInfoNode *tmp = node;
        node = node->next;
        free(tmp->infos);
        free(tmp);
    }

    sort(summaries, summaryCount * sizeof(RunInfoSummary), sizeof(RunInfoSummary), compRunInfoSummaries);

    for(u32 i = 0; i < summaryCount; i++)
    {
        RunInfoSummary *s = &summaries[i];

        printf("| %15s | %15f | %15f | %15f | %15f | %15f |\n",
               s->name, s->timeAvg, s->timeSD, s->gflopAvg, s->gflopSD, s->maxError);
    }

    SetCurrentDirectory("../..");

    free(summaries);
}

static bool
areValidationLayersSupported()
{
    u32 layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, 0x0);

    VkLayerProperties *layers = malloc(layerCount * sizeof(layers[0]));
    vkEnumerateInstanceLayerProperties(&layerCount, layers);

    for(u32 i = 0; i < layerCount; i++)
    {
        if(strcmp(layers[i].layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            return true;
        }
    }

    return false;
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

    bool validationLayersSupported = areValidationLayersSupported();

    const char *layerName = "VK_LAYER_KHRONOS_validation";
    if(validationLayersSupported) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &layerName;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = NULL;
    }

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
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = NULL;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    const char *atomicExtensionName = "VK_EXT_shader_atomic_float";
    deviceCreateInfo.enabledExtensionCount = 1;
    deviceCreateInfo.ppEnabledExtensionNames = &atomicExtensionName;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicDeviceFeature = { 0 };
    atomicDeviceFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomicDeviceFeature.shaderBufferFloat32AtomicAdd = true;

    deviceCreateInfo.pNext = &atomicDeviceFeature;

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

    // NOTE(radomski): The upper function returns the supported types of
    // memory. Since we can have many different heaps eg. local on chip memory
    // and shared ram we have to specify which heap supports given type.
    //
    // Since we want to find memory that can be used with a given buffer we are
    // using `memoryTypeBits`. It's a bitmask where the i'th bit tells you if
    // the i'th memoryProperty memory type is supported for this buffer type.
    // Otherwise we want to find a memory type that will support the properties
    // we want or more - since the and in the second if condition.
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
    result.bufferSize = bufferSize; // the actual size of this buffer is memoryReqs.size.
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
        descriptorBufferInfo[i].range = buffers[i].bufferSize - offsets[i];
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
    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

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
    VK_CALL(vkBeginCommandBuffer(result, &beginInfo));

    VkCommandBufferBeginInfo commandBufferInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL,
    };

    vkCmdBindPipeline(result, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineDefinition->pipeline);
    vkCmdBindDescriptorSets(result, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineDefinition->pipelineLayout, 0, 1, descriptorSet, 0, NULL);

    vkCmdResetQueryPool(result, state->queryPool, 0, 2);
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
    Vector result = { 0 };

    result.len = len;
    result.floatdata = malloc(result.len * sizeof(result.floatdata[0]));
    for(int i = 0; i < result.len; i++) {
        result.floatdata[i] = v;
    }

    return result;
}

static Vector
createRandomUnilateralVector(u32 len)
{
    Vector result = getSetVector(0.0f, len);

    for(u32 i = 0; i < len; i++)
    {
        result.floatdata[i] = randomUnilateral();
    }

    return result;
}

static void
destroyVector(Vector vec)
{
    free(vec.floatdata);
}

static MatrixCOO
ReadMatrixFormatToCOO(const char *filename)
{
    MatrixCOO result = { 0 };

    Str str = readEntireFileStr(filename);
    double start = getWallTime();

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
        StrSplitIter partIter = StringSplit(StringTrim(line), " ");
        Str MStr = NextInSplit(&partIter);
        Str NStr = NextInSplit(&partIter);
        Str ElementNumStr = NextInSplit(&partIter);
        assert(MStr.bytes && NStr.bytes && ElementNumStr.bytes);
        result.M = atoi(MStr.bytes);
        result.N = atoi(NStr.bytes);

        u32 factor = isSymmetric ? 2 : 1;
        result.elementNum = atoi(ElementNumStr.bytes) * factor - (factor - 1) * atoi(NStr.bytes);
        u32 toAllocate = result.elementNum * sizeof(result.floatdata[0]);
        result.floatdata = malloc(toAllocate);
        result.row = malloc(toAllocate);
        result.col = malloc(toAllocate);
        totalDataAllocated += 3 * toAllocate;

        printf("[MatrixCOO Parse]: MStr = %.*s, NStr = %.*s, ElementNum = %u\n", MStr.length, MStr.bytes, NStr.length, NStr.bytes, result.elementNum);
    }

    u32 elementIndex = 0;
    while((line = NextInSplit(&lineIter)).bytes != NULL) // wczytuj następne linie pliku tak długą jak to możliwe
    {
        StrSplitIter partIter = StringSplit(StringTrim(line), " ");
        Str RowStr = NextInSplit(&partIter);
        Str ColStr = NextInSplit(&partIter);
        Str ValueStr = NextInSplit(&partIter);
        assert(RowStr.bytes && ColStr.bytes);

        u32 row = atoi(RowStr.bytes) - 1;
        u32 col = atoi(ColStr.bytes) - 1;
        result.row[elementIndex] = row;
        result.col[elementIndex] = col;
        float value = ValueStr.length == 0 ? 1.0f : atof(ValueStr.bytes);
        result.floatdata[elementIndex] = value;
        elementIndex += 1;

        if(isSymmetric && col != row) {
            result.row[elementIndex] = col;
            result.col[elementIndex] = row;
            result.floatdata[elementIndex] = value;
            elementIndex += 1;
        }
    }
    assert(elementIndex == result.elementNum);

    double end = getWallTime();
    printf("[MatrixCOO Parse]: Parsing took %.2lfs and allocated %uMB\n", end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixCOO(MatrixCOO mat)
{
    u32 size = mat.elementNum * sizeof(mat.floatdata[0]);
    return size * 3;
}

static void
destroyMatrixCOO(MatrixCOO mat)
{
    free(mat.floatdata);
    free(mat.row);
    free(mat.col);
}

static MatrixELL
COOToMatrixELL(MatrixCOO matrix)
{
    double start = getWallTime();
    MatrixELL result = { 0 };

    result.M = matrix.M;
    result.N = matrix.N;

    u32 *PArray = malloc(result.M*sizeof(u32));
    memset(PArray, 0, result.M*sizeof(u32));

    for (int i = 0; i < matrix.elementNum; i++)
    {
        PArray[matrix.row[i]] += 1;
    }

    u32 P = 0;
    for(u32 rowIndices = 0; rowIndices < result.M; rowIndices++)
    {
        P = MAX(P, PArray[rowIndices]);
    }

    free(PArray);

    u32 totalDataAllocated = 0;

    result.P = P;
    result.floatdata = malloc(result.M*P*sizeof(result.floatdata[0]));
    result.columnIndices = malloc(result.M * P * sizeof(result.columnIndices[0]));
    result.elementNum = matrix.elementNum;

    totalDataAllocated += result.M*P*sizeof(result.floatdata[0]);
    totalDataAllocated += result.M * P * sizeof(result.columnIndices[0]);

    printf("[MatrixELL Parse]: M = %u, N = %u, P = %u\n", result.M, result.N, result.P);

    memset(result.floatdata, 0, result.M*P*sizeof(result.floatdata[0]));
    memset(result.columnIndices, 0xff, result.M*P*sizeof(result.columnIndices[0]));

    for (int i = 0; i < matrix.elementNum; i++)
    {
        u32 startIndex = matrix.row[i] * result.P;
        u32 endIndex = (matrix.row[i] + 1) * result.P;
        for(u32 k = startIndex; k < endIndex; k++)
        {
            if(result.columnIndices[k] == INVALID_COLUMN) {
                result.columnIndices[k] = matrix.col[i];
                result.floatdata[k] = matrix.floatdata[i];
                break;
            }
        }
    }

    double end = getWallTime();
    printf("[MatrixELL Parse]: Parsing took %.2lfs and allocated %uMB\n", end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixELL(MatrixELL mat)
{
    u32 size = 0;
    size += mat.M * mat.P * sizeof(mat.floatdata[0]);
    size += mat.M * mat.P * sizeof(mat.columnIndices[0]);
    return size;
}

static void
destroyMatrixELL(MatrixELL mat)
{
    free(mat.floatdata);
    free(mat.columnIndices);
}

static MatrixSELL
ELLToMatrixSELL(MatrixELL matrix, uint32_t C)
{
    double start = getWallTime();
    MatrixSELL result = { 0 };

    result.C = C;
    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum;

    printf("[MatrixSELL Parse]: M = %u, N = %u, C = %u\n", result.M, result.N, result.C);

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
                P[sliceIdx] += matrix.columnIndices[Pi + sliceIdx*matrix.P + offset] != INVALID_COLUMN;
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
    u32 rawDataSize = elementsToAllocate * sizeof(result.columnIndices[0]);
    result.columnIndices = malloc(rawDataSize);
    result.floatdata = malloc(rawDataSize);
    totalDataAllocated += 2 * rawDataSize;

    for(u32 i = 0; i < sliceCount; i++)
    {
        u32 sliceP = (result.rowOffsets[i+1] - result.rowOffsets[i]) / result.C;

        for(u32 sliceIdx = 0; sliceIdx < result.C && sliceIdx < (result.M - i * result.C); sliceIdx++)
        {
            u32 ELLOffset = (sliceIdx * matrix.P) + (i * result.C * matrix.P);
            u32 SELLOffset = result.rowOffsets[i] + sliceIdx * sliceP;
            u32 size = sliceP * sizeof(result.columnIndices[0]);

            void *colDst  = result.columnIndices + SELLOffset;
            void *colSrc  = matrix.columnIndices + ELLOffset;
            void *dataDst = result.floatdata + SELLOffset;
            void *dataSrc = matrix.floatdata + ELLOffset;
            memcpy(colDst, colSrc, size);
            memcpy(dataDst, dataSrc, size);
        }
    }

    double end = getWallTime();
    printf("[MatrixSELL Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixSELL(MatrixSELL mat)
{
    u32 sliceCount = DIV_CEIL(mat.M, mat.C);
    u32 elementsToAllocate = mat.rowOffsets[sliceCount];
    u32 rawDataSize = elementsToAllocate * sizeof(mat.columnIndices[0]);
    return 2 * rawDataSize;
}

static void
destroyMatrixSELL(MatrixSELL mat)
{
    free(mat.floatdata);
    free(mat.columnIndices);
    free(mat.rowOffsets);
}

static MatrixSELLColumnMajor
ELLToMatrixSELLColumnMajor(MatrixELL matrix, uint32_t C)
{
    double start = getWallTime();
    MatrixSELLColumnMajor result = { 0 };

    result.C = C;
    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum;

    printf("[MatrixSELLColumnMajor Parse]: M = %u, N = %u, C = %u\n", result.M, result.N, result.C);

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
                P[sliceIdx] += matrix.columnIndices[Pi + sliceIdx*matrix.P + offset] != INVALID_COLUMN;
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
    u32 rawDataSize = elementsToAllocate * sizeof(result.columnIndices[0]);
    result.columnIndices = malloc(rawDataSize);
    result.floatdata = malloc(rawDataSize);
    totalDataAllocated += 2 * rawDataSize;

    for(u32 i = 0; i < sliceCount; i++)
    {
        u32 sliceP = (result.rowOffsets[i+1] - result.rowOffsets[i]) / result.C;

        for(u32 sliceIdx = 0; sliceIdx < result.C && sliceIdx < (result.M - i * result.C); sliceIdx++)
        {
            u32 ELLOffset = (sliceIdx * matrix.P) + (i * result.C * matrix.P);
            u32 SELLOffset = result.rowOffsets[i] + sliceIdx;
            u32 *colDst  = result.columnIndices + SELLOffset;
            u32 *colSrc  = matrix.columnIndices + ELLOffset;
            float *dataDst = result.floatdata + SELLOffset;
            float *dataSrc = matrix.floatdata + ELLOffset;
            for(u32 sliceCol = 0; sliceCol < sliceP; sliceCol++) {
                dataDst[sliceCol * C] = dataSrc[sliceCol];
                colDst[sliceCol * C] = colSrc[sliceCol];
            }
        }
    }

    double end = getWallTime();
    printf("[MatrixSELLColumnMajor Parse]: Parsing took %.2lfs and allocated %uMB\n", end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixSELLColumnMajor(MatrixSELLColumnMajor mat)
{
    u32 sliceCount = DIV_CEIL(mat.M, mat.C);
    u32 elementsToAllocate = mat.rowOffsets[sliceCount];
    u32 rawDataSize = elementsToAllocate * sizeof(mat.columnIndices[0]);
    return 2 * rawDataSize;
}

static void
destroyMatrixSELLColumnMajor(MatrixSELLColumnMajor mat)
{
    free(mat.floatdata);
    free(mat.columnIndices);
    free(mat.rowOffsets);
}

static MatrixCSR
ELLToMatrixCSR(MatrixELL matrix)
{
    double start = getWallTime();
    MatrixCSR result = { 0 };

    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum;

    printf("[MatrixCSR Parse]: M = %u, N = %u\n", result.M, result.N);

    u32 valuesSize          = result.elementNum * sizeof(result.floatdata[0]);
    u32 columnIndicesesSize = result.elementNum * sizeof(u32);
    u32 rowOffsetsSize      = (result.M+1) * sizeof(u32);
    u32 totalDataAllocated  = valuesSize + columnIndicesesSize + rowOffsetsSize;

    result.floatdata     = malloc(valuesSize);
    result.columnIndices = malloc(columnIndicesesSize);
    result.rowOffsets    = malloc(rowOffsetsSize);
    result.rowOffsets[0] = 0;

    u32 head = 0;
    u32 rowHead = 1;
    for(u32 row = 0; row < matrix.M; row++)
    {
        u32 p = 0;
        for(; p < matrix.P && matrix.columnIndices[row * matrix.P + p] != INVALID_COLUMN; p++) {
            result.floatdata[head]     = matrix.floatdata[row * matrix.P + p];
            result.columnIndices[head] = matrix.columnIndices[row * matrix.P + p];
            head += 1;
        }

        result.rowOffsets[rowHead] = result.rowOffsets[rowHead - 1] + p;
        rowHead += 1;
    }

    double end = getWallTime();
    printf("[MatrixCSR Parse]: Parsing took %.2lfs and allocated %uMB\n", end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixCSR(MatrixCSR mat)
{
    u32 valuesSize          = mat.elementNum * sizeof(mat.floatdata[0]);
    u32 columnIndicesesSize = mat.elementNum * sizeof(u32);
    u32 rowOffsetsSize      = (mat.M+1) * sizeof(u32);
    u32 totalDataAllocated  = valuesSize + columnIndicesesSize + rowOffsetsSize;
    return totalDataAllocated;
}

static void
destroyMatrixCSR(MatrixCSR mat)
{
    free(mat.floatdata);
    free(mat.columnIndices);
    free(mat.rowOffsets);
}

static MatrixCSC
ELLToMatrixCSC(MatrixELL matrix)
{
    double start = getWallTime();
    MatrixCSC result = { 0 };

    result.M = matrix.M;
    result.N = matrix.N;
    result.elementNum = matrix.elementNum;

    printf("[MatrixCSC Parse]: M = %u, N = %u\n", result.M, result.N);

    u32 valuesSize         = result.elementNum * sizeof(result.floatdata[0]);
    u32 rowIndicesesSize   = result.elementNum * sizeof(u32);
    u32 columnOffsets      = (result.N+1) * sizeof(u32);
    u32 totalDataAllocated = valuesSize + rowIndicesesSize + columnOffsets;

    // The idea is to 'transpose' the columnIndices table
    u32 *rowIndices = malloc(matrix.N * matrix.P * sizeof(u32));
    u32 *colFront = calloc(1, matrix.N * sizeof(u32));
    memset(rowIndices, INVALID_COLUMN, matrix.N * matrix.P * sizeof(u32));
    for(u32 row = 0; row < matrix.M; row++) {
        for(u32 p = 0; p < matrix.P; p++) {
            u32 colIndex = matrix.columnIndices[row * matrix.P + p];
            if(colIndex != INVALID_COLUMN) {
                rowIndices[colFront[colIndex] + colIndex * matrix.P] = row;
                colFront[colIndex] += 1;
            }
        }
    }

    result.floatdata        = malloc(valuesSize);
    result.rowIndices       = malloc(rowIndicesesSize);
    result.columnOffsets    = malloc(columnOffsets);
    result.columnOffsets[0] = 0;

    u32 head = 0;
    u32 colHead = 1;
    for(u32 col = 0; col < matrix.N; col++)
    {
        u32 p = 0;
        for(; p < matrix.P && rowIndices[col * matrix.P + p] != INVALID_COLUMN; p++) {
            u32 ri = rowIndices[col * matrix.P + p];
            u32 pp = 0;
            for(;(pp < matrix.P) && (matrix.columnIndices[ri * matrix.P + pp] != col); pp++){}

            assert(matrix.columnIndices[ri * matrix.P + pp] == col);
            result.floatdata[head]  = matrix.floatdata[ri * matrix.P + pp];
            result.rowIndices[head] = ri;
            head += 1;
        }

        result.columnOffsets[colHead] = result.columnOffsets[colHead - 1] + p;
        colHead += 1;
    }

    double end = getWallTime();
    printf("[MatrixCSC Parse]: Parsing took %.2lfs and allocated %uMB\n",
           end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixCSC(MatrixCSC mat)
{
    u32 valuesSize         = mat.elementNum * sizeof(mat.floatdata[0]);
    u32 rowIndicesesSize   = mat.elementNum * sizeof(u32);
    u32 columnOffsets      = (mat.N+1) * sizeof(u32);
    u32 totalDataAllocated = valuesSize + rowIndicesesSize + columnOffsets;
    return totalDataAllocated;
}

static void
destroyMatrixCSC(MatrixCSC mat)
{
    free(mat.floatdata);
    free(mat.rowIndices);
    free(mat.columnOffsets);
}

static MatrixBSR
ELLToMatrixBSR(MatrixELL matrix, u32 blockSize)
{
    double start = getWallTime();
    MatrixBSR result = { 0 };

    result.blockSize = blockSize;
    result.MB = DIV_CEIL(matrix.M, result.blockSize);
    result.NB = DIV_CEIL(matrix.N, result.blockSize);
    result.elementNum = matrix.elementNum;

    printf("[MatrixBSR Parse]: MB = %u, NB = %u\n", result.MB, result.NB);

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
                        smallestCol = MIN(smallestCol, matrix.columnIndices[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi]);
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
                        if(matrix.columnIndices[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi] < smallestMultipleOfBlockSize + blockSize) {
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
    u32 columnIndicesSize = result.nnzb * sizeof(u32);
    u32 totalDataAllocated = floatDataSize + rowOffsetsSize + columnIndicesSize;

    result.floatdata        = calloc(1, floatDataSize);
    result.rowOffsets  = calloc(1, rowOffsetsSize);
    result.columnIndices = calloc(1, columnIndicesSize);

    u32 scratchBlockSize = sizeof(float) * blockSize * blockSize;
    float *scratchBlock = malloc(scratchBlockSize);
    memset(scratchBlock, 0, scratchBlockSize);

    u8 *writeHead = (u8 *)result.floatdata;
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
                        smallestCol = MIN(smallestCol, matrix.columnIndices[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi]);
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
                        u32 col = matrix.columnIndices[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + cbi];
                        if(col < smallestMultipleOfBlockSize + blockSize) {
                            for(u32 ei = 0; ei < cbi+1; ei++)
                            {
                                u32 col = matrix.columnIndices[(globalRow + rbi) * matrix.P + rowFront[globalRow + rbi] + ei];
                                scratchBlock[(col - smallestMultipleOfBlockSize) + rbi * blockSize] = matrix.floatdata[rowFront[globalRow + rbi] + (globalRow + rbi) * matrix.P + ei];
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
            result.columnIndices[colIndexHead++] = smallestMultipleOfBlockSize / blockSize;
            result.rowOffsets[rowOffsetsHead] += 1;
            memset(scratchBlock, 0, scratchBlockSize);
        }
        if(rowOffsetsHead < result.MB) {
            result.rowOffsets[rowOffsetsHead+1] = result.rowOffsets[rowOffsetsHead];
        }
        rowOffsetsHead += 1;
    }

    double end = getWallTime();
    printf("[MatrixBSR Parse]: Parsing took %.2lfs and allocated %uMB\n", end - start, TO_MEGABYTES(totalDataAllocated));

    return result;
}

static u32
getMemorySizeMatrixBSR(MatrixBSR mat)
{
    u32 floatDataSize   = mat.nnzb * mat.blockSize * mat.blockSize * sizeof(float);
    u32 rowOffsetsSize  = (mat.MB+1) * sizeof(u32);
    u32 columnIndicesSize = mat.nnzb * sizeof(u32);
    u32 totalDataAllocated = floatDataSize + rowOffsetsSize + columnIndicesSize;
    return totalDataAllocated;
}

static void
destroyMatrixBSR(MatrixBSR mat)
{
    free(mat.floatdata);
    free(mat.rowOffsets);
    free(mat.columnIndices);
}

static Vector
MatrixELLMulVec(MatrixELL mat, Vector vec)
{
    Vector result = getSetVector(0.0f, vec.len);

    for(u32 row = 0; row < mat.M; row++)
    {
        for(u32 P = 0; P < mat.P; P++)
        {
            u32 cellOffset = row * mat.P + P;
            u32 col = mat.columnIndices[cellOffset];
            if(col == INVALID_COLUMN) { break; }
            result.floatdata[row] += vec.floatdata[col] * mat.floatdata[cellOffset];
        }
    }

    return result;
}

static void
runTestsForMatrixCPUMul()
{
    MatrixCOO matCOO = ReadMatrixFormatToCOO("data/bcsstk30.mtx");
    MatrixELL matELL = COOToMatrixELL(matCOO);
    Vector vec = getSetVector(1.0, matELL.N);

    Vector result = MatrixELLMulVec(matELL, vec);
    assert(result.len == vec.len);
    for(u32 i = 0; i < vec.len; i++)
    {
        if(result.floatdata[i] != expectedVector[i]) {
            printf("i, lhs == rhs | %d, %f == %f\n", i, result.floatdata[i], expectedVector[i]);
            assert(result.floatdata[i] == expectedVector[i]);
        }
    }

    destroyMatrixCOO(matCOO);
    destroyMatrixELL(matELL);
    destroyVector(vec);
    destroyVector(result);
}

static void
InVecToSSBO(VKState *state, Vector vec, VKBufferAndMemory ssbo)
{
    void *mappedMemory = NULL;
    vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
    memcpy(mappedMemory, vec.floatdata, vec.len * sizeof(vec.floatdata[0]));
    vkUnmapMemory(state->device, ssbo.bufferMemory);
}

static double
checkIfVectorIsSame(VKState *state, VKBufferAndMemory ssbo, Vector expVec)
{
    bool success = true;
    float *floatData = NULL;
    vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, (void **)&floatData);

    float errorLimit = 1 / 1e2;
    float maxError = 0.0;

    for(u32 i = 0; i < expVec.len; i++)
    {
        float error = fabs(floatData[i] - expVec.floatdata[i]) / expVec.floatdata[i];
        float k = maxError;
        maxError = MAX(error, maxError);

        if(maxError != k) {
            success = true;
        }

        if(error > errorLimit) {
            printf("[Vector match check]: (i, lhs == rhs) => (%d, %f == %f)\n", i, floatData[i], expVec.floatdata[i]);
            success = false;
            break;
        }
    }
    vkUnmapMemory(state->device, ssbo.bufferMemory);

    return maxError * 100.0f;
}

static ScenarioCOO
createScenarioCOO(VKState *state, MatrixCOO *matrix, Vector vec)
{
    ScenarioCOO result = { 0 };

    const u32 HEADER_SIZE = sizeof(matrix->elementNum) + sizeof(matrix->N) + sizeof(matrix->M);
    u32 matrixFloatSize           = matrix->elementNum*sizeof(matrix->floatdata[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixRowSize             = matrix->elementNum*sizeof(matrix->row[0]);
    u32 matrixColSize             = matrix->elementNum*sizeof(matrix->col[0]);
    u32 vectorSize                = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost = createBuffer(state, matrixFloatSizeWithHeader, srcUsageFlags, memoryFlags);
    result.matRowHost   = createBuffer(state, matrixRowSize,             srcUsageFlags, memoryFlags);
    result.matColHost   = createBuffer(state, matrixColSize,             srcUsageFlags, memoryFlags);
    result.inVecHost    = createBuffer(state, vectorSize,                srcUsageFlags, memoryFlags);
    result.outVecHost   = createBuffer(state, vectorSize,                bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice = createBuffer(state, matrixFloatSizeWithHeader, dstUsageFlags, deviceMemoryFlags);
    result.matColDevice   = createBuffer(state, matrixColSize,             dstUsageFlags, deviceMemoryFlags);
    result.matRowDevice   = createBuffer(state, matrixRowSize,             dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice    = createBuffer(state, vectorSize,                dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice   = createBuffer(state, vectorSize,                bidUsageFlags, deviceMemoryFlags);

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
        memcpy(mappedMemory, matrix->floatdata, matrixFloatSize);
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


    const u32 INPUT_MAT_DESC = 3;
    const u32 INPUT_VEC_DESC = 1;
    const u32 OUTPUT_VEC_DESC = 1;
    u32 descriptorCount = INPUT_MAT_DESC + INPUT_VEC_DESC + OUTPUT_VEC_DESC;
    result.descriptorPool      = createDescriptorPool(state->device);
    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, descriptorCount);
    result.descriptorSet       = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

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
runScenarioCOO(VKState *state, ScenarioCOO *scn, MatrixCOO *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("COO", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixCOO(*matrix), filename);
}

static void
destroyScenarioCOO(VKState *state, ScenarioCOO *scn)
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

static ScenarioELL
createScenarioELL(VKState *state, MatrixELL *matrix, Vector vec)
{
    ScenarioELL result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 3);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 matrixSize = 2*matrix->M*matrix->P*sizeof(matrix->floatdata[0])+3*sizeof(u32);
    u32 vectorSize = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost = createBuffer(state, matrixSize, srcUsageFlags, memoryFlags);
    result.inVecHost  = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice = createBuffer(state, matrixSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice  = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

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

        memcpy(data, matrix->columnIndices, MP * sizeof(matrix->columnIndices[0]));
        data += MP * sizeof(matrix->columnIndices[0]);
        memcpy(data, matrix->floatdata, MP * sizeof(matrix->floatdata[0]));

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

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_ell.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELL(VKState *state, ScenarioELL *scn, MatrixELL *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("ELL", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixELL(*matrix), filename);
}

static void
destroyScenarioELL(VKState *state, ScenarioELL *scn)
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
createScenarioELLBufferOffset(VKState *state, MatrixELL *matrix, Vector vec)
{
    ScenarioELLBufferOffset result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 4);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 matrixSize = alignTo(2*matrix->M*matrix->P*sizeof(matrix->floatdata[0])+3*sizeof(u32), 0x10);
    u32 vectorSize = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost = createBuffer(state, matrixSize, srcUsageFlags, memoryFlags);
    result.inVecHost  = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice = createBuffer(state, matrixSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice  = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

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

        memcpy(data, matrix->columnIndices, MP * sizeof(matrix->columnIndices[0]));
        data += MP * sizeof(matrix->columnIndices[0]);
        data = (u8 *)alignTo((u64)data, 0x10);
        memcpy(data, matrix->floatdata, MP * sizeof(matrix->floatdata[0]));

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
    u32 floatOffset = alignTo(3 * sizeof(matrix->P) + (matrix->M * matrix->P * sizeof(matrix->columnIndices[0])), 0x10);
    u32 offsets[] = { 0, floatOffset, 0, 0 };
    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_ell_offset.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELLBufferOffset(VKState *state, ScenarioELLBufferOffset *scn, MatrixELL *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("ELLBufferOffset", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixELL(*matrix), filename);
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
createScenarioELL2Buffer(VKState *state, MatrixELL *matrix, Vector vec)
{
    ScenarioELL2Buffer result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 4);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 matrixSizeIntData = matrix->M*matrix->P*sizeof(matrix->floatdata[0])+3*sizeof(u32);
    u32 matrixSizeFloatData = matrix->M*matrix->P*sizeof(matrix->floatdata[0]);
    u32 vectorSize = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost      = createBuffer(state, matrixSizeIntData, srcUsageFlags, memoryFlags);
    result.matFloatHost = createBuffer(state, matrixSizeFloatData, srcUsageFlags, memoryFlags);
    result.inVecHost       = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost      = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice      = createBuffer(state, matrixSizeIntData, dstUsageFlags, deviceMemoryFlags);
    result.matFloatDevice = createBuffer(state, matrixSizeFloatData, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice       = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice      = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

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
        memcpy(data, matrix->columnIndices, MP * sizeof(matrix->columnIndices[0]));

        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matFloatHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);

        u32 MP = matrix->M * matrix->P;
        memcpy(mappedMemory, matrix->floatdata, MP * sizeof(matrix->floatdata[0]));

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

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_ell_offset.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioELL2Buffer(VKState *state, ScenarioELL2Buffer *scn, MatrixELL *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("ELL2Buffer", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixELL(*matrix), filename);
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
createScenarioSELL(VKState *state, MatrixSELL *matrix, Vector vec)
{
    ScenarioSELL result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 5);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 sliceCount        = DIV_CEIL(matrix->M, matrix->C);
    u32 elementsAllocated = matrix->rowOffsets[sliceCount];

    u32 headerSize      = 3*sizeof(u32);
    u32 columnIndicesSize = elementsAllocated * sizeof(matrix->columnIndices[0]);
    u32 rowOffsetsSize  = (sliceCount+1) * sizeof(matrix->rowOffsets[0]);
    u32 floatDataSize   = elementsAllocated * sizeof(matrix->floatdata[0]);
    u32 vectorSize      = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHeaderAndColIndexHost = createBuffer(state, headerSize + columnIndicesSize, srcUsageFlags, memoryFlags);
    result.matRowOffsetsHost        = createBuffer(state, rowOffsetsSize, srcUsageFlags, memoryFlags);
    result.matFloatHost             = createBuffer(state, floatDataSize, srcUsageFlags, memoryFlags);
    result.inVecHost                = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost               = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matHeaderAndColIndexDevice = createBuffer(state, headerSize + columnIndicesSize, dstUsageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice        = createBuffer(state, rowOffsetsSize, dstUsageFlags, deviceMemoryFlags);
    result.matFloatDevice             = createBuffer(state, floatDataSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice                = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice               = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHeaderAndColIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->C;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);

        memcpy(data, matrix->columnIndices, columnIndicesSize);

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
        memcpy(mappedMemory, matrix->floatdata, floatDataSize);

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

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_sell.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioSELL(VKState *state, ScenarioSELL *scn, MatrixSELL *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    char *name = malloc(16);
    sprintf(name, "SELL%d", matrix->C);
    saveRunInfo(name, runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixSELL(*matrix), filename);
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

static ScenarioSELLColumnMajor
createScenarioSELLColumnMajor(VKState *state, MatrixSELLColumnMajor *matrix, Vector vec)
{
    ScenarioSELLColumnMajor result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 5);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 sliceCount        = DIV_CEIL(matrix->M, matrix->C);
    u32 elementsAllocated = matrix->rowOffsets[sliceCount];

    u32 headerSize      = 3*sizeof(u32);
    u32 columnIndicesSize = elementsAllocated * sizeof(matrix->columnIndices[0]);
    u32 rowOffsetsSize  = (sliceCount+1) * sizeof(matrix->rowOffsets[0]);
    u32 floatDataSize   = elementsAllocated * sizeof(matrix->floatdata[0]);
    u32 vectorSize      = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHeaderAndColIndexHost = createBuffer(state, headerSize + columnIndicesSize, srcUsageFlags, memoryFlags);
    result.matRowOffsetsHost        = createBuffer(state, rowOffsetsSize, srcUsageFlags, memoryFlags);
    result.matFloatHost             = createBuffer(state, floatDataSize, srcUsageFlags, memoryFlags);
    result.inVecHost                = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost               = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matHeaderAndColIndexDevice = createBuffer(state, headerSize + columnIndicesSize, dstUsageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice        = createBuffer(state, rowOffsetsSize, dstUsageFlags, deviceMemoryFlags);
    result.matFloatDevice             = createBuffer(state, floatDataSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice                = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice               = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHeaderAndColIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->C;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);

        memcpy(data, matrix->columnIndices, columnIndicesSize);

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
        memcpy(mappedMemory, matrix->floatdata, floatDataSize);

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

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_sell_column_major.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioSELLColumnMajor(VKState *state, ScenarioSELLColumnMajor *scn, MatrixSELLColumnMajor *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    char *name = malloc(16);
    sprintf(name, "SELL%d", matrix->C);
    saveRunInfo(name, runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixSELLColumnMajor(*matrix), filename);
}

static void
destroyScenarioSELLColumnMajor(VKState *state, ScenarioSELLColumnMajor *scn)
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
createScenarioSELLOffsets(VKState *state, MatrixSELL *matrix, Vector vec)
{
    ScenarioSELLOffsets result = { 0 };

    result.descriptorSetLayout = createConsecutiveDescriptorSetLayout(state->device, 5);
    result.descriptorPool = createDescriptorPool(state->device);
    result.descriptorSet = createDescriptorSet(state->device, result.descriptorSetLayout, result.descriptorPool);

    u32 sliceCount        = DIV_CEIL(matrix->M, matrix->C);
    u32 elementsAllocated = matrix->rowOffsets[sliceCount];

    u32 headerSize      = 3*sizeof(u32);
    u32 columnIndicesSize = elementsAllocated * sizeof(matrix->columnIndices[0]);
    u32 rowOffsetsSize  = (sliceCount+1) * sizeof(matrix->rowOffsets[0]);
    u32 floatDataSize   = elementsAllocated * sizeof(matrix->floatdata[0]);
    u32 matSize         = alignTo(headerSize + columnIndicesSize, 0x10) + alignTo(rowOffsetsSize, 0x10) + alignTo(floatDataSize, 0x10);
    u32 vectorSize      = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matHost    = createBuffer(state, matSize, srcUsageFlags, memoryFlags);
    result.inVecHost  = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matDevice    = createBuffer(state, matSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice  = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

    {
        VKBufferAndMemory ssbo = result.matHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        u32 *u32MappedMemory = (u32 *)mappedMemory;
        u32MappedMemory[0] = matrix->M;
        u32MappedMemory[1] = matrix->C;
        u32MappedMemory[2] = matrix->N;
        u8 *data = (u8 *)(u32MappedMemory + 3);

        memcpy(data, matrix->columnIndices, columnIndicesSize);
        data += columnIndicesSize;
        data = (u8 *)alignTo((u64)data, 0x10);
        memcpy(data, matrix->rowOffsets, rowOffsetsSize);
        data += rowOffsetsSize;
        data = (u8 *)alignTo((u64)data, 0x10);
        memcpy(data, matrix->floatdata, floatDataSize);

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
    u32 offsets[] = { 0,
        alignTo(columnIndicesSize + headerSize, 0x10),
        alignTo(columnIndicesSize + headerSize, 0x10) + alignTo(rowOffsetsSize, 0x10),
        0, 0 };
    bindDescriptorSetWithBuffers(state, result.descriptorSet, buffers, offsets, ARRAY_LEN(buffers));

    result.pipelineDefinition = createComputePipeline(state->device, "build/shaders/sparse_matmul_sell.spv", result.descriptorSetLayout);

    return result;
}

static void
runScenarioSELLOffsets(VKState *state, ScenarioSELLOffsets *scn, MatrixSELL *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("SELLOffsets", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixSELL(*matrix), filename);
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
createScenarioCSR(VKState *state, MatrixCSR *matrix, Vector vec)
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
    u32 matrixFloatSize           = matrix->elementNum*sizeof(matrix->floatdata[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixColumnIndexSize     = matrix->elementNum*sizeof(matrix->columnIndices[0]);
    u32 matrixRowOffsetsSize      = (matrix->M+1)*sizeof(matrix->rowOffsets[0]);
    u32 vectorSize                = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost      = createBuffer(state, matrixFloatSizeWithHeader, srcUsageFlags, memoryFlags);
    result.matRowOffsetsHost = createBuffer(state, matrixRowOffsetsSize, srcUsageFlags, memoryFlags);
    result.matColIndexHost   = createBuffer(state, matrixColumnIndexSize, srcUsageFlags, memoryFlags);
    result.inVecHost         = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost        = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice      = createBuffer(state, matrixFloatSizeWithHeader, dstUsageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice = createBuffer(state, matrixRowOffsetsSize, dstUsageFlags, deviceMemoryFlags);
    result.matColIndexDevice   = createBuffer(state, matrixColumnIndexSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice         = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice        = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

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
        memcpy(mappedMemory, matrix->floatdata, matrixFloatSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matColIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->columnIndices, matrixColumnIndexSize);
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
runScenarioCSR(VKState *state, ScenarioCSR *scn, MatrixCSR *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("CSR", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixCSR(*matrix), filename);
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
createScenarioCSC(VKState *state, MatrixCSC *matrix, Vector vec)
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
    u32 matrixFloatSize           = matrix->elementNum*sizeof(matrix->floatdata[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixRowIndexSize        = matrix->elementNum*sizeof(matrix->rowIndices[0]);
    u32 matrixColOffsetsSize      = (matrix->N+1)*sizeof(matrix->columnOffsets[0]);
    u32 vectorSize                = matrix->N*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost      = createBuffer(state, matrixFloatSizeWithHeader, srcUsageFlags, memoryFlags);
    result.matColOffsetsHost = createBuffer(state, matrixColOffsetsSize, srcUsageFlags, memoryFlags);
    result.matRowIndexHost   = createBuffer(state, matrixRowIndexSize, srcUsageFlags, memoryFlags);
    result.inVecHost         = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost        = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice      = createBuffer(state, matrixFloatSizeWithHeader, dstUsageFlags, deviceMemoryFlags);
    result.matColOffsetsDevice = createBuffer(state, matrixColOffsetsSize, dstUsageFlags, deviceMemoryFlags);
    result.matRowIndexDevice   = createBuffer(state, matrixRowIndexSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice         = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice        = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

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
        memcpy(mappedMemory, matrix->floatdata, matrixFloatSize);
        vkUnmapMemory(state->device, ssbo.bufferMemory);
    }

    {
        VKBufferAndMemory ssbo = result.matRowIndexHost;

        void *mappedMemory = NULL;
        vkMapMemory(state->device, ssbo.bufferMemory, 0, ssbo.bufferSize, 0, &mappedMemory);
        memcpy(mappedMemory, matrix->rowIndices, matrixRowIndexSize);
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
runScenarioCSC(VKState *state, ScenarioCSC *scn, MatrixCSC *matrix, Vector expVec, char *filename)
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
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    saveRunInfo("CSC", runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixCSC(*matrix), filename);
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
createScenarioBSR(VKState *state, MatrixBSR *matrix, Vector vec)
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
    u32 matrixFloatSize           = matrix->nnzb*matrix->blockSize*matrix->blockSize*sizeof(matrix->floatdata[0]);
    u32 matrixFloatSizeWithHeader = matrixFloatSize + HEADER_SIZE;
    u32 matrixRowOffsetsSize        = (matrix->MB+1)*sizeof(matrix->rowOffsets[0]);
    u32 matrixColIndiciesSize      = matrix->nnzb*sizeof(matrix->columnIndices[0]);
    u32 vectorSize                = vec.len*sizeof(matrix->floatdata[0]);

    VkBufferUsageFlags srcUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags dstUsageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBufferUsageFlags bidUsageFlags = srcUsageFlags | dstUsageFlags;
    VkMemoryPropertyFlagBits memoryFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkMemoryPropertyFlagBits deviceMemoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Staging buffers
    result.matFloatHost      = createBuffer(state, matrixFloatSizeWithHeader, srcUsageFlags, memoryFlags);
    result.matColIndiciesHost = createBuffer(state, matrixColIndiciesSize, srcUsageFlags, memoryFlags);
    result.matRowOffsetsHost   = createBuffer(state, matrixRowOffsetsSize, srcUsageFlags, memoryFlags);
    result.inVecHost         = createBuffer(state, vectorSize, srcUsageFlags, memoryFlags);
    result.outVecHost        = createBuffer(state, vectorSize, bidUsageFlags, memoryFlags);

    // On device memory buffers
    result.matFloatDevice      = createBuffer(state, matrixFloatSizeWithHeader, dstUsageFlags, deviceMemoryFlags);
    result.matColIndiciesDevice = createBuffer(state, matrixColIndiciesSize, dstUsageFlags, deviceMemoryFlags);
    result.matRowOffsetsDevice   = createBuffer(state, matrixRowOffsetsSize, dstUsageFlags, deviceMemoryFlags);
    result.inVecDevice         = createBuffer(state, vectorSize, dstUsageFlags, deviceMemoryFlags);
    result.outVecDevice        = createBuffer(state, vectorSize, bidUsageFlags, deviceMemoryFlags);

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
        memcpy(mappedMemory, matrix->floatdata, matrixFloatSize);
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
        memcpy(mappedMemory, matrix->columnIndices, matrixColIndiciesSize);
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
runScenarioBSR(VKState *state, ScenarioBSR *scn, MatrixBSR *matrix, Vector expVec, char *filename)
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
        u32 nonZeroCount = matrix->elementNum;
        runInfo[i].time = runCommandBuffer(state, &scn->commandBuffer);
        runInfo[i].gflops = ((2 * nonZeroCount) / runInfo[i].time) / 1e6;
    }

    copyStagingBufferToDevice(state, scn->outVecDevice, scn->outVecHost);
    double maxError = checkIfVectorIsSame(state, scn->outVecHost, expVec);

    char *name = malloc(16);
    sprintf(name, "BSR%d", matrix->blockSize);
    saveRunInfo(name, runInfo, ARRAY_LEN(runInfo), maxError, getMemorySizeMatrixBSR(*matrix), filename);
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
runTestsForMatrix(VKState *state, char *filename)
{
    printf("=== [%31s] ===\n", filename);

    MatrixCOO matCOO   = ReadMatrixFormatToCOO(filename);
    MatrixELL matELL   = COOToMatrixELL(matCOO);
#if 0
    MatrixSELL matSELL_2 = ELLToMatrixSELL(matELL, 2);
    MatrixSELL matSELL_4 = ELLToMatrixSELL(matELL, 4);
    MatrixSELL matSELL_8 = ELLToMatrixSELL(matELL, 8);
#endif

    MatrixSELLColumnMajor matSELL_CM_2 = ELLToMatrixSELLColumnMajor(matELL, 2);
    MatrixSELLColumnMajor matSELL_CM_4 = ELLToMatrixSELLColumnMajor(matELL, 4);
    MatrixSELLColumnMajor matSELL_CM_8 = ELLToMatrixSELLColumnMajor(matELL, 8);
    MatrixSELLColumnMajor matSELL_CM_16 = ELLToMatrixSELLColumnMajor(matELL, 16);
    MatrixSELLColumnMajor matSELL_CM_32 = ELLToMatrixSELLColumnMajor(matELL, 32);
    MatrixCSR matCSR   = ELLToMatrixCSR(matELL);
    MatrixCSC matCSC   = ELLToMatrixCSC(matELL);
    MatrixBSR matBSR_2 = ELLToMatrixBSR(matELL, 2);
    MatrixBSR matBSR_4 = ELLToMatrixBSR(matELL, 4);
    MatrixBSR matBSR_8 = ELLToMatrixBSR(matELL, 8);

    Vector vec = createRandomUnilateralVector(matELL.N);
    Vector expVec = MatrixELLMulVec(matELL, vec);

    ScenarioCOO scnCOO = createScenarioCOO(state, &matCOO, vec);
    runScenarioCOO(state, &scnCOO, &matCOO, expVec, filename);
    destroyScenarioCOO(state, &scnCOO);

    ScenarioELL scnELL = createScenarioELL(state, &matELL, vec);
    runScenarioELL(state, &scnELL, &matELL, expVec, filename);
    destroyScenarioELL(state, &scnELL);

#if 0
    ScenarioSELL scnSELL_2 = createScenarioSELL(state, &matSELL_2, vec);
    runScenarioSELL(state, &scnSELL_2, &matSELL_2, expVec, filename);
    destroyScenarioSELL(state, &scnSELL_2);

    ScenarioSELL scnSELL_4 = createScenarioSELL(state, &matSELL_4, vec);
    runScenarioSELL(state, &scnSELL_4, &matSELL_4, expVec, filename);
    destroyScenarioSELL(state, &scnSELL_4);

    ScenarioSELL scnSELL_8 = createScenarioSELL(state, &matSELL_8, vec);
    runScenarioSELL(state, &scnSELL_8, &matSELL_8, expVec, filename);
    destroyScenarioSELL(state, &scnSELL_8);
#endif

    ScenarioSELLColumnMajor scnSELL_CM_2 = createScenarioSELLColumnMajor(state, &matSELL_CM_2, vec);
    runScenarioSELLColumnMajor(state, &scnSELL_CM_2, &matSELL_CM_2, expVec, filename);
    destroyScenarioSELLColumnMajor(state, &scnSELL_CM_2);

    ScenarioSELLColumnMajor scnSELL_CM_4 = createScenarioSELLColumnMajor(state, &matSELL_CM_4, vec);
    runScenarioSELLColumnMajor(state, &scnSELL_CM_4, &matSELL_CM_4, expVec, filename);
    destroyScenarioSELLColumnMajor(state, &scnSELL_CM_4);

    ScenarioSELLColumnMajor scnSELL_CM_8 = createScenarioSELLColumnMajor(state, &matSELL_CM_8, vec);
    runScenarioSELLColumnMajor(state, &scnSELL_CM_8, &matSELL_CM_8, expVec, filename);
    destroyScenarioSELLColumnMajor(state, &scnSELL_CM_8);

    ScenarioSELLColumnMajor scnSELL_CM_16 = createScenarioSELLColumnMajor(state, &matSELL_CM_16, vec);
    runScenarioSELLColumnMajor(state, &scnSELL_CM_16, &matSELL_CM_16, expVec, filename);
    destroyScenarioSELLColumnMajor(state, &scnSELL_CM_16);

    ScenarioSELLColumnMajor scnSELL_CM_32 = createScenarioSELLColumnMajor(state, &matSELL_CM_32, vec);
    runScenarioSELLColumnMajor(state, &scnSELL_CM_32, &matSELL_CM_32, expVec, filename);
    destroyScenarioSELLColumnMajor(state, &scnSELL_CM_32);

    ScenarioCSR scnCSR = createScenarioCSR(state, &matCSR, vec);
    runScenarioCSR(state, &scnCSR, &matCSR, expVec, filename);
    destroyScenarioCSR(state, &scnCSR);

    ScenarioCSC scnCSC = createScenarioCSC(state, &matCSC, vec);
    runScenarioCSC(state, &scnCSC, &matCSC, expVec, filename);
    destroyScenarioCSC(state, &scnCSC);

    ScenarioBSR scnBSR_2 = createScenarioBSR(state, &matBSR_2, vec);
    runScenarioBSR(state, &scnBSR_2, &matBSR_2, expVec, filename);
    destroyScenarioBSR(state, &scnBSR_2);

    ScenarioBSR scnBSR_4 = createScenarioBSR(state, &matBSR_4, vec);
    runScenarioBSR(state, &scnBSR_4, &matBSR_4, expVec, filename);
    destroyScenarioBSR(state, &scnBSR_4);

    ScenarioBSR scnBSR_8 = createScenarioBSR(state, &matBSR_8, vec);
    runScenarioBSR(state, &scnBSR_8, &matBSR_8, expVec, filename);
    destroyScenarioBSR(state, &scnBSR_8);

    destroyMatrixCOO(matCOO);
    destroyMatrixELL(matELL);
#if 0
    destroyMatrixSELL(matSELL_2);
    destroyMatrixSELL(matSELL_4);
    destroyMatrixSELL(matSELL_8);
#endif
    destroyMatrixSELLColumnMajor(matSELL_CM_2);
    destroyMatrixSELLColumnMajor(matSELL_CM_4);
    destroyMatrixSELLColumnMajor(matSELL_CM_8);
    destroyMatrixSELLColumnMajor(matSELL_CM_16);
    destroyMatrixSELLColumnMajor(matSELL_CM_32);
    destroyMatrixCSR(matCSR);
    destroyMatrixCSC(matCSC);
    destroyMatrixBSR(matBSR_2);
    destroyMatrixBSR(matBSR_4);
    destroyMatrixBSR(matBSR_8);
}

int main()
{
    VKState state = initalizeVulkan();

#ifdef TESTS
    runTestsForMatrixCPUMul();
#endif

#if 0
    runTestsForMatrix(&state, "data/test2.mtx");
    printRunStats();
#endif

#if 1
#if 1
    char *matricies[] = {
        "data/beaflw.mtx",
        "data/bcsstk32.mtx",
        "data/dense2.mtx",
        "data/scircuit.mtx",
        "data/Ga41As41H72.mtx"
    };
#else
    char *matricies[] = {
         "data/beaflw.mtx",
         "data/scircuit.mtx" 
         };
#endif

    for(u32 i = 0; i < ARRAY_LEN(matricies); i++)
    {
        runTestsForMatrix(&state, matricies[i]);
        printRunStats();
        glob_rand_state = 0x34fae2;
    }
#endif

    return 0;
}
