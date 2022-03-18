#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

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
} VulkanBufferAndMemory;

#define VK_ASSERT_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
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

    VkPhysicalDevice *devicesArray = NULL;
    vkEnumeratePhysicalDevices(instance, &deviceCount, devicesArray);
    
    // TODO(radomski): Choose the most powerfull GPU
    return devicesArray[0];
}

static uint32_t
getComputeQueueFamilyIndex(VkPhysicalDevice phyDevice)
{
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(phyDevice, &queueFamilyCount, NULL);

    VkQueueFamilyProperties *queueFamiliesArray = NULL;
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

    // Specify any desired device features here. We do not need any for this application, though.
    VkPhysicalDeviceFeatures deviceFeatures = { 0 };

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = 0; // TODO(radomski): add validation layers
    deviceCreateInfo.ppEnabledLayerNames = NULL;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VkDevice device = { 0 };
    VK_ASSERT_RESULT(vkCreateDevice(phyDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

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
    return result;
}

static void
initalizeVulkan()
{
    VkInstance instance = createInstance();
    VkPhysicalDevice phyDevice = findPhysicalDevice(instance);
    VulkanDeviceAndComputeQueue deviceAndQueue = createDevice(phyDevice);

    uint32_t matrixASize = 64 * 64 * sizeof(float);
    VulkanBufferAndMemory bufferAndMemory = createBuffer(phyDevice, deviceAndQueue.device, matrixASize);
}

int main()
{
    initalizeVulkan();
    return 0;
}