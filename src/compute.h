#ifndef COMPUTE_H
#define COMPUTE_H

#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

#define U32_MAX ((u32)-1)

typedef struct Data
{
    u8 *bytes;
    u32 length;
} Data;

typedef struct Str
{
    char *bytes;
    u32 length;
} Str;

#define LIT_STR_TO_STRING_SLICE(str) { .bytes = str, .length = sizeof(str) - 1 }

typedef struct StrSplitIter
{
    char *str, *head, *delim;
    u32 strLength, delimLength;
} StrSplitIter;

typedef struct Vector
{
    float *floatdata;
    u32 len;
} Vector;

typedef struct MatrixCOO
{
    u32 M, N, elementNum; // wymiar M x N i ilość elementów niezerowych
    float *floatdata;     // tablica wartości elementów
    u32 *row, *col;       // odpowiednio tablice rzędów i kolumn elementów
} MatrixCOO;

// If in columnIndices means no data (zero) at that space
#define INVALID_COLUMN 0xffffffff

typedef struct MatrixELL
{
    u32 M, N, elementNum; // wymiar M x N i ilość elementów niezerowych
    u32 P;                // maksymalna liczba elementów niezerowych
    float *floatdata;     // tablica wartości elementów
    u32 *columnIndices;   // tablica kolumn elementów
} MatrixELL;

typedef struct MatrixSELL
{
    u32 M, N, C;        // wymiar M x N i wysokość paska
    u32 elementNum;     // ilość elementów niezerowych
    float *floatdata;   // tablica wartości elementów
    u32 *columnIndices; // tablica kolumn elementów
    u32 *rowOffsets;    // tablica pierwszego indeksu elementu w pasku
} MatrixSELL;

typedef struct MatrixCSR
{
    u32 M, N, elementNum; // wymiar M x N i ilość elementów niezerowych
    float *floatdata;     // tablica wartości elementów
    u32 *columnIndices;   // tablica kolumn elementów
    u32 *rowOffsets;      // tablica pierwszego indeksu elementu w rzędzie
} MatrixCSR;

typedef struct MatrixCSC
{
    u32 M, N, elementNum; // wymiar M x N i ilość elementów niezerowych
    float *floatdata;     // tablica wartości elementów
    u32 *rowIndices;      // tablica rzędów elementów
    u32 *columnOffsets;   // tablica pierwszego indeksu elementu w kolumnie
} MatrixCSC;

// NOTE(radomski):
//
// |         Array |             Length |
// |---------------|--------------------|
// |          data | nnzb * blockSize^2 |
// | columnIndices |               nnzb |
// |    rowOffsets |               MB+1 |
typedef struct MatrixBSR
{
    u32 MB, NB, nnzb;   // wymiar MB x NB i ilość bloków niezerowych
    u32 blockSize;      // rozmiar bloku
    float *floatdata;   // tablica wartości elementów
    u32 *rowOffsets;    // tablica pierwszego indeksu elementu w rzędzie
    u32 *columnIndices; // tablica kolumn elementów
} MatrixBSR;

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

typedef struct ScenarioCOO
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matFloatHost;
    VKBufferAndMemory matRowHost;
    VKBufferAndMemory matColHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matFloatDevice;
    VKBufferAndMemory matColDevice;
    VKBufferAndMemory matRowDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioCOO;

typedef struct ScenarioELL
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
} ScenarioELL;

typedef struct ScenarioELLBufferOffset
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
} ScenarioELLBufferOffset;

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

typedef struct ScenarioSELLOffsets
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
} ScenarioSELLOffsets;

typedef struct ScenarioCSR
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matFloatHost;
    VKBufferAndMemory matColIndexHost;
    VKBufferAndMemory matRowOffsetsHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matFloatDevice;
    VKBufferAndMemory matColIndexDevice;
    VKBufferAndMemory matRowOffsetsDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioCSR;

typedef struct ScenarioCSC
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matFloatHost;
    VKBufferAndMemory matRowIndexHost;
    VKBufferAndMemory matColOffsetsHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matFloatDevice;
    VKBufferAndMemory matRowIndexDevice;
    VKBufferAndMemory matColOffsetsDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioCSC;

typedef struct ScenarioBSR
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VKBufferAndMemory matFloatHost;
    VKBufferAndMemory matRowOffsetsHost;
    VKBufferAndMemory matColIndiciesHost;
    VKBufferAndMemory inVecHost;
    VKBufferAndMemory outVecHost;

    VKBufferAndMemory matFloatDevice;
    VKBufferAndMemory matRowOffsetsDevice;
    VKBufferAndMemory matColIndiciesDevice;
    VKBufferAndMemory inVecDevice;
    VKBufferAndMemory outVecDevice;

    VKPipelineDefinition pipelineDefinition;
    VkCommandBuffer commandBuffer;
} ScenarioBSR;


#define STMNT(S) do{ S }while(0)

#define SLL_STACK_PUSH_(H,N) N->next=H,H=N
#define SLL_STACK_POP_(H) H=H=H->next
#define SLL_QUEUE_PUSH_MULTIPLE_(F,L,FF,LL) if(LL){if(F){L->next=FF;}else{F=FF;}L=LL;L->next=0;}
#define SLL_QUEUE_PUSH_(F,L,N) SLL_QUEUE_PUSH_MULTIPLE_(F,L,N,N)
#define SLL_QUEUE_POP_(F,L) if (F==L) { F=L=0; } else { F=F->next; }

#define SLL_STACK_PUSH(H,N) (SLL_STACK_PUSH_((H),(N)))
#define SLL_STACK_POP(H) (SLL_STACK_POP_((H)))
#define SLL_QUEUE_PUSH_MULTIPLE(F,L,FF,LL) STMNT( SLL_QUEUE_PUSH_MULTIPLE_((F),(L),(FF),(LL)) )
#define SLL_QUEUE_PUSH(F,L,N) STMNT( SLL_QUEUE_PUSH_((F),(L),(N)) )
#define SLL_QUEUE_POP(F,L) STMNT( SLL_QUEUE_POP_((F),(L)) )

typedef struct RunInformation 
{
    double time;
    double gflops;
} RunInformation;

typedef struct RunInfoNode 
{
    RunInformation *infos;
    u32 len;
    char *name;
    double maxEpsilon;
    struct RunInfoNode *next;
} RunInfoNode;

typedef struct RunInfoList 
{
    RunInfoNode *head;
    RunInfoNode *tail;
    u32 count;
} RunInfoList;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define DIV_CEIL(a, b) ((a + b - 1) / b)

#define TO_MEGABYTES(a) ((a) / (1024 * 1024))

#endif
