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
    float *data;
    u32 len;
} Vector;

typedef struct COOMatrix
{
    float *data;
    u32 *row, *col, elementNum;
    u32 M, N;
} COOMatrix;

// If in columnIndex means no data (zero) at that space
#define INVALID_COLUMN 0xffffffff

typedef struct ELLMatrix
{
    float *data;
    u32 M, P, N, *columnIndex;
    u32 elementNum;
} ELLMatrix;

typedef struct SELLMatrix
{
    float *data;
    u32 M, N, C;
    u32 *columnIndex, *rowOffsets;
    u32 elementNum;
} SELLMatrix;

typedef struct CSRMatrix
{
    float *data;
    u32 M, N;
    u32 *columnIndex, *rowOffsets;
    u32 elementNum;
} CSRMatrix;

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
