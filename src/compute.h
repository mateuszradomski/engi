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

typedef struct RunInformation 
{
    double time;
    double gflops;
} RunInformation;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define DIV_CEIL(a, b) ((a + b - 1) / b)

#define TO_MEGABYTES(a) ((a) / (1024 * 1024))

#endif
