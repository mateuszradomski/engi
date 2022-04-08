#ifndef COMPUTE_H
#define COMPUTE_H

#include <stdint.h>

typedef struct Data
{
    uint8_t *bytes;
    uint32_t length;
} Data;

typedef struct StringSlice
{
    char *bytes;
    uint32_t length;
} StringSlice;

#define COMP_STR_TO_STRING_SLICE(str) { .bytes = str, .length = sizeof(str) - 1 }

typedef struct StringSplitIterator
{
    char *str, *head, *delim;
    uint32_t strLength, delimLength;
} StringSplitIterator;

typedef struct COOMatrix
{
    float *data;
    uint32_t *row, *col, elementNum;
} COOMatrix;

// If in columnIndex means no data (zero) at that space
#define INVALID_COLUMN 0xffffffff

typedef struct ELLMatrix
{
    float *data;
    uint32_t M, P, *columnIndex;
} ELLMatrix;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#endif