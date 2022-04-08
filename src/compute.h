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

typedef struct StringSplitIterator
{
    char *str, *head, delim;
    uint32_t strLength;
} StringSplitIterator;

typedef struct COOMatrix
{
    float *data;
    uint32_t *row;
    uint32_t *col;
    uint32_t elementNum;
} COOMatrix;

#define INVALID_COLUMN 0xffffffff;

typedef struct ELLMatrix
{
    float *data;
    uint32_t *columnIndex; // Index of column for that data [INVALID_COLUMN for no data]
    uint32_t M;           // Number of rows
    uint32_t P;           // Number of columns
} ELLMatrix;

#endif