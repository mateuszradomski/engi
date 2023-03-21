#version 450

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer bufA {
    uint blockSize, nnzb, MB, NB;
    float floatdata[];
};
layout(set = 0, binding = 1) buffer bufAColumnIndex { uint rowOffsets[]; };
layout(set = 0, binding = 2) buffer bufARowOffsets  { uint colIndicies[]; };
layout(set = 0, binding = 3) buffer inputVector     { float inVec[]; };
layout(set = 0, binding = 4) buffer outputVector    { float outVec[]; };

//layout(set = 0, binding = 1) buffer inputVector    { float outVec[]; };

layout(set = 0, binding = 1)
    buffer type_t {
        float data[];
    } singleDescriptor[];

void main()
{
    const uint rowi = gl_GlobalInvocationID.x;
    const uint rowInBlockIndex = rowi % blockSize;
    const uint rowBlockIndex = rowi / blockSize;

    if(rowInBlockIndex < blockSize && rowBlockIndex < MB) {
        uint rowOffset = rowOffsets[rowBlockIndex];
        uint nnzCol = rowOffsets[rowBlockIndex + 1] - rowOffset;
        float prod = 0.0;
        for(uint coli = 0; coli < nnzCol; coli++) {
            const uint cellOffset = rowOffset + coli;
            const uint blockOffset = cellOffset * blockSize * blockSize;
            const uint colbi = colIndicies[cellOffset];
            for(uint cbi = 0; cbi < blockSize; cbi++) {
                prod += inVec[colbi * blockSize + cbi] * floatdata[blockOffset + cbi + rowInBlockIndex * blockSize];
            }
        }
        outVec[rowBlockIndex * blockSize + rowInBlockIndex] = prod;
    }
}
