#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_KHR_memory_scope_semantics : enable

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer bufA
{
    uint elementNum;
    uint M;
    uint N;
    float floatdata[];
};

layout(set = 0, binding = 1) buffer bufAColumnIndex
{
    uint columnIndex[];
};

layout(set = 0, binding = 2) buffer bufARowOffsets
{
    uint rowOffsets[];
};

layout(set = 0, binding = 3) buffer inputVector
{
    float inVec[];
};

layout(set = 0, binding = 4) buffer outputVector
{
    float outVec[];
};

void main()
{
    const uint rowi = gl_GlobalInvocationID.x;

    if(rowi < M) { 
        const uint rowOffset = rowOffsets[rowi];
        const uint nzCount = rowOffsets[rowi+1] - rowOffset;

        float sum = 0.0;
        for(uint coli = 0; coli < nzCount; coli++)
        {
            const uint cellOffset = rowOffset + coli;
            sum += inVec[columnIndex[cellOffset]] * floatdata[cellOffset];
        }

        outVec[rowi] = sum;
    }
}
