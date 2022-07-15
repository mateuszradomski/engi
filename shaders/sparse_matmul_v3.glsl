#version 450
#extension GL_ARB_separate_shader_objects : enable

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer matHeaderAndColIndex
{
    uint M, C, N;
    uint columnIndex[];
};

layout(set = 0, binding = 1) buffer matRowOffsets
{
    uint rowOffsets[];
};

layout(set = 0, binding = 2) buffer matFloat
{
    float floatdata[];
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
        const uint sliceIndex = rowi / C;
        const uint sliceC = (rowOffsets[sliceIndex + 1] - rowOffsets[sliceIndex]) / C;
        const uint rowOffset = rowOffsets[sliceIndex] + sliceC * (rowi % C);

        float sum = 0.0;
        for(uint coli = 0; coli < sliceC; coli++)
        {
            const uint cellOffset = rowOffset + coli;
            if(columnIndex[cellOffset] == 0xffffffff) {
                break;
            }

            sum += inVec[columnIndex[cellOffset]] * floatdata[cellOffset];
        }

        outVec[rowi] = sum;
    }
}
