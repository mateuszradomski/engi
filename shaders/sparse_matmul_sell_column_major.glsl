#version 450

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer matHeaderAndColIndex {
    uint M, C, N;
    uint columnIndex[];
};
layout(set = 0, binding = 1) buffer matRowOffsets { uint rowOffsets[]; };
layout(set = 0, binding = 2) buffer matFloat { float floatdata[]; };
layout(set = 0, binding = 3) buffer inputVector { float inVec[]; };
layout(set = 0, binding = 4) buffer outputVector { float outVec[]; };

void main()
{
    const uint rowi = gl_GlobalInvocationID.x;

    if(rowi < M) { 
        const uint sliceIndex      = rowi / C;
        const uint rowIndexInSlice = rowi % C;
        const uint columnCount     = (rowOffsets[sliceIndex + 1] - rowOffsets[sliceIndex]) / C;
        const uint rowOffset       = rowOffsets[sliceIndex] + rowIndexInSlice;

        float prod = 0.0;
        for(uint coli = 0; coli < columnCount; coli++) {
            const uint cellOffset = rowOffset + coli * C;
            prod += inVec[columnIndex[cellOffset]] * floatdata[cellOffset];
        }
        outVec[rowi] = prod;
    }
}
