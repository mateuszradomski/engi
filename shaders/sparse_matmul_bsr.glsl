#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_KHR_memory_scope_semantics : enable

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer bufA
{
    uint blockSize;
    uint nnzb;
    uint MB;
    uint NB;
    float floatdata[];
};

layout(set = 0, binding = 1) buffer bufAColumnIndex
{
    uint rowOffsets[];
};

layout(set = 0, binding = 2) buffer bufARowOffsets
{
    uint colIndicies[];
};

layout(set = 0, binding = 3) buffer inputVector
{
    float inVec[];
};

layout(set = 0, binding = 4) buffer outputVector
{
    float outVec[];
};

layout(set = 0, binding = 4) buffer outputVectorU32
{
    uint outVecU32[];
};

void main()
{
    const uint rowbi = gl_GlobalInvocationID.x;
    if(rowbi < MB) {
        uint rowOffset = rowOffsets[rowbi];
        uint nnzCol = rowOffsets[rowbi + 1] - rowOffset;

        for(uint coli = 0; coli < nnzCol; coli++)
        {
            const uint cellOffset = rowOffset + coli;
            const uint blockOffset = cellOffset * blockSize * blockSize;

            const uint colbi = colIndicies[cellOffset];
            for(uint rbi = 0; rbi < blockSize; rbi++)
            {
                float rowSum = 0.0;
                for(uint cbi = 0; cbi < blockSize; cbi++)
                {
                    rowSum += inVec[colbi * blockSize + cbi] * floatdata[blockOffset + cbi + rbi * blockSize];
                }
                outVec[rowbi * blockSize + rbi] += rowSum;
            }
        }
    }
}
