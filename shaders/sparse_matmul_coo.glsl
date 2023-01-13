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
    float data[];
};

layout(set = 0, binding = 1) buffer bufARow
{
    uint rows[];
};

layout(set = 0, binding = 2) buffer bufACol
{
    uint cols[];
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
    const uint ii = gl_GlobalInvocationID.x;

    if(ii < elementNum) { 
        const uint row = rows[ii] - 1;
        const uint col = cols[ii] - 1;

        float sum = data[ii] * inVec[col];
        uint expected_mem = outVecU32[row];
        float input_mem = outVec[row] + sum;
        uint returned_mem = atomicCompSwap(outVecU32[row], expected_mem, floatBitsToUint(input_mem));
        while(returned_mem != expected_mem){
            expected_mem = returned_mem;
            input_mem = (uintBitsToFloat(expected_mem) + sum);
            returned_mem = atomicCompSwap(outVecU32[row], expected_mem, floatBitsToUint(input_mem));
        }
    }
}
