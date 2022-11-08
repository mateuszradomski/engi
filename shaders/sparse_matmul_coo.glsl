#version 450
#extension GL_ARB_separate_shader_objects : enable

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
    uint row[];
};

layout(set = 0, binding = 2) buffer bufACol
{
    uint col[];
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
    const uint ii = gl_GlobalInvocationID.x;

    if(ii < elementNum) { 
        const uint row = row[ii];
        const uint col = col[ii];
        const uint index = (row - 1) * M + (col - 1);
        float INDATA = data[index] * inVec[row - 1];

        uint MEM = floatBitsToUint(outVec[row]);
        uint expected_mem = MEM;
        float input_mem = (uintBitsToFloat(MEM) + INDATA);  //initially this is what we assume we want to put in here. 
        uint returned_mem = atomicCompSwap(MEM, expected_mem, floatBitsToUint(input_mem)); //if data returned is what we expected it to be, we're good, we added to it successfully
        while(returned_mem != expected_mem){ // if the data returned is something different, we know another thread completed its transaction, so we'll have to add to that instead. 
            expected_mem = returned_mem;
            input_mem = (uintBitsToFloat(expected_mem) + INDATA);
            returned_mem = atomicCompSwap(MEM, expected_mem, floatBitsToUint(input_mem));
        }
    }
}
