#version 450

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer bufA {
    uint elementNum;
    uint M;
    uint N;
    float data[];
};
layout(set = 0, binding = 1) buffer bufARow      { uint rows[]; };
layout(set = 0, binding = 2) buffer bufACol      { uint cols[]; };
layout(set = 0, binding = 3) buffer bufInVec     { float inVec[]; };
layout(set = 0, binding = 4) buffer bufOutVec    { float outVec[]; };
layout(set = 0, binding = 4) buffer bufOutVecU32 { uint outVecU32[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    if(i < elementNum) { 
        const uint row = rows[i] - 1;
        const uint col = cols[i] - 1;

        float sum = data[i] * inVec[col];
        uint expected_mem = outVecU32[row];
        float input_mem = outVec[row] + sum;
        uint returned_mem = atomicCompSwap(outVecU32[row], expected_mem,
                                           floatBitsToUint(input_mem));
        while(returned_mem != expected_mem) {
            expected_mem = returned_mem;
            input_mem = (uintBitsToFloat(expected_mem) + sum);
            returned_mem = atomicCompSwap(outVecU32[row], expected_mem,
                                          floatBitsToUint(input_mem));
        }
    }
}
