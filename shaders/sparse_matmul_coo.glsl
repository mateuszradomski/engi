#version 450
#extension GL_EXT_shader_atomic_float: enable

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer bufA {
    uint elementNum, M, N;
    float data[];
};
layout(set = 0, binding = 1) buffer bufARow      { uint rows[]; };
layout(set = 0, binding = 2) buffer bufACol      { uint cols[]; };
layout(set = 0, binding = 3) buffer bufInVec     { float inVec[]; };
layout(set = 0, binding = 4) buffer bufOutVec    { float outVec[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    if(i < elementNum) { 
        const uint row = rows[i];
        const uint col = cols[i];

        float sum = data[i] * inVec[col];
        atomicAdd(outVec[row], sum);
    }
}
