#version 450
#extension GL_ARB_separate_shader_objects : enable

#define WORKGROUP_SIZE 32
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer bufA {
    uint M, P, N;
    uint cellIndices[];
};

layout(set = 0, binding = 1) buffer bufAFloat    { float floatdata[]; };
layout(set = 0, binding = 2) buffer inputVector  { float inVec[]; };
layout(set = 0, binding = 3) buffer outputVector { float outVec[]; };

void main() {
    const uint rowi = gl_GlobalInvocationID.x;

    if(rowi < M) { 
        const uint rowOffset = rowi * P;

        float sum = 0.0;
        for(uint coli = 0; coli < P; coli++) {
            const uint cellOffset = rowOffset + coli;
            if(cellIndices[cellOffset] == 0xffffffff) { break; }
            sum += inVec[cellIndices[cellOffset]] * floatdata[cellOffset];
        }
        outVec[rowi] = sum;
    }
}
