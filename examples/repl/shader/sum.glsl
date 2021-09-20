#version 320
#ifdef GL_ARB_shading_language_420pack
#extension GL_ARB_shading_language_420pack : require
#endif
#extension GL_ARB_compute_shader : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer CompOutputBuffer
{
    float numbers[];
} comp_output;

layout(set = 0, binding = 1, std430) buffer LeftBuffer
{
    float numbers[];
} lhs;

layout(set = 0, binding = 2, std430) buffer RightBuffer
{
    float numbers[];
} rhs;

void main()
{
    uint index = gl_GlobalInvocationID.x;
    comp_output.numbers[index] = lhs.numbers[index] + rhs.numbers[index];
}

