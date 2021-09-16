#version 330 es
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 1, std430) buffer _13_4
{
    float _m0[3285];
} _4;

layout(set = 0, binding = 2, std430) buffer _13_5
{
    float _m0[3285];
} _5;

layout(set = 0, binding = 3, std430) buffer _13_6
{
    float _m0[3285];
} _6;

layout(set = 0, binding = 4, std430) buffer _13_7
{
    float _m0[3285];
} _7;

layout(set = 0, binding = 5, std430) buffer _16_8
{
    float _m0[3285][198];
} _8;

layout(set = 0, binding = 6, std430) buffer _19_9
{
    float _m0[5788][300];
} _9;

layout(set = 0, binding = 0, std430) buffer _11_3
{
    float _m0[5788][198];
} _3;

uint _55(uint _58)
{
    uint _60 = 0u;
    while (int(_4._m0[_60]) != (-1))
    {
        float _72 = _7._m0[_60];
        float _75 = _9._m0[_58][uint(_72)];
        float _77 = _6._m0[_60];
        _60 = uint(((_75 <= _77) ? &_4 : &_5)._m0[_60]);
    }
    return _60;
}

void main()
{
    if (gl_GlobalInvocationID.x < 5788u)
    {
        _3._m0[gl_GlobalInvocationID.x][0u] = float(int(_55(gl_GlobalInvocationID.x)));
    }
    else
    {
    }
}

