; Magic: 0x07230203
; Version: 0x00010500 (Version: 1.5.0)
; Generator: 0x00080001 (käsin tehty artisaanikoodi)
; Bound: 100
; Schema: 0
    OpCapability Shader
    OpCapability GroupNonUniform
    OpCapability GroupNonUniformArithmetic
    OpCapability GroupNonUniformBallot
    OpCapability GroupNonUniformQuad
    OpCapability GroupNonUniformVote
    OpCapability Groups

    OpCapability VulkanMemoryModel

    OpMemoryModel Logical Vulkan
    OpEntryPoint GLCompute %main "main" %invocation_id %SubgroupSize %SubgroupID %SubgroupLocalID %out %input
    OpExecutionMode %main LocalSize 1024 1 1
    OpDecorate %invocation_id BuiltIn GlobalInvocationId

    OpDecorate %oa ArrayStride 4
    OpMemberDecorate %os 0 Offset 0
    OpDecorate %os Block
    OpDecorate %out DescriptorSet 0
    OpDecorate %out Binding 0
    OpDecorate %out Aliased

    OpDecorate %lra ArrayStride 4
    OpMemberDecorate %lrs 0 Offset 0
    OpDecorate %lrs Block
    OpDecorate %input DescriptorSet 0
    OpDecorate %input Binding 1

    OpDecorate %SubgroupSize RelaxedPrecision
    OpDecorate %SubgroupSize Flat
    OpDecorate %SubgroupSize BuiltIn SubgroupSize

    OpDecorate %SubgroupLocalID RelaxedPrecision
    OpDecorate %SubgroupLocalID Flat
    OpDecorate %SubgroupLocalID BuiltIn SubgroupLocalInvocationId

    OpDecorate %SubgroupID RelaxedPrecision
    OpDecorate %SubgroupID Flat
    OpDecorate %SubgroupID BuiltIn SubgroupId

; All types, variables, and constants
    %1 = OpTypeInt 32 0
    %void = OpTypeVoid
    %11 = OpTypeFunction %void
    %bool = OpTypeBool
    %float = OpTypeFloat 32

    %uint_0 = OpConstant %1 0
    %uint_1 = OpConstant %1 1
    %uint_2 = OpConstant %1 2
    %uint_3 = OpConstant %1 3
    %uint_5 = OpConstant %1 5
    %uint_64 = OpConstant %1 64
    %uint_1024 = OpConstant %1 1024

    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1

    %true = OpConstantTrue %bool
    %false = OpConstantFalse %bool

    %_ptr_Input_uint = OpTypePointer Input %1
    %SubgroupSize = OpVariable %_ptr_Input_uint Input
    %SubgroupLocalID = OpVariable %_ptr_Input_uint Input
    %SubgroupID = OpVariable %_ptr_Input_uint Input

; Wg
    %wg_vec = OpTypeVector %1 3
    %wg_vec_p = OpTypePointer Input %wg_vec
    %invocation_id = OpVariable %wg_vec_p Input
    %wg = OpTypePointer Input %1

    %lra = OpTypeArray %1 %uint_1
    %lrs = OpTypeStruct %lra
    %lrsp = OpTypePointer StorageBuffer %lrs
    %input = OpVariable %lrsp StorageBuffer

    %oa = OpTypeArray %1 %uint_1024
    %os = OpTypeStruct %oa
    %osp = OpTypePointer StorageBuffer %os
    %out = OpVariable %osp StorageBuffer

; Pointer types
    %_ptr_Function_uint = OpTypePointer Function %1
    %_ptr_Uniform_uint = OpTypePointer StorageBuffer %1
    %_ptr_Uniform_float = OpTypePointer StorageBuffer %float
    %_ptr_Uniform_bool = OpTypePointer StorageBuffer %bool
    %_ptr_Function_float = OpTypePointer Function %float

; Some access flags
    %none = OpConstant %1 0x0
    %Volatile = OpConstant %1 0x1
    %Acquire = OpConstant %1 0x2
    %Release = OpConstant %1 0x4
    %AcquireRelease = OpConstant %1 0x8
    %MakePointerVisible = OpConstant %1 0x10
    %NonPrivatePointer = OpConstant %1 0x20
    %UniformMemory = OpConstant %1 0x40

    %main = OpFunction %void None %11
    %16 = OpLabel

        ; invocation id ptr, "thread" id
        %52 = OpAccessChain %wg %invocation_id %uint_0
        %53 = OpLoad %1 %52

        ; apl iota starts from 1, so lets add u1
        %apl_iota = OpIAdd %1 %53 %uint_1

        %sync_dest = OpAccessChain %_ptr_Uniform_uint %out %uint_0 %53
        OpStore %sync_dest %apl_iota

    OpReturn
    OpFunctionEnd