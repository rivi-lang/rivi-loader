; SPIR-V
; Version: 1.5
; Generator: Khronos SPIR-V Tools Assembler; 0
; Bound: 85
; Schema: 0
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical Simple
               OpEntryPoint GLCompute %1 "main" %gl_GlobalInvocationID %3 %4 %5 %6 %7 %8 %9
               OpExecutionMode %1 LocalSize 1024 1 1
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %_arr__arr_float_uint_198_uint_5788 ArrayStride 792
               OpMemberDecorate %_struct_11 0 Offset 0
               OpDecorate %_struct_11 Block
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %3 Aliased
               OpDecorate %_arr_float_uint_3285 ArrayStride 4
               OpMemberDecorate %_struct_13 0 Offset 0
               OpDecorate %_struct_13 Block
               OpDecorate %4 DescriptorSet 0
               OpDecorate %4 Binding 1
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 2
               OpDecorate %6 DescriptorSet 0
               OpDecorate %6 Binding 3
               OpDecorate %7 DescriptorSet 0
               OpDecorate %7 Binding 4
               OpDecorate %_arr_float_uint_198 ArrayStride 4
               OpDecorate %_arr__arr_float_uint_198_uint_3285 ArrayStride 792
               OpMemberDecorate %_struct_16 0 Offset 0
               OpDecorate %_struct_16 Block
               OpDecorate %8 DescriptorSet 0
               OpDecorate %8 Binding 5
               OpDecorate %_arr_float_uint_300 ArrayStride 4
               OpDecorate %_arr__arr_float_uint_300_uint_5788 ArrayStride 1200
               OpMemberDecorate %_struct_19 0 Offset 0
               OpDecorate %_struct_19 Block
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 6
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
       %void = OpTypeVoid
         %23 = OpTypeFunction %void
       %bool = OpTypeBool
      %float = OpTypeFloat 32
   %uint_300 = OpConstant %uint 300
  %uint_5788 = OpConstant %uint 5788
  %uint_3285 = OpConstant %uint 3285
   %uint_198 = OpConstant %uint 198
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%_ptr_Input_uint = OpTypePointer Input %uint
%_arr_float_uint_3285 = OpTypeArray %float %uint_3285
 %_struct_13 = OpTypeStruct %_arr_float_uint_3285
%_ptr_StorageBuffer__struct_13 = OpTypePointer StorageBuffer %_struct_13
          %4 = OpVariable %_ptr_StorageBuffer__struct_13 StorageBuffer
          %5 = OpVariable %_ptr_StorageBuffer__struct_13 StorageBuffer
          %6 = OpVariable %_ptr_StorageBuffer__struct_13 StorageBuffer
          %7 = OpVariable %_ptr_StorageBuffer__struct_13 StorageBuffer
%_arr_float_uint_198 = OpTypeArray %float %uint_198
%_arr__arr_float_uint_198_uint_3285 = OpTypeArray %_arr_float_uint_198 %uint_3285
 %_struct_16 = OpTypeStruct %_arr__arr_float_uint_198_uint_3285
%_ptr_StorageBuffer__struct_16 = OpTypePointer StorageBuffer %_struct_16
          %8 = OpVariable %_ptr_StorageBuffer__struct_16 StorageBuffer
%_arr_float_uint_300 = OpTypeArray %float %uint_300
%_arr__arr_float_uint_300_uint_5788 = OpTypeArray %_arr_float_uint_300 %uint_5788
 %_struct_19 = OpTypeStruct %_arr__arr_float_uint_300_uint_5788
%_ptr_StorageBuffer__struct_19 = OpTypePointer StorageBuffer %_struct_19
          %9 = OpVariable %_ptr_StorageBuffer__struct_19 StorageBuffer
%_arr__arr_float_uint_198_uint_5788 = OpTypeArray %_arr_float_uint_198 %uint_5788
 %_struct_11 = OpTypeStruct %_arr__arr_float_uint_198_uint_5788
%_ptr_StorageBuffer__struct_11 = OpTypePointer StorageBuffer %_struct_11
          %3 = OpVariable %_ptr_StorageBuffer__struct_11 StorageBuffer
%_ptr_Function_uint = OpTypePointer Function %uint
%_ptr_StorageBuffer__arr_float_uint_198 = OpTypePointer StorageBuffer %_arr_float_uint_198
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%_ptr_Function_float = OpTypePointer Function %float
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %int_n1 = OpConstant %int -1
    %float_0 = OpConstant %float 0
   %float_n1 = OpConstant %float -1
         %46 = OpTypeFunction %uint %uint
          %1 = OpFunction %void None %23
         %47 = OpLabel
         %48 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
         %49 = OpLoad %uint %48
         %50 = OpULessThan %bool %49 %uint_5788
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %53
         %52 = OpLabel
         %54 = OpFunctionCall %uint %55 %49
         %56 = OpAccessChain %_ptr_StorageBuffer__arr_float_uint_198 %3 %uint_0 %49
         %57 = OpAccessChain %_ptr_StorageBuffer__arr_float_uint_198 %8 %uint_0 %54
         %58 = OpLoad %_arr_float_uint_198 %57
               OpStore %56 %58
               OpBranch %51
         %53 = OpLabel
               OpBranch %51
         %51 = OpLabel
               OpReturn
               OpFunctionEnd
         %55 = OpFunction %uint None %46
         %59 = OpFunctionParameter %uint
         %60 = OpLabel
         %61 = OpVariable %_ptr_Function_uint Function %uint_0
               OpBranch %62
         %62 = OpLabel
               OpLoopMerge %63 %64 None
               OpBranch %65
         %65 = OpLabel
         %66 = OpLoad %uint %61
         %67 = OpAccessChain %_ptr_StorageBuffer_float %4 %uint_0 %66
         %68 = OpLoad %float %67
         %69 = OpConvertFToS %int %68
         %70 = OpINotEqual %bool %69 %int_n1
               OpBranchConditional %70 %71 %63
         %71 = OpLabel
         %72 = OpAccessChain %_ptr_StorageBuffer_float %7 %uint_0 %66
         %73 = OpLoad %float %72
         %74 = OpConvertFToU %uint %73
         %75 = OpAccessChain %_ptr_StorageBuffer_float %9 %uint_0 %59 %74
         %76 = OpLoad %float %75
         %77 = OpAccessChain %_ptr_StorageBuffer_float %6 %uint_0 %66
         %78 = OpLoad %float %77
         %79 = OpFOrdLessThanEqual %bool %76 %78
         %80 = OpSelect %_ptr_StorageBuffer__struct_13 %79 %4 %5
         %81 = OpAccessChain %_ptr_StorageBuffer_float %80 %uint_0 %66
         %82 = OpLoad %float %81
         %83 = OpConvertFToU %uint %82
               OpStore %61 %83
               OpBranch %64
         %64 = OpLabel
               OpBranch %62
         %63 = OpLabel
         %84 = OpLoad %uint %61
               OpReturnValue %84
               OpFunctionEnd
