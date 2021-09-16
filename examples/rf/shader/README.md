# SPIR-V Notes

Herein are some documentation along the journey of hand-writing SPIR-V from the groundup:

## Atomic operations and floats

Atomic operations are not supported for floats in Vulkan. This means much of traditional locking mechanism are unavailable. Instead, you must use the Vulkan memory model to your benefit: when you store, you make make an availability operation on non-private pointer on the memory scope of your liking. For example: `OpStore %target_ptr %val MakePointerAvailable|NonPrivatePointer %DeviceScope`. These operations adhere to memory barriers, after which you can do `OpLoad` and you are guaranteed by the system to find `val` in `target_ptr`.

## Float64

Float64 do not seem to be trivially supported on Apple hardware: `SPIRV-Cross threw an exception: double types are not supported in buffers in MSL.`.

## Subgroup operations

Eventually some communication between threads are needed, such as when doing a sum reduction. Such operations are listed under [3.36.21. Group and Subgroup Instructions](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_group_a_group_and_subgroup_instructions). Examples can be found: [Vulkan Subgroup Explained](https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/06-subgroups.pdf) and [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).

Yet, the operations on [3.36.21. Group and Subgroup Instructions](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_group_a_group_and_subgroup_instructions) actually give an error if you try to apply them: GLSL cross-compilation and MSL cross-compilation say `SPIRV-Cross threw an exception: Cannot resolve expression type.`. However, from [an old Github issue](https://github.com/KhronosGroup/SPIRV-Cross/issues/499) you can find files which show that what actually implements the subgroup operations are the ones listed under [3.36.24. Non-Uniform Instructions](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_non_uniform_a_non_uniform_instructions). Using these instructions will not resolve into an opcode error, but something else: for GLSL you get `SPIRV-Cross threw an exception: Can only use subgroup operations in Vulkan semantics.` (regardless of whether your SPIR-V file states VulkanMemoryModel and OpMemoryModel as Vulkan), and similarly, for MSL you get `SPIRV-Cross threw an exception: Subgroups are only supported in Metal 2.0 and up.`. From the MSL error we can deduce it's a cross-compilation issue, thus for MSL `spirv-cross` command you must add `--msl-version020000` and for GLSL you must add `-V` (V for Vulkan).

## Bugs found

### SPIRV-Cross

1. [MSL: OpAtomicStore produces an unexpected variable](https://github.com/KhronosGroup/SPIRV-Cross/issues/1343)
