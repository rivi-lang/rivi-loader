use std::error::Error;

use gpu_allocator::*;

pub mod spirv;
mod buffer;
mod command;
mod fence;
mod gpu;
mod shader;
mod compute;
mod debug_layer;
mod vulkan;

use crate::compute::Compute;
use crate::vulkan::Vulkan;


pub fn new(
    debug_flag: bool
) -> Result<(Vulkan, Vec<Compute>), Box<dyn Error>> {
    let vk = unsafe { Vulkan::new(debug_flag)? };
    let logical_devices = unsafe { vk.logical_devices(vk.gpus()?) };
    Ok((vk, logical_devices))
}
