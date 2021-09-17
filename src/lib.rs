use std::error::Error;

use debug_layer::DebugOption;
use gpu_allocator::*;

pub mod compute;
pub mod spirv;
pub mod debug_layer;
pub mod vulkan;
mod buffer;
mod command;
mod fence;
mod gpu;
mod shader;

use crate::compute::Compute;
use crate::vulkan::Vulkan;


pub fn new(
    debug: DebugOption
) -> Result<(Vulkan, Vec<Compute>), Box<dyn Error>> {
    let vk = unsafe { Vulkan::new(debug)? };
    let logical_devices = unsafe { vk.logical_devices(vk.gpus()?) };
    Ok((vk, logical_devices))
}
