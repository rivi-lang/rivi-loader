use std::error::Error;

pub mod compute;
pub mod debug_layer;
pub mod vulkan;
pub mod shader;
pub mod buffer;
pub mod command;
pub mod fence;
pub mod gpu;

use crate::compute::Compute;
use crate::vulkan::Vulkan;
use crate::debug_layer::DebugOption;

pub fn new(
    debug: DebugOption
) -> Result<(Vulkan, Vec<Compute>), Box<dyn Error>> {
    let vk = Vulkan::new(debug)?;
    let logical_devices = unsafe { vk.logical_devices(vk.gpus()?) };
    Ok((vk, logical_devices))
}
