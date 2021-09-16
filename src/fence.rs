use std::error::Error;

use ash::{version::DeviceV1_0, vk};


pub struct Fence {
    pub(crate) fence: vk::Fence,
    pub(crate) present_queue: vk::Queue,
    pub(crate) phy_index: usize,
}

impl Fence {

    pub(crate) unsafe fn new(
        device: &ash::Device,
        queue_family_index: u32,
        queue_index: u32
    ) -> Result<Fence, Box<dyn Error>> {
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let present_queue = device.get_device_queue(queue_family_index, queue_index);
        Ok(Fence{fence, present_queue, phy_index: queue_family_index as usize})
    }

    pub(crate) unsafe fn submit(
        &self,
        device: &ash::Device,
        command_buffers: &[vk::CommandBuffer]
    ) -> Result<(), Box<dyn Error>> {
        let info = [vk::SubmitInfo::builder().command_buffers(command_buffers).build()];
        let result = device.queue_submit(
            self.present_queue,
            &info,
            self.fence,
        )?;
        Ok(result)
    }

}