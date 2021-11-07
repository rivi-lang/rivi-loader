use std::error::Error;

use ash::vk;


pub struct Fence {
    pub fence: vk::Fence,
    pub present_queue: vk::Queue,
    pub phy_index: usize,
}

impl Fence {

    pub fn new(
        device: &ash::Device,
        queue_family_index: u32,
        queue_index: u32
    ) -> Result<Fence, Box<dyn Error>> {
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let present_queue = unsafe{ device.get_device_queue(queue_family_index, queue_index) };
        Ok(Fence{fence, present_queue, phy_index: queue_family_index as usize})
    }

    pub fn submit(
        &self,
        device: &ash::Device,
        command_buffers: &[vk::CommandBuffer]
    ) -> Result<(), Box<dyn Error>> {
        let info = [vk::SubmitInfo::builder().command_buffers(command_buffers).build()];
        let result = unsafe { device.queue_submit(self.present_queue, &info, self.fence)? };
        Ok(result)
    }

}