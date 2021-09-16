use std::error::Error;

use ash::{version::DeviceV1_0, vk};


pub(crate) struct Command {
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
    pub(crate) descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Command {

    fn descriptor_pool(
        device: &ash::Device,
        descriptor_count: u32,
        max_sets: u32,
    ) -> Result<vk::DescriptorPool, Box<dyn Error>> {
        let descriptor_pool_size = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(descriptor_count)
            .build()];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&descriptor_pool_size);
        unsafe { Ok(device.create_descriptor_pool(&descriptor_pool_info, None)?) }
    }

    fn command_pool(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandPool, Box<dyn Error>> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index);
        unsafe { Ok(device.create_command_pool(&command_pool_info, None)?) }
    }

    fn allocate_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        command_buffer_count: u32,
    ) -> Result<Vec<vk::CommandBuffer>, Box<dyn Error>> {
        let command_buffers_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(command_buffer_count)
            .command_pool(command_pool);
        unsafe { Ok(device.allocate_command_buffers(&command_buffers_info)?) }
    }

    pub(crate) fn new(
        queue_family_index: u32,
        descriptor_count: u32,
        max_sets: u32,
        set_layouts: &[vk::DescriptorSetLayout],
        command_buffer_count: u32,
        device: &ash::Device,
    ) -> Result<Command, Box<dyn Error>> {

        let descriptor_pool = Command::descriptor_pool(device, descriptor_count, max_sets)?;

        let descriptor_set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(set_layouts);

        let descriptor_sets = (0..command_buffer_count)
            .into_iter()
            .filter_map(|_| match unsafe { device.allocate_descriptor_sets(&descriptor_set_info) } {
                Ok(ds) => Some(ds[0]),
                Err(_) => None,
            })
            .collect();

        let command_pool = Command::command_pool(device, queue_family_index)?;
        let command_buffers = Command::allocate_command_buffers(device, command_pool, command_buffer_count)?;

        Ok(Command {
            descriptor_pool,
            command_pool,
            command_buffers,
            descriptor_sets,
        })
    }
}