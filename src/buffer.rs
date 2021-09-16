use std::{error::Error, sync::Mutex};

use ash::{version::DeviceV1_0, vk};
use gpu_allocator::{AllocationCreateDesc, MemoryLocation};


pub(crate) struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: gpu_allocator::SubAllocation,
    pub(crate) device_size: vk::DeviceSize,
}

impl Buffer {

    pub(crate) fn new(
        device: &ash::Device,
        allocator_arc: &Mutex<gpu_allocator::VulkanAllocator>,
        device_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryLocation,
        queue_family_indices: &[u32],
    ) -> Result<Buffer, Box<dyn Error>> {
        let sharing_mode = match queue_family_indices.len() {
            1 => vk::SharingMode::EXCLUSIVE,
            _ => vk::SharingMode::CONCURRENT,
        };
        let create_info = vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(queue_family_indices);
        let buffer = unsafe { device.create_buffer(&create_info, None) }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut allocator = allocator_arc.lock().unwrap();
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "Allocation",
            requirements,
            location: memory_usage,
            linear: true,
        })?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?; }
        Ok(Buffer {
            buffer,
            allocation,
            device_size,
        })
    }

    pub(crate) fn fill<T: Sized>(
        &self,
        data: &[T],
    ) -> Result<(), Box<dyn Error>> {
        let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(())
    }

}