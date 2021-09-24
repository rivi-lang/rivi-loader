use std::{error::Error, sync::{RwLock}};

use ash::{version::DeviceV1_0, vk};
use gpu_allocator::{AllocationCreateDesc, MemoryLocation, VulkanAllocator};


pub struct Buffer<'a, 'b>  {
    pub buffer: vk::Buffer,
    pub allocation: gpu_allocator::SubAllocation,
    pub device_size: vk::DeviceSize,

    device: &'a ash::Device,
    allocator: &'b Option<RwLock<gpu_allocator::VulkanAllocator>>,
}

impl <'a, 'b> Buffer<'_, '_> {

    pub fn new(
        device: &'a ash::Device,
        allocator: &'b Option<RwLock<VulkanAllocator>>,
        device_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryLocation,
        queue_family_indices: &[u32],
    ) -> Result<Buffer<'a, 'b>, Box<dyn Error>> {
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
        let mut malloc = allocator.as_ref().unwrap().write().unwrap();
        let allocation = malloc.allocate(&AllocationCreateDesc {
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
            device,
            allocator,
        })
    }

    pub fn fill<T: Sized>(
        &self,
        data: &[T],
    ) -> Result<(), Box<dyn Error>> {
        let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(())
    }

}

impl <'a, 'b> Drop for Buffer<'a, 'b> {
    fn drop(
        &mut self
    ) {
        let some = self.allocator.as_ref().unwrap();
        let mut malloc = some.write().unwrap();
        malloc.free(self.allocation.to_owned()).unwrap();
        unsafe { self.device.destroy_buffer(self.buffer, None) };
    }
}