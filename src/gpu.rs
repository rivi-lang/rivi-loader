use std::{error::Error, ffi::CString, sync::{RwLock}};

use ash::{version::{InstanceV1_0, InstanceV1_1}, vk};
use gpu_allocator::{VulkanAllocator, VulkanAllocatorCreateDesc};

use crate::{Compute, fence::Fence};


pub(crate) struct GPU {
    pub physical: vk::PhysicalDevice,
    pub queue_families: Vec<QueueFamily>,
}

pub(crate) struct QueueFamily {
    pub queue_count: u32,
    pub physical_index: usize,
}

impl GPU {

    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice
    ) -> Option<GPU> {

        // Retrieving Subgroup operations will segfault a Mac
        // https://www.khronos.org/blog/vulkan-subgroup-tutorial
        let mut sp = vk::PhysicalDeviceSubgroupProperties::builder();
        let mut dp2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut sp)
            .build();
        instance
            .fp_v1_1()
            .get_physical_device_properties2(pdevice, &mut dp2);

        let device_name = dp2
            .properties
            .device_name
            .iter()
            .filter_map(|f| {
                let u = *f as u8;
                match u {
                    0 => None,
                    _ => Some(u as char),
                }
            })
            .collect::<String>();

        println!("Found device: {} ({:?})", device_name, dp2.properties.device_type);
        println!("Physical device has subgroup size of: {:?}", sp.subgroup_size);
        println!("Supported subgroup operations: {:?}", sp.supported_operations);
        println!("Supported subgroup stages: {:?}", sp.supported_stages);

        let queues = instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .filter_map(|(index, prop)| {
                println!("Queue family at index {} has {} threads and capabilities: {:?}", index, prop.queue_count, prop.queue_flags);
                match prop.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    false => None,
                    true => Some(QueueFamily{queue_count: prop.queue_count, physical_index: index}),
                }
            })
            .collect::<Vec<QueueFamily>>();

        match queues.is_empty() {
            false => Some(GPU{physical: pdevice, queue_families: queues}),
            true => None,
        }
    }

    pub(crate) unsafe fn device(
        &self,
        instance: &ash::Instance
    ) -> Result<Compute, Box<dyn Error>> {

        let queue_infos: Vec<_> = self.queue_families
            .iter()
            .map(|queue| {

                let queues = (0..queue.queue_count)
                    .into_iter()
                    .map(|_| 1.0f32)
                    .collect::<Vec<f32>>();

                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue.physical_index as u32)
                    .queue_priorities(&queues)
                    .build()
            })
            .collect();

        let features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let memory = instance.get_physical_device_memory_properties(self.physical);

        let mut variable_pointers = vk::PhysicalDeviceVariablePointersFeatures::builder()
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true)
            .build();

        let mut ext_names: Vec<CString> = vec![
            CString::new("VK_KHR_variable_pointers")?,
            CString::new("VK_KHR_get_memory_requirements2")?,
            CString::new("VK_KHR_dedicated_allocation")?,
        ];

        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            ext_names.push(CString::new("VK_KHR_portability_subset")?);
        }

        let ext_names_raw: Vec<_> = ext_names
            .iter().map(|raw_name| raw_name.as_ptr()).collect();
        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&ext_names_raw)
            .enabled_features(&features)
            .push_next(&mut variable_pointers);

        let device = instance.create_device(self.physical, &device_info, None)?;

        let allocator_create_info = VulkanAllocatorCreateDesc {
            physical_device: self.physical,
            device: device.clone(),
            instance: instance.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let allocator = VulkanAllocator::new(&allocator_create_info);

        let fences = queue_infos
            .iter()
            .flat_map(|queue_info| {
                (0..queue_info.queue_count)
                    .into_iter()
                    .filter_map(|index| {
                        match Fence::new(&device, queue_info.queue_family_index, index) {
                            Ok(f) => Some(f),
                            Err(_) => None,
                        }
                    })
                    .collect::<Vec<Fence>>()
            })
            .collect::<Vec<Fence>>();

        Ok(Compute{ device, allocator: Some(RwLock::new(allocator)), fences, memory })
    }

}