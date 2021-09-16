use ash::vk::MemoryMapFlags;
use ash::{vk, Entry, extensions::ext::DebugUtils};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use rayon::prelude::*;
use gpu_allocator::*;

use spirv::SPIRV;

use std::sync::Mutex;
use std::{convert::TryInto, error::Error, slice};
use std::default::Default;
use std::ffi::CString;
use std::time::Instant;

pub mod spirv;

pub struct App {
    // Entry: Loads the Vulkan library.
    // Needs to outlive Instance and Device.
    entry: ash::Entry,
    // Instance: Loads instance level functions.
    // Needs to outlive the Devices it has created.
    instance: ash::Instance,
    debug: Option<Debug>,
}

pub struct LogicalDevice {
    pub device: ash::Device,
    pub allocator: Mutex<gpu_allocator::VulkanAllocator>,
    pub fences: Vec<Fence>,
}

struct Shader {
    pub shader_module: vk::ShaderModule,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl Shader {
    unsafe fn drop(self, device: &ash::Device) {
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_shader_module(self.shader_module, None);
        for set_layout in self.set_layouts {
            device.destroy_descriptor_set_layout(set_layout, None);
        }
        device.destroy_pipeline(self.pipeline, None);
    }
}
struct Debug {
    loader: ash::extensions::ext::DebugUtils,
    callback: vk::DebugUtilsMessengerEXT,
}

pub fn new(debug_flag: bool) -> Result<(App, Vec<LogicalDevice>), Box<dyn Error>> {
    unsafe {
        let app: App = create_instance(debug_flag)?;
        let logical_devices = app.logical_devices(app.gpus()?)?;
        Ok((app, logical_devices))
    }
}

unsafe fn create_instance(debug_flag: bool) -> Result<App, Box<dyn Error>> {

    let mut layer_names: Vec<CString> = Vec::new();
    if debug_flag {
        layer_names.push(CString::new("VK_LAYER_KHRONOS_validation")?);
        //layer_names.push(CString::new("VK_LAYER_LUNARG_api_dump")?);
    };
    let layers_names_raw: Vec<_> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let entry = Entry::new()?;

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
        ..Default::default()
    };
    if debug_flag {
        debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(App::vulkan_debug_utils_callback),
            ..Default::default()
        };
    }

    let instance = entry
        .create_instance(&vk::InstanceCreateInfo::builder()
            .push_next(&mut debug_info)
            .application_info(&vk::ApplicationInfo {
                api_version: vk::make_version(1, 2, 0),
                engine_version: 0,
                ..Default::default()
            })
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&[DebugUtils::name().as_ptr()])
        , None)?;

    println!("Instance created");
    match entry.try_enumerate_instance_version()? {
        Some(v) => println!("Using Vulkan {}.{}.{}", vk::version_major(v), vk::version_minor(v), vk::version_patch(v)),
        None => println!("Using Vulkan 1.0"),
    };

    let debug = match debug_flag {
        false => None,
        true => {
            let loader = DebugUtils::new(&entry, &instance);
            let callback = loader.create_debug_utils_messenger(&debug_info, None).unwrap();
            println!("Debug attached");
            Some(Debug{loader, callback})
        }
    };

    Ok(App{entry, instance, debug})
}

impl App {

    unsafe extern "system" fn vulkan_debug_utils_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
        let severity = format!("{:?}", message_severity).to_lowercase();
        let ty = format!("{:?}", message_type).to_lowercase();
        println!("[Debug][{}][{}] {:?}", severity, ty, message);
        vk::FALSE
    }

    unsafe fn gpus(&self) -> Result<Vec<GPU>, Box<dyn Error>> {
        let gpus = self.instance
            .enumerate_physical_devices()?
            .iter()
            .filter_map(|pdevice| {
                // Retrieving Subgroup operations will segfault a Mac
                // https://www.khronos.org/blog/vulkan-subgroup-tutorial
                let mut sp = vk::PhysicalDeviceSubgroupProperties::builder();
                let mut dp2 = vk::PhysicalDeviceProperties2::builder()
                    .push_next(&mut sp)
                    .build();
                self.instance
                    .fp_v1_1()
                    .get_physical_device_properties2(*pdevice, &mut dp2);
                println!("Supported subgroup operations: {:?}", sp.supported_operations);
                println!("Supported subgroup stages: {:?}", sp.supported_stages);

                let queues = self.instance
                    .get_physical_device_queue_family_properties(*pdevice)
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
                    false => Some(GPU{physical: *pdevice, queue_families: queues}),
                    true => None,
                }
            })
            .collect::<Vec<GPU>>();

        match gpus.is_empty() {
            false => Ok(gpus),
            true => Err(format!("No compute capable GPUs"))?,
        }
    }

    unsafe fn logical_devices(&self, gpus: Vec<GPU>) -> Result<Vec<LogicalDevice>, Box<dyn Error>> {
        let load_timer = Instant::now();
        let ldevices = gpus
            .iter()
            .map(|gpu| {

                let queue_infos: Vec<_> = gpu.queue_families
                    .iter()
                    .map(|queue|
                        vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(queue.physical_index as u32)
                            .queue_priorities(&[1.0f32])
                            .build()
                    )
                    .collect();

                let features = vk::PhysicalDeviceFeatures {
                    ..Default::default()
                };

                let mut variable_pointers = vk::PhysicalDeviceVariablePointersFeatures::builder()
                    .variable_pointers(true)
                    .variable_pointers_storage_buffer(true)
                    .build();

                let mut ext_names: Vec<CString> = vec![
                    CString::new("VK_KHR_variable_pointers").unwrap(),
                    CString::new("VK_KHR_get_memory_requirements2").unwrap(),
                    CString::new("VK_KHR_dedicated_allocation").unwrap(),
                ];

                if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
                    ext_names.push(CString::new("VK_KHR_portability_subset").unwrap());
                }

                let ext_names_raw: Vec<_> = ext_names
                    .iter().map(|raw_name| raw_name.as_ptr()).collect();
                let device_info = vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&ext_names_raw)
                    .enabled_features(&features)
                    .push_next(&mut variable_pointers);

                let device = self.instance.create_device(gpu.physical, &device_info, None).unwrap();

                println!("App load device created {}ms", load_timer.elapsed().as_millis());

                let allocator_create_info = VulkanAllocatorCreateDesc {
                    physical_device: gpu.physical,
                    device: device.clone(),
                    instance: self.instance.clone(),
                    debug_settings: Default::default(),
                    buffer_device_address: false,
                };

                let allocator = VulkanAllocator::new(&allocator_create_info);

                println!("App load allocator created {}ms", load_timer.elapsed().as_millis());

                let fences = queue_infos
                    .iter()
                    .flat_map(|queue_info| {
                        (0..queue_info.queue_count)
                            .into_iter()
                            .map(|index| {
                                let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).expect("Create fence failed.");
                                let present_queue = device.get_device_queue(queue_info.queue_family_index, index);
                                Fence{fence, present_queue, phy_index: queue_info.queue_family_index as usize}
                            })
                            .collect::<Vec<Fence>>()
                    })
                    .collect::<Vec<Fence>>();

                LogicalDevice{ device, allocator: Mutex::new(allocator), fences }

            })
            .collect::<Vec<LogicalDevice>>();

        Ok(ldevices)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            println!("dropping app");
            if self.debug.is_some() {
                let debug = self.debug.as_ref().unwrap();
                debug.loader.destroy_debug_utils_messenger(debug.callback, None);
                println!("debug messenger destroyed");
            }
            self.instance.destroy_instance(None);
            println!("instance destroyed");
        }
    }
}

#[derive(Debug)]
struct Buffer {
    buffer: vk::Buffer,
    allocation: gpu_allocator::SubAllocation,
    device_size: vk::DeviceSize,
    flags: MemoryMapFlags,
}

impl Buffer {
    fn new(
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
            flags: MemoryMapFlags::empty(),
        })
    }
    fn fill<T: Sized>(
        &self,
        data: &[T],
    ) -> Result<(), Box<dyn Error>> {
        let data_ptr = self.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(())
    }

}

struct Command {
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    descriptor_sets: Vec<vk::DescriptorSet>,
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

    fn new(
        queue_family_index: u32,
        descriptor_count: u32,
        max_sets: u32,
        set_layouts: &[vk::DescriptorSetLayout],
        command_buffer_count: u32,
        device: &ash::Device,
    ) -> Result<Command, Box<dyn Error>> {

        let descriptor_pool = Command::descriptor_pool(device, descriptor_count, max_sets)?;

        unsafe {
            let foo = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(set_layouts);

            let descriptor_sets = (0..command_buffer_count)
                .into_iter()
                .map(|_| {
                    let ds = device.allocate_descriptor_sets(&foo).unwrap();
                    ds.first().unwrap().to_owned()
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
}

fn load(device: &ash::Device, spirv: &SPIRV) -> Result<Shader, Box<dyn Error>>  {
    unsafe {
        let set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&spirv.dslbs),
            None,
        )?;
        let set_layouts = vec![set_layout];
        println!("Set layout created");

        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts),
            None,
        )?;
        println!("Pipeline layout done");

        let shader_entry_name = CString::new("main")?;
        let shader_module = device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spirv.binary), None)?;
        let entry_point = vk::PipelineShaderStageCreateInfo {
            p_name: shader_entry_name.as_ptr(),
            module: shader_module,
            stage: vk::ShaderStageFlags::COMPUTE,
            // According to https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html
            // "Another problem is querying the subgroup size from inside the kernel, which has a
            // surprising gotcha. Unless the VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT
            // flag is set at pipeline creation time, the gl_SubgroupSize variable is defined to have
            // the value from VkPhysicalDeviceSubgroupProperties, which in my experiment is always 32 on
            // Intel no matter the actual subgroup size. But setting that flag makes it give the value expected."
            flags: vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT|
            vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS_EXT,
            ..Default::default()
        };
        println!("Entrypoint done");

        let pipeline = device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder()
                .stage(entry_point)
                .layout(pipeline_layout)
                .build()
            ],
            None,
        ).unwrap()[0];
        println!("Pipelines created");

        Ok(Shader{shader_module, pipeline_layout, pipeline, set_layouts})
    }
}

struct GPU {
    pub physical: vk::PhysicalDevice,
    pub queue_families: Vec<QueueFamily>,
}

#[derive(Copy, Clone, Debug)]
struct QueueFamily {
    pub queue_count: u32,
    pub physical_index: usize,
}

pub struct Fence {
    pub fence: vk::Fence,
    pub present_queue: vk::Queue,
    pub phy_index: usize,
}

impl Fence {
    unsafe fn submit(&self, device: &ash::Device, command_buffers: &[vk::CommandBuffer]) {
        device.queue_submit(self.present_queue, &[vk::SubmitInfo::builder()
            .command_buffers(command_buffers).build()
        ], self.fence).expect("queue submit failed.");
    }
}

const STRIDE: usize = std::mem::size_of::<f32>() as usize;

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        unsafe {
            println!("dropping logical device");
            self.device.device_wait_idle().unwrap();
            for fence in &self.fences {
                self.device.destroy_fence(fence.fence, None);
            }
            self.device.destroy_device(None);
        }
    }
}

impl LogicalDevice {

    fn create_cpu_inputs(&self, queue: &[u32], inputs: &Vec<Vec<f32>>) -> Vec<Buffer> {
        inputs
            .iter()
            .map(|input| {
                let buffer = Buffer::new(
                    &self.device,
                    &self.allocator,
                    (input.len() * STRIDE) as u64,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                    queue,
                ).unwrap();

                buffer.fill(input).unwrap();

                buffer

            })
            .collect()
    }

    pub unsafe fn execute(&self, input: &Vec<Vec<Vec<f32>>>, out_length: usize, spirv: &spirv::SPIRV, fences: &[Fence]) -> &[f32] {

        let run_timer = Instant::now();
        let func = load(&self.device, spirv).unwrap();
        let descriptor_count = spirv.dslbs.len() as u32;

        let queue_family_indices = fences
            .iter()
            .map(|f| f.phy_index as u32)
            .collect::<Vec<u32>>();

        let size_in_bytes = out_length * input.len() * STRIDE;
        let cpu_buffer = Buffer::new(
            &self.device,
            &self.allocator,
            size_in_bytes.try_into().unwrap(),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            &queue_family_indices,
        ).unwrap();

        let range = input.len() / fences.len();

        println!("Commands {}ms", run_timer.elapsed().as_millis());

        let res = fences
            .par_iter()
            .enumerate()
            .flat_map(|(fence_i, fence)| {

                let command = Command::new(
                    fence.phy_index.try_into().unwrap(),
                    descriptor_count,
                    input.len().try_into().unwrap(),
                    &func.set_layouts,
                    range.try_into().unwrap(),
                    &self.device,
                ).unwrap();

                let res = (0..range)
                    .into_iter()
                    .flat_map(|index| {

                        let index_offset = range * fence_i + index;

                        let cpu_buffers = self.create_cpu_inputs(&queue_family_indices, &input[index_offset]);

                        let cpu_offset: vk::DeviceSize = (cpu_buffer.device_size / input.len() as u64) * index_offset as u64;
                        let gpu_offset: vk::DeviceSize = (cpu_buffer.device_size / input.len() as u64) * index_offset as u64;
                        let gpu_chunk_size = cpu_buffer.device_size / input.len() as u64;

                        let buffer_infos = (0..=cpu_buffers.len())
                            .into_iter()
                            .map(|f| match f {
                                0 => [vk::DescriptorBufferInfo::builder()
                                    .buffer(cpu_buffer.buffer)
                                    .offset(gpu_offset.try_into().unwrap())
                                    .range(gpu_chunk_size.try_into().unwrap())
                                    .build()],
                                _ => [vk::DescriptorBufferInfo::builder()
                                    .buffer(cpu_buffers.get(f-1).unwrap().buffer)
                                    .offset(0)
                                    .range(vk::WHOLE_SIZE)
                                    .build()],
                            })
                            .collect::<Vec<[vk::DescriptorBufferInfo; 1]>>();

                        let ds = command.descriptor_sets
                            .get(index)
                            .unwrap()
                            .to_owned();

                        let wds = buffer_infos
                            .iter()
                            .enumerate()
                            .map(|(index, buf)| {
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(ds)
                                    .dst_binding(index.try_into().unwrap())
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .buffer_info(buf)
                                    .build()
                            })
                            .collect::<Vec<vk::WriteDescriptorSet>>();

                        self.device.update_descriptor_sets(&wds, &[]);

                        self.device.begin_command_buffer(command.command_buffers[index], &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

                        cpu_buffers.iter().for_each(|cpu|
                            self.device.cmd_copy_buffer(
                                command.command_buffers[index],
                                cpu.buffer,
                                cpu.buffer,
                                &[vk::BufferCopy::builder()
                                    .src_offset(0)
                                    .dst_offset(0)
                                    .size(cpu.device_size)
                                    .build()
                                ],
                            ));

                        self.device.cmd_bind_pipeline(command.command_buffers[index], vk::PipelineBindPoint::COMPUTE, func.pipeline);
                        self.device.cmd_bind_descriptor_sets(command.command_buffers[index], vk::PipelineBindPoint::COMPUTE, func.pipeline_layout, 0, &[ds], &[]);
                        self.device.cmd_dispatch(command.command_buffers[index], 1024, 1, 1);

                        self.device.cmd_copy_buffer(
                            command.command_buffers[index],
                            cpu_buffer.buffer,
                            cpu_buffer.buffer,
                            &[vk::BufferCopy::builder()
                                .src_offset(gpu_offset)
                                .dst_offset(cpu_offset)
                                .size(gpu_chunk_size)
                                .build()
                            ],
                        );
                        self.device.end_command_buffer(command.command_buffers[index]).expect("End commandbuffer");

                        cpu_buffers

                    })
                    .collect::<Vec<_>>();

                println!("Command buffers {}ms", run_timer.elapsed().as_millis());

                fence.submit(&self.device, &command.command_buffers);
                self.device.wait_for_fences(&[fence.fence], true, std::u64::MAX).expect("Wait for fence failed.");
                self.device.reset_fences(&[fence.fence]).unwrap();

                println!("After fences {}ms", run_timer.elapsed().as_millis());

                self.device.destroy_command_pool(command.command_pool, None);
                self.device.destroy_descriptor_pool(command.descriptor_pool, None);

                res

            })
            .collect::<Vec<_>>();

        let mapping = cpu_buffer.allocation.mapped_ptr().unwrap().as_ptr();
        let result = slice::from_raw_parts::<f32>(mapping as *const f32, out_length * input.len());

        println!("Results gathered {}ms", run_timer.elapsed().as_millis());

        let mut malloc = self.allocator.lock().unwrap();
        malloc.free(cpu_buffer.allocation).unwrap();
        self.device.destroy_buffer(cpu_buffer.buffer, None);
        res.into_iter().for_each(|f| {
            malloc.free(f.allocation).unwrap();
            self.device.destroy_buffer(f.buffer, None);
        });

        func.drop(&self.device);
        println!("Resource cleanup done {}ms", run_timer.elapsed().as_millis());

        result
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::new;

    #[test]
    fn app_new() {
        let init_timer = Instant::now();
        let res = new(false);
        assert!(res.is_ok());
        let (_app, devices) = res.unwrap();
        println!("Found {} logical device(s)", devices.len());
        println!("Found {} thread(s)", devices.iter().map(|f| f.fences.len()).sum::<usize>());
        println!("App new {}ms", init_timer.elapsed().as_millis());
    }

}