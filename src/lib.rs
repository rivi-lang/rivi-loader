use ash::{vk, Entry, extensions::ext::DebugUtils};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use rayon::prelude::*;

use std::{convert::TryInto, error::Error, slice};
use std::default::Default;
use std::ffi::CString;
use std::time::Instant;

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
    pub allocator: vk_mem::Allocator,
    pub fences: Vec<Fence>,
}

pub struct Shader {
    pub shader_module: vk::ShaderModule,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl Shader {
    pub unsafe fn Drop(self, device: &ash::Device) {
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
            .map(|pdevice| {
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
            .filter_map(|f| Some(f)? )
            .collect::<Vec<GPU>>();

        match gpus.is_empty() {
            false => Ok(gpus),
            true => Err(format!("No compute capable GPUs"))?,
        }
    }

    pub unsafe fn logical_devices(&self, gpus: Vec<GPU>) -> Result<Vec<LogicalDevice>, Box<dyn Error>> {
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

                let allocator_create_info = vk_mem::AllocatorCreateInfo {
                    physical_device: gpu.physical,
                    device: device.clone(),
                    instance: self.instance.clone(),
                    preferred_large_heap_block_size: 1 * 1024 * 1024 * 1024,
                    flags: vk_mem::AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION,
                    ..Default::default()
                };
                let allocator = vk_mem::Allocator::new(&allocator_create_info).unwrap();

                let phy_properties = allocator.get_physical_device_properties().unwrap();
                println!("{:?} {}", phy_properties.device_type, String::from_utf8(phy_properties.device_name.iter().map(|&c| c as u8).filter(|&c| c > 0).collect()).unwrap());

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

                LogicalDevice{device, allocator, fences}

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
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    device_size: vk::DeviceSize,
}

impl Buffer {
    fn new(
        allocator: &vk_mem::Allocator,
        size_in_bytes: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        sharing_mode: vk::SharingMode,
        queue_family_indices: &[u32],
    ) -> Result<Buffer, vk_mem::error::Error> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            ..Default::default()
        };
        let (buffer, allocation, allocation_info) = allocator.create_buffer(
            &ash::vk::BufferCreateInfo::builder()
                .size(size_in_bytes)
                .usage(usage)
                .sharing_mode(sharing_mode)
                .queue_family_indices(queue_family_indices)
                .build(),
            &allocation_create_info,
        )?;
        Ok(Buffer {
            buffer,
            allocation,
            allocation_info,
            device_size: size_in_bytes,
        })
    }
    fn fill<T: Sized>(
        &self,
        allocator: &vk_mem::Allocator,
        data: &[T],
    ) -> Result<(), vk_mem::error::Error> {
        let data_ptr = allocator.map_memory(&self.allocation)? as *mut T;
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        allocator.unmap_memory(&self.allocation);
        Ok(())
    }
}

pub struct Command {
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

    pub fn command_pool(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandPool, Box<dyn Error>> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index);
        unsafe { Ok(device.create_command_pool(&command_pool_info, None)?) }
    }

    pub fn allocate_command_buffers(
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

pub fn load(device: &ash::Device, shader: &[u32], binding_count: u32) -> Result<Shader, Box<dyn Error>>  {
    let dslb = (0..binding_count)
        .into_iter()
        .map(|i|
            vk::DescriptorSetLayoutBinding::builder()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        )
        .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
    unsafe {
        let set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&dslb),
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
        let shader_module = device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&shader), None)?;
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

pub struct GPU {
    pub physical: vk::PhysicalDevice,
    pub queue_families: Vec<QueueFamily>,
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamily {
    pub queue_count: u32,
    pub physical_index: usize,
}

pub struct Fence {
    pub fence: vk::Fence,
    pub present_queue: vk::Queue,
    pub phy_index: usize,
}

impl Fence {
    pub unsafe fn submit(&self, device: &ash::Device, command_buffers: &[vk::CommandBuffer]) {
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
            self.allocator.destroy();
            self.device.destroy_device(None);
        }
    }
}

impl LogicalDevice {

    unsafe fn create_cpu_inputs(&self, queue: &[u32], inputs: &Vec<Vec<f32>>) -> Vec<Buffer> {
        inputs
            .iter()
            .map(|input| {

                let cpu_buffer = Buffer::new(
                    &self.allocator,
                    (input.len() * STRIDE) as u64,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk_mem::MemoryUsage::CpuToGpu,
                    vk::SharingMode::CONCURRENT,
                    queue,
                ).unwrap();

                cpu_buffer.fill(&self.allocator, input).unwrap();

                cpu_buffer
            })
            .collect()
    }

    fn create_gpu_inputs(&self, queue: &[u32], data_lens: &Vec<usize>) -> Vec<Buffer> {
        data_lens
            .iter()
            .map(|data_len|
                Buffer::new(
                    &self.allocator,
                    (data_len * STRIDE) as u64,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk_mem::MemoryUsage::GpuOnly,
                    vk::SharingMode::CONCURRENT,
                    queue,
                ).unwrap()
            )
            .collect()
    }

    pub unsafe fn execute(&self, input: &Vec<Vec<Vec<f32>>>, out_length: u64, spirv: &[u32]) -> &[f32] {

        let run_timer = Instant::now();

        let binding_count = 7;
        let func = load(&self.device, spirv, binding_count).unwrap();

        let queue_family_indices = self.fences
            .iter()
            .map(|f| f.phy_index as u32)
            .collect::<Vec<u32>>();

        let cpu_buffer = Buffer::new(
            &self.allocator,
            (out_length*input.len() as u64) * std::mem::size_of::<f32>() as u64,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::CpuToGpu,
            vk::SharingMode::CONCURRENT,
            &queue_family_indices,
        ).unwrap();

        let range = input.len() / self.fences.len();

        println!("Commands {}ms", run_timer.elapsed().as_millis());

        let res = self.fences
            .par_iter()
            .enumerate()
            .flat_map(|(fence_i, fence)| {

                let command = Command::new(
                    fence.phy_index.try_into().unwrap(),
                    binding_count,
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

                        cpu_buffers.iter().for_each(|(cpu)|
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
                                .src_offset(gpu_offset.try_into().unwrap())
                                .dst_offset(cpu_offset.try_into().unwrap())
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

        let mapping = self.allocator.map_memory(&cpu_buffer.allocation).unwrap();
        let result = slice::from_raw_parts::<f32>(mapping as *const f32, (out_length*input.len() as u64).try_into().unwrap());
        self.allocator.unmap_memory(&cpu_buffer.allocation);

        println!("Results gathered {}ms", run_timer.elapsed().as_millis());

        self.allocator.destroy_buffer(cpu_buffer.buffer, &cpu_buffer.allocation);
        res.iter().for_each(|f| self.allocator.destroy_buffer(f.buffer, &f.allocation));

        func.Drop(&self.device);
        println!("Resource cleanup done {}ms", run_timer.elapsed().as_millis());

        result
    }

    pub fn infered_execute(&self, input: Vec<Vec<f32>>, out_length: usize, func: &Shader) -> Vec<f32> { unsafe {

        let run_timer = Instant::now();

        let command = Command::new(
            self.fences.first().unwrap().phy_index.try_into().unwrap(),
            1,
            1,
            &func.set_layouts,
            1,
            &self.device,
        ).unwrap();

        let queue_family_indices = self.fences
            .iter()
            .map(|fence| {
                fence.phy_index as u32
            })
            .collect::<Vec<u32>>();

        // output destination
        let cpu_buffer = Buffer::new(
            &self.allocator,
            (input[0].len() * STRIDE) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::CpuToGpu,
            vk::SharingMode::CONCURRENT,
            &queue_family_indices,
        ).unwrap();
        // output source
        let gpu_buffer = Buffer::new(
            &self.allocator,
            (input[0].len() * STRIDE) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::GpuToCpu,
            vk::SharingMode::CONCURRENT,
            &queue_family_indices,
        ).unwrap();
        // output
        let mut buffers = vec![
            [vk::DescriptorBufferInfo::builder()
                .buffer(gpu_buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()]
        ];

        let input_lens: Vec<_> = input
            .iter()
            .map(|input| input.len())
            .collect();
        let cpu_buffers = self.create_cpu_inputs(&queue_family_indices, &input);
        let gpu_buffers = self.create_gpu_inputs(&queue_family_indices, &input_lens);
        // inputs
        buffers.extend(gpu_buffers
            .iter()
            .map(|buf|
                [vk::DescriptorBufferInfo::builder()
                    .buffer(buf.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()]
            ));

        let ds = command.descriptor_sets
            .get(0)
            .unwrap()
            .to_owned();

        let wds: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(index, buf)|
                vk::WriteDescriptorSet::builder()
                    .dst_set(ds)
                    .dst_binding(index.try_into().unwrap())
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(buf)
                    .build()
            )
            .collect();

        let cbuffer = command.command_buffers
            .get(0)
            .unwrap()
            .to_owned();

        self.device.update_descriptor_sets(&wds[..], &[]);

        self.device.begin_command_buffer(cbuffer, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

        cpu_buffers.iter().zip(gpu_buffers.iter()).for_each(|(cpu, gpu)| {
            self.device.cmd_copy_buffer(
                cbuffer,
                cpu.buffer,
                gpu.buffer,
                &[vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(cpu.device_size)
                    .build()
                ],
            )
        });

        self.device.cmd_bind_pipeline(cbuffer, vk::PipelineBindPoint::COMPUTE, func.pipeline);
        self.device.cmd_bind_descriptor_sets(cbuffer, vk::PipelineBindPoint::COMPUTE, func.pipeline_layout, 0, &[ds], &[]);
        self.device.cmd_dispatch(cbuffer, *input_lens.iter().max().unwrap() as u32, 1, 1);

        self.device.cmd_copy_buffer(
            cbuffer,
            gpu_buffer.buffer,
            cpu_buffer.buffer,
            &[vk::BufferCopy::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(gpu_buffer.device_size)
                .build()],
        );
        self.device.end_command_buffer(cbuffer).expect("End commandbuffer");

        self.fences.first().unwrap().submit(&self.device, &[cbuffer]);

        println!("Command buffers {}ms", run_timer.elapsed().as_millis());

        self.device.wait_for_fences(&[self.fences.first().unwrap().fence], true, std::u64::MAX).expect("Wait for fence failed.");

        println!("After fences {}ms", run_timer.elapsed().as_millis());

        let mapping = self.allocator.map_memory(&cpu_buffer.allocation).unwrap();
        let result = slice::from_raw_parts::<f32>(mapping as *const f32, out_length).to_vec();
        self.allocator.unmap_memory(&cpu_buffer.allocation);

        println!("Results gathered {}ms", run_timer.elapsed().as_millis());

        cpu_buffers
            .iter()
            .for_each(|f| self.allocator.destroy_buffer(f.buffer, &f.allocation));
        gpu_buffers
            .iter()
            .for_each(|f| self.allocator.destroy_buffer(f.buffer, &f.allocation));
        self.allocator.destroy_buffer(cpu_buffer.buffer, &cpu_buffer.allocation);
        self.allocator.destroy_buffer(gpu_buffer.buffer, &gpu_buffer.allocation);

        self.device.destroy_command_pool(command.command_pool, None);
        self.device.destroy_descriptor_pool(command.descriptor_pool, None);

        result
    }}
}