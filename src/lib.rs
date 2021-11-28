mod lib_test;

use std::error::Error;
use std::fmt;
use std::sync::RwLock;

use ash::vk;
use gpu_allocator::vulkan::*;
use rayon::prelude::*;

const LAYER_VALIDATION: *const std::os::raw::c_char = concat!("VK_LAYER_KHRONOS_validation", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const LAYER_DEBUG: *const std::os::raw::c_char = concat!("VK_LAYER_LUNARG_api_dump", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;

const EXT_VARIABLE_POINTERS: *const std::os::raw::c_char = concat!("VK_KHR_variable_pointers", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const EXT_GET_MEMORY_REQUIREMENTS2: *const std::os::raw::c_char = concat!("VK_KHR_get_memory_requirements2", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const EXT_DEDICATED_ALLOCATION: *const std::os::raw::c_char = concat!("VK_KHR_dedicated_allocation", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const EXT_PORTABILITY_SUBSET: *const std::os::raw::c_char = concat!("VK_KHR_portability_subset", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;

const SHADER_ENTRYPOINT: *const std::os::raw::c_char = concat!("main", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;

pub fn new(
    debug: DebugOption
) -> Result<(Vulkan, Vec<Compute>), Box<dyn Error>> {

    let vk = Vulkan::new(debug)?;

    println!("Found Vulkan version: {:?}", vk.version()?);

    let pdevices = unsafe { vk.instance.enumerate_physical_devices()? };
    let logical_devices = pdevices
        .into_iter()
        .filter_map(|pdevice| {

            println!("Found device: {}", vk.device_name(pdevice));

            let sp = vk.subgroup_properties(pdevice);
            println!("Physical device has subgroup size of: {:?}", sp.subgroup_size);
            println!("Supported subgroup operations: {:?}", sp.supported_operations);
            println!("Supported subgroup stages: {:?}", sp.supported_stages);

            let queue_infos = unsafe { vk.queue_infos(pdevice) };

            match vk.compute(pdevice, &queue_infos) {
                Ok(c) => Some(c),
                Err(_) => None,
            }
        })
        .filter(|c| !c.fences.is_empty())
        .collect::<Vec<Compute>>();

    match logical_devices.is_empty() {
        true => Err("No compute capable devices".to_string().into()),
        false => Ok((vk, logical_devices)),
    }
}

pub struct Vulkan {
    // Entry: Loads the Vulkan library.
    // Needs to outlive Instance and Device.
    _entry: ash::Entry,
    // Instance: Loads instance level functions.
    // Needs to outlive the Devices it has created.
    instance: ash::Instance,
    debug_layer: Option<DebugLayer>,
}

impl Vulkan {

    pub fn new(
        debug: DebugOption
    ) -> Result<Vulkan, Box<dyn Error>> {

        let vk_layers = match debug {
            DebugOption::None => vec![],
            DebugOption::Validation => vec![LAYER_VALIDATION],
            DebugOption::Debug => vec![LAYER_VALIDATION, LAYER_DEBUG],
        };

        let mut info = match debug {
            DebugOption::None => vk::DebugUtilsMessengerCreateInfoEXT {
                ..Default::default()
            },
            _ => {
                vk::DebugUtilsMessengerCreateInfoEXT {
                    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    pfn_user_callback: Some(DebugLayer::callback),
                    ..Default::default()
                }
            }
        };

        let _entry = unsafe { ash::Entry::new()? };

        let instance = unsafe {
            _entry.create_instance(&vk::InstanceCreateInfo::builder()
                .push_next(&mut info)
                .application_info(&vk::ApplicationInfo {
                    api_version: vk::make_api_version(0, 1, 2, 0),
                    engine_version: 0,
                    ..Default::default()
                })
                .enabled_layer_names(&vk_layers)
                .enabled_extension_names(&[ash::extensions::ext::DebugUtils::name().as_ptr()])
            , None)?
        };

        let debug_layer = match debug {
            DebugOption::None => None,
            _ => {
                let loader = ash::extensions::ext::DebugUtils::new(&_entry, &instance);
                let callback = unsafe { loader.create_debug_utils_messenger(&info, None)? };
                Some(DebugLayer{loader, callback})
            },
        };

        Ok(Vulkan{_entry, instance, debug_layer})
    }

    fn version(
        &self
    ) -> Result<(u32, u32, u32), Box<dyn Error>>  {
        match self._entry.try_enumerate_instance_version()? {
            Some(v) => Ok((vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))),
            None => Ok((vk::api_version_major(1), vk::api_version_minor(0), vk::api_version_patch(0))),
        }
    }

    unsafe fn queue_infos(
        &self,
        pdevice: vk::PhysicalDevice,
    ) -> Vec<vk::DeviceQueueCreateInfo> {
        self.instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .filter(|(idx, prop)| {
                println!("Queue family at index {} has {} thread(s) and capabilities: {:?}", idx, prop.queue_count, prop.queue_flags);
                prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .map(|(idx, prop)| {
                let queues = vec![1.0f32; prop.queue_count as usize];
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(idx as u32)
                    .queue_priorities(&queues)
                    .build()
            })
            .collect()
    }

    fn device_name(
        &self,
        pdevice: vk::PhysicalDevice,
    ) -> String {
        let mut dp2 = vk::PhysicalDeviceProperties2::builder().build();
        unsafe { self.instance.fp_v1_1().get_physical_device_properties2(pdevice, &mut dp2) };
        let device_name = dp2.properties.device_name
            .iter()
            .filter_map(|f| match *f as u8 {
                0 => None,
                _ => Some(*f as u8 as char),
            })
            .collect::<String>();
        format!("{} ({:?})", device_name, dp2.properties.device_type)
    }

    fn subgroup_properties(
        &self,
        pdevice: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceSubgroupProperties {
        // Retrieving Subgroup operations will segfault a Mac
        // https://www.khronos.org/blog/vulkan-subgroup-tutorial
        let mut sp = vk::PhysicalDeviceSubgroupProperties::builder();
        let mut dp2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut sp).build();
        unsafe { self.instance.fp_v1_1().get_physical_device_properties2(pdevice, &mut dp2); }
        sp.build()
    }

    fn compute(
        &self,
        pdevice: vk::PhysicalDevice,
        queue_infos: &[vk::DeviceQueueCreateInfo],
    ) -> Result<Compute, Box<dyn Error>> {

        let features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let memory = unsafe { self.instance.get_physical_device_memory_properties(pdevice) };

        let mut variable_pointers = vk::PhysicalDeviceVariablePointersFeatures::builder()
            .variable_pointers(true)
            .variable_pointers_storage_buffer(true)
            .build();

        let mut ext_names: Vec<_> = vec![
            EXT_VARIABLE_POINTERS,
            EXT_GET_MEMORY_REQUIREMENTS2,
            EXT_DEDICATED_ALLOCATION,
        ];

        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            ext_names.push(EXT_PORTABILITY_SUBSET)
        }

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_infos)
            .enabled_extension_names(&ext_names)
            .enabled_features(&features)
            .push_next(&mut variable_pointers);

        let device = unsafe { self.instance.create_device(pdevice, &device_info, None)? };

        let allocator_create_info = AllocatorCreateDesc {
            physical_device: pdevice,
            device: device.clone(),
            instance: self.instance.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let allocator = Allocator::new(&allocator_create_info)?;

        let fences = queue_infos
            .iter()
            .flat_map(|queue_info| {
                (0..queue_info.queue_count)
                    .into_iter()
                    .filter_map(|queue_index| {
                        match Fence::new(&device, queue_info.queue_family_index, queue_index) {
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

impl Drop for Vulkan {
    fn drop(
        &mut self
    ) {
        self.debug_layer = None;
        unsafe { self.instance.destroy_instance(None) }
    }
}

struct DebugLayer {
    loader: ash::extensions::ext::DebugUtils,
    callback: vk::DebugUtilsMessengerEXT,
}

pub enum DebugOption {
    None,
    Validation,
    Debug,
}

impl DebugLayer {

    extern "system" fn callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let message = unsafe { std::ffi::CStr::from_ptr((*p_callback_data).p_message) };
        let severity = format!("{:?}", message_severity).to_lowercase();
        let ty = format!("{:?}", message_type).to_lowercase();
        println!("[Debug][{}][{}] {:?}", severity, ty, message);
        vk::FALSE
    }

}

impl Drop for DebugLayer {
    fn drop(
        &mut self
    ) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.callback, None) }
    }
}

pub struct Shader<'a> {
    pub shader_module: vk::ShaderModule,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub binding_count: usize,

    device: &'a ash::Device,
}

impl <'a> Shader<'_> {

    fn module(
        binary: &[u32]
    ) -> Result<rspirv::dr::Module, Box<dyn Error>> {
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(binary, &mut loader)?;
        Ok(loader.module())
    }

    fn binding_count(
        module: &rspirv::dr::Module
    ) -> Result<usize, Box<dyn Error>> {
        let binding_count = module
            .annotations
            .iter()
            .flat_map(|f| f.operands.clone())
            .filter(|op| op.eq(&rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding)))
            .count();
        Ok(binding_count)
    }

    fn descriptor_set_layout_bindings(
        binding_count: usize
    ) -> Vec<vk::DescriptorSetLayoutBinding> {
        (0..binding_count)
            .into_iter()
            .map(|i|
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            )
            .collect::<Vec<vk::DescriptorSetLayoutBinding>>()
    }

    pub fn create(
        device: &'a ash::Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
        binary: &[u32],
    ) -> Result<Shader<'a>, Box<dyn Error>> {
        let set_layout = unsafe { device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings),
            None,
        )? };
        let set_layouts = vec![set_layout];

        let pipeline_layout = unsafe { device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts),
            None,
        )? };

        let shader_module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(binary), None)? };
        let entry_point = vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRYPOINT,
            module: shader_module,
            stage: vk::ShaderStageFlags::COMPUTE,
            // According to https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html
            // "Another problem is querying the subgroup size from inside the kernel, which has a
            // surprising gotcha. Unless the VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT
            // flag is set at pipeline creation time, the gl_SubgroupSize variable is defined to have
            // the value from VkPhysicalDeviceSubgroupProperties, which in my experiment is always 32 on
            // Intel no matter the actual subgroup size. But setting that flag makes it give the value expected."
            flags: vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT|vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS_EXT,
            ..Default::default()
        };

        let pipelines = unsafe { device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder()
                .stage(entry_point)
                .layout(pipeline_layout)
                .build()
            ],
            None,
        ) };

        let pipeline = match pipelines {
            Ok(f) => f[0],
            Err((_, err)) => return Err(Box::new(err)),
        };

        Ok(Shader{shader_module, pipeline_layout, pipeline, set_layouts, device, binding_count: bindings.len()})
    }

    pub fn new<R: std::io::Read + std::io::Seek>(
        compute: &'a Compute,
        x: &mut R
    ) -> Result<Shader<'a>, Box<dyn Error>> {
        let binary = ash::util::read_spv(x)?;
        let module = Shader::module(&binary)?;
        let binding_count = Shader::binding_count(&module)?;
        let bindings = Shader::descriptor_set_layout_bindings(binding_count);
        let shader = Shader::create(&compute.device, &bindings, &binary)?;
        Ok(shader)
    }
}

impl <'a> Drop for Shader<'a> {
    fn drop(
        &mut self
    ) {
        unsafe {
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_shader_module(self.shader_module, None);
            for set_layout in self.set_layouts.to_owned() {
                self.device.destroy_descriptor_set_layout(set_layout, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

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
        let present_queue = unsafe { device.get_device_queue(queue_family_index, queue_index) };
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

pub struct Compute {
    pub device: ash::Device,
    pub allocator: Option<RwLock<Allocator>>,
    pub fences: Vec<Fence>,

    memory: vk::PhysicalDeviceMemoryProperties,
}

impl fmt::Debug for Compute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        println!("Memory types: {}", self.memory.memory_type_count);
        self.memory
            .memory_types
            .iter()
            .filter(|mt| !mt.property_flags.is_empty())
            .enumerate()
            .for_each(|(idx, mt)| {
                println!("Index {} {:?} (heap {})", idx, mt.property_flags, mt.heap_index);
            });

        println!("Memory heaps: {}", self.memory.memory_heap_count);
        self.memory
            .memory_heaps
            .iter()
            .filter(|mh| mh.size.ne(&0))
            .enumerate()
            .for_each(|(idx, mh)| {
                println!("{:?} GiB {:?} (heap {})", mh.size / 1_073_741_824, mh.flags, idx);
            });

        let qfs = self.fences.iter().map(|f| f.phy_index);

        let mut uniqs: Vec<usize> = Vec::new();
        qfs.into_iter().for_each(|f|
            if !uniqs.contains(&f) {
                uniqs.push(f);
            }
        );

        f.write_fmt(format_args!("  Found {} compute core(s) with {} total of thread(s)", uniqs.len(), self.fences.len()))
    }
}

impl Compute {

    fn create_cpu_inputs<T>(
        &self,
        queue: &[u32],
        inputs: &[Vec<T>]
    ) -> Vec<Buffer> {
        inputs.iter().map(|input| {
            let buffer = Buffer::new(
                &self.device,
                &self.allocator,
                (input.len() * std::mem::size_of::<T>()) as u64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                queue,
            ).unwrap();

            buffer.fill(input).unwrap();

            buffer

        })
        .collect()
    }

    fn task<T>(
        &self,
        command: &Command,
        func: &Shader<'_>,
        queue_family_indices: &[u32],
        cpu_buffer: &Buffer<'_, '_>,
        input: &[Vec<Vec<T>>],
        memory_mappings: Vec<(usize, vk::DeviceSize, vk::DeviceSize)>,
    ) -> Vec<Vec<Buffer>> {

        command.command_buffers.iter().enumerate().map(|(index, command_buffer)| {

            let ds = command.descriptor_sets[index];
            let (index_offset, cpu_offset, cpu_chunk_size) = memory_mappings[index];
            let cpu_buffers = self.create_cpu_inputs(queue_family_indices, &input[index_offset]);

            let buffer_infos = (0..=cpu_buffers.len()).into_iter()
                .map(|f| match f {
                    0 => [vk::DescriptorBufferInfo::builder()
                        .buffer(cpu_buffer.buffer)
                        .offset(cpu_offset)
                        .range(cpu_chunk_size)
                        .build()],
                    _ => [vk::DescriptorBufferInfo::builder()
                        .buffer(cpu_buffers[f-1].buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()],
                })
                .collect::<Vec<[vk::DescriptorBufferInfo; 1]>>();

            let wds = buffer_infos.iter().enumerate()
                .map(|(index, buf)|
                    vk::WriteDescriptorSet::builder()
                        .dst_set(ds)
                        .dst_binding(index as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(buf)
                        .build()
                )
                .collect::<Vec<vk::WriteDescriptorSet>>();

            unsafe {
                self.device.update_descriptor_sets(&wds, &[]);
                self.device.begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            };

            cpu_buffers.iter().for_each(|cpu|
                unsafe {
                    self.device.cmd_copy_buffer(
                        *command_buffer,
                        cpu.buffer,
                        cpu.buffer,
                        &[vk::BufferCopy::builder()
                            .src_offset(0)
                            .dst_offset(0)
                            .size(cpu.device_size)
                            .build()
                        ],
                    )
                }
            );

            unsafe {
                self.device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, func.pipeline);
                self.device.cmd_bind_descriptor_sets(*command_buffer, vk::PipelineBindPoint::COMPUTE, func.pipeline_layout, 0, &[ds], &[]);
                self.device.cmd_dispatch(*command_buffer, 1024, 1, 1);

                self.device.cmd_copy_buffer(
                    *command_buffer,
                    cpu_buffer.buffer,
                    cpu_buffer.buffer,
                    &[vk::BufferCopy::builder()
                        .src_offset(cpu_offset)
                        .dst_offset(cpu_offset)
                        .size(cpu_chunk_size)
                        .build()
                    ],
                );
                self.device.end_command_buffer(*command_buffer).unwrap();
            }

            cpu_buffers

        })
        .collect::<Vec<Vec<Buffer>>>()
    }

    pub fn execute<T: std::marker::Sync>(
        &self,
        input: &[Vec<Vec<T>>],
        out_length: usize,
        shader: &Shader,
        fences: &[Fence]
    ) -> &[T] {

        let queue_family_indices = fences
            .iter()
            .map(|f| f.phy_index as u32)
            .collect::<Vec<u32>>();

        let size_in_bytes = out_length * input.len() * std::mem::size_of::<T>();
        let cpu_buffer = Buffer::new(
            &self.device,
            &self.allocator,
            size_in_bytes as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
            &queue_family_indices,
        ).unwrap();

        fences.par_iter().enumerate().for_each(|(fence_idx, fence)| {

            let command = Command::new(
                fence.phy_index as u32,
                shader.binding_count as u32,
                input.len() as u32,
                &shader.set_layouts,
                (input.len() / fences.len()) as u32,
                &self.device,
            ).unwrap();

            let max_sets = input.len() as vk::DeviceSize;

            let memory_mappings = command.command_buffers
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    let index_offset = command.command_buffers.len() * fence_idx + index;
                    let cpu_offset: vk::DeviceSize = (cpu_buffer.device_size / max_sets) * index_offset as vk::DeviceSize;
                    let cpu_chunk_size: vk::DeviceSize = cpu_buffer.device_size / max_sets;
                    (index_offset, cpu_offset, cpu_chunk_size)
                })
                .collect::<Vec<_>>();

            let _buffers = self.task(
                &command,
                shader,
                &queue_family_indices,
                &cpu_buffer,
                input,
                memory_mappings,
            );

            fence.submit(&self.device, &command.command_buffers).unwrap();
            unsafe {
                self.device.wait_for_fences(&[fence.fence], true, std::u64::MAX).unwrap();
                self.device.reset_fences(&[fence.fence]).unwrap();
            }

        });

        let mapping = cpu_buffer.allocation.mapped_ptr().unwrap().as_ptr();
        unsafe { std::slice::from_raw_parts::<T>(mapping as *const T, out_length * input.len()) }
    }
}

impl Drop for Compute {
    fn drop(
        &mut self
    ) {
        unsafe { self.device.device_wait_idle().unwrap() }
        for fence in &self.fences {
            unsafe { self.device.destroy_fence(fence.fence, None) }
        }
        self.allocator = None;
        unsafe { self.device.destroy_device(None) }
    }
}

pub struct Command<'a> {
    pub descriptor_pool: vk::DescriptorPool,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,

    device: &'a ash::Device,
}

impl <'a> Command<'_> {

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

    pub fn new(
        queue_family_index: u32,
        descriptor_count: u32,
        max_sets: u32,
        set_layouts: &[vk::DescriptorSetLayout],
        command_buffer_count: u32,
        device: &'a ash::Device,
    ) -> Result<Command<'a>, Box<dyn Error>> {

        let descriptor_pool = Command::descriptor_pool(device, descriptor_count, max_sets)?;

        let descriptor_set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(set_layouts);

        let mut descriptor_sets = vec![];
        for _ in 0..command_buffer_count {
            let sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_info) }?;
            descriptor_sets.push(sets[0]);
        }

        let command_pool = Command::command_pool(device, queue_family_index)?;
        let command_buffers = Command::allocate_command_buffers(device, command_pool, command_buffer_count)?;

        Ok(Command {
            descriptor_pool,
            command_pool,
            command_buffers,
            descriptor_sets,
            device,
        })
    }
}

impl <'a> Drop for Command<'a> {
    fn drop(
        &mut self
    ) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

pub struct Buffer<'a, 'b>  {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub device_size: vk::DeviceSize,

    device: &'a ash::Device,
    allocator: &'b Option<RwLock<Allocator>>,
}

impl <'a, 'b> Buffer<'_, '_> {

    pub fn new(
        device: &'a ash::Device,
        allocator: &'b Option<RwLock<Allocator>>,
        device_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: gpu_allocator::MemoryLocation,
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
        let lock = self.allocator.as_ref().unwrap();
        let mut malloc = lock.write().unwrap();
        malloc.free(self.allocation.to_owned()).unwrap();
        unsafe { self.device.destroy_buffer(self.buffer, None) };
    }
}