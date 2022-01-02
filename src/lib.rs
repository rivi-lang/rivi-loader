mod lib_test;

use std::{error::Error, fmt, sync::RwLock};

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
) -> Result<Vulkan, Box<dyn Error>> {
    let vk = Vulkan::new(debug)?;
    let version = vk.version()?;
    println!("Using Vulkan {:?}", version);
    Ok(vk)
}

pub struct Vulkan {
    _entry: ash::Entry, // Needs to outlive Instance and Devices.
    instance: ash::Instance, // Needs to outlive Devices.
    debug_layer: Option<DebugLayer>,
    compute: Option<Vec<Compute>>,
}

impl Vulkan {

    fn new(
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

        let _entry = unsafe { ash::Entry::load()? };

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
                let messenger = unsafe { loader.create_debug_utils_messenger(&info, None)? };
                Some(DebugLayer{loader, messenger})
            },
        };

        let computes = Self::logical_devices(&instance)?;
        let compute = match computes.len() {
            0 => None,
            _ => Some(computes),
        };

        Ok(Vulkan{_entry, instance, debug_layer, compute})
    }

    fn version(
        &self
    ) -> Result<(u32, u32, u32), Box<dyn Error>>  {
        match self._entry.try_enumerate_instance_version()? {
            Some(v) => Ok((vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))),
            None => Ok((vk::api_version_major(1), vk::api_version_minor(0), vk::api_version_patch(0))),
        }
    }

    fn logical_devices(
        instance: &ash::Instance,
    ) -> Result<Vec<Compute>, Box<dyn Error>> {
        let pdevices = unsafe { instance.enumerate_physical_devices()? };
        Ok(pdevices.into_iter()
            .filter(|pdevice| {
                let (_, properties) = Self::device_name(instance, *pdevice);
                properties.device_type.ne(&vk::PhysicalDeviceType::CPU)
            })
            .map(|pdevice| {

                let (device_name, _) = Self::device_name(instance, pdevice);
                println!("Found device: {}", device_name);
                let device = Self::create_device(instance, pdevice)?;
                let queue_infos = unsafe { Self::queue_infos(instance, pdevice) };
                let fences = Self::create_fences(&device, queue_infos)?;
                let allocator = Allocator::new(&AllocatorCreateDesc {
                    physical_device: pdevice,
                    device: device.clone(),
                    instance: instance.clone(),
                    debug_settings: Default::default(),
                    buffer_device_address: false,
                })?;
                let memory = unsafe { instance.get_physical_device_memory_properties(pdevice) };

                let sp = Self::subgroup_properties(instance, pdevice);
                println!("Subgroup size of {} is: {:?}", device_name, sp.subgroup_size);
                println!("Supported subgroup operations: {:?}", sp.supported_operations);
                println!("Supported subgroup stages: {:?}", sp.supported_stages);

                Ok(Compute { device, allocator: Some(RwLock::new(allocator)), fences, memory})

            })
            .collect::<Result<Vec<Compute>, Box<dyn Error>>>()?.into_iter()
            .filter(|c| !c.fences.is_empty())
            .collect::<Vec<Compute>>())
    }

    unsafe fn queue_infos(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
    ) -> Vec<(usize, Vec<f32>)> {
        instance.get_physical_device_queue_family_properties(pdevice).iter().enumerate()
            .filter(|(_, prop)| prop.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, prop)| (idx, vec![1.0f32; prop.queue_count as usize]))
            .collect()
    }

    fn device_name(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
    ) -> (String, vk::PhysicalDeviceProperties)  {
        let mut dp2 = vk::PhysicalDeviceProperties2::builder().build();
        unsafe { instance.fp_v1_1().get_physical_device_properties2(pdevice, &mut dp2) };
        let device_name = dp2.properties.device_name.iter()
            .filter_map(|f| match *f as u8 {
                0 => None,
                _ => Some(*f as u8 as char),
            })
            .collect::<String>();
        (device_name, dp2.properties)
    }

    fn subgroup_properties(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceSubgroupProperties {
        // Retrieving Subgroup operations will segfault a Mac
        // https://www.khronos.org/blog/vulkan-subgroup-tutorial
        let mut sp = vk::PhysicalDeviceSubgroupProperties::builder();
        let mut dp2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut sp).build();
        unsafe { instance.fp_v1_1().get_physical_device_properties2(pdevice, &mut dp2); }
        sp.build()
    }

    fn create_fences(
        device: &ash::Device,
        queue_infos: Vec<(usize, Vec<f32>)>,
    ) -> Result<Vec<Fence>, Box<dyn Error>> {
        Ok(queue_infos.into_iter().flat_map(|(phy_index, queue_priorities)| {
            (0..queue_priorities.len()).into_iter().map(|queue_index| {
                let vk_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
                let present_queue = unsafe { device.get_device_queue(phy_index as u32, queue_index as u32) };
                Ok(Fence{ fence: vk_fence, present_queue, phy_index: phy_index as u32 })
            })
            .collect::<Result<Vec<Fence>, Box<dyn Error>>>().into_iter()
            .flatten()
        })
        .collect())
    }

    fn create_device(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
    ) -> Result<ash::Device, vk::Result> {

        let features = vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

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

        // See: https://github.com/MaikKlein/ash/issues/539
        let priorities = unsafe { Self::queue_infos(instance, pdevice) }.into_iter().map(|f| f.1).collect::<Vec<_>>();
        let queue_create_infos = unsafe { Self::queue_infos(instance, pdevice) }.into_iter().enumerate().map(|(idx, (phy_index, _))| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(phy_index as u32)
                .queue_priorities(&priorities[idx])
                .build()
        })
        .collect::<Vec<vk::DeviceQueueCreateInfo>>();

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&ext_names)
            .enabled_features(&features)
            .push_next(&mut variable_pointers);

        unsafe { instance.create_device(pdevice, &device_info, None) }
    }

    pub fn load_shader<R: std::io::Read + std::io::Seek>(
        &self,
        x: &mut R
    ) -> Result<Shader<'_>, Box<dyn Error>> {
        let binary = ash::util::read_spv(x)?;
        let bindings = Shader::module(&binary).map(|module| Shader::descriptor_set_layout_bindings(Shader::binding_count(&module)))?;
        match &self.compute {
            Some(c) => {
                let shaders = c.iter()
                    .map(|f| Shader::create(&f.device, &bindings, &binary))
                    .collect::<Result<Vec<Shader>, Box<dyn Error>>>()?;
                match shaders.into_iter().next() {
                    Some(s) => Ok(s),
                    None => Err("No compute capable devices".to_string().into()),
                }
            }
            None => Err("No compute capable devices".to_string().into()),
        }
    }

    pub fn compute<T: std::marker::Sync>(
        &self,
        input: &[Vec<Vec<T>>],
        output: &mut [T],
        shader: &Shader,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        match &self.compute {
            Some(c) => c.first().unwrap().execute(input, output, shader),
            None => Err("No compute capable devices".to_string().into()),
        }
    }

    pub fn device_count(
        &self
    ) -> usize {
        match &self.compute {
            Some(c) => c.len(),
            None => 0,
        }
    }

    pub fn threads(
        &self
    ) -> usize {
        match &self.compute {
            Some(c) => c.iter().map(|d| d.fences.len()).sum::<usize>(),
            None => 0,
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        self.compute = None;
        self.debug_layer = None;
        unsafe { self.instance.destroy_instance(None) }
    }
}

pub enum DebugOption {
    None,
    Validation,
    Debug,
}

struct DebugLayer {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
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
    fn drop(&mut self) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.messenger, None) }
    }
}

pub struct Shader<'a> {
    module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    binding_count: u32,

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
    ) -> usize {
        module.annotations.iter()
            .flat_map(|f| f.operands.clone())
            .filter(|op| op.eq(&rspirv::dr::Operand::Decoration(rspirv::spirv::Decoration::Binding)))
            .count()
    }

    fn descriptor_set_layout_bindings(
        binding_count: usize
    ) -> Vec<vk::DescriptorSetLayoutBinding> {
        (0..binding_count).into_iter().map(|i|
            vk::DescriptorSetLayoutBinding::builder()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        )
        .collect()
    }

    fn create(
        device: &'a ash::Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
        binary: &[u32],
    ) -> Result<Shader<'a>, Box<dyn Error>> {
        let set_layouts = unsafe { device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings),
            None,
        ) }.map(|set_layout| vec![set_layout])?;

        let pipeline_layout = unsafe { device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts),
            None,
        )? };

        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(binary), None)? };
        let stage = vk::PipelineShaderStageCreateInfo {
            p_name: SHADER_ENTRYPOINT,
            module,
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

        let pipeline = unsafe { device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder().stage(stage).layout(pipeline_layout).build()],
            None,
        ) }.map(|pipelines| pipelines[0]).map_err(|(_, err)| err)?;

        Ok(Shader{module, pipeline_layout, pipeline, set_layouts, device, binding_count: bindings.len() as u32})
    }
}

impl <'a> Drop for Shader<'a> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_pipeline_layout(self.pipeline_layout, None) };
        unsafe { self.device.destroy_shader_module(self.module, None) };
        for set_layout in self.set_layouts.iter().copied() {
            unsafe { self.device.destroy_descriptor_set_layout(set_layout, None) };
        }
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}

struct Fence {
    fence: vk::Fence,
    present_queue: vk::Queue,
    phy_index: u32,
}

struct Compute {
    device: ash::Device,
    allocator: Option<RwLock<Allocator>>,
    fences: Vec<Fence>,
    memory: vk::PhysicalDeviceMemoryProperties,
}

impl fmt::Debug for Compute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        println!("Memory types: {}", self.memory.memory_type_count);
        self.memory.memory_types.iter()
            .filter(|mt| !mt.property_flags.is_empty())
            .enumerate()
            .for_each(|(idx, mt)| {
                println!("Index {} {:?} (heap {})", idx, mt.property_flags, mt.heap_index);
            });

        println!("Memory heaps: {}", self.memory.memory_heap_count);
        self.memory.memory_heaps.iter()
            .filter(|mh| mh.size.ne(&0))
            .enumerate()
            .for_each(|(idx, mh)| {
                println!("{:?} GiB {:?} (heap {})", mh.size / 1_073_741_824, mh.flags, idx);
            });

        f.write_fmt(format_args!("  Found {} compute core(s) with {} total of thread(s)", self.cores().len(), self.fences.len()))
    }
}

impl Compute {

    fn cores(
        &self
    ) -> Vec<u32> {
        self.fences.iter().fold(vec![], |mut acc, f| {
            if !acc.contains(&f.phy_index) {
                acc.push(f.phy_index);
            }
            acc
        })
    }

    // task must return a result vector to avoid
    // rust ownership system to delete it before
    // it is used by vulkan
    fn task<T>(
        &self,
        descriptor_set: &vk::DescriptorSet,
        command_buffer: &vk::CommandBuffer,
        shader: &Shader<'_>,
        output: vk::Buffer,
        input: &[Vec<T>],
        memory_mapping: (vk::DeviceSize, vk::DeviceSize),
    ) -> Result<Vec<Buffer>, Box<dyn Error + Send + Sync>> {

        let (cpu_offset, cpu_chunk_size) = memory_mapping;
        let input_buffers = input.iter().map(|data| {
            Buffer::new(
                "cpu input",
                &self.device,
                &self.allocator,
                (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                &self.cores(),
            )?.fill(data)
        })
        .collect::<Result<Vec<Buffer>, Box<dyn Error + Send + Sync>>>()?;

        let buffer_infos = (0..=input_buffers.len()).into_iter()
            .map(|f| match f {
                0 => [vk::DescriptorBufferInfo::builder()
                    .buffer(output)
                    .offset(cpu_offset)
                    .range(cpu_chunk_size)
                    .build()],
                _ => [vk::DescriptorBufferInfo::builder()
                    .buffer(input_buffers[f-1].buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()],
            })
            .collect::<Vec<[vk::DescriptorBufferInfo; 1]>>();

        let wds = buffer_infos.iter().enumerate()
            .map(|(i, buf)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(buf)
                    .build()
            })
            .collect::<Vec<vk::WriteDescriptorSet>>();

        unsafe {
            self.device.update_descriptor_sets(&wds, &[]);
            self.device.begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            self.device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, shader.pipeline);
            self.device.cmd_bind_descriptor_sets(*command_buffer, vk::PipelineBindPoint::COMPUTE, shader.pipeline_layout, 0, &[*descriptor_set], &[]);
            self.device.cmd_dispatch(*command_buffer, 1024, 1, 1);
            self.device.end_command_buffer(*command_buffer)?;
        }

        Ok(input_buffers)
    }

    fn execute<T: std::marker::Sync>(
        &self,
        input: &[Vec<Vec<T>>],
        output: &mut [T],
        shader: &Shader,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {

        let threads = &self.fences[0..input.len()];

        let output_buffer = Buffer::new(
            "output buffer",
            &self.device,
            &self.allocator,
            (output.len() * std::mem::size_of::<T>()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::GpuToCpu,
            &self.cores(),
        )?;

        threads.par_iter().enumerate().try_for_each(|(fence_idx, fence)| -> Result<(), Box<dyn Error + Send + Sync>> {

            let command = Command::new(
                fence.phy_index,
                shader.binding_count,
                &shader.set_layouts,
                (input.len() / threads.len()) as u32,
                &self.device,
            )?;

            // even though buffers are not used, it must outlive queue submits and fence waiting.
            // otherwise, the memory "disappears" and will cause problems on discrete cards.
            let _buffers = command.command_buffers.iter().enumerate().map(|(index, command_buffer)| {
                let index_offset = command.command_buffers.len() * fence_idx + index;
                let cpu_offset = (output_buffer.device_size / input.len() as vk::DeviceSize) * index_offset as vk::DeviceSize;
                let cpu_chunk_size = output_buffer.device_size / input.len() as vk::DeviceSize;
                let memory_mappings = (cpu_offset, cpu_chunk_size);
                self.task(&command.descriptor_sets[index], command_buffer, shader, output_buffer.buffer, &input[index_offset], memory_mappings)
            })
            .collect::<Result<Vec<Vec<Buffer>>, Box<dyn Error + Send + Sync>>>()?;

            let submits = [vk::SubmitInfo::builder().command_buffers(&command.command_buffers).build()];
            unsafe {
                self.device.queue_submit(fence.present_queue, &submits, fence.fence)?;
                self.device.wait_for_fences(&[fence.fence], true, u64::MAX)?;
                self.device.reset_fences(&[fence.fence])?;
            }

            Ok(())

        })?;

        let data_ptr = match output_buffer.allocation.mapped_ptr() {
            Some(c_ptr) => c_ptr.as_ptr() as *mut T,
            None => return Err("could not map output buffer".to_string().into()),
        };
        unsafe { data_ptr.copy_to_nonoverlapping(output.as_mut_ptr(), output.len()) };
        Ok(())
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() }
        for fence in &self.fences {
            unsafe { self.device.destroy_fence(fence.fence, None) }
        }
        self.allocator = None;
        unsafe { self.device.destroy_device(None) }
    }
}

struct Command<'a> {
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    device: &'a ash::Device,
}

impl <'a> Command<'_> {

    fn descriptor_pool(
        device: &ash::Device,
        descriptor_count: u32,
        max_sets: u32,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        let descriptor_pool_size = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(descriptor_count)
            .build()];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&descriptor_pool_size);
        unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }
    }

    fn command_pool(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandPool, vk::Result> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index);
        unsafe { device.create_command_pool(&command_pool_info, None) }
    }

    fn allocate_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        command_buffer_count: u32,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let command_buffers_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(command_buffer_count)
            .command_pool(command_pool);
        unsafe { device.allocate_command_buffers(&command_buffers_info) }
    }

    fn new(
        queue_family_index: u32,
        descriptor_count: u32,
        set_layouts: &[vk::DescriptorSetLayout],
        command_buffer_count: u32,
        device: &'a ash::Device,
    ) -> Result<Command<'a>, Box<dyn Error + Send + Sync>> {

        let descriptor_pool = Command::descriptor_pool(device, descriptor_count, command_buffer_count)?;

        let descriptor_set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(set_layouts);

        let descriptor_sets = (0..command_buffer_count).into_iter().flat_map(|_| {
            unsafe { device.allocate_descriptor_sets(&descriptor_set_info) }.map(|sets| sets[0])
        }).collect();

        let command_pool = Command::command_pool(device, queue_family_index)?;
        let command_buffers = Command::allocate_command_buffers(device, command_pool, command_buffer_count)?;

        Ok(Command { descriptor_pool, command_pool, command_buffers, descriptor_sets, device })
    }
}

impl <'a> Drop for Command<'a> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_command_pool(self.command_pool, None) };
        unsafe { self.device.destroy_descriptor_pool(self.descriptor_pool, None) };
    }
}

struct Buffer<'a, 'b>  {
    buffer: vk::Buffer,
    allocation: Allocation,
    device_size: vk::DeviceSize,

    device: &'a ash::Device,
    allocator: &'b Option<RwLock<Allocator>>,
}

impl <'a, 'b> Buffer<'_, '_> {

    fn new(
        name: &str,
        device: &'a ash::Device,
        allocator: &'b Option<RwLock<Allocator>>,
        device_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: gpu_allocator::MemoryLocation,
        queue_family_indices: &[u32],
    ) -> Result<Buffer<'a, 'b>, Box<dyn Error + Send + Sync>> {
        let create_info = vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(usage)
            .sharing_mode(match queue_family_indices.len() {
                1 => vk::SharingMode::EXCLUSIVE,
                _ => vk::SharingMode::CONCURRENT,
            })
            .queue_family_indices(queue_family_indices);
        let buffer = unsafe { device.create_buffer(&create_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut malloc = allocator.as_ref().unwrap().write().unwrap();
        let allocation = malloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
        })?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };
        Ok(Buffer { buffer, allocation, device_size, device, allocator })
    }

    fn fill<T: Sized>(
        self,
        data: &[T],
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let data_ptr = match self.allocation.mapped_ptr() {
            Some(c_ptr) => c_ptr.as_ptr() as *mut T,
            None => return Err("could not fill buffer".to_string().into()),
        };
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(self)
    }
}

impl <'a, 'b> Drop for Buffer<'a, 'b> {
    fn drop(&mut self) {
        let lock = self.allocator.as_ref().unwrap();
        let mut malloc = lock.write().unwrap();
        malloc.free(self.allocation.to_owned()).unwrap();
        unsafe { self.device.destroy_buffer(self.buffer, None) };
    }
}