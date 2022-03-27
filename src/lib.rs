mod lib_test;

use std::{error::Error, fmt, sync::RwLock};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use num_traits::Bounded;
use rspirv::binary::Assemble;

const LAYER_VALIDATION: *const std::os::raw::c_char = concat!("VK_LAYER_KHRONOS_validation", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const LAYER_DEBUG: *const std::os::raw::c_char = concat!("VK_LAYER_LUNARG_api_dump", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;

const EXT_VARIABLE_POINTERS: *const std::os::raw::c_char = concat!("VK_KHR_variable_pointers", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const EXT_GET_MEMORY_REQUIREMENTS2: *const std::os::raw::c_char = concat!("VK_KHR_get_memory_requirements2", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const EXT_DEDICATED_ALLOCATION: *const std::os::raw::c_char = concat!("VK_KHR_dedicated_allocation", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;
const EXT_PORTABILITY_SUBSET: *const std::os::raw::c_char = concat!("VK_KHR_portability_subset", "\0") as *const str as *const [std::os::raw::c_char] as *const std::os::raw::c_char;

const COMPUTE_BIT: ash::vk::QueueFlags = vk::QueueFlags::COMPUTE;
const TRANSFER_BIT: ash::vk::QueueFlags = vk::QueueFlags::TRANSFER;

pub fn new(
    debug: DebugOption
) -> Result<Vulkan, Box<dyn Error>> {
    let vk = Vulkan::new(debug)?;
    Ok(vk)
}

pub struct Specialization {
    constant_id: u32,
    offset: u32,
    size: usize,
    data: u8,
}

// TODO: pass validation layers
pub fn load_shader(
    gpu: &Compute,
    module: rspirv::dr::Module,
    specializations: Vec<Specialization>,
) -> Result<Shader<'_>, Box<dyn Error>> {
    let bindings = Shader::descriptor_set_layout_bindings(Shader::binding_count(&module));
    let specialization_infos = specializations.iter().map(|specialization| {
        (specialization.data, vk::SpecializationMapEntry::builder()
            .constant_id(specialization.constant_id)
            .offset(specialization.offset)
            .size(specialization.size))
        })
        .collect::<Vec<_>>();
    let mut data = Vec::new();
    let mut map_entries = Vec::new();
    for specialization_info in specialization_infos {
        data.push(specialization_info.0);
        map_entries.push(specialization_info.1.build());
    }
    Shader::create(&gpu.device, &bindings, &module, vk::SpecializationInfo::builder()
        .data(&data)
        .map_entries(&map_entries))
}

pub struct Vulkan {
    entry: ash::Entry, // Needs to outlive Instance and Devices.
    instance: ash::Instance, // Needs to outlive Devices.
    debug_layer: Option<DebugLayer>,
    compute: Option<Vec<Compute>>,
}

impl Vulkan {

    fn new(
        debug: DebugOption
    ) -> Result<Self, Box<dyn Error>> {

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

        let entry = unsafe { ash::Entry::load()? };

        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::builder()
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
                let loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
                let messenger = unsafe { loader.create_debug_utils_messenger(&info, None)? };
                Some(DebugLayer{loader, messenger})
            },
        };

        let computes = Self::logical_devices(&instance)?;
        let compute = match computes.is_empty() {
            true => None,
            false => Some(computes),
        };

        Ok(Self{entry, instance, debug_layer, compute})
    }

    pub fn version(
        &self
    ) -> Result<(u32, u32, u32), Box<dyn Error>>  {
        match self.entry.try_enumerate_instance_version()? {
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
                let sp = Self::subgroup_properties(instance, *pdevice);
                properties.device_type.ne(&vk::PhysicalDeviceType::CPU)
                && sp.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
            })
            .map(|pdevice| {
                let device = Self::create_device(instance, pdevice)?;
                let queue_infos = unsafe { Self::queue_infos(instance, pdevice, COMPUTE_BIT | TRANSFER_BIT) };
                let fences = Some(Self::create_fences(instance, pdevice, &device, queue_infos)?);
                let memory = unsafe { instance.get_physical_device_memory_properties(pdevice) };
                let (name, properties) = Self::device_name(instance, pdevice);
                let subgroup_properties = Self::subgroup_properties(instance, pdevice);
                let subgroup_size = subgroup_properties.subgroup_size as usize;

                Ok(Compute { device, fences, memory, name, subgroup_size, supported_operations: subgroup_properties.supported_operations, properties})

            })
            .collect::<Result<Vec<Compute>, Box<dyn Error>>>()?.into_iter()
            .filter(|c| c.fences.is_some())
            .collect::<Vec<Compute>>())
    }

    unsafe fn queue_infos(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
        bits: vk::QueueFlags,
    ) -> Vec<(usize, Vec<f32>)> {
        instance.get_physical_device_queue_family_properties(pdevice).iter().enumerate()
            .filter(|(_, prop)| prop.queue_flags.contains(bits))
            .map(|(idx, prop)| (idx, vec![1.0_f32; prop.queue_count as usize]))
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
        // https://www.khronos.org/blog/vulkan-subgroup-tutorial
        let mut sp = vk::PhysicalDeviceSubgroupProperties::builder();
        let mut dp2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut sp).build();
        unsafe { instance.fp_v1_1().get_physical_device_properties2(pdevice, &mut dp2); }
        sp.build()
    }

    fn create_fences(
        instance: &ash::Instance,
        pdevice: vk::PhysicalDevice,
        device: &ash::Device,
        queue_infos: Vec<(usize, Vec<f32>)>,
    ) -> Result<Vec<Fence>, Box<dyn Error>> {
        queue_infos.into_iter().map(|(phy_index, queue_priorities)| {
            let vk_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
            let queues = (0..queue_priorities.len()).into_iter().map(|queue_index| {
                unsafe { device.get_device_queue(phy_index as u32, queue_index as u32) }
            })
            .collect::<Vec<vk::Queue>>();
            let allocator = Some(RwLock::new(Allocator::new(&AllocatorCreateDesc {
                physical_device: pdevice,
                device: device.clone(),
                instance: instance.clone(),
                debug_settings: Default::default(),
                buffer_device_address: false,
            })?));
            Ok(Fence{ vk_fence, queues, phy_index, allocator})
        })
        .collect::<Result<Vec<Fence>, Box<dyn Error>>>()
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
            ext_names.push(EXT_PORTABILITY_SUBSET);
        }

        // See: https://github.com/MaikKlein/ash/issues/539
        let priorities = unsafe { Self::queue_infos(instance, pdevice, COMPUTE_BIT | TRANSFER_BIT) }.into_iter().map(|f| f.1).collect::<Vec<_>>();
        let queue_create_infos = unsafe { Self::queue_infos(instance, pdevice, COMPUTE_BIT | TRANSFER_BIT) }.into_iter().enumerate().map(|(idx, (phy_index, _))| {
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

    pub fn local_gpus(
        &self
    ) -> Result<&[Compute], Box<dyn Error>> {
        match &self.compute {
            Some(gpus) => Ok(gpus),
            None => Err("No compute capable devices".to_string().into()),
        }
    }
}

impl fmt::Display for Vulkan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let cpus = format!("cpu_logical_cores: {}", "?");
        let f32_size = format!("f32_size: {}", std::mem::size_of::<f32>());
        let gpu_device_count = format!("gpu_device_count: {}", self.compute.as_ref().unwrap().len());
        let gpus = self.compute.as_ref().unwrap().iter().map(|gpu| {

            let name = format!("name: {}", gpu.name);
            let device_type = format!("device_type: {:?}", gpu.properties.device_type);

            let queue_size = format!("queue_size: {:?}", gpu.fences.as_ref().unwrap().len());
            let queue_str = gpu.fences.as_ref().unwrap().iter()
                .map(|f| format!("{} {}", f.phy_index, f.queues.len()))
                .collect::<Vec<String>>();
            let queues = format!("queues: {:?}", queue_str);

            let subgroup_size = format!("subgroup_size: {:?}", gpu.subgroup_size);
            let subgroup_operations = format!("subgroup_operations: {:?}", gpu.supported_operations);

            let memory_heap_count = format!("memory_heap_count: {}", gpu.memory.memory_heap_count);
            let mem_str = gpu.memory.memory_heaps.iter()
                .filter(|mh| mh.size.ne(&0))
                .enumerate()
                .map(|(idx, mh)| {
                    format!("{} {}", idx, mh.size / 1_073_741_824)
                })
                .collect::<Vec<String>>();
            let memory_heaps = format!("memory_heaps: {:?}", mem_str);

            format!("{name}\n{device_type}\n{queue_size}\n{queues}\n{subgroup_size}\n{subgroup_operations}\n{memory_heap_count}\n{memory_heaps}")

        })
        .collect::<String>();
        write!(f, "{cpus}\n{f32_size}\n{gpu_device_count}\n{gpus}")
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
        (0..binding_count).into_iter().map(|i| {
            vk::DescriptorSetLayoutBinding::builder()
                .binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        })
        .collect()
    }

    fn create(
        device: &'a ash::Device,
        bindings: &[vk::DescriptorSetLayoutBinding],
        module: &rspirv::dr::Module,
        spec: vk::SpecializationInfoBuilder,
    ) -> Result<Shader<'a>, Box<dyn Error>> {
        let set_layouts = unsafe { device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings),
            None,
        ) }.map(|set_layout| vec![set_layout])?;

        // TODO: make dynamic with spir-v reflection
        let push_constant_ranges = vk::PushConstantRange::builder()
            .offset(0)
            .size(4)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();
        let pipeline_layout = unsafe { device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&[push_constant_ranges]),
            None)?
        };

        let binary = module.assemble();
        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&binary), None)? };
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            // According to https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html
            // "Another problem is querying the subgroup size from inside the kernel, which has a
            // surprising gotcha. Unless the VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT
            // flag is set at pipeline creation time, the gl_SubgroupSize variable is defined to have
            // the value from VkPhysicalDeviceSubgroupProperties, which in my experiment is always 32 on
            // Intel no matter the actual subgroup size. But setting that flag makes it give the value expected."
            .flags(vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT|vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS_EXT)
            .module(module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0")?)
            .specialization_info(&spec)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let pipeline = unsafe { device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder().stage(stage.build()).layout(pipeline_layout).build()],
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

pub struct Fence {
    pub queues: Vec<vk::Queue>,
    pub phy_index: usize,

    vk_fence: vk::Fence,
    allocator: Option<RwLock<Allocator>>,
}

pub struct Compute {
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub name: String,
    pub subgroup_size: usize,
    pub properties: vk::PhysicalDeviceProperties,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub fences: Option<Vec<Fence>>,

    device: ash::Device,
}

pub struct Schedule<'a, T: Bounded> {
    pub shader: &'a Shader<'a>,
    pub fence: &'a Fence,
    pub tasks: Vec<Task<T>>,
}

pub struct Task<T: Bounded> {
    pub input: Vec<Vec<T>>,
    pub output: Vec<T>,
    pub push_constants: Vec<PushConstant>,
    pub queue: vk::Queue,
    pub group_count: GroupCount,
}

impl<T: Bounded> Task<T> {

    fn dispatch(
        &self,
        device: &ash::Device,
        descriptor_set: vk::DescriptorSet,
        command_buffer: vk::CommandBuffer,
        shader: &Shader<'_>,
        output: &Buffer,
        input: &[Buffer],
    ) -> Result<(), vk::Result> {

        let buffer_infos = (0..=input.len()).into_iter()
            .map(|f| match f {
                0 => [output.buffer_info],
                _ => [input[f-1].buffer_info],
            })
            .collect::<Vec<[vk::DescriptorBufferInfo; 1]>>();

        let wds = buffer_infos.iter().enumerate()
            .map(|(i, buf)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(buf)
                    .build()
            })
            .collect::<Vec<vk::WriteDescriptorSet>>();

        unsafe {
            device.update_descriptor_sets(&wds, &[]);
            device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, shader.pipeline);
            device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::COMPUTE, shader.pipeline_layout, 0, &[descriptor_set], &[]);
            for push_constant in &self.push_constants {
                device.cmd_push_constants(command_buffer, shader.pipeline_layout, vk::ShaderStageFlags::COMPUTE, push_constant.offset, &push_constant.constants);
            }
            device.cmd_dispatch(command_buffer, self.group_count.x, self.group_count.y, self.group_count.z);
            device.end_command_buffer(command_buffer)
        }
    }
}

pub struct GroupCount {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for GroupCount {
    fn default() -> Self {
        GroupCount { x: 1, y: 1, z: 1 }
    }
}

pub struct PushConstant {
    pub offset: u32,
    pub constants: Vec<u8>,
}

impl Compute {

    pub fn scheduled<T: Bounded>(
        &self,
        shader: &Shader,
        fence: &Fence,
        schedule: &mut Schedule<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {

        let allocator = fence.allocator.as_ref().unwrap();
        let command = Command::new(
            fence.phy_index as u32,
            shader.binding_count,
            &shader.set_layouts,
            schedule.tasks.len() as u32,
            &self.device,
        )?;
        let descriptor_set = command.descriptor_sets.first().unwrap().to_owned();

        schedule.tasks.iter_mut()
            .zip(command.command_buffers.iter())
            .try_for_each(|(task, command_buffer)| {
                self.execute(&descriptor_set, command_buffer, allocator, shader, fence, task)
            })
    }

    fn execute<T: Bounded>(
        &self,
        descriptor_set: &vk::DescriptorSet,
        command_buffer: &vk::CommandBuffer,
        allocator: &RwLock<Allocator>,
        shader: &Shader,
        fence: &Fence,
        task: &mut Task<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {

        let output_buffer = Buffer::new(
            &format!("output {}", fence.phy_index),
            &self.device,
            &allocator,
            Buffer::buffer_create_info(
                (task.output.len() * std::mem::size_of::<T>()) as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                &[fence.phy_index as u32]
            ),
            gpu_allocator::MemoryLocation::GpuToCpu,
            0,
            vk::WHOLE_SIZE,
        )?;

        let input_buffers = task.input
            .iter()
            .map(|data| {
                let buffer = Buffer::new(
                    &format!("input {}", fence.phy_index),
                    &self.device,
                    &allocator,
                    Buffer::buffer_create_info(
                        (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize,
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                        &[fence.phy_index as u32]
                    ),
                    gpu_allocator::MemoryLocation::CpuToGpu,
                    0,
                    vk::WHOLE_SIZE,
                )?.fill(data)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<Buffer<'_, '_>>, Box<dyn Error + Send + Sync>>>()?;

        task.dispatch(
            &self.device,
            *descriptor_set,
            *command_buffer,
            shader,
            &output_buffer,
            &input_buffers
        )?;

        // TODO: possible optimization here: group submits per queue -> map into intermediate result
        // useful if multiple queues are used
        let submits = [vk::SubmitInfo::builder().command_buffers(&[*command_buffer]).build()];
        unsafe { self.device.queue_submit(task.queue, &submits, fence.vk_fence)? };

        unsafe {
            self.device.wait_for_fences(&[fence.vk_fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence.vk_fence])?;
        }

        let data_ptr = output_buffer.c_ptr.as_ptr().cast::<T>();
        unsafe { data_ptr.copy_to_nonoverlapping(task.output.as_mut_ptr(), task.output.len()) };

        Ok(())
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() }
        for fence in self.fences.as_ref().unwrap() {
            unsafe { self.device.destroy_fence(fence.vk_fence, None) }
        }
        self.fences = None;
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

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_info)? };

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
    allocation: Option<Allocation>,
    c_ptr: std::ptr::NonNull<std::ffi::c_void>,
    buffer_info: vk::DescriptorBufferInfo,

    device: &'a ash::Device,
    allocator: &'b RwLock<Allocator>,
}

impl <'a, 'b> Buffer<'_, '_> {

    fn new(
        name: &str,
        device: &'a ash::Device,
        allocator: &'b &RwLock<Allocator>,
        create_info: vk::BufferCreateInfoBuilder,
        location: gpu_allocator::MemoryLocation,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<Buffer<'a, 'b>, Box<dyn Error + Send + Sync>> {
        let buffer = unsafe { device.create_buffer(&create_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut malloc = allocator.write().unwrap();
        let allocation = malloc.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
        })?;
        let c_ptr = allocation.mapped_ptr().unwrap();
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };
        let buffer_info = Self::buffer_info(buffer, offset, range);
        Ok(Buffer { buffer, allocation: Some(allocation), c_ptr, device, buffer_info, allocator })
    }

    fn fill<T: Sized>(
        self,
        data: &[T],
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let data_ptr = self.c_ptr.as_ptr().cast::<T>();
        unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
        Ok(self)
    }

    fn buffer_create_info(
        device_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        queue_family_indices: &[u32],
    ) -> vk::BufferCreateInfoBuilder {
        vk::BufferCreateInfo::builder()
            .size(device_size)
            .usage(usage)
            .sharing_mode(match queue_family_indices.len() {
                1 => vk::SharingMode::EXCLUSIVE,
                _ => vk::SharingMode::CONCURRENT,
            })
            .queue_family_indices(queue_family_indices)
    }

    fn buffer_info(
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize
    ) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(offset)
            .range(range)
            .build()
    }
}

impl <'a, 'b> Drop for Buffer<'a, 'b> {
    fn drop(&mut self) {
        let mut malloc = self.allocator.write().unwrap();
        malloc.free(self.allocation.take().unwrap()).unwrap();
        unsafe { self.device.destroy_buffer(self.buffer, None) };
    }
}