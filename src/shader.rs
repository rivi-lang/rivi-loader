use std::{error::Error, ffi::CString, io};

use ash::vk;

use crate::compute::Compute;

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
            .flat_map(|f| f
                .operands
                .iter()
                .filter_map(|op| match op {
                    rspirv::dr::Operand::Decoration(d) => match d {
                        rspirv::spirv::Decoration::Binding => Some(rspirv::spirv::Decoration::Binding),
                        _ => None
                    },
                    _ => None
                })
                .collect::<Vec<rspirv::spirv::Decoration>>()
        ).count();
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

        let shader_entry_name = CString::new("main")?;
        let shader_module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(binary), None)? };
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

    pub fn new<R: io::Read + io::Seek>(
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
