use std::{error::Error, ffi::CString};

use ash::{version::DeviceV1_0, vk};

use crate::spirv::SPIRV;


pub(crate) struct Shader {
    pub shader_module: vk::ShaderModule,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl Shader {

    pub(crate) unsafe fn new(
        device: &ash::Device,
        spirv: &SPIRV
    ) -> Result<Shader, Box<dyn Error>>  {
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

        let pipelines = device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder()
                .stage(entry_point)
                .layout(pipeline_layout)
                .build()
            ],
            None,
        );

        let pipeline = match pipelines {
            Ok(f) => f[0],
            Err((_, err)) => return Err(Box::new(err)),
        };

        println!("Pipelines created");

        Ok(Shader{shader_module, pipeline_layout, pipeline, set_layouts})
    }

    pub(crate) unsafe fn drop(
        self,
        device: &ash::Device
    ) {
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_shader_module(self.shader_module, None);
        for set_layout in self.set_layouts {
            device.destroy_descriptor_set_layout(set_layout, None);
        }
        device.destroy_pipeline(self.pipeline, None);
    }
}