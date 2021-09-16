use std::{error::Error, io};

use ash::vk;
use rspirv::dr::Module;


pub struct SPIRV {
    pub(crate) binary: Vec<u32>,
    pub(crate) dslbs: Vec<vk::DescriptorSetLayoutBinding>,
}

impl SPIRV {

    fn module(
        binary: &[u32]
    ) -> Result<Module, Box<dyn Error>> {
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(binary, &mut loader)?;
        Ok(loader.module())
    }

    fn binding_count(
        module: &Module
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

    pub fn new<R: io::Read + io::Seek>(
        x: &mut R
    ) -> Result<SPIRV, Box<dyn Error>> {
        let binary = ash::util::read_spv(x)?;
        let module = SPIRV::module(&binary)?;
        let binding_count = SPIRV::binding_count(&module)?;
        let dslbs = SPIRV::descriptor_set_layout_bindings(binding_count);
        Ok(SPIRV { binary, dslbs })
    }

}

#[cfg(test)]
mod tests {

    use super::SPIRV;

    #[test]
    fn spirv_new_from_file() {
        let mut spirv = std::io::Cursor::new(&include_bytes!("../examples/rf/shader/apply.spv")[..]);
        SPIRV::new(&mut spirv).expect("can load shader");
    }
}