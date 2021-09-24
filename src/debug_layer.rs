use std::{error::Error, ffi::CString};

use ash::{extensions::ext::DebugUtils, vk};


pub(crate) struct DebugLayer {
    pub(crate) loader: ash::extensions::ext::DebugUtils,
    pub(crate) callback: vk::DebugUtilsMessengerEXT,
}

pub enum DebugOption {
    None,
    Validation,
    Debug,
}

impl DebugOption {

    pub(crate) fn cons(
        &self
    ) -> Result<(Vec<CString>, vk::DebugUtilsMessengerCreateInfoEXT), Box<dyn Error>> {

        let mut layer_names: Vec<CString> = Vec::new();
        match self {
            DebugOption::None => {},
            DebugOption::Validation => {
                layer_names.push(CString::new("VK_LAYER_KHRONOS_validation")?);
            },
            DebugOption::Debug => {
                layer_names.push(CString::new("VK_LAYER_KHRONOS_validation")?);
                layer_names.push(CString::new("VK_LAYER_LUNARG_api_dump")?);
            },
        }

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
            ..Default::default()
        };
        match self {
            DebugOption::None => {},
            _ => {
                debug_info = vk::DebugUtilsMessengerCreateInfoEXT {
                    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    pfn_user_callback: Some(DebugLayer::vulkan_debug_utils_callback),
                    ..Default::default()
                };
            }
        }

        Ok((layer_names, debug_info))

    }

}

impl DebugLayer {

    pub(crate) unsafe extern "system" fn vulkan_debug_utils_callback(
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

    pub(crate) fn new(
        option: DebugOption,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
        entry: &ash::Entry,
        instance: &ash::Instance
    ) -> Result<Option<DebugLayer>, Box<dyn Error>> {
        match option {
            DebugOption::None => Ok(None),
            _ => {
                let loader = DebugUtils::new(entry, instance);
                let callback = unsafe { loader.create_debug_utils_messenger(info, None)? };
                println!("Debug attached");
                Ok(Some(DebugLayer{loader, callback}))
            }
        }
    }

}

impl Drop for DebugLayer {
    fn drop(
        &mut self
    ) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.callback, None) }
    }
}