use rivi_loader::{DebugOption, Vulkan};

fn main() {
    let vk = Vulkan::new(DebugOption::None).unwrap();
    let gpus = vk.compute.as_ref().unwrap();
    for gpu in gpus {
        println!("{} ({:?}):", gpu.name, gpu.properties.device_type);
    }
}
