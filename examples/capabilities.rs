use rivi_loader::{DebugOption, Vulkan};

fn main() {
    let vk = Vulkan::new(DebugOption::None).unwrap();
    println!("{}", vk);
}
