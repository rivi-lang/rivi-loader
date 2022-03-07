use rivi_loader::DebugOption;

fn main() {
    let vk = rivi_loader::new(DebugOption::None).unwrap();
    println!("{}", vk);
}
