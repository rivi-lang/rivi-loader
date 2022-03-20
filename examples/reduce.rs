use rivi_loader::{DebugOption, Schedule, PushConstant};
use rayon::prelude::*;

fn main() {

    let vk = rivi_loader::new(DebugOption::None).unwrap();
    let gpus = vk.local_gpus().unwrap();

    let binary = &include_bytes!("./reduce/reduce.spv")[..];
    let module = rspirv::dr::load_bytes(binary).unwrap();

    for gpu in gpus {

        println!("{} ({:?}):", gpu.name, gpu.properties.device_type);

        let specializations = Vec::new();
        let shader = rivi_loader::load_shader(gpu, module.clone(), specializations).unwrap();

        gpu.fences.as_ref().unwrap().par_iter().for_each(|fence| {

            let vec4 = 4;
            let workgroup_size = gpu.subgroup_size * gpu.subgroup_size;
            let a = vec![1.0f32; workgroup_size * vec4];
            let input = vec![a];
            let mut output = vec![0.0f32; 1];
            let push_constants = vec![
                PushConstant { offset: 0, constants: vec![2] },
            ];
            let mut schedule = Schedule {
                output: &mut output, input: &input, shader: &shader, push_constants, fence
            };

            let run_timer = std::time::Instant::now();
            gpu.execute(&mut schedule).unwrap();
            let end_timer = run_timer.elapsed().as_millis();

            println!("  Core {}: {:?} {}", fence.phy_index, schedule.output[0], end_timer);
            schedule.output.iter().enumerate().for_each(|(index, val)| {
                if *val != 0.0 {
                    println!("{} {}", index, val)
                }
            })
        });
    }
}
