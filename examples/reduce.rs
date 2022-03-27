use rivi_loader::{DebugOption, Schedule, PushConstant, Task, GroupCount};
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

        gpu.fences
            .as_ref()
            .unwrap()
            .par_iter()
            .for_each(|fence| {
                let tasks = fence.queues
                    .iter()
                    .map(|queue| {

                        let vec4 = 4;
                        let workgroup_size = gpu.subgroup_size * gpu.subgroup_size;
                        let a = vec![1.0f32; workgroup_size * vec4];
                        let input = vec![a];
                        let output = vec![0.0f32; 1];
                        let push_constants = vec![
                            PushConstant { offset: 0, constants: vec![2] },
                        ];
                        let group_count = GroupCount { ..Default::default() };

                        Task {
                            input,
                            output,
                            push_constants,
                            queue: *queue,
                            group_count,
                        }
                    })
                    .collect::<Vec<_>>();

                let mut schedule = Schedule { shader: &shader, fence, tasks };
                let run_timer = std::time::Instant::now();
                gpu.scheduled(&shader, fence, &mut schedule).unwrap();
                let end_timer = run_timer.elapsed().as_millis();

                schedule.tasks
                    .iter()
                    .for_each(|task| {
                        println!("  Core {}, queue {:?}: result {:?} in {}ms", fence.phy_index, task.queue, task.output[0], end_timer);
                        task.output.iter().enumerate().for_each(|(index, val)| {
                            if *val != 0.0 {
                                println!("{} {}", index, task.output[0])
                            }
                        })
                    });
            })
    }
}
