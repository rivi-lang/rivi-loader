use rivi_loader::{DebugOption, GroupCount, PushConstant, Task, Vulkan};

fn main() {
    let vk = Vulkan::new(DebugOption::None).unwrap();
    let gpus = vk.compute.as_ref().unwrap();

    loop {
        let binary = &include_bytes!("./reduce/reduce.spv")[..];
        let module = rspirv::dr::load_bytes(binary).unwrap();

        let gpu = gpus.first().unwrap();
        println!("{} ({:?}):", gpu.name, gpu.properties.device_type);
        let specializations = Vec::new();
        let shader = rivi_loader::load_shader(gpu, module, specializations).unwrap();

        let queue_family = gpu.fences.as_ref().unwrap().first().unwrap();
        let queue = queue_family.queues.first().unwrap();

        let vec4 = 4;
        let mut tasks = vec![Task {
            input: vec![vec![1.0f32; gpu.subgroup_size * gpu.subgroup_size * vec4]],
            output: vec![0.0f32; 4096],
            push_constants: vec![PushConstant {
                offset: 0,
                constants: vec![2],
            }],
            queue: *queue,
            group_count: GroupCount { x: 1, y: 1, z: 1 },
        }];

        let run_timer = std::time::Instant::now();
        gpu.scheduled(&shader, queue_family, &mut tasks).unwrap();
        let end_timer = run_timer.elapsed().as_micros();

        let task = tasks.first().unwrap();
        println!(
            "Queue Family {}, Queue {:?}: {:?} in {}qs",
            queue_family.phy_index, task.queue, task.output[0], end_timer
        );
        assert_eq!(
            task.output[0],
            (gpu.subgroup_size * gpu.subgroup_size * vec4) as f32
        );
        let errors = task
            .output
            .iter()
            .enumerate()
            .filter(|(i, v)| (*v).ne(&0f32) && i.ne(&0))
            .collect::<Vec<_>>();
        println!("Errors: {:?}", errors)
    }
}
