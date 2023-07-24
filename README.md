# rivi-loader

[![Latest version](https://img.shields.io/crates/v/rivi-loader.svg)](https://crates.io/crates/rivi-loader)
[![Docs](https://docs.rs/rivi-loader/badge.svg)](https://docs.rs/rivi-loader/)
[![LICENSE](https://img.shields.io/badge/license-GPL-blue.svg)](LICENSE-GPL)

```toml
[dependencies]
rivi-loader = "0.2.0"
```

rivi-loader is a Vulkan-based program loader for GPGPU applications. Roughly speaking, if you have input(s) and an output it will run a SPIR-V kernel on those values. This way, the long days of wondering how Vulkan works can be forgotten / skipped. That being said, the library does not help you in writing SPIR-V or any other shading language. It neither helps you in scheduling. What it does do is help you to setup the Vulkan instance, attach debug layers on it, query compute capable logical devices, expose fences and compute-capable queue families and queues, manage memory and command buffers, and deal with shader creation with specialization constant support. There is also a keen interest in making sure that the command buffers can be queued and polled in parallel on the CPU side.

## Example

```Rust
use rivi_loader::{DebugOption, PushConstant, Task, GroupCount, Vulkan};

fn main() {

    let vk = Vulkan::new(DebugOption::None).unwrap();
    let gpus = vk.compute.as_ref().unwrap();

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
        input: vec![
            vec![1.0f32; gpu.subgroup_size * gpu.subgroup_size * vec4],
        ],
        output: vec![0.0f32; 1],
        push_constants: vec![
            PushConstant { offset: 0, constants: vec![2] },
        ],
        queue: *queue,
        group_count: GroupCount { ..Default::default() },
    }];

    let run_timer = std::time::Instant::now();
    gpu.scheduled(&shader, queue_family, &mut tasks).unwrap();
    let end_timer = run_timer.elapsed().as_micros();

    let task = tasks.first().unwrap();
    println!("Queue Family {}, Queue {:?}: {:?} in {}qs", queue_family.phy_index, task.queue, task.output[0], end_timer);
    assert_eq!(task.output[0], (gpu.subgroup_size * gpu.subgroup_size * vec4) as f32);
}

```

## Features

- Rust lifetime management of Vulkan resources
- scheduling on multiple gpus
- scheduling on multiple queue families
- supports shader specialization constants
- supports push constants
- supports per device querying of "advanced" compute capabilities, e.g., subgroup sizes
- allows Rust generics to be used on input and output buffers

## Limitations (i.e., what this repository assumes)

- there should always exist an input vector and an output vector in your SPIR-V code
- the length of the output vector has to be always statically known

## Installation

1. Make sure Rust is installed, if not, consider [rustup](https://rustup.rs/)
2. Make sure Vulkan is installed, if not, consider [LunarG](https://vulkan.lunarg.com/sdk/home) and the installation instructions found on [ash](https://github.com/MaikKlein/ash#example)
3. Run an example, e.g. `cargo run --example reduce`.

## The library

To understand what is happening in the `lib.rs`, consider [a blog post series about graphics applications](https://hoj-senna.github.io/ashen-aetna/).

## Periferia Labs

Periferia Labs is an ad-hoc group of friends tinkering with GPGPU. See our other projects:

- [laskin.live](https://github.com/periferia-labs/laskin.live) - An online calculator, but you can only use it on your remote friend’s GPU
