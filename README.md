# rivi-loader

[![Latest version](https://img.shields.io/crates/v/rivi-loader.svg)](https://crates.io/crates/rivi-loader)
[![Docs](https://docs.rs/rivi-loader/badge.svg)](https://docs.rs/rivi-loader/)
[![LICENSE](https://img.shields.io/badge/license-GPL-blue.svg)](LICENSE-GPL)

```toml
[dependencies]
rivi-loader = "0.1.4"
```

rivi-loader is a Vulkan-based program loader for GPGPU applications. It builds on the Rust-based Vulkan wrapper [ash](https://github.com/MaikKlein/ash). The project is a part of research agenda of interoperable GPU computing, that is, an effort to evaluate how Vulkan could be utilized to replace GLSL and CUDA accelerated programs with Vulkan.

## Example

```Rust
fn main() {
    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0];
    let input = &vec![vec![a, b]];
    let expected_output: Vec<f32> = vec![4.0, 6.0];
    let out_length = expected_output.len();

    let vk = rivi_loader::new(DebugOption::None).unwrap();

    let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
    let shaders = vk.load_shader(&mut cursor).unwrap();
    let shader = shaders.first().unwrap();

    let result = vk.compute(input, out_length, shader);

    println!("Result: {:?}", result);
    assert_eq!(result, expected_output);
}
```

The project aims to highlight performance optimizations available by using Vulkan with hand-written SPIR-V code. Various recent features of Vulkan, such as variable storage buffer pointers, dedicated memory allocations, subgroup operations, and asynchronous queue family usage are all relevant parts of the effort.

Thanks to Vulkan, the example programs run on both discrete and integrated graphics cards and across various operating systems. Testing on discrete cards has primarily been done on Windows 10 and Linux (Arch and Ubuntu LTS using proprietary drivers) using AMD and Nvidia cards, with both AMD and Intel CPUs on computers with single and multiple GPUs. Integrated cards that have been tested include Apple M1, Raspberry Pi 4, and Nvidia Jetson TX2.

The project is aimed as an example repository from which motivated people can use to start to tip their toes into GPGPU. In particular, the `examples/repl.rs` should be a good starting point to see what can be abstracted away. At the moment, programming your own application requires SPIR-V know-how, but one of the primary goals of the effort is eventually integrating a flavor of APL as the user-interfacing language (see: [haavisto2021vulkan](https://github.com/jhvst/haavisto2021vulkan) and [hal-03155647](https://hal.inria.fr/hal-03155647/)).

## Features

- lifetime management for Vulkan resources
- interoperable platform support across operating systems and graphics cards
- multi-gpu support (mixing AMD and Nvidia on a single machine is OK)

## Current limitations (to be addressed)

- output buffer dimensions have to be statically known and oftentimes hand-written, no type inference is given at the moment
- memory limitations are not endorsed, which means **you may run out of memory and hang your computer**
- this is only a loader program, it assumes you can write your own compute kernels

## Installation

1. Make sure Rust is installed, if not, consider [rustup](https://rustup.rs/)
2. Make sure Vulkan is installed, if not, consider [LunarG](https://vulkan.lunarg.com/sdk/home) and the installation instructions found on [ash](https://github.com/MaikKlein/ash#example)
3. Run an example. **Recommended**: simple vector addition `cargo run --release --example repl`. For random forest prediction (see: [haavisto2021vulkan](https://github.com/jhvst/haavisto2021vulkan)) run `cargo run --release --example rf` (**this may crash your computer**).

### Example descriptions

- `repl` a rough translation of WebGPU-based [laskin.live](https://github.com/periferia-labs/laskin.live) on Vulkan
- `rf` a random forest prediction algorithm translated from scikit (proof-of-concept of variable pointers feature and performance comparison against Cython)

## The library

To understand what is happening in the `lib.rs`, consider [a blog post series about graphics applications](https://hoj-senna.github.io/ashen-aetna/).

## Periferia Labs

Periferia Labs is an ad-hoc group of friends tinkering with GPGPU. See our other projects:

- [laskin.live](https://github.com/periferia-labs/laskin.live) - An online calculator, but you can only use it on your remote friend’s GPU
