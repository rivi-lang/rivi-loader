# rivi-loader

[![Latest version](https://img.shields.io/crates/v/rivi-loader.svg)](https://crates.io/crates/rivi-loader)
[![Docs](https://docs.rs/rivi-loader/badge.svg)](https://docs.rs/rivi-loader/)
[![LICENSE](https://img.shields.io/badge/license-GPL-blue.svg)](LICENSE-GPL)

```toml
[dependencies]
rivi-loader = "0.1.5"
```

rivi-loader is a Vulkan-based program loader for GPGPU applications. Roughly speaking, if you have input(s) and an output it will run a SPIR-V kernel on those values. This way, the long days of wondering how Vulkan works can be forgotten / skipped. That being said, the library does not help you in writing SPIR-V or any other shading language. It neither helps you in scheduling. What it does do is help you to setup the Vulkan instance, attach debug layers on it, query compute capable logical devices, expose fences and compute-capable queue families and queues, manage memory and command buffers, and deal with shader creation with specialization constant support. There is also a keen interest in making sure that the command buffers can be queued and polled in parallel on the CPU side.

## Example

```Rust
fn main() {
    let a = vec![1.0f32; 64];
    let input = &vec![vec![a]];
    let mut output = vec![0.0f32; 1];

    let vk = rivi_loader::new(DebugOption::None).unwrap();

    let mut cursor = std::io::Cursor::new(&include_bytes!("./reduce/reduce.spv")[..]);
    let shader = vk.load_shader(&mut cursor, Some(vec![vec![2]])).unwrap();

    vk.compute(input, &mut output, &shader).unwrap();

    println!("Result: {:?}", output);
}
```

## Features

- lifetime management of Vulkan resources
- multi-gpu support
- allows scheduling on multiple queue families
- allows scheduling on multiple queues of a single queue family
- allows shader specialization constants to be set
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
