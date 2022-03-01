# rivi-loader

[![Latest version](https://img.shields.io/crates/v/rivi-loader.svg)](https://crates.io/crates/rivi-loader)
[![Docs](https://docs.rs/rivi-loader/badge.svg)](https://docs.rs/rivi-loader/)
[![LICENSE](https://img.shields.io/badge/license-GPL-blue.svg)](LICENSE-GPL)

```toml
[dependencies]
rivi-loader = "0.1.4"
```

rivi-loader is a Vulkan-based program loader for GPGPU applications.

## Example

```Rust
fn main() {
    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0];
    let input = &vec![vec![a, b]];
    let mut output = vec![0.0f32; 2];

    let vk = rivi_loader::new(DebugOption::None).unwrap();

    let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
    let shader = vk.load_shader(&mut cursor).unwrap();

    vk.compute(input, &mut output, &shader).unwrap();

    println!("Result: {:?}", output);
    assert_eq!(output, vec![4.0, 6.0]);
}
```

Thanks to Vulkan, the example programs run on both discrete and integrated graphics cards and across various operating systems. Testing on discrete cards has primarily been done on Windows 10 and Linux (Arch and Ubuntu LTS using proprietary drivers) using AMD and Nvidia cards, with both AMD and Intel CPUs on computers with single and multiple GPUs. Integrated cards that have been tested include Apple M1, Raspberry Pi 4, and Nvidia Jetson TX2.

## Features

- lifetime management for Vulkan resources
- interoperable platform support across operating systems and graphics cards
- multi-gpu support (mixing AMD and Nvidia on a single machine is OK)

## Installation

1. Make sure Rust is installed, if not, consider [rustup](https://rustup.rs/)
2. Make sure Vulkan is installed, if not, consider [LunarG](https://vulkan.lunarg.com/sdk/home) and the installation instructions found on [ash](https://github.com/MaikKlein/ash#example)
3. Run an example, e.g. `cargo run --release --example repl`.

## The library

To understand what is happening in the `lib.rs`, consider [a blog post series about graphics applications](https://hoj-senna.github.io/ashen-aetna/).

## Periferia Labs

Periferia Labs is an ad-hoc group of friends tinkering with GPGPU. See our other projects:

- [laskin.live](https://github.com/periferia-labs/laskin.live) - An online calculator, but you can only use it on your remote friend’s GPU
