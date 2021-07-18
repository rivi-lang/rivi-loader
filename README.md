# rivi-loader

This part of the project is a commanding interface to the GPU, written in Rust. It's responsible for program initialization, memory allocation, transfer, and synchronization between the GPU and the CPU.

## Installation

1. Make sure Rust is installed, if not, consider [rustup](https://rustup.rs/)
2. Make sure Vulkan is installed, if not, consider [LunarG](https://vulkan.lunarg.com/sdk/home) and the installation instructions found on [ash](https://github.com/MaikKlein/ash#example)
3. Run `cargo run`.

To understand what is happening here, consider [a blog post series about graphics applications](https://hoj-senna.github.io/ashen-aetna/).
