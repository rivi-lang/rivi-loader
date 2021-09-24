use std::{convert::TryInto, fmt, slice, sync::RwLock};

use ash::{version::DeviceV1_0, vk::{self, PhysicalDeviceMemoryProperties}};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;

use crate::{buffer::Buffer, command::Command, fence::Fence, shader::Shader};


const STRIDE: usize = std::mem::size_of::<f32>() as usize;

pub struct Compute {
    pub device: ash::Device,
    pub allocator: Option<RwLock<gpu_allocator::VulkanAllocator>>,
    pub fences: Vec<Fence>,

    pub(crate) memory: PhysicalDeviceMemoryProperties,
}

impl fmt::Debug for Compute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        println!("Memory types: {}", self.memory.memory_type_count);
        self.memory
            .memory_types
            .iter()
            .filter(|mt| !mt.property_flags.is_empty())
            .enumerate()
            .for_each(|(idx, mt)| {
                println!("Index {} {:?} (heap {})", idx, mt.property_flags, mt.heap_index);
            });

        println!("Memory heaps: {}", self.memory.memory_heap_count);
        self.memory
            .memory_heaps
            .iter()
            .filter(|mh| mh.size.ne(&0))
            .enumerate()
            .for_each(|(idx, mh)| {
                println!("{:?} GiB {:?} (heap {})", mh.size / 1_073_741_824, mh.flags, idx);
            });

        let qfs = self.fences
            .iter()
            .map(|f| f.phy_index);

        let mut uniqs: Vec<usize> = Vec::new();
        qfs
            .into_iter()
            .for_each(|f| {
                 if !uniqs.contains(&f) {
                     uniqs.push(f);
                 }
            });

        f.write_fmt(format_args!("  Found {} compute core(s) with {} thread(s)", uniqs.len(), self.fences.len()))
    }
}

impl Compute {

    fn create_cpu_inputs(
        &self,
        queue: &[u32],
        inputs: &[Vec<f32>]
    ) -> Vec<Buffer> {
        inputs
            .iter()
            .map(|input| {
                let buffer = Buffer::new(
                    &self.device,
                    &self.allocator,
                    (input.len() * STRIDE) as u64,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                    queue,
                ).unwrap();

                buffer.fill(input).unwrap();

                buffer

            })
            .collect()
    }

    fn task(
        &self,
        command: &Command,
        fence_idx: u32,
        max_sets: u32,
        func: &Shader<'_>,
        queue_family_indices: &[u32],
        cpu_buffer: &Buffer<'_, '_>,
        input: &[Vec<Vec<f32>>],
    ) -> Vec<Vec<Buffer>> {

        command.command_buffers
            .iter()
            .enumerate()
            .map(|(index, command_buffer)| {

                let index_offset = (command.command_buffers.len() as u32 * fence_idx + index as u32) as usize;
                let cpu_offset: vk::DeviceSize = (cpu_buffer.device_size / max_sets as u64) * index_offset as u64;
                let cpu_chunk_size = cpu_buffer.device_size / max_sets as u64;

                let cpu_buffers = self.create_cpu_inputs(queue_family_indices, &input[index_offset]);

                let buffer_infos = (0..=cpu_buffers.len())
                    .into_iter()
                    .map(|f| match f {
                        0 => [vk::DescriptorBufferInfo::builder()
                            .buffer(cpu_buffer.buffer)
                            .offset(cpu_offset)
                            .range(cpu_chunk_size)
                            .build()],
                        _ => [vk::DescriptorBufferInfo::builder()
                            .buffer(cpu_buffers.get(f-1).unwrap().buffer)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)
                            .build()],
                    })
                    .collect::<Vec<[vk::DescriptorBufferInfo; 1]>>();

                let ds = command.descriptor_sets
                    .get(index as usize)
                    .unwrap()
                    .to_owned();

                let wds = buffer_infos
                    .iter()
                    .enumerate()
                    .map(|(index, buf)| {
                        vk::WriteDescriptorSet::builder()
                            .dst_set(ds)
                            .dst_binding(index.try_into().unwrap())
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(buf)
                            .build()
                    })
                    .collect::<Vec<vk::WriteDescriptorSet>>();

                unsafe {
                    self.device.update_descriptor_sets(&wds, &[]);
                    self.device.begin_command_buffer(*command_buffer, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                };

                cpu_buffers.iter().for_each(|cpu|
                    unsafe {
                        self.device.cmd_copy_buffer(
                            *command_buffer,
                            cpu.buffer,
                            cpu.buffer,
                            &[vk::BufferCopy::builder()
                                .src_offset(0)
                                .dst_offset(0)
                                .size(cpu.device_size)
                                .build()
                            ],
                        )
                    }
                );

                unsafe {
                    self.device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::COMPUTE, func.pipeline);
                    self.device.cmd_bind_descriptor_sets(*command_buffer, vk::PipelineBindPoint::COMPUTE, func.pipeline_layout, 0, &[ds], &[]);
                    self.device.cmd_dispatch(*command_buffer, 1024, 1, 1);

                    self.device.cmd_copy_buffer(
                        *command_buffer,
                        cpu_buffer.buffer,
                        cpu_buffer.buffer,
                        &[vk::BufferCopy::builder()
                            .src_offset(cpu_offset)
                            .dst_offset(cpu_offset)
                            .size(cpu_chunk_size)
                            .build()
                        ],
                    );
                    self.device.end_command_buffer(*command_buffer).expect("End commandbuffer");
                }

                cpu_buffers

            })
            .collect::<Vec<_>>()
    }

    pub fn execute(
        &self,
        input: &[Vec<Vec<f32>>],
        out_length: usize,
        shader: &Shader,
        fences: &[Fence]
    ) -> &[f32] { unsafe {

        let queue_family_indices = fences
            .iter()
            .map(|f| f.phy_index as u32)
            .collect::<Vec<u32>>();

        let size_in_bytes = out_length * input.len() * STRIDE;
        let cpu_buffer = Buffer::new(
            &self.device,
            &self.allocator,
            size_in_bytes.try_into().unwrap(),
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            &queue_family_indices,
        ).unwrap();

        fences
            .par_iter()
            .enumerate()
            .for_each(|(fence_idx, fence)| {

                let command = Command::new(
                    fence.phy_index as u32,
                    shader.binding_count as u32,
                    input.len() as u32,
                    &shader.set_layouts,
                    (input.len() / fences.len()) as u32,
                    &self.device,
                ).unwrap();

                let _buffers = self.task(
                    &command,
                    fence_idx as u32,
                    input.len() as u32,
                    shader,
                    &queue_family_indices,
                    &cpu_buffer,
                    input
                );

                fence.submit(&self.device, &command.command_buffers).unwrap();
                self.device.wait_for_fences(&[fence.fence], true, std::u64::MAX).unwrap();
                self.device.reset_fences(&[fence.fence]).unwrap();

            });

        let mapping = cpu_buffer.allocation.mapped_ptr().unwrap().as_ptr();
        slice::from_raw_parts::<f32>(mapping as *const f32, out_length * input.len())
    }}
}

impl Drop for Compute {
    fn drop(
        &mut self
    ) {
        unsafe { self.device.device_wait_idle().unwrap() }
        for fence in &self.fences {
            unsafe { self.device.destroy_fence(fence.fence, None) }
        }
        self.allocator = None;
        unsafe { self.device.destroy_device(None) }
    }
}