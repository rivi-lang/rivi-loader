use std::{convert::TryInto, slice, sync::Mutex, time::Instant};

use ash::{version::DeviceV1_0, vk};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;

use crate::{buffer::Buffer, command::Command, fence::Fence, shader::Shader, spirv};


const STRIDE: usize = std::mem::size_of::<f32>() as usize;

pub struct Compute {
    pub device: ash::Device,
    pub allocator: Mutex<gpu_allocator::VulkanAllocator>,
    pub fences: Vec<Fence>,
}

impl Compute {

    fn create_cpu_inputs(
        &self,
        queue: &[u32],
        inputs: &Vec<Vec<f32>>
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

    pub unsafe fn execute(
        &self,
        input: &Vec<Vec<Vec<f32>>>,
        out_length: usize,
        spirv: &spirv::SPIRV,
        fences: &[Fence]
    ) -> &[f32] {

        let run_timer = Instant::now();
        let func = Shader::new(&self.device, spirv).unwrap();
        let descriptor_count = spirv.dslbs.len() as u32;

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

        let range = input.len() / fences.len();

        println!("Commands {}ms", run_timer.elapsed().as_millis());

        let res = fences
            .par_iter()
            .enumerate()
            .flat_map(|(fence_i, fence)| {

                let command = Command::new(
                    fence.phy_index.try_into().unwrap(),
                    descriptor_count,
                    input.len().try_into().unwrap(),
                    &func.set_layouts,
                    range.try_into().unwrap(),
                    &self.device,
                ).unwrap();

                let res = (0..range)
                    .into_iter()
                    .flat_map(|index| {

                        let index_offset = range * fence_i + index;

                        let cpu_buffers = self.create_cpu_inputs(&queue_family_indices, &input[index_offset]);

                        let cpu_offset: vk::DeviceSize = (cpu_buffer.device_size / input.len() as u64) * index_offset as u64;
                        let cpu_chunk_size = cpu_buffer.device_size / input.len() as u64;

                        let buffer_infos = (0..=cpu_buffers.len())
                            .into_iter()
                            .map(|f| match f {
                                0 => [vk::DescriptorBufferInfo::builder()
                                    .buffer(cpu_buffer.buffer)
                                    .offset(cpu_offset.try_into().unwrap())
                                    .range(cpu_chunk_size.try_into().unwrap())
                                    .build()],
                                _ => [vk::DescriptorBufferInfo::builder()
                                    .buffer(cpu_buffers.get(f-1).unwrap().buffer)
                                    .offset(0)
                                    .range(vk::WHOLE_SIZE)
                                    .build()],
                            })
                            .collect::<Vec<[vk::DescriptorBufferInfo; 1]>>();

                        let ds = command.descriptor_sets
                            .get(index)
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

                        self.device.update_descriptor_sets(&wds, &[]);

                        self.device.begin_command_buffer(command.command_buffers[index], &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

                        cpu_buffers.iter().for_each(|cpu|
                            self.device.cmd_copy_buffer(
                                command.command_buffers[index],
                                cpu.buffer,
                                cpu.buffer,
                                &[vk::BufferCopy::builder()
                                    .src_offset(0)
                                    .dst_offset(0)
                                    .size(cpu.device_size)
                                    .build()
                                ],
                            ));

                        self.device.cmd_bind_pipeline(command.command_buffers[index], vk::PipelineBindPoint::COMPUTE, func.pipeline);
                        self.device.cmd_bind_descriptor_sets(command.command_buffers[index], vk::PipelineBindPoint::COMPUTE, func.pipeline_layout, 0, &[ds], &[]);
                        self.device.cmd_dispatch(command.command_buffers[index], 1024, 1, 1);

                        self.device.cmd_copy_buffer(
                            command.command_buffers[index],
                            cpu_buffer.buffer,
                            cpu_buffer.buffer,
                            &[vk::BufferCopy::builder()
                                .src_offset(cpu_offset)
                                .dst_offset(cpu_offset)
                                .size(cpu_chunk_size)
                                .build()
                            ],
                        );
                        self.device.end_command_buffer(command.command_buffers[index]).expect("End commandbuffer");

                        cpu_buffers

                    })
                    .collect::<Vec<_>>();

                println!("Command buffers {}ms", run_timer.elapsed().as_millis());

                fence.submit(&self.device, &command.command_buffers).unwrap();
                self.device.wait_for_fences(&[fence.fence], true, std::u64::MAX).expect("Wait for fence failed.");
                self.device.reset_fences(&[fence.fence]).unwrap();

                println!("After fences {}ms", run_timer.elapsed().as_millis());

                self.device.destroy_command_pool(command.command_pool, None);
                self.device.destroy_descriptor_pool(command.descriptor_pool, None);

                res

            })
            .collect::<Vec<_>>();

        let mapping = cpu_buffer.allocation.mapped_ptr().unwrap().as_ptr();
        let result = slice::from_raw_parts::<f32>(mapping as *const f32, out_length * input.len());

        println!("Results gathered {}ms", run_timer.elapsed().as_millis());

        let mut malloc = self.allocator.lock().unwrap();
        malloc.free(cpu_buffer.allocation).unwrap();
        self.device.destroy_buffer(cpu_buffer.buffer, None);
        res.into_iter().for_each(|f| {
            malloc.free(f.allocation).unwrap();
            self.device.destroy_buffer(f.buffer, None);
        });

        func.drop(&self.device);
        println!("Resource cleanup done {}ms", run_timer.elapsed().as_millis());

        result
    }
}

impl Drop for Compute {
    fn drop(
        &mut self
    ) {
        println!("dropping logical device");
        unsafe { self.device.device_wait_idle().unwrap() }
        for fence in &self.fences {
            unsafe { self.device.destroy_fence(fence.fence, None) }
        }
        unsafe { self.device.destroy_device(None) }
    }
}