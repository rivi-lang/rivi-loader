use std::error::Error;

use rivi_loader::{DebugOption, GroupCount, Task, Vulkan};
use rayon::prelude::*;

/// `rf.rs` runs Python Scikit derived random forest prediction algorithm.
/// The implementation of this algorithm was derived from Python/Cython to APL, and
/// then hand-translated from APL to SPIR-V.
///
/// The baseline used 150 iterations of the prediction. This file replicates
/// that functionality by load-balancing it among all fences found.
///
/// The whole ordeal is further elaborated here: https://hal.inria.fr/hal-03155647/
fn main() {
    // initialize vulkan process
    let vk = Vulkan::new(DebugOption::None).unwrap();
    // bind shader to a compute device
    let binary = &include_bytes!("./rf/shader/apply.spv")[..];
    let shader = rspirv::dr::load_bytes(binary).unwrap();

    loop {
        println!("Batched runtime: {}ms", batched(&vk, &shader));
    }
}

fn batched(vk: &rivi_loader::Vulkan, shader: &rspirv::dr::Module) -> u128 {

    let gpu = vk.compute.as_ref().unwrap().first().unwrap();
    let threads = gpu.fences.as_ref().unwrap().len();

    // replicate work among cores
    let dataset = load_input(150);

    // create upper bound for iterations
    let bound = (150.0 / threads as f32).ceil() as i32;

    (0..bound).map(|_| {

        let specializations = Vec::new();
        let shader = rivi_loader::load_shader(gpu, shader.clone(), specializations).unwrap();

        let time = gpu.fences.as_ref().unwrap().par_iter().map(|fence| {

            let mut tasks = fence.queues.iter().map(|queue| {
                Task {
                    input: dataset[bound as usize].clone(),
                    output: vec![0.0f32; 1_146_024],
                    push_constants: vec![],
                    queue: *queue,
                    group_count: GroupCount {
                        x: 1024,
                        ..Default::default()
                    },
                }
            })
            .collect::<Vec<_>>();

            let run_timer = std::time::Instant::now();
            gpu.scheduled(&shader, fence, &mut tasks).unwrap();
            let end_timer = run_timer.elapsed().as_millis();

            tasks.into_iter().for_each(|t| assert_eq!(t.output.into_iter().map(|f| f as f64).sum::<f64>(), 490058.0_f64));

            end_timer
        }).collect::<Vec<_>>();

        time.iter().sum::<u128>() / gpu.fences.as_ref().unwrap().len() as u128

    }).sum()
}

fn csv(f: &str, v: &mut Vec<f32>) -> Result<(), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(f)?;
    for record in reader.records() {
        let record = record?;
        for field in record.iter() {
            let n: f32 = field.parse()?;
            v.push(n);
        }
    }
    Ok(())
}

fn load_input(chunks: usize) -> Vec<Vec<Vec<f32>>> {

    let mut feature: Vec<f32> = Vec::new();
    if let Err(err) = csv("examples/rf/dataset/feature.csv", &mut feature) {
        panic!("error running example: {}", err);
    }

    let mut th: Vec<f32> = Vec::new();
    if let Err(err) = csv("examples/rf/dataset/threshold.csv", &mut th) {
        panic!("error running example: {}", err);
    }

    let mut left: Vec<f32> = Vec::new();
    if let Err(err) = csv("examples/rf/dataset/left.csv", &mut left) {
        panic!("error running example: {}", err);
    }

    let mut right: Vec<f32> = Vec::new();
    if let Err(err) = csv("examples/rf/dataset/right.csv", &mut right) {
        panic!("error running example: {}", err);
    }

    let mut values: Vec<f32> = Vec::new();
    if let Err(err) = csv("examples/rf/dataset/values.csv", &mut values) {
        panic!("error running example: {}", err);
    }

    let mut x: Vec<f32> = Vec::new();
    if let Err(err) = csv("examples/rf/dataset/x.csv", &mut x) {
        panic!("error running example: {}", err);
    }

    (0..chunks).into_iter().map(|_| vec![
        left.clone(),
        right.clone(),
        th.clone(),
        feature.clone(),
        values.clone(),
        x.clone()
    ]).collect()
}