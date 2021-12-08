use std::{error::Error, time::Instant};

use rivi_loader::{DebugOption, Shader};

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
    let vk = rivi_loader::new(DebugOption::None).unwrap();
    println!("Found {} compute device(s)", vk.compute.len());
    let cores = vk.compute.iter().map(|d| d.cores()).sum::<usize>();
    println!("Found {} core(s)", cores);

    // replicate work among cores
    let input = load_input(cores);

    // bind shader to a compute device
    let compute = vk.compute.first().unwrap();
    let mut cursor = std::io::Cursor::new(&include_bytes!("./rf/shader/apply.spv")[..]);
    let shader = Shader::new(compute, &mut cursor).unwrap();

    // create upper bound for iterations
    let bound = (150.0 / cores as f32).ceil() as i32;

    let run_timer = Instant::now();
    for x in 0..bound {
        let _result = compute.execute(&input, 1_146_024, &shader).unwrap();
        println!("App {} execute {}ms", x, run_timer.elapsed().as_millis());
        // to check the results below against precomputed answer (slow)
        //dbg!((_result.iter().sum::<f32>() - 490058.0*cores as f32).abs() < 0.1);
    }
    println!("App executions {}ms", run_timer.elapsed().as_millis());
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