use std::{error::Error, time::Instant};

use rivi_loader::DebugOption;

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
    // bind shader to a compute device
    let mut cursor = std::io::Cursor::new(&include_bytes!("./rf/shader/apply.spv")[..]);
    let shader = vk.load_shader(&mut cursor).unwrap();

    loop {
        let a = batched(&vk, &shader);
        let b = at_once(&vk, &shader);
        println!("Batched runtime: {}ms", a);
        println!("At once runtime: {}ms", b);
    }
}

fn batched(vk: &rivi_loader::Vulkan, shader: &rivi_loader::Shader) -> u128 {

    // replicate work among cores
    let input = load_input(vk.threads());

    // create upper bound for iterations
    let bound = (150.0 / vk.threads() as f32).ceil() as i32;

    (0..bound).map(|_| {

        let mut output = vec![0.0f32; 1_146_024 * vk.threads()];

        let run_timer = Instant::now();
        vk.compute(&input, &mut output, shader).unwrap();
        let end_timer = run_timer.elapsed().as_millis();

        // to check the results below against precomputed answer
        assert_eq!(output.into_iter().map(|f| f as f64).sum::<f64>(), 490058.0*vk.threads() as f64);

        end_timer
    }).sum()
}

fn at_once(vk: &rivi_loader::Vulkan, shader: &rivi_loader::Shader) -> u128 {

    // replicate work among cores
    let input = load_input(150);
    let mut output = vec![0.0f32; 1_146_024 * 150];

    let run_timer = Instant::now();
    vk.compute(&input, &mut output, shader).unwrap();
    let end_timer = run_timer.elapsed().as_millis();

    // to check the results below against precomputed answer
    assert_eq!(output.into_iter().map(|f| f as f64).sum::<f64>(), 490058.0*150 as f64);

    end_timer
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