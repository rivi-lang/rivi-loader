use std::{error::Error, time::Instant};

use rivi_loader::debug_layer::DebugOption;
use rivi_loader::shader::Shader;


const NUM: i32 = 32;

fn main() {

    let input = load_input();

    let (_vulkan, devices) = rivi_loader::new(DebugOption::None).unwrap();
    println!("Found {} compute device(s)", devices.len());
    println!("Found {} core(s)", devices.iter().map(|f| f.fences.len()).sum::<usize>());

    // ensure the work is evenly split among cores
    assert_eq!(NUM % devices.iter().map(|f| f.fences.len()).sum::<usize>() as i32, 0);

    let compute = devices.first().unwrap();
    let cores = &compute.fences;

    let mut cursor = std::io::Cursor::new(&include_bytes!("./rf/shader/apply.spv")[..]);
    let shader = Shader::new(compute, &mut cursor).unwrap();

    let run_timer = Instant::now();
    for x in 0..5 {
        let _result = compute.execute(&input, 1146024, &shader, cores);
        println!("App {} execute {}ms", x, run_timer.elapsed().as_millis());
        //dbg!((_result.iter().sum::<f32>() - 490058.0*NUM as f32).abs() < 0.1);
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

fn load_input() -> Vec<Vec<Vec<f32>>> {

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

    (0..NUM).into_iter().map(|_| vec![
        left.clone(),
        right.clone(),
        th.clone(),
        feature.clone(),
        values.clone(),
        x.clone()
    ]).collect()
}