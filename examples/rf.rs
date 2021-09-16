#[cfg(target_os = "macos")]
extern crate metal;

use std::{error::Error, time::Instant};
use std::process;

use rivi_loader::spirv::SPIRV;

fn example(f: &str, v: &mut Vec<f32>) -> Result<(), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(f)
        .expect("Cannot read fild");
    for record in reader.records() {
        let record = record?;
        for field in record.iter() {
            let n: f32 = field.parse().unwrap();
            v.push(n);
        }
    }
    Ok(())
}

fn main() {

    let mut feature: Vec<f32> = Vec::new();
    if let Err(err) = example("examples/dataset/feature.csv", &mut feature) {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let mut th: Vec<f32> = Vec::new();
    if let Err(err) = example("examples/dataset/threshold.csv", &mut th) {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let mut left: Vec<f32> = Vec::new();
    if let Err(err) = example("examples/dataset/left.csv", &mut left) {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let mut right: Vec<f32> = Vec::new();
    if let Err(err) = example("examples/dataset/right.csv", &mut right) {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let mut values: Vec<f32> = Vec::new();
    if let Err(err) = example("examples/dataset/values.csv", &mut values) {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let mut x: Vec<f32> = Vec::new();
    if let Err(err) = example("examples/dataset/x.csv", &mut x) {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let params: Vec<_> = (0..NUM).into_iter().map(|_|
        vec![left.clone(), right.clone(), th.clone(), feature.clone(), values.clone(), x.clone()]
    ).collect();

    run(params);
}

const NUM: i32 = 32;

fn run(input: Vec<Vec<Vec<f32>>>) {

    let init_timer = Instant::now();

    unsafe {

        let (app, logical_devices) = rivi_loader::new(true).unwrap();
        println!("Found {} logical device(s)", logical_devices.len());
        println!("Found {} thread(s)", logical_devices.iter().map(|f| f.fences.len()).sum::<usize>());
        println!("App new {}ms", init_timer.elapsed().as_millis());

        let mut spirv = std::io::Cursor::new(&include_bytes!("./shader/apply.spv")[..]);
        let shader = SPIRV::new(&mut spirv).unwrap();
        println!("App load {}ms", init_timer.elapsed().as_millis());

        //assert_eq!(NUM % app.logical_devices.iter().map(|f| f.fences.len()).sum::<usize>() as i32, 0);

        let ldevice = logical_devices.first().unwrap();

        let run_timer = Instant::now();
        for x in 0..5 {

            let _result = ldevice.execute(
                &input,
                1146024,
                &shader,
                &ldevice.fences
            );
            println!("App {} execute {}ms", x, run_timer.elapsed().as_millis());

            //dbg!(_result.iter().sum::<f32>() == 490058.0*NUM as f32);

        }
        println!("App executions {}ms", run_timer.elapsed().as_millis());

        println!("Total time {}ms", init_timer.elapsed().as_millis());
    }
}