#[cfg(target_os = "macos")]
extern crate metal;

use std::time::Instant;
use std::fmt;

enum Operator {
  Sum
}

impl fmt::Display for Operator {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      match self {
        Operator::Sum => write!(f, "+")
      }
      // or, alternatively:
      // fmt::Debug::fmt(self, f)
  }
}

fn spirv_load(op: Operator) -> Vec<u32> {
    let mut spirv = match op {
        Operator::Sum => std::io::Cursor::new(&include_bytes!("./shader/sum.spv")[..]),
    };
    ash::util::read_spv(&mut spirv).expect("Failed to read vertex shader spv file")
}

fn main() {
    let param_a = [1.0, 2.0].to_vec();
    let param_b = [3.0, 4.0].to_vec();
    let shader = spirv_load(Operator::Sum);

    let init_timer = Instant::now();
    let (app, logical_devices) = rivi_loader::new(true).unwrap();
    println!("Found {} logical device(s)", logical_devices.len());
    println!("Found {} thread(s)", logical_devices.iter().map(|f| f.fences.len()).sum::<usize>());
    println!("App new {}ms", init_timer.elapsed().as_millis());

    // result needs a type inference -- it is likely
    // the type of either parameter
    let result_len = param_a.len();

    let ldevice = logical_devices.first().unwrap();
    let func = rivi_loader::load(&ldevice.device, &shader, 3).unwrap();

    let run_timer = Instant::now();
    let result = ldevice.infered_execute(
        vec![param_a, param_b],
        result_len,
        &func,
    );
    unsafe { func.Drop(&ldevice.device); }

    println!("Result: {:?}", result);

    println!("App execute {}ms", run_timer.elapsed().as_millis());

    let cpu_timer = Instant::now();
    let cpu_result: Vec<f32> = [1.0, 2.0].to_vec().iter().zip([3.0, 4.0].to_vec().iter()).map(|(&a, &b)| a + b).collect();
    println!("CPU execute {}ms", cpu_timer.elapsed().as_millis());

    assert_eq!(result, cpu_result);
    dbg!(cpu_result == result);
}
