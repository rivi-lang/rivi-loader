#[cfg(target_os = "macos")]
extern crate metal;

use std::convert::TryInto;
use std::error::Error;
use std::time::Instant;
use std::fmt;

use rivi_loader::spirv::SPIRV;

enum Operator {
  Sum
}

impl fmt::Display for Operator {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      match self {
        Operator::Sum => write!(f, "+")
      }
  }
}

fn spirv_load(op: Operator) -> Result<SPIRV, Box<dyn Error>> {
    match op {
      Operator::Sum => {
        let mut spirv = std::io::Cursor::new(&include_bytes!("./shader/sum.spv")[..]);
        SPIRV::new(&mut spirv)
      },
      _ => todo!("operator not implemented")
    }
}

fn main() {

    unsafe {

      let param_a = [1.0, 2.0].to_vec();
      let param_b = [3.0, 4.0].to_vec();
      let shader = spirv_load(Operator::Sum).expect("could not load spirv");

      let init_timer = Instant::now();
      let (app, logical_devices) = rivi_loader::new(true).unwrap();
      println!("Found {} logical device(s)", logical_devices.len());
      println!("Found {} thread(s)", logical_devices.iter().map(|f| f.fences.len()).sum::<usize>());
      println!("App new {}ms", init_timer.elapsed().as_millis());

      // result needs a type inference -- it is likely
      // the type of either parameter
      let result_len = param_a.len();

      let ldevice = logical_devices.first().unwrap();

      let run_timer = Instant::now();
      let result = ldevice.execute(
          &vec![vec![param_a, param_b]],
          result_len,
          &shader,
          &ldevice.fences[0..1],
      );

      println!("Result: {:?}", result);

      println!("App execute {}ms", run_timer.elapsed().as_millis());

      let cpu_timer = Instant::now();
      let cpu_result: Vec<f32> = [1.0, 2.0].to_vec().iter().zip([3.0, 4.0].to_vec().iter()).map(|(&a, &b)| a + b).collect();
      println!("CPU execute {}ms", cpu_timer.elapsed().as_millis());

      assert_eq!(result, cpu_result);
      dbg!(cpu_result == result);

  }
}
