use rivi_loader::{DebugOption, Shader};

fn main() {
    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0];
    let input = &vec![vec![a, b]];
    let expected_output: Vec<f32> = vec![4.0, 6.0];
    let out_length = expected_output.len();

    let (_vulkan, devices) = rivi_loader::new(DebugOption::None).unwrap();
    println!("Found {} compute device(s)", devices.len());
    println!("Found {} core(s)", devices.iter().map(|d| d.cores()).sum::<usize>());

    let compute = devices.first().unwrap();

    let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
    let shader = Shader::new(compute, &mut cursor).unwrap();

    let result = compute.execute(input, out_length, &shader).unwrap();

    println!("Result: {:?}", result);
    assert_eq!(result, expected_output);
}
