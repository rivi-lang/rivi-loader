use rivi_loader::{debug_layer::DebugOption, shader::Shader};


fn main() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let input = &vec![vec![a, b]];
    let expected_output = vec![4.0, 6.0];
    let out_length = expected_output.len();

    let (_vulkan, devices) = rivi_loader::new(DebugOption::None).unwrap();
    println!("Found {} compute device(s)", devices.len());
    println!("Found {} core(s)", devices.iter().map(|f| f.fences.len()).sum::<usize>());

    let compute = devices.first().unwrap();
    let cores = &compute.fences[0..1];

    let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
    let shader = Shader::new(compute, &mut cursor).unwrap();

    let result = compute.execute(input, out_length, &shader, cores);

    println!("Result: {:?}", result);
    assert_eq!(result, expected_output);
}
