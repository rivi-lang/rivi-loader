use rivi_loader::DebugOption;

fn main() {
    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0];
    let input = &vec![vec![a, b]];
    let expected_output: Vec<f32> = vec![4.0, 6.0];
    let out_length = expected_output.len();

    let vk = rivi_loader::new(DebugOption::None).unwrap();

    let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
    let shaders = vk.load_shader(&mut cursor).unwrap();
    let shader = shaders.first().unwrap();

    let result = vk.compute(input, out_length, shader);

    println!("Result: {:?}", result);
    assert_eq!(result, expected_output);
}
