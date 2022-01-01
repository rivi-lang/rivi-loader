use rivi_loader::DebugOption;

fn main() {
    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0];
    let input = &vec![vec![a, b]];
    let mut output = vec![0.0f32; 2];

    let vk = rivi_loader::new(DebugOption::None).unwrap();

    let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
    let shader = vk.load_shader(&mut cursor).unwrap();

    vk.compute(input, &mut output, &shader).unwrap();

    println!("Result: {:?}", output);
    assert_eq!(output, vec![4.0, 6.0]);
}
