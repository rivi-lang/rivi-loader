use rivi_loader::{DebugOption, Shader};

fn main() {

    let (_vulkan, devices) = rivi_loader::new(DebugOption::None).unwrap();
    println!("Total number of compute device(s): {}", devices.len());
    assert_eq!(devices.len(), 2);

    let a: Vec<f32> = vec![1.0, 2.0];
    let b: Vec<f32> = vec![3.0, 4.0];

    let c: Vec<f32> = vec![1.0, 2.0];
    let d: Vec<f32> = vec![1.0, 2.0];

    let input = vec![
      // gpu split
      vec![
        // core split (queue family, we use a single queue)
        vec![a, b]
      ],
      vec![vec![c, d]],
    ];

    let joined = devices
      .iter()
      .enumerate()
      .map(|(idx, c)| {
        println!("Device {}\n{:?}", idx+1, c);

        let mut cursor = std::io::Cursor::new(&include_bytes!("./repl/shader/sum.spv")[..]);
        let shader = Shader::new(c, &mut cursor).unwrap();

        let chunk = input.get(idx).unwrap();

        let res = c.execute(chunk, 2, &shader).unwrap();
        res.to_vec()

      })
      .collect::<Vec<Vec<f32>>>();

    joined
      .iter()
      .enumerate()
      .for_each(|(idx, res)| println!("Result of GPU {}: {:?}", idx+1, res));

    let expected_output: Vec<f32> = vec![4.0, 6.0, 2.0, 4.0];
    let flat_joined = joined.into_iter().flatten().collect::<Vec<_>>();
    println!("Joined result: {:?}", flat_joined);
    dbg!(flat_joined == expected_output);
}
