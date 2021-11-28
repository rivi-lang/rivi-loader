#[test]
fn app_new() {
    let init_timer = std::time::Instant::now();
    let res = crate::new(crate::DebugOption::Validation);
    assert!(res.is_ok());
    let (_app, devices) = res.unwrap();
    println!("Found {} logical device(s)", devices.len());
    println!("Found {} thread(s)", devices.iter().map(|f| f.fences.len()).sum::<usize>());
    println!("App new {}ms", init_timer.elapsed().as_millis());
}