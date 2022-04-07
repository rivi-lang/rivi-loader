#[test]
fn app_new() {
    let res = crate::Vulkan::new(crate::DebugOption::Validation);
    assert!(res.is_ok());
}