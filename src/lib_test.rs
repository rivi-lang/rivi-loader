#[test]
fn app_new() {
    let res = crate::new(crate::DebugOption::Validation);
    assert!(res.is_ok());
}