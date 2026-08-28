use std::env;
use std::process::Command;

fn main() {
    let out = env::var("OUT_DIR").unwrap();
    // compute_75 PTX: oldest arch we run (GTX 1650); the driver JIT-compiles
    // it for sm_89 on the server. Also keeps the server's nvcc 11.5 usable.
    let status = Command::new("nvcc")
        .args(["-ptx", "-O3", "-arch=compute_75", "src/q4.cu", "-o"])
        .arg(format!("{out}/q4.ptx"))
        .status()
        .expect("nvcc not found on PATH (needed to build ft-cuda)");
    assert!(status.success(), "nvcc failed");
    println!("cargo:rerun-if-changed=src/q4.cu");
}
