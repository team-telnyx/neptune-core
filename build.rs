fn main() {
    // Tell Cargo to rerun this build script if the build.rs file changes
    println!("cargo:rerun-if-changed=build.rs");

    // Tell Cargo to rerun this build script if the environment variables change
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    // Set a feature flag to indicate we're using the mock implementation
    println!("cargo:rustc-cfg=feature=\"hip_mock\"");
}
