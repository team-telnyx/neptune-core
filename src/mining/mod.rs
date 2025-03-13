// GPU mining implementation
mod gpu_kernel;
mod gpu_miner;
mod hip_mock;

// Export the GPU miner implementation
pub use gpu_miner::GpuMiner;
pub use gpu_kernel::create_gpu_kernel_source;