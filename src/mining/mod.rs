// GPU mining implementation
mod gpu_kernel;
mod gpu_miner;

// Mock implementation for testing
#[cfg(feature = "hip_mock")]
mod hip_mock;

// Export the GPU miner implementation
pub use gpu_kernel::create_gpu_kernel_source;
pub use gpu_miner::GpuMiner;

// Re-export hip_mock for use in other modules when the feature is enabled
#[cfg(feature = "hip_mock")]
pub use hip_mock::*;
