// Use mock implementation for testing
mod mock_impl;

pub use mock_impl::GpuMiner;
pub use mock_impl::create_gpu_kernel_source;