use std::sync::Arc;
use std::time::Duration;
use std::thread;
use rocm_sys::*;
use hip_runtime_sys::*;
use twenty_first::math::digest::Digest;
use tracing::*;
use rand::Rng;

use super::gpu_kernel::MiningKernel;

pub struct GpuMiner {
    device: i32,
    context: Arc<Context>,
    stream: Stream,
    device_name: String,
    compute_units: i32,
}

struct Context {
    _context: (),
}

impl Context {
    fn new() -> Result<Self, String> {
        // ROCm/HIP doesn't have explicit context management like CUDA
        Ok(Self { _context: () })
    }
}

struct Stream {
    stream: hipStream_t,
}

impl Stream {
    fn new(flags: u32) -> Result<Self, String> {
        unsafe {
            let mut stream = std::ptr::null_mut();
            if hipStreamCreate(&mut stream) != hipSuccess {
                return Err("Failed to create HIP stream".to_string());
            }
            Ok(Self { stream })
        }
    }

    fn destroy(&self) -> Result<(), String> {
        unsafe {
            if hipStreamDestroy(self.stream) != hipSuccess {
                return Err("Failed to destroy HIP stream".to_string());
            }
            Ok(())
        }
    }
}

impl GpuMiner {
    pub fn new(device_id: i32) -> Result<Self, String> {
        unsafe {
            let mut device_count = 0;
            if hipGetDeviceCount(&mut device_count) != hipSuccess {
                return Err("Failed to get GPU device count".to_string());
            }
            
            if device_count == 0 {
                return Err("No GPU devices found".to_string());
            }
            
            if device_id >= device_count {
                return Err(format!("Invalid device ID {}. Available devices: {}", device_id, device_count));
            }

            if hipSetDevice(device_id) != hipSuccess {
                return Err("Failed to set GPU device".to_string());
            }

            // Get device name and compute units
            let mut name_buffer = [0u8; 256];
            if hipDeviceGetName(name_buffer.as_mut_ptr() as *mut i8, name_buffer.len() as i32, device_id) != hipSuccess {
                return Err("Failed to get device name".to_string());
            }
            
            let device_name = std::ffi::CStr::from_ptr(name_buffer.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned();
                
            let mut compute_units = 0;
            if hipDeviceGetAttribute(&mut compute_units, hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount, device_id) != hipSuccess {
                return Err("Failed to get compute unit count".to_string());
            }

            let context = Arc::new(Context::new()?);
            let stream = Stream::new(hipStreamNonBlocking)?;

            info!("Initialized GPU miner on device {}: {} with {} compute units", device_id, device_name, compute_units);

            Ok(Self {
                device: device_id,
                context,
                stream,
                device_name,
                compute_units,
            })
        }
    }
    
    pub fn get_device_info(&self) -> (i32, &str, i32) {
        (self.device, &self.device_name, self.compute_units)
    }

    pub fn mine_block(
        &self,
        kernel_auth_path: [Digest; 2],
        header_auth_path: [Digest; 3],
        threshold: Digest,
        difficulty: u64,
    ) -> Result<Option<Digest>, String> {
        unsafe {
            // Allocate device memory for auth paths
            let mut d_kernel_auth_path = std::ptr::null_mut();
            let mut d_header_auth_path = std::ptr::null_mut();
            let mut d_threshold = std::ptr::null_mut();
            let mut d_result = std::ptr::null_mut();
            let mut d_found_nonce = std::ptr::null_mut();
            
            let kernel_size = std::mem::size_of::<Digest>() * 2;
            let header_size = std::mem::size_of::<Digest>() * 3;
            let digest_size = std::mem::size_of::<Digest>();
            let result_size = std::mem::size_of::<u64>();
            
            // Allocate memory on device
            if hipMalloc(&mut d_kernel_auth_path, kernel_size) != hipSuccess ||
               hipMalloc(&mut d_header_auth_path, header_size) != hipSuccess ||
               hipMalloc(&mut d_threshold, digest_size) != hipSuccess ||
               hipMalloc(&mut d_result, result_size) != hipSuccess ||
               hipMalloc(&mut d_found_nonce, digest_size) != hipSuccess {
                self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                return Err("Failed to allocate device memory".to_string());
            }
            
            // Copy data to device
            if hipMemcpy(
                d_kernel_auth_path,
                kernel_auth_path.as_ptr() as *const _,
                kernel_size,
                hipMemcpyHostToDevice,
            ) != hipSuccess ||
            hipMemcpy(
                d_header_auth_path,
                header_auth_path.as_ptr() as *const _,
                header_size,
                hipMemcpyHostToDevice,
            ) != hipSuccess ||
            hipMemcpy(
                d_threshold,
                &threshold as *const _ as *const _,
                digest_size,
                hipMemcpyHostToDevice,
            ) != hipSuccess {
                self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                return Err("Failed to copy data to device".to_string());
            }
            
            // Initialize result to 0 (not found)
            let result_init: u64 = 0;
            if hipMemcpy(
                d_result,
                &result_init as *const _ as *const _,
                result_size,
                hipMemcpyHostToDevice,
            ) != hipSuccess {
                self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                return Err("Failed to initialize result memory".to_string());
            }
            
            // Generate random nonce start
            let nonce_start = rand::thread_rng().gen::<u64>();
            let nonce_range = 1_000_000_000; // 1 billion nonces per kernel launch
            
            // Create and launch kernel
            let kernel = MiningKernel::new(difficulty, nonce_start, nonce_range);
            
            info!("Launching GPU mining kernel with {} nonces starting from {}", nonce_range, nonce_start);
            
            if let Err(e) = kernel.launch(
                self.stream.stream,
                d_kernel_auth_path as *const _,
                d_header_auth_path as *const _,
                d_threshold as *const _,
                d_result as *mut _,
                d_found_nonce as *mut _,
            ) {
                self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                return Err(e);
            }
            
            // Wait for kernel to complete
            if hipStreamSynchronize(self.stream.stream) != hipSuccess {
                self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                return Err("Failed to synchronize HIP stream".to_string());
            }
            
            // Check if a nonce was found
            let mut result: u64 = 0;
            if hipMemcpy(
                &mut result as *mut _ as *mut _,
                d_result,
                result_size,
                hipMemcpyDeviceToHost,
            ) != hipSuccess {
                self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                return Err("Failed to copy result from device".to_string());
            }
            
            let found_nonce = if result == 1 {
                // A valid nonce was found, copy it back
                let mut nonce = Digest::default();
                if hipMemcpy(
                    &mut nonce as *mut _ as *mut _,
                    d_found_nonce,
                    digest_size,
                    hipMemcpyDeviceToHost,
                ) != hipSuccess {
                    self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
                    return Err("Failed to copy nonce from device".to_string());
                }
                Some(nonce)
            } else {
                None
            };
            
            // Free device memory
            self.free_device_memory(d_kernel_auth_path, d_header_auth_path, d_threshold, d_result, d_found_nonce);
            
            Ok(found_nonce)
        }
    }
    
    unsafe fn free_device_memory(
        &self,
        d_kernel_auth_path: *mut ::std::os::raw::c_void,
        d_header_auth_path: *mut ::std::os::raw::c_void,
        d_threshold: *mut ::std::os::raw::c_void,
        d_result: *mut ::std::os::raw::c_void,
        d_found_nonce: *mut ::std::os::raw::c_void,
    ) {
        if !d_kernel_auth_path.is_null() {
            hipFree(d_kernel_auth_path);
        }
        if !d_header_auth_path.is_null() {
            hipFree(d_header_auth_path);
        }
        if !d_threshold.is_null() {
            hipFree(d_threshold);
        }
        if !d_result.is_null() {
            hipFree(d_result);
        }
        if !d_found_nonce.is_null() {
            hipFree(d_found_nonce);
        }
    }
    
    // Get the number of available GPU devices
    pub fn get_device_count() -> Result<i32, String> {
        unsafe {
            let mut device_count = 0;
            if hipGetDeviceCount(&mut device_count) != hipSuccess {
                return Err("Failed to get GPU device count".to_string());
            }
            Ok(device_count)
        }
    }
}

impl Drop for GpuMiner {
    fn drop(&mut self) {
        unsafe {
            // Cleanup resources
            self.stream.destroy().ok();
        }
    }
}