use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
// Use our mock implementation instead of the real HIP runtime
use super::hip_mock::*;
use crate::prelude::twenty_first::math::digest::Digest;
use rand::Rng;
use tracing::*;

use super::gpu_kernel::{create_gpu_kernel_source, MiningKernel};

// Flag to track if we've attempted to compile the GPU kernel
static mut GPU_KERNEL_COMPILED: bool = false;

pub struct GpuMiner {
    device: i32,
    context: Arc<Context>,
    stream: Stream,
    device_name: String,
    compute_units: i32,
    is_using_gpu: bool,
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
    // Get the number of available GPU devices
    pub fn get_device_count() -> Result<i32, String> {
        unsafe {
            let mut device_count = 0;
            let status = hipGetDeviceCount(&mut device_count);
            if status != hipSuccess {
                error!("Failed to get GPU device count: error code {}", status);
                return Err(format!(
                    "Failed to get GPU device count: error code {}",
                    status
                ));
            }
            Ok(device_count)
        }
    }

    // Log information about a specific GPU device
    unsafe fn log_device_info(device_id: i32) -> Result<(), String> {
        let mut name_buffer = [0u8; 256];
        if hipDeviceGetName(
            name_buffer.as_mut_ptr() as *mut i8,
            name_buffer.len() as i32,
            device_id,
        ) != hipSuccess
        {
            return Err(format!("Failed to get device name for device {}", device_id));
        }

        let device_name = std::ffi::CStr::from_ptr(name_buffer.as_ptr() as *const i8)
            .to_string_lossy()
            .into_owned();

        let mut compute_units = 0;
        if hipDeviceGetAttribute(
            &mut compute_units,
            hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount,
            device_id,
        ) != hipSuccess
        {
            warn!("Failed to get compute unit count for device {}", device_id);
        }

        let mut clock_rate = 0;
        if hipDeviceGetAttribute(
            &mut clock_rate,
            hipDeviceAttribute_t_hipDeviceAttributeClockRate,
            device_id,
        ) != hipSuccess
        {
            warn!("Failed to get clock rate for device {}", device_id);
        }

        let mut total_mem = 0;
        hipDeviceTotalMem(&mut total_mem, device_id);

        // Get PCI bus ID
        let mut pci_bus_id = [0u8; 64];
        if hipDeviceGetPCIBusId(
            pci_bus_id.as_mut_ptr() as *mut i8,
            pci_bus_id.len() as i32,
            device_id,
        ) != hipSuccess
        {
            warn!("Failed to get PCI bus ID for device {}", device_id);
        }

        let pci_id = std::ffi::CStr::from_ptr(pci_bus_id.as_ptr() as *const i8)
            .to_string_lossy()
            .into_owned();

        info!("🔍 GPU Device #{}: {} (PCI: {})", device_id, device_name, pci_id);
        info!("  - Compute Units: {}", compute_units);
        info!("  - Total Memory: {} MB", total_mem / (1024 * 1024));
        info!("  - Clock Rate: {} MHz", clock_rate / 1000);

        // Get architecture information
        let mut major = 0;
        let mut minor = 0;
        if hipDeviceGetAttribute(
            &mut major,
            hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor,
            device_id,
        ) == hipSuccess
            && hipDeviceGetAttribute(
                &mut minor,
                hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor,
                device_id,
            ) == hipSuccess
        {
            info!("  - Architecture: {}.{}", major, minor);
        }

        // Get warp size
        let mut warp_size = 0;
        if hipDeviceGetAttribute(
            &mut warp_size,
            hipDeviceAttribute_t_hipDeviceAttributeWarpSize,
            device_id,
        ) == hipSuccess
        {
            info!("  - Warp Size: {}", warp_size);
        }

        // Get max threads per block
        let mut max_threads = 0;
        if hipDeviceGetAttribute(
            &mut max_threads,
            hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerBlock,
            device_id,
        ) == hipSuccess
        {
            info!("  - Max Threads Per Block: {}", max_threads);
        }

        // Get max shared memory per block
        let mut max_shared_mem = 0;
        if hipDeviceGetAttribute(
            &mut max_shared_mem,
            hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerBlock,
            device_id,
        ) == hipSuccess
        {
            info!("  - Max Shared Memory Per Block: {} KB", max_shared_mem / 1024);
        }

        // Get ROCm device ID
        let mut rocm_device_id = 0;
        if hipDeviceGetAttribute(
            &mut rocm_device_id,
            hipDeviceAttribute_t_hipDeviceAttributeDeviceId,
            device_id,
        ) == hipSuccess
        {
            info!("  - ROCm Device ID: {}", rocm_device_id);
        }

        // Check if this is an MI100 GPU
        if device_name.contains("MI100") {
            info!("  - 🚀 Detected AMD MI100 GPU!");
        }

        // Get HIP runtime and driver versions
        let mut runtime_version = 0;
        if hipRuntimeGetVersion(&mut runtime_version) == hipSuccess {
            info!("  - HIP Runtime Version: {}", runtime_version);
        }

        let mut driver_version = 0;
        if hipDriverGetVersion(&mut driver_version) == hipSuccess {
            info!("  - HIP Driver Version: {}", driver_version);
        }

        Ok(())
    }

    pub fn new(device_id: i32) -> Result<Self, String> {
        unsafe {
            // Get device count and log information about available devices
            let device_count = Self::get_device_count()?;

            info!(
                "Initializing GPU miner. Found {} GPU device(s)",
                device_count
            );

            if device_count == 0 {
                return Err("No GPU devices found".to_string());
            }

            // Log information about all available devices
            for dev_id in 0..device_count {
                Self::log_device_info(dev_id)?;
            }

            // Validate the requested device ID
            if device_id >= device_count {
                return Err(format!(
                    "Invalid device ID {}. Available devices: {}",
                    device_id, device_count
                ));
            }

            // Set the active device
            info!("Setting active GPU device to {}", device_id);
            let status = hipSetDevice(device_id);
            if status != hipSuccess {
                error!(
                    "Failed to set GPU device {}: error code {}",
                    device_id, status
                );
                return Err(format!(
                    "Failed to set GPU device {}: error code {}",
                    device_id, status
                ));
            }

            // Get device name and compute units
            let mut name_buffer = [0u8; 256];
            if hipDeviceGetName(
                name_buffer.as_mut_ptr() as *mut i8,
                name_buffer.len() as i32,
                device_id,
            ) != hipSuccess
            {
                return Err("Failed to get device name".to_string());
            }

            let device_name = std::ffi::CStr::from_ptr(name_buffer.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned();

            let mut compute_units = 0;
            if hipDeviceGetAttribute(
                &mut compute_units,
                hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount,
                device_id,
            ) != hipSuccess
            {
                return Err("Failed to get compute unit count".to_string());
            }

            // Create context and stream
            info!("Creating HIP context and stream for device {}", device_id);
            let context = Arc::new(Context::new()?);
            let stream = Stream::new(hipStreamNonBlocking)?;

            // Try to compile the GPU kernel if not already done
            info!("Ensuring GPU kernel is compiled and available");
            let is_using_gpu = Self::ensure_gpu_kernel_compiled();

            if is_using_gpu {
                info!("✅ Successfully initialized GPU miner on device {}: {} with {} compute units (using GPU acceleration)", 
                    device_id, device_name, compute_units);
            } else {
                warn!("⚠️ Initialized GPU miner on device {}: {} with {} compute units (using CPU fallback)", 
                    device_id, device_name, compute_units);
            }

            Ok(Self {
                device: device_id,
                context,
                stream,
                device_name,
                compute_units,
                is_using_gpu,
            })
        }
    }

    // Ensure the GPU kernel is compiled and available
    // This function will now panic instead of falling back to CPU if GPU initialization fails
    unsafe fn ensure_gpu_kernel_compiled() -> bool {
        if !GPU_KERNEL_COMPILED {
            GPU_KERNEL_COMPILED = true;

            // Check if the kernel file already exists
            if !Path::new("mining_kernel.hsaco").exists() {
                info!("No pre-compiled GPU kernel found, will generate and compile one");

                // Generate the kernel source code
                let kernel_source = create_gpu_kernel_source();
                info!(
                    "Generated GPU kernel source code ({} bytes)",
                    kernel_source.len()
                );

                // Write the kernel source to a file
                match File::create("mining_kernel.cpp") {
                    Ok(mut file) => {
                        match file.write_all(kernel_source.as_bytes()) {
                            Ok(_) => info!("Successfully wrote kernel source to mining_kernel.cpp"),
                            Err(e) => {
                                error!("FATAL: Failed to write kernel source: {}", e);
                                panic!("GPU initialization failed: Cannot write kernel source file: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("FATAL: Failed to create kernel source file: {}", e);
                        panic!(
                            "GPU initialization failed: Cannot create kernel source file: {}",
                            e
                        );
                    }
                }

                // Check if hipcc is available
                let hipcc_check = std::process::Command::new("which").arg("hipcc").output();

                match hipcc_check {
                    Ok(output) => {
                        if !output.status.success() {
                            error!("FATAL: hipcc compiler not found in PATH. Cannot compile GPU kernel.");
                            panic!("GPU initialization failed: hipcc compiler not found in PATH");
                        }
                        let hipcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        info!("Found hipcc compiler at: {}", hipcc_path);
                    }
                    Err(e) => {
                        error!("FATAL: Failed to check for hipcc compiler: {}", e);
                        panic!(
                            "GPU initialization failed: Cannot check for hipcc compiler: {}",
                            e
                        );
                    }
                }

                // Get ROCm version for compiler flags
                let rocm_version = std::process::Command::new("sh")
                    .arg("-c")
                    .arg("rocm-smi --showdriverversion | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+' || echo 'unknown'")
                    .output()
                    .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                    .unwrap_or_else(|_| "unknown".to_string());

                info!("Detected ROCm version: {}", rocm_version);

                // Check for AMD MI100 GPUs and add specific optimization flags
                let mut has_mi100 = false;
                let mut device_count = 0;
                if hipGetDeviceCount(&mut device_count) == hipSuccess {
                    for device_id in 0..device_count {
                        let mut name_buffer = [0u8; 256];
                        if hipDeviceGetName(
                            name_buffer.as_mut_ptr() as *mut i8,
                            name_buffer.len() as i32,
                            device_id,
                        ) == hipSuccess
                        {
                            let device_name =
                                std::ffi::CStr::from_ptr(name_buffer.as_ptr() as *const i8)
                                    .to_string_lossy();

                            if device_name.contains("MI100") {
                                has_mi100 = true;
                                info!("🚀 Detected AMD MI100 GPU, will use specific optimizations");
                                break;
                            }
                        }
                    }
                }

                // Prepare compiler flags based on detected hardware
                let mut compiler_args = vec![
                    "--genco".to_string(),
                    "-O3".to_string(),
                    "-fgpu-rdc".to_string(),
                ];

                // Get available GPU architectures from hipcc
                let arch_output = std::process::Command::new("hipcc")
                    .arg("--help")
                    .output()
                    .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
                    .unwrap_or_else(|_| String::new());

                // Try alternative method if the first one doesn't work
                let arch_output = if arch_output.is_empty() || !arch_output.contains("--offload-arch") {
                    info!("Trying alternative method to detect supported architectures");
                    std::process::Command::new("sh")
                        .arg("-c")
                        .arg("hipconfig --version || echo 'unknown'")
                        .output()
                        .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
                        .unwrap_or_else(|_| String::new())
                } else {
                    arch_output
                };

                // Check if gfx908 is supported
                let has_gfx908_support = arch_output.contains("gfx908");
                
                // If we still can't determine, try to check if we're on ROCm 5.0 or newer
                let has_gfx908_support = if !has_gfx908_support {
                    let rocm_version = std::process::Command::new("sh")
                        .arg("-c")
                        .arg("rocm-smi --showdriverversion | grep -oE '[0-9]+\\.[0-9]+' || echo '0.0'")
                        .output()
                        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
                        .unwrap_or_else(|_| "0.0".to_string());
                    
                    // Parse major and minor version
                    let parts: Vec<&str> = rocm_version.split('.').collect();
                    if parts.len() >= 2 {
                        if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                            // ROCm 5.0 and newer should support gfx908
                            let supports_gfx908 = major >= 5;
                            info!("Detected ROCm version {}.{}, gfx908 support: {}", major, minor, supports_gfx908);
                            supports_gfx908
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    has_gfx908_support
                };
                
                // Add MI100-specific optimizations if detected and supported
                if has_mi100 {
                    info!("Detected AMD MI100 GPU, checking for gfx908 architecture support");
                    
                    if has_gfx908_support {
                        info!("gfx908 architecture is supported by hipcc, using MI100-specific optimizations");
                        compiler_args.push("-march=gfx908".to_string());
                        compiler_args.push("-mcumode".to_string());
                        compiler_args.push("--offload-arch=gfx908".to_string());
                        compiler_args.push("-mwavefrontsize64".to_string());
                    } else {
                        // Use generic AMD GPU flags instead
                        info!("gfx908 architecture not supported by hipcc, using generic AMD GPU flags");
                        compiler_args.push("--amdgpu-target=gfx900".to_string()); // More widely supported
                    }
                }

                // Add output file and source file
                compiler_args.push("-o".to_string());
                compiler_args.push("mining_kernel.hsaco".to_string());
                compiler_args.push("mining_kernel.cpp".to_string());

                // Try to compile the kernel using hipcc with appropriate flags
                info!(
                    "Compiling GPU kernel with hipcc using flags: {:?}",
                    compiler_args
                );
                let compile_output = std::process::Command::new("hipcc")
                    .args(&compiler_args)
                    .output();

                match compile_output {
                    Ok(output) => {
                        if !output.status.success() {
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            error!("FATAL: Failed to compile GPU kernel: {}", stderr);
                            panic!(
                                "GPU initialization failed: Cannot compile GPU kernel: {}",
                                stderr
                            );
                        }
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        if !stdout.is_empty() {
                            info!("Compiler output: {}", stdout);
                        }
                        info!("Successfully compiled GPU kernel to mining_kernel.hsaco");
                    }
                    Err(e) => {
                        error!("FATAL: Failed to execute hipcc: {}", e);
                        panic!("GPU initialization failed: Cannot execute hipcc: {}", e);
                    }
                }
            } else {
                info!("Using existing GPU kernel binary: mining_kernel.hsaco");
            }

            // Check if the compiled kernel exists and has non-zero size
            match std::fs::metadata("mining_kernel.hsaco") {
                Ok(metadata) => {
                    let size = metadata.len();
                    if size > 0 {
                        info!("GPU kernel is available for mining (size: {} bytes)", size);
                        return true;
                    } else {
                        error!("FATAL: GPU kernel file exists but has zero size");
                        panic!("GPU initialization failed: Kernel file exists but has zero size");
                    }
                }
                Err(e) => {
                    error!("FATAL: Failed to get GPU kernel file metadata: {}", e);
                    panic!(
                        "GPU initialization failed: Cannot access kernel file: {}",
                        e
                    );
                }
            }
        }

        // Check if the kernel file exists and has non-zero size
        match std::fs::metadata("mining_kernel.hsaco") {
            Ok(metadata) => {
                if metadata.len() > 0 {
                    true
                } else {
                    error!("FATAL: GPU kernel file has zero size");
                    panic!("GPU initialization failed: Kernel file has zero size");
                }
            }
            Err(e) => {
                error!("FATAL: Failed to access GPU kernel file: {}", e);
                panic!(
                    "GPU initialization failed: Cannot access kernel file: {}",
                    e
                );
            }
        }
    }

    pub fn get_device_info(&self) -> (i32, &str, i32, bool) {
        (
            self.device,
            &self.device_name,
            self.compute_units,
            self.is_using_gpu,
        )
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
            if hipMalloc(&mut d_kernel_auth_path, kernel_size) != hipSuccess
                || hipMalloc(&mut d_header_auth_path, header_size) != hipSuccess
                || hipMalloc(&mut d_threshold, digest_size) != hipSuccess
                || hipMalloc(&mut d_result, result_size) != hipSuccess
                || hipMalloc(&mut d_found_nonce, digest_size) != hipSuccess
            {
                self.free_device_memory(
                    d_kernel_auth_path,
                    d_header_auth_path,
                    d_threshold,
                    d_result,
                    d_found_nonce,
                );
                return Err("Failed to allocate device memory".to_string());
            }

            // Copy data to device
            if hipMemcpy(
                d_kernel_auth_path,
                kernel_auth_path.as_ptr() as *const _,
                kernel_size,
                hipMemcpyHostToDevice,
            ) != hipSuccess
                || hipMemcpy(
                    d_header_auth_path,
                    header_auth_path.as_ptr() as *const _,
                    header_size,
                    hipMemcpyHostToDevice,
                ) != hipSuccess
                || hipMemcpy(
                    d_threshold,
                    &threshold as *const _ as *const _,
                    digest_size,
                    hipMemcpyHostToDevice,
                ) != hipSuccess
            {
                self.free_device_memory(
                    d_kernel_auth_path,
                    d_header_auth_path,
                    d_threshold,
                    d_result,
                    d_found_nonce,
                );
                return Err("Failed to copy data to device".to_string());
            }

            // Initialize result to 0 (not found)
            let result_init: u64 = 0;
            if hipMemcpy(
                d_result,
                &result_init as *const _ as *const _,
                result_size,
                hipMemcpyHostToDevice,
            ) != hipSuccess
            {
                self.free_device_memory(
                    d_kernel_auth_path,
                    d_header_auth_path,
                    d_threshold,
                    d_result,
                    d_found_nonce,
                );
                return Err("Failed to initialize result memory".to_string());
            }

            // Generate random nonce start
            let nonce_start = rand::thread_rng().gen::<u64>();

            // Adjust nonce range based on device capabilities
            // MI100 GPUs can handle larger batches
            let nonce_range = if self.device_name.contains("MI100") {
                // MI100 has 120 compute units, can handle larger batches
                5_000_000_000 // 5 billion nonces per kernel launch for MI100
            } else {
                1_000_000_000 // 1 billion nonces for other GPUs
            };

            // Create and launch kernel
            let kernel = MiningKernel::new(difficulty, nonce_start, nonce_range);

            if self.is_using_gpu {
                info!(
                    "Launching GPU mining kernel on {} with {} compute units",
                    self.device_name, self.compute_units
                );
                info!(
                    "Processing {} nonces starting from {}",
                    nonce_range, nonce_start
                );

                // Log additional information for MI100 GPUs
                if self.device_name.contains("MI100") {
                    info!("Using optimized kernel for AMD MI100 GPU architecture");
                }
            } else {
                error!("FATAL: GPU mining is not available, cannot proceed with CPU fallback");
                panic!("GPU mining is not available. Please check GPU initialization logs for details.");
            }

            if let Err(e) = kernel.launch(
                self.stream.stream,
                d_kernel_auth_path as *const _,
                d_header_auth_path as *const _,
                d_threshold as *const _,
                d_result as *mut _,
                d_found_nonce as *mut _,
            ) {
                self.free_device_memory(
                    d_kernel_auth_path,
                    d_header_auth_path,
                    d_threshold,
                    d_result,
                    d_found_nonce,
                );
                return Err(e);
            }

            // Wait for kernel to complete
            if hipStreamSynchronize(self.stream.stream) != hipSuccess {
                self.free_device_memory(
                    d_kernel_auth_path,
                    d_header_auth_path,
                    d_threshold,
                    d_result,
                    d_found_nonce,
                );
                return Err("Failed to synchronize HIP stream".to_string());
            }

            // Check if a nonce was found
            let mut result: u64 = 0;
            if hipMemcpy(
                &mut result as *mut _ as *mut _,
                d_result,
                result_size,
                hipMemcpyDeviceToHost,
            ) != hipSuccess
            {
                self.free_device_memory(
                    d_kernel_auth_path,
                    d_header_auth_path,
                    d_threshold,
                    d_result,
                    d_found_nonce,
                );
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
                ) != hipSuccess
                {
                    self.free_device_memory(
                        d_kernel_auth_path,
                        d_header_auth_path,
                        d_threshold,
                        d_result,
                        d_found_nonce,
                    );
                    return Err("Failed to copy nonce from device".to_string());
                }

                info!("🎉 GPU found valid nonce!");

                Some(nonce)
            } else {
                None
            };

            // Free device memory
            self.free_device_memory(
                d_kernel_auth_path,
                d_header_auth_path,
                d_threshold,
                d_result,
                d_found_nonce,
            );

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

    // Implementation removed to fix duplicate function definition
}

impl Drop for GpuMiner {
    fn drop(&mut self) {
        unsafe {
            // Cleanup resources
            self.stream.destroy().ok();
        }
    }
}
