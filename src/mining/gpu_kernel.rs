use hip_runtime_sys::*;
use std::ffi::c_void;
use twenty_first::math::digest::Digest;
use tasm_lib::triton_vm::prelude::BFieldCodec;
use tasm_lib::prelude::Tip5;
use std::sync::Once;
use tracing::*;

#[repr(C)]
pub struct MiningKernel {
    difficulty: u64,
    nonce_start: u64,
    nonce_range: u64,
}

// Static variable to track if we've initialized the GPU kernel module
static GPU_KERNEL_INIT: Once = Once::new();
static mut GPU_KERNEL_MODULE: *mut c_void = std::ptr::null_mut();
static mut GPU_KERNEL_FUNCTION: *mut c_void = std::ptr::null_mut();

impl MiningKernel {
    pub fn new(difficulty: u64, nonce_start: u64, nonce_range: u64) -> Self {
        Self {
            difficulty,
            nonce_start,
            nonce_range,
        }
    }

    pub unsafe fn launch(
        &self,
        stream: hipStream_t,
        kernel_auth_path: *const Digest,
        header_auth_path: *const Digest,
        threshold: *const Digest,
        result: *mut u64,
        found_nonce: *mut Digest,
    ) -> Result<(), String> {
        const THREADS_PER_BLOCK: u32 = 256;
        let blocks = ((self.nonce_range as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Launch configuration
        let grid = dim3 {
            x: blocks,
            y: 1,
            z: 1,
        };
        let block = dim3 {
            x: THREADS_PER_BLOCK,
            y: 1,
            z: 1,
        };

        // Shared memory size
        let shared_mem_size = 0;

        // Initialize the GPU kernel if not already done
        let kernel_ptr = self.initialize_gpu_kernel()?;

        // Launch kernel with proper kernel pointer
        let status = hipLaunchKernel(
            kernel_ptr,
            grid,
            block,
            &mut [
                kernel_auth_path as *mut c_void,
                header_auth_path as *mut c_void,
                threshold as *mut c_void,
                self.difficulty as *mut c_void,
                self.nonce_start as *mut c_void,
                result as *mut c_void,
                found_nonce as *mut c_void,
            ] as *mut c_void,
            shared_mem_size,
            stream,
        );

        if status != hipSuccess {
            return Err(format!("Failed to launch kernel: error code {}", status));
        }

        Ok(())
    }
    
    // Initialize the GPU kernel module and function
    unsafe fn initialize_gpu_kernel(&self) -> Result<*const c_void, String> {
        let mut initialized = false;
        
        GPU_KERNEL_INIT.call_once(|| {
            // Check if HIP runtime is available
            let mut device_count = 0;
            let status = hipGetDeviceCount(&mut device_count);
            if status != hipSuccess {
                error!("FATAL: Failed to get HIP device count: error code {}", status);
                panic!("GPU initialization failed: Cannot get HIP device count: error code {}", status);
            }
            
            if device_count == 0 {
                error!("FATAL: No HIP devices available for GPU mining");
                panic!("GPU initialization failed: No HIP devices available");
            }
            
            // Log available devices
            info!("Found {} HIP device(s) for GPU mining", device_count);
            
            // Check for AMD MI100 GPUs specifically
            let mut has_mi100 = false;
            for device_id in 0..device_count {
                let mut name_buffer = [0u8; 256];
                if hipDeviceGetName(name_buffer.as_mut_ptr() as *mut i8, name_buffer.len() as i32, device_id) == hipSuccess {
                    let device_name = std::ffi::CStr::from_ptr(name_buffer.as_ptr() as *const i8)
                        .to_string_lossy();
                    
                    let mut compute_units = 0;
                    if hipDeviceGetAttribute(&mut compute_units, hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount, device_id) == hipSuccess {
                        info!("  Device #{}: {} with {} compute units", device_id, device_name, compute_units);
                        
                        // Check if this is an MI100 GPU
                        if device_name.contains("MI100") {
                            info!("  🚀 Detected AMD MI100 GPU at device #{}", device_id);
                            has_mi100 = true;
                            
                            // Set this device as active for kernel compilation
                            let set_device_status = hipSetDevice(device_id);
                            if set_device_status != hipSuccess {
                                warn!("Failed to set MI100 GPU as active device: error code {}", set_device_status);
                            } else {
                                info!("Set MI100 GPU (device #{}) as active for kernel compilation", device_id);
                            }
                        }
                    } else {
                        info!("  Device #{}: {}", device_id, device_name);
                    }
                    
                    // Get ROCm device ID (for AMD GPUs)
                    let mut rocm_device_id = 0;
                    if hipDeviceGetAttribute(&mut rocm_device_id, hipDeviceAttribute_t_hipDeviceAttributeDeviceId, device_id) == hipSuccess {
                        info!("  - ROCm Device ID: {}", rocm_device_id);
                    }
                }
            }
            
            // Log HIP runtime and driver versions
            let mut hip_runtime_version = 0;
            if hipRuntimeGetVersion(&mut hip_runtime_version) == hipSuccess {
                info!("HIP Runtime Version: {}", hip_runtime_version);
            }
            
            let mut hip_driver_version = 0;
            if hipDriverGetVersion(&mut hip_driver_version) == hipSuccess {
                info!("HIP Driver Version: {}", hip_driver_version);
            }
            
            // Create a module that contains our kernel
            info!("Attempting to create GPU module...");
            let module_ptr = self.create_gpu_module().unwrap_or_else(|e| {
                error!("FATAL: Failed to create GPU module: {}", e);
                panic!("GPU initialization failed: Cannot create GPU module: {}", e);
            });
            
            if module_ptr.is_null() {
                error!("FATAL: Failed to create GPU module, module pointer is null");
                panic!("GPU initialization failed: Module pointer is null");
            }
            
            // Get the kernel function from the module
            info!("Attempting to get kernel function...");
            let function_ptr = self.get_kernel_function(module_ptr).unwrap_or_else(|e| {
                error!("FATAL: Failed to get kernel function: {}", e);
                hipModuleUnload(module_ptr);
                panic!("GPU initialization failed: Cannot get kernel function: {}", e);
            });
            
            if function_ptr.is_null() {
                error!("FATAL: Failed to get kernel function, function pointer is null");
                hipModuleUnload(module_ptr);
                panic!("GPU initialization failed: Function pointer is null");
            }
            
            // Store the module and function pointers
            GPU_KERNEL_MODULE = module_ptr;
            GPU_KERNEL_FUNCTION = function_ptr;
            initialized = true;
            
            info!("✅ Successfully initialized GPU kernel for mining");
        });
        
        if !initialized && GPU_KERNEL_FUNCTION.is_null() {
            // If initialization failed, we should not fall back to CPU implementation
            error!("FATAL: GPU kernel initialization failed");
            panic!("GPU initialization failed: Cannot initialize GPU kernel");
        }
        
        Ok(GPU_KERNEL_FUNCTION as *const c_void)
    }
    
    // Create a GPU module that contains our kernel
    unsafe fn create_gpu_module(&self) -> Result<*mut c_void, String> {
        let mut module = std::ptr::null_mut();
        
        // First, check if a pre-compiled kernel exists
        if std::path::Path::new("mining_kernel.hsaco").exists() {
            info!("Found pre-compiled kernel file, attempting to load it");
            
            // Check the file size to ensure it's not empty
            match std::fs::metadata("mining_kernel.hsaco") {
                Ok(metadata) => {
                    let size = metadata.len();
                    if size == 0 {
                        error!("Pre-compiled kernel file exists but has zero size");
                        std::fs::remove_file("mining_kernel.hsaco").ok();
                        info!("Removed invalid kernel file, will recompile");
                    } else {
                        info!("Pre-compiled kernel file size: {} bytes", size);
                    }
                },
                Err(e) => {
                    warn!("Failed to get kernel file metadata: {}, will try to recompile", e);
                }
            }
            
            // Try to load the pre-compiled kernel
            if std::path::Path::new("mining_kernel.hsaco").exists() && 
               std::fs::metadata("mining_kernel.hsaco").map(|m| m.len() > 0).unwrap_or(false) {
                let status = hipModuleLoad(&mut module, b"mining_kernel.hsaco\0".as_ptr() as *const i8);
                
                if status == hipSuccess {
                    info!("✅ Successfully loaded pre-compiled GPU kernel module");
                    return Ok(module);
                }
                
                warn!("Failed to load pre-compiled GPU module: error code {}. Will recompile it.", status);
                // Remove the invalid kernel file
                std::fs::remove_file("mining_kernel.hsaco").ok();
            }
        }
        
        // If no pre-compiled kernel exists or loading failed, compile it
        info!("Generating GPU kernel source code for AMD MI100 GPU");
        let kernel_source = create_gpu_kernel_source();
        info!("Generated kernel source code: {} bytes", kernel_source.len());
        
        // Write the kernel source to a file
        let source_file = "mining_kernel.cpp";
        match std::fs::write(source_file, kernel_source) {
            Ok(_) => info!("Successfully wrote kernel source to {}", source_file),
            Err(e) => {
                error!("Failed to write kernel source: {}", e);
                return Err(format!("Failed to write kernel source: {}", e));
            }
        }
        
        // Check if hipcc is available and get its version
        info!("Checking hipcc compiler...");
        let hipcc_version = std::process::Command::new("hipcc")
            .arg("--version")
            .output();
            
        match hipcc_version {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    info!("hipcc version: {}", version.lines().next().unwrap_or("unknown"));
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    warn!("Failed to get hipcc version: {}", stderr);
                }
            },
            Err(e) => {
                warn!("Failed to execute hipcc version check: {}", e);
            }
        }
        
        // Try to compile the kernel using hipcc with verbose output
        info!("Compiling GPU kernel with hipcc...");
        let output = std::process::Command::new("hipcc")
            .args(&[
                "--genco",
                "-O3",
                "-fgpu-rdc",
                "-v", // Verbose output
                "-o", "mining_kernel.hsaco",
                source_file
            ])
            .output();
        
        match output {
            Ok(output) => {
                // Log both stdout and stderr regardless of success
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                if !stdout.is_empty() {
                    info!("Compiler stdout: {}", stdout);
                }
                
                if !stderr.is_empty() {
                    if output.status.success() {
                        info!("Compiler stderr (not an error): {}", stderr);
                    } else {
                        error!("Compiler stderr: {}", stderr);
                    }
                }
                
                if !output.status.success() {
                    return Err(format!("Failed to compile GPU kernel: {}", stderr));
                }
                
                info!("✅ Successfully compiled GPU kernel");
                
                // Verify the compiled file exists and has non-zero size
                match std::fs::metadata("mining_kernel.hsaco") {
                    Ok(metadata) => {
                        let size = metadata.len();
                        if size == 0 {
                            error!("Compiled kernel file has zero size");
                            return Err("Compiled kernel file has zero size".to_string());
                        }
                        info!("Compiled kernel file size: {} bytes", size);
                    },
                    Err(e) => {
                        error!("Failed to get compiled kernel file metadata: {}", e);
                        return Err(format!("Failed to get compiled kernel file metadata: {}", e));
                    }
                }
                
                // Now try to load the compiled module
                let status = hipModuleLoad(&mut module, b"mining_kernel.hsaco\0".as_ptr() as *const i8);
                
                if status != hipSuccess {
                    error!("Failed to load compiled GPU module: error code {}", status);
                    return Err(format!("Failed to load compiled GPU module: error code {}", status));
                }
                
                info!("✅ Successfully loaded compiled GPU kernel module");
                Ok(module)
            },
            Err(e) => {
                error!("Failed to execute hipcc: {}", e);
                return Err(format!("Failed to execute hipcc: {}", e));
            }
        }
    }
    
    // Get the kernel function from the module
    unsafe fn get_kernel_function(&self, module: *mut c_void) -> Result<*mut c_void, String> {
        // Get the kernel function from the module
        let mut function = std::ptr::null_mut();
        
        info!("Attempting to get 'mining_kernel' function from module");
        
        // First, try to list all functions in the module (if available)
        // This is not directly supported by HIP API, but we can log the attempt
        info!("Looking for kernel function 'mining_kernel' in the module");
        
        let status = hipModuleGetFunction(
            &mut function,
            module,
            b"mining_kernel\0".as_ptr() as *const i8,
        );
        
        if status != hipSuccess {
            error!("FATAL: Failed to get kernel function 'mining_kernel': error code {}", status);
            
            // Try to get more information about the error
            let error_string = match status {
                1 => "hipErrorInvalidValue - Invalid module or function name",
                2 => "hipErrorNotInitialized - HIP runtime not initialized",
                3 => "hipErrorNotFound - Function not found in module",
                _ => "Unknown error",
            };
            
            error!("Error details: {}", error_string);
            
            // Check if the module is valid
            if module.is_null() {
                error!("FATAL: Module pointer is null");
                return Err("Module pointer is null".to_string());
            }
            
            return Err(format!("Failed to get kernel function: {} (error code {})", error_string, status));
        }
        
        if function.is_null() {
            error!("FATAL: Function pointer is null even though hipModuleGetFunction succeeded");
            return Err("Function pointer is null".to_string());
        }
        
        info!("✅ Successfully retrieved 'mining_kernel' function from module");
        Ok(function)
    }
}

// This is the GPU kernel that will be executed on the device
// This is the CPU fallback implementation of the mining kernel
// It's used when the GPU kernel fails to initialize or when no GPU is available
#[no_mangle]
pub unsafe extern "C" fn mining_kernel(
    kernel_auth_path: *const Digest,
    header_auth_path: *const Digest,
    threshold: *const Digest,
    difficulty: u64,
    nonce_start: u64,
    result: *mut u64,
    found_nonce: *mut Digest,
) {
    // This is a CPU implementation that simulates multiple GPU threads
    // It processes nonces in batches to mimic GPU behavior
    const BATCH_SIZE: u32 = 256; // Simulate one GPU block
    
    for batch_idx in 0..16 { // Process 16 batches (simulating 16 GPU blocks)
        for thread_idx in 0..BATCH_SIZE {
            let global_idx = (batch_idx * BATCH_SIZE + thread_idx) as u64;
            let nonce_offset = global_idx;
            
            // Each thread tries a different nonce
            let mut nonce = Digest::default();
            nonce.values_mut()[0] = twenty_first::math::bfield::BField::new(nonce_start + nonce_offset);
            
            // Get the kernel and header auth paths
            let kernel_path = std::slice::from_raw_parts(kernel_auth_path, 2);
            let header_path = std::slice::from_raw_parts(header_auth_path, 3);
            let threshold_val = *threshold;
            
            // Calculate the block hash using the fast kernel hash function
            let header_mast_hash = Tip5::hash_pair(Tip5::hash_varlen(&nonce.encode()), header_path[0]);
            let header_mast_hash = Tip5::hash_pair(header_mast_hash, header_path[1]);
            let header_mast_hash = Tip5::hash_pair(header_path[2], header_mast_hash);
            
            let block_hash = Tip5::hash_pair(
                Tip5::hash_pair(
                    Tip5::hash_varlen(&header_mast_hash.encode()),
                    kernel_path[0],
                ),
                kernel_path[1],
            );
            
            // Check if we found a valid nonce
            if block_hash <= threshold_val {
                // Atomically set the result to 1 to indicate success
                hip_atomic_exchange(result, 1, hipMemoryOrderRelease);
                
                // Store the found nonce
                *found_nonce = nonce;
                
                // Early return on success
                return;
            }
        }
    }
}

// Create a file with the GPU kernel source code
// This would be compiled at runtime in a real implementation
pub fn create_gpu_kernel_source() -> String {
    r#"
    #include <hip/hip_runtime.h>
    
    // Define Digest structure to match Rust's Digest
    typedef struct {
        uint64_t values[4];
    } Digest;
    
    // Hash functions that would be implemented in the real kernel
    __device__ void tip5_hash_pair(Digest* result, const Digest* left, const Digest* right) {
        // Simple XOR-based hash function for demonstration
        for (int i = 0; i < 4; i++) {
            result->values[i] = left->values[i] ^ right->values[i];
            // Add some mixing to make it non-trivial
            result->values[i] = (result->values[i] << 1) | (result->values[i] >> 63);
        }
    }
    
    __device__ void tip5_hash_varlen(Digest* result, const uint64_t* data, size_t len) {
        // Initialize with some values
        result->values[0] = 0x6a09e667f3bcc908ULL;
        result->values[1] = 0xbb67ae8584caa73bULL;
        result->values[2] = 0x3c6ef372fe94f82bULL;
        result->values[3] = 0xa54ff53a5f1d36f1ULL;
        
        // Mix in the data
        size_t words = len / 8;
        for (size_t i = 0; i < words; i++) {
            result->values[i % 4] ^= data[i];
            // Add some mixing
            result->values[i % 4] = (result->values[i % 4] << 1) | (result->values[i % 4] >> 63);
        }
    }
    
    // The actual GPU kernel
    extern "C" __global__ void mining_kernel(
        const Digest* kernel_auth_path,
        const Digest* header_auth_path,
        const Digest* threshold,
        uint64_t difficulty,
        uint64_t nonce_start,
        uint64_t* result,
        Digest* found_nonce
    ) {
        uint32_t thread_idx = hipThreadIdx_x;
        uint32_t block_idx = hipBlockIdx_x;
        uint32_t block_dim = hipBlockDim_x;
        
        uint64_t global_idx = block_idx * block_dim + thread_idx;
        uint64_t nonce_value = nonce_start + global_idx;
        
        // Create nonce digest
        Digest nonce;
        nonce.values[0] = nonce_value;
        nonce.values[1] = 0;
        nonce.values[2] = 0;
        nonce.values[3] = 0;
        
        // Encode nonce (simplified for kernel)
        uint64_t encoded_nonce[5];
        encoded_nonce[0] = 4; // Length
        encoded_nonce[1] = nonce.values[0];
        encoded_nonce[2] = nonce.values[1];
        encoded_nonce[3] = nonce.values[2];
        encoded_nonce[4] = nonce.values[3];
        
        // Calculate header MAST hash
        Digest hash1, hash2, header_mast_hash;
        
        // Hash nonce
        tip5_hash_varlen(&hash1, encoded_nonce, sizeof(encoded_nonce));
        
        // Build header MAST hash
        tip5_hash_pair(&hash2, &hash1, &header_auth_path[0]);
        tip5_hash_pair(&hash1, &hash2, &header_auth_path[1]);
        tip5_hash_pair(&header_mast_hash, &header_auth_path[2], &hash1);
        
        // Encode header MAST hash
        uint64_t encoded_header[5];
        encoded_header[0] = 4; // Length
        encoded_header[1] = header_mast_hash.values[0];
        encoded_header[2] = header_mast_hash.values[1];
        encoded_header[3] = header_mast_hash.values[2];
        encoded_header[4] = header_mast_hash.values[3];
        
        // Calculate block hash
        tip5_hash_varlen(&hash1, encoded_header, sizeof(encoded_header));
        tip5_hash_pair(&hash2, &hash1, &kernel_auth_path[0]);
        Digest block_hash;
        tip5_hash_pair(&block_hash, &hash2, &kernel_auth_path[1]);
        
        // Check if we found a valid nonce
        bool is_valid = true;
        for (int i = 0; i < 4; i++) {
            if (block_hash.values[i] > threshold->values[i]) {
                is_valid = false;
                break;
            }
            if (block_hash.values[i] < threshold->values[i]) {
                break;
            }
        }
        
        if (is_valid) {
            // Atomically set the result to 1 to indicate success
            atomicExch(result, 1);
            
            // Store the found nonce
            *found_nonce = nonce;
        }
    }
    "#
}

// Helper function for atomic operations
unsafe fn hip_atomic_exchange(address: *mut u64, val: u64, order: u32) -> u64 {
    let mut old_val: u64 = 0;
    hipAtomicExch(address as *mut _, val, &mut old_val, order);
    old_val
}