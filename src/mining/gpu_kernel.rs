use hip_runtime_sys::*;
use std::ffi::c_void;
use twenty_first::math::digest::Digest;
use tasm_lib::triton_vm::prelude::BFieldCodec;
use tasm_lib::prelude::Tip5;

#[repr(C)]
pub struct MiningKernel {
    difficulty: u64,
    nonce_start: u64,
    nonce_range: u64,
}

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

        // Launch kernel
        let status = hipLaunchKernel(
            mining_kernel as *const c_void,
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
            return Err("Failed to launch kernel".to_string());
        }

        Ok(())
    }
}

// This is the GPU kernel that will be executed on the device
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
    let thread_idx = hipThreadIdx_x();
    let block_idx = hipBlockIdx_x();
    let block_dim = hipBlockDim_x();
    
    let global_idx = (block_idx * block_dim + thread_idx) as u64;
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
    }
}

// Helper function for atomic operations
unsafe fn hip_atomic_exchange(address: *mut u64, val: u64, order: u32) -> u64 {
    let mut old_val: u64 = 0;
    hipAtomicExch(address as *mut _, val, &mut old_val, order);
    old_val
}