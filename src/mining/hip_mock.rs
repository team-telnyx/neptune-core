// Mock implementation of HIP runtime for testing
// This would be replaced with actual HIP runtime in a real environment

pub type hipStream_t = *mut ::std::os::raw::c_void;
pub type hipModule_t = *mut ::std::os::raw::c_void;
pub type hipFunction_t = *mut ::std::os::raw::c_void;
pub type hipDeviceAttribute_t = i32;

pub const hipSuccess: i32 = 0;
pub const hipErrorNotFound: i32 = 3;
pub const hipStreamNonBlocking: u32 = 1;

pub const hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount: i32 = 16;
pub const hipDeviceAttribute_t_hipDeviceAttributeClockRate: i32 = 13;
pub const hipDeviceAttribute_t_hipDeviceAttributeDeviceId: i32 = 1;
pub const hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor: i32 = 28;
pub const hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor: i32 = 29;
pub const hipDeviceAttribute_t_hipDeviceAttributeWarpSize: i32 = 10;
pub const hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerBlock: i32 = 1;
pub const hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerBlock: i32 = 8;

pub const hipMemcpyHostToDevice: i32 = 1;
pub const hipMemcpyDeviceToHost: i32 = 2;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

// Mock implementations of HIP functions
#[no_mangle]
pub unsafe extern "C" fn hipGetDeviceCount(count: *mut i32) -> i32 {
    *count = 1; // Pretend we have 1 device
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipGetDevice(device: *mut i32) -> i32 {
    *device = 0; // Always return device 0
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipSetDevice(device: i32) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipDeviceGetName(name: *mut i8, len: i32, device: i32) -> i32 {
    // Set name to "AMD MI100" with more detailed information
    let device_name = b"AMD Instinct MI100 Accelerator\0";
    let name_len = device_name.len().min(len as usize);
    std::ptr::copy_nonoverlapping(device_name.as_ptr() as *const i8, name, name_len);
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipDeviceGetAttribute(
    value: *mut i32,
    attr: hipDeviceAttribute_t,
    device: i32,
) -> i32 {
    match attr {
        // MI100 has 120 compute units
        hipDeviceAttribute_t_hipDeviceAttributeMultiprocessorCount => *value = 120,
        // MI100 has 1502 MHz clock rate
        hipDeviceAttribute_t_hipDeviceAttributeClockRate => *value = 1502000,
        hipDeviceAttribute_t_hipDeviceAttributeDeviceId => *value = 0,
        // MI100 is based on CDNA architecture (gfx908)
        hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMajor => *value = 9,
        hipDeviceAttribute_t_hipDeviceAttributeComputeCapabilityMinor => *value = 0,
        // MI100 has 64 threads per wavefront
        hipDeviceAttribute_t_hipDeviceAttributeWarpSize => *value = 64,
        // MI100 supports 1024 threads per block
        hipDeviceAttribute_t_hipDeviceAttributeMaxThreadsPerBlock => *value = 1024,
        // MI100 has 64KB of shared memory per block
        hipDeviceAttribute_t_hipDeviceAttributeMaxSharedMemoryPerBlock => *value = 65536,
        _ => *value = 0,
    }
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipDeviceTotalMem(bytes: *mut usize, device: i32) -> i32 {
    *bytes = 32 * 1024 * 1024 * 1024; // 32GB (MI100 has 32GB HBM2 memory)
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipDeviceGetPCIBusId(pciBusId: *mut i8, len: i32, device: i32) -> i32 {
    let pci_id = b"0000:00:00.0\0";
    let id_len = pci_id.len().min(len as usize);
    std::ptr::copy_nonoverlapping(pci_id.as_ptr() as *const i8, pciBusId, id_len);
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipRuntimeGetVersion(version: *mut i32) -> i32 {
    *version = 60203; // ROCm 6.2.3 (matches the version in the Jira ticket)
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipDriverGetVersion(version: *mut i32) -> i32 {
    *version = 60203; // ROCm 6.2.3 (matches the version in the Jira ticket)
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipStreamCreate(stream: *mut hipStream_t) -> i32 {
    *stream = std::ptr::null_mut();
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipStreamDestroy(stream: hipStream_t) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipMalloc(ptr: *mut *mut ::std::os::raw::c_void, size: usize) -> i32 {
    *ptr = std::ptr::null_mut();
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipFree(ptr: *mut ::std::os::raw::c_void) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipMemcpy(
    dst: *mut ::std::os::raw::c_void,
    src: *const ::std::os::raw::c_void,
    size: usize,
    kind: i32,
) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipModuleLoad(module: *mut hipModule_t, fname: *const i8) -> i32 {
    *module = std::ptr::null_mut();
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipModuleUnload(module: hipModule_t) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipModuleGetFunction(
    function: *mut hipFunction_t,
    module: hipModule_t,
    name: *const i8,
) -> i32 {
    *function = std::ptr::null_mut();
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipLaunchKernel(
    function: *const ::std::os::raw::c_void,
    grid_dim: dim3,
    block_dim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    shared_mem: usize,
    stream: hipStream_t,
) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipStreamSynchronize(stream: hipStream_t) -> i32 {
    hipSuccess
}

#[no_mangle]
pub unsafe extern "C" fn hipAtomicExch(
    address: *mut u64,
    val: u64,
    old_val: *mut u64,
    order: u32,
) -> i32 {
    *old_val = 0;
    hipSuccess
}
