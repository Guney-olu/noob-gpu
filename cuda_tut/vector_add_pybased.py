import ctypes
import ctypes.util
import numpy as np
import os

# 1. CUDA C++ Kernel for Vector Addition

VECTOR_ADD_KERNEL = """
extern "C"
__global__ void add_vectors(const float* a, const float* b, float* c, int n) {
    // Calculate the unique global index for this thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: Make sure we don't go past the end of the arrays.
    // This is important if n is not a perfect multiple of the block size.
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Python<->C interface for CUDA Driver API and the NVRTC

def find_cuda_library(lib_name):
    path = ctypes.util.find_library(lib_name)
    if path:
        return path
    if os.name == "nt": 
        search_paths = [os.path.join(os.environ.get("CUDA_PATH", ""), "bin")]
        lib_name_ext = lib_name + ".dll"
    else:
        search_paths = ["/usr/local/cuda/lib64", "/opt/cuda/lib64"]
        lib_name_ext = "lib" + lib_name + ".so"
        
    for p in search_paths:
        full_path = os.path.join(p, lib_name_ext)
        if os.path.exists(full_path):
            return full_path  
    raise FileNotFoundError(f"Could not find CUDA library: {lib_name}")


# TO Load the CUDA and NVRTC libraries
try:
    libcuda = ctypes.CDLL(find_cuda_library('cuda'))
    libnvrtc = ctypes.CDLL(find_cuda_library('nvrtc'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the CUDA Toolkit is installed and its 'lib64' or 'bin' directory is in your system's path.")
    exit()


CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_ulonglong 

def check_result(result):
    if result != 0:
        raise RuntimeError(f"CUDA/NVRTC API call failed with error code: {result}")

nvrtcCreateProgram = libnvrtc.nvrtcCreateProgram
nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
nvrtcCreateProgram.restype = int

nvrtcCompileProgram = libnvrtc.nvrtcCompileProgram
nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
nvrtcCompileProgram.restype = int

nvrtcGetPTXSize = libnvrtc.nvrtcGetPTXSize
nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtcGetPTXSize.restype = int

nvrtcGetPTX = libnvrtc.nvrtcGetPTX
nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtcGetPTX.restype = int

check_result(libcuda.cuInit(0))

cuDeviceGet = libcuda.cuDeviceGet
cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
cuDeviceGet.restype = int

cuCtxCreate = libcuda.cuCtxCreate_v2
cuCtxCreate.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
cuCtxCreate.restype = int

cuModuleLoadData = libcuda.cuModuleLoadData
cuModuleLoadData.argtypes = [ctypes.POINTER(CUmodule), ctypes.c_void_p]
cuModuleLoadData.restype = int

cuModuleGetFunction = libcuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p]
cuModuleGetFunction.restype = int

cuMemAlloc = libcuda.cuMemAlloc_v2
cuMemAlloc.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = libcuda.cuMemcpyHtoD_v2
cuMemcpyHtoD.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
cuMemcpyHtoD.restype = int

cuMemcpyDtoH = libcuda.cuMemcpyDtoH_v2
cuMemcpyDtoH.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
cuMemcpyDtoH.restype = int

cuLaunchKernel = libcuda.cuLaunchKernel
cuLaunchKernel.argtypes = [
    CUfunction,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  
    ctypes.c_uint,
    ctypes.c_void_p, # stream
    ctypes.POINTER(ctypes.c_void_p), # kernel parameters
    ctypes.POINTER(ctypes.c_void_p) 
]
cuLaunchKernel.restype = int

cuCtxSynchronize = libcuda.cuCtxSynchronize
cuCtxSynchronize.restype = int

cuMemFree = libcuda.cuMemFree_v2
cuMemFree.argtypes = [CUdeviceptr]
cuMemFree.restype = int

cuCtxDestroy = libcuda.cuCtxDestroy_v2
cuCtxDestroy.argtypes = [CUcontext]
cuCtxDestroy.restype = int

if __name__ == "__main__":
    print("1. Preparing data on the CPU...")
    N = 1024 * 1024
    
    h_a = np.random.rand(N).astype(np.float32)
    h_b = np.random.rand(N).astype(np.float32)

    h_c = np.empty_like(h_a)
    
    data_size = h_a.nbytes
    
    print("2. Compiling CUDA kernel at runtime...")
    prog = ctypes.c_void_p()
    check_result(nvrtcCreateProgram(ctypes.byref(prog), VECTOR_ADD_KERNEL.encode('utf-8'), b'add_vectors.cu', 0, None, None))
    
    compile_options = []
    c_compile_options = (ctypes.c_char_p * len(compile_options))(*[o.encode('utf-8') for o in compile_options])
    result = nvrtcCompileProgram(prog, len(compile_options), c_compile_options)
    if result != 0:
        print(f"Kernel compilation failed with error code: {result}")
        exit()

    ptx_size = ctypes.c_size_t()
    check_result(nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)))
    ptx = (ctypes.c_char * ptx_size.value)()
    check_result(nvrtcGetPTX(prog, ptx))

    print("3. Initializing CUDA device and context...")
    device = CUdevice()
    check_result(cuDeviceGet(ctypes.byref(device), 0)) 
    
    context = CUcontext()
    check_result(cuCtxCreate(ctypes.byref(context), 0, device))

    print("4. Loading module and allocating memory on the GPU...")
    module = CUmodule()
    check_result(cuModuleLoadData(ctypes.byref(module), ptx))
    
    kernel = CUfunction()
    check_result(cuModuleGetFunction(ctypes.byref(kernel), module, b"add_vectors"))

    d_a = CUdeviceptr()
    d_b = CUdeviceptr()
    d_c = CUdeviceptr()
    check_result(cuMemAlloc(ctypes.byref(d_a), data_size))
    check_result(cuMemAlloc(ctypes.byref(d_b), data_size))
    check_result(cuMemAlloc(ctypes.byref(d_c), data_size))

    print("5. Copying data from CPU to GPU...")
    check_result(cuMemcpyHtoD(d_a, h_a.ctypes.data, data_size))
    check_result(cuMemcpyHtoD(d_b, h_b.ctypes.data, data_size))

    print("6. Launching the kernel on the GPU...")
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    n_arg = ctypes.c_int(N)
    args = [ctypes.byref(d_a), ctypes.byref(d_b), ctypes.byref(d_c), ctypes.byref(n_arg)]
    kernel_params = (ctypes.c_void_p * len(args))(*[ctypes.cast(arg, ctypes.c_void_p) for arg in args])

    check_result(cuLaunchKernel(
        kernel,
        blocks_per_grid, 1, 1,          # Grid dimensions
        threads_per_block, 1, 1,         # Block dimensions
        0,                               # Shared memory (0 for this kernel)
        None,                            # Stream (0 or None for default)
        kernel_params,                   # Kernel parameters
        None                             # Extra options
    ))

    print("7. Copying result from GPU back to CPU...")
    check_result(cuCtxSynchronize()) # Wait for the kernel to finish
    
    check_result(cuMemcpyDtoH(h_c.ctypes.data, d_c, data_size))

    print("8. Verifying results and cleaning up...")
    
    cpu_c = h_a + h_b
    if np.allclose(h_c, cpu_c):
        print("Success! GPU and CPU results match.")
    else:
        print("Error! GPU and CPU results do not match.")

    check_result(cuMemFree(d_a))
    check_result(cuMemFree(d_b))
    check_result(cuMemFree(d_c))
    check_result(cuCtxDestroy(context))