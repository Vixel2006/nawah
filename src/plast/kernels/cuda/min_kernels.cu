#include "plast/kernels/cuda/reduction_kernels.h"
#include <cfloat> // Added for FLT_MAX
#include <cuda_runtime.h>
#include <math.h>
#include <plast/core/device_management.h>
#include <plast/core/shape_utils_c.h> // Added for get_total_elements
#include <plast/kernels/cuda/cuda_kernel_utils.h>

// Full reduction kernel for float (multi-block)
__global__ void full_min_kernel_float(float* out, const float* in, size_t num_elements)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    float thread_min = FLT_MAX;
    while (idx < num_elements)
    {
        thread_min = fminf(thread_min, in[idx]);
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = thread_min;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) out[0] = sdata[tid];
}

// Host function for full sum reduction (contiguous float)
extern "C" void plast_cuda_min_full_reduction_float(const float* input_data, float* output_data,
                                                    const size_t* input_shape, size_t input_ndim)
{
    size_t num_elements = get_total_elements(input_shape, input_ndim);

    // Initialize output_data to 0 on device
    cudaMemset(output_data, 0, sizeof(float));

    // Determine block and grid dimensions
    int blockSize = 256; // Example block size
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535; // Max gridDim.x

    // Shared memory size for reduction
    size_t smemSize = blockSize * sizeof(float);

    full_min_kernel_float<<<numBlocks, blockSize, smemSize>>>(output_data, input_data,
                                                              num_elements);
}
