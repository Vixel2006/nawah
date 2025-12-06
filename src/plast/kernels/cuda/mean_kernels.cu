#include "plast/core/shape_utils_c.h"
#include "plast/kernels/cuda/reduction_kernels.h"
#include <cuda_runtime.h>
#include <plast/core/device_management.h>
#include <plast/kernels/cuda/cuda_kernel_utils.h>

// Full reduction kernel for float (multi-block)
__global__ void full_mean_kernel_float(float* out, const float* in, size_t num_elements)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    float thread_sum = 0.0f;
    while (idx < num_elements)
    {
        thread_sum += in[idx];
        idx += gridDim.x * blockDim.x;
    }
    sdata[tid] = thread_sum;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
    }
    __syncthreads();

    if (tid == 0) out[0] = sdata[tid] / num_elements;
}

// Host function for full sum reduction (contiguous float)
extern "C" void plast_cuda_mean_full_reduction_float(const float* input_data, float* output_data,
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

    full_mean_kernel_float<<<numBlocks, blockSize, smemSize>>>(output_data, input_data,
                                                               num_elements);
}
