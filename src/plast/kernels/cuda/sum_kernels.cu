#include "plast/kernels/cuda/reduction_kernels.h"
#include <cuda_runtime.h>
#include <plast/core/device_management.h>
#include <plast/kernels/cuda/cuda_kernel_utils.h>
#include "plast/core/shape_utils_c.h"

// Full reduction kernel for float (multi-block)
__global__ void full_sum_kernel_float(float* out, const float* in, size_t num_elements)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // Each thread loads multiple elements from global memory to shared memory
    // and performs a partial sum
    float thread_sum = 0.0f;
    while (idx < num_elements)
    {
        thread_sum += in[idx];
        idx += gridDim.x * blockDim.x; // Stride across grid
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        // For a true full reduction across multiple blocks, we need a global sum.
        // Let's use atomicAdd for simplicity for now, assuming 'out' is a single element.
        atomicAdd(out, sdata[0]);
    }
}

// Host function for full sum reduction (contiguous float)
extern "C" void plast_cuda_sum_full_reduction_float(const float* input_data, float* output_data,
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

    full_sum_kernel_float<<<numBlocks, blockSize, smemSize>>>(output_data, input_data,
                                                              num_elements);
}
