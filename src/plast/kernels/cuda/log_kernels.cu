#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

__global__ void log_cuda_kernel_float_kernel(float* out, const float* in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = logf(in[idx]);
    }
}

extern "C" void plast_cuda_log_kernel_float(float* out, const float* in, size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    log_cuda_kernel_float_kernel<<<numBlocks, blockSize>>>(out, in, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_log_kernel_float: %s\n", cudaGetErrorString(err));
    }
}
