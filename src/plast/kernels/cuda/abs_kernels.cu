#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

__global__ void abs_cuda_kernel_float_kernel(float* out, const float* in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = fabsf(in[idx]);
    }
}

extern "C" void plast_cuda_abs_kernel_float(float* out, const float* in, size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    abs_cuda_kernel_float_kernel<<<numBlocks, blockSize>>>(out, in, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

__global__ void abs_cuda_kernel_int32_kernel(int32_t* out, const int32_t* in, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = abs(in[idx]);
    }
}

extern "C" void plast_cuda_abs_kernel_int32(int32_t* out, const int32_t* in, size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    abs_cuda_kernel_int32_kernel<<<numBlocks, blockSize>>>(out, in, num_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_int32: %s\n", cudaGetErrorString(err));
    }
}
