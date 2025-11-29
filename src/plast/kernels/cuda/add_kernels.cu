#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void add_cuda_kernel_float_kernel(float* out, const float* in1, const float* in2,
                                             size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = in1[idx] + in2[idx];
    }
}

extern "C" void plast_cuda_add_kernel_float(float* out, const float* in1, const float* in2,
                                            size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    add_cuda_kernel_float_kernel<<<numBlocks, blockSize>>>(out, in1, in2, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

__global__ void add_cuda_kernel_int32_kernel(int32_t* out, const int32_t* in1, const int32_t* in2,
                                             size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = in1[idx] + in2[idx];
    }
}

extern "C" void plast_cuda_add_kernel_int32(int32_t* out, const int32_t* in1, const int32_t* in2,
                                            size_t num_elements)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    add_cuda_kernel_int32_kernel<<<numBlocks, blockSize>>>(out, in1, in2, num_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_add_kernel_int32: %s\n", cudaGetErrorString(err));
    }
}
