#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

__global__ void leaky_relu_cuda_kernel_float_kernel(float* out, const float* in, size_t num_elements, float alpha)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        out[idx] = (in[idx] > 0.0f) ? in[idx] : in[idx] * alpha;
    }
}

// Wrapper function to launch the CUDA kernel for float
extern "C" void plast_cuda_leaky_relu_kernel_float(float* out, const float* in, size_t num_elements, float alpha)
{
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;
    leaky_relu_cuda_kernel_float_kernel<<<numBlocks, blockSize>>>(out, in, num_elements, alpha);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in plast_cuda_leaky_relu_kernel_float: %s\n", cudaGetErrorString(err));
    }
}

