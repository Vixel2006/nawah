#include "plast/kernels/cuda/transpose_kernels.h"
#include "plast/kernels/cuda/cuda_kernel_utils.h"
#include "plast/core/shape_utils_c.h" // For get_total_elements
#include <cuda_runtime.h>
#include <string.h> // For memcpy

// CUDA kernel for transpose forward
__global__ void transpose_forward_kernel_cuda_impl(void* output_data, const void* input_data,
                                                   const size_t* input_shape_d,
                                                   const size_t* input_strides_d,
                                                   size_t input_ndim,
                                                   const size_t* output_shape_d,
                                                   const size_t* output_strides_d,
                                                   size_t output_ndim,
                                                   const int* inverse_axes_d, // Inverse of original axes
                                                   size_t item_size, size_t total_elements)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) {
        return;
    }

    // Calculate multi-dimensional indices for output from linear index
    size_t current_output_indices[MAX_NDIM];
    size_t temp_tid = tid;
    for (int i = output_ndim - 1; i >= 0; --i) {
        current_output_indices[i] = temp_tid % output_shape_d[i];
        temp_tid /= output_shape_d[i];
    }

    // Apply inverse_axes to get input indices
    size_t input_indices[MAX_NDIM];
    for (size_t j = 0; j < input_ndim; ++j) {
        input_indices[j] = current_output_indices[inverse_axes_d[j]];
    }

    size_t input_linear_index = cuda_get_index(input_indices, input_strides_d, input_ndim);
    size_t output_linear_index = cuda_get_index(current_output_indices, output_strides_d, output_ndim);

    memcpy((char*)output_data + output_linear_index * item_size,
           (const char*)input_data + input_linear_index * item_size,
           item_size);
}


extern "C" void cuda_transpose_forward_kernel(void* output_data, const void* input_data,
                                               const size_t* input_shape,
                                               const size_t* input_strides, size_t input_ndim,
                                               const size_t* output_shape,
                                               const size_t* output_strides, size_t output_ndim,
                                               const int* axes, // Original axes for transpose
                                               size_t item_size)
{
    size_t total_elements = get_total_elements(output_shape, output_ndim);

    if (total_elements == 0) {
        return;
    }

    // Create inverse_axes array on host
    int inverse_axes[MAX_NDIM];
    for (size_t i = 0; i < output_ndim; ++i) {
        inverse_axes[axes[i]] = i;
    }

    // Allocate device memory for kernel parameters
    size_t *input_shape_d, *input_strides_d, *output_shape_d, *output_strides_d;
    int *inverse_axes_d;

    PLAST_CUDA_CHECK(cudaMalloc((void**)&input_shape_d, input_ndim * sizeof(size_t)));
    PLAST_CUDA_CHECK(cudaMalloc((void**)&input_strides_d, input_ndim * sizeof(size_t)));
    PLAST_CUDA_CHECK(cudaMalloc((void**)&output_shape_d, output_ndim * sizeof(size_t)));
    PLAST_CUDA_CHECK(cudaMalloc((void**)&output_strides_d, output_ndim * sizeof(size_t)));
    PLAST_CUDA_CHECK(cudaMalloc((void**)&inverse_axes_d, output_ndim * sizeof(int)));

    // Copy host data to device
    PLAST_CUDA_CHECK(cudaMemcpy(input_shape_d, input_shape, input_ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    PLAST_CUDA_CHECK(cudaMemcpy(input_strides_d, input_strides, input_ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    PLAST_CUDA_CHECK(cudaMemcpy(output_shape_d, output_shape, output_ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    PLAST_CUDA_CHECK(cudaMemcpy(output_strides_d, output_strides, output_ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    PLAST_CUDA_CHECK(cudaMemcpy(inverse_axes_d, inverse_axes, output_ndim * sizeof(int), cudaMemcpyHostToDevice));

    // Configure kernel launch
    int threadsPerBlock = 256;
    int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    transpose_forward_kernel_cuda_impl<<<numBlocks, threadsPerBlock>>>(
        output_data, input_data,
        input_shape_d, input_strides_d, input_ndim,
        output_shape_d, output_strides_d, output_ndim,
        inverse_axes_d,
        item_size, total_elements);

    PLAST_CUDA_CHECK(cudaGetLastError());
    PLAST_CUDA_CHECK(cudaDeviceSynchronize());

    // Free device memory
    PLAST_CUDA_CHECK(cudaFree(input_shape_d));
    PLAST_CUDA_CHECK(cudaFree(input_strides_d));
    PLAST_CUDA_CHECK(cudaFree(output_shape_d));
    PLAST_CUDA_CHECK(cudaFree(output_strides_d));
    PLAST_CUDA_CHECK(cudaFree(inverse_axes_d));
}