#pragma once
// CUDA unary backward kernels

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // CUDA kernel for element-wise absolute backward of float tensors
    void plast_cuda_abs_backward_kernel_float(float* grad_in, const float* grad_out, const float* in,
                                              size_t num_elements);

    // CUDA kernel for element-wise absolute backward of int32_t tensors
    void plast_cuda_abs_backward_kernel_int32(int32_t* grad_in, const int32_t* grad_out, const int32_t* in,
                                              size_t num_elements);

    // CUDA kernel for element-wise exponential backward of float tensors
    void plast_cuda_exp_backward_kernel_float(float* grad_in, const float* grad_out, const float* in,
                                              size_t num_elements);

    // CUDA kernel for element-wise exponential backward of int32_t tensors
    void plast_cuda_exp_backward_kernel_int32(int32_t* grad_in, const int32_t* grad_out, const int32_t* in,
                                              size_t num_elements);

    // CUDA kernel for element-wise logarithm backward of float tensors
    void plast_cuda_log_backward_kernel_float(float* grad_in, const float* grad_out, const float* in,
                                              size_t num_elements);

    // CUDA kernel for element-wise logarithm backward of int32_t tensors
    void plast_cuda_log_backward_kernel_int32(int32_t* grad_in, const int32_t* grad_out, const int32_t* in,
                                              size_t num_elements);

    // CUDA kernel for element-wise ReLU backward of float tensors
    void plast_cuda_relu_backward_kernel_float(float* grad_in, const float* grad_out, const float* in,
                                               size_t num_elements);

    // CUDA kernel for element-wise ReLU backward of int32_t tensors
    void plast_cuda_relu_backward_kernel_int32(int32_t* grad_in, const int32_t* grad_out, const int32_t* in,
                                               size_t num_elements);

    // CUDA kernel for element-wise Leaky ReLU backward of float tensors
    void plast_cuda_leaky_relu_backward_kernel_float(float* grad_in, const float* grad_out, const float* in,
                                                     size_t num_elements, float alpha);

    // CUDA kernel for element-wise Leaky ReLU backward of int32_t tensors
    void plast_cuda_leaky_relu_backward_kernel_int32(int32_t* grad_in, const int32_t* grad_out, const int32_t* in,
                                                     size_t num_elements, float alpha);

#ifdef __cplusplus
}
#endif