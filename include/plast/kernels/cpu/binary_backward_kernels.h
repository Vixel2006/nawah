#pragma once
// CPU binary backward kernels

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // CPU kernel for element-wise addition backward of float tensors
    void plast_cpu_add_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                             size_t num_elements);

    // CPU kernel for element-wise addition backward of int32_t tensors
    void plast_cpu_add_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2, const int32_t* grad_out,
                                             size_t num_elements);

    // CPU kernel for element-wise subtraction backward of float tensors
    void plast_cpu_sub_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                             size_t num_elements);

    // CPU kernel for element-wise subtraction backward of int32_t tensors
    void plast_cpu_sub_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2, const int32_t* grad_out,
                                             size_t num_elements);

    // CPU kernel for element-wise multiplication backward of float tensors
    void plast_cpu_mul_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                             const float* in1, const float* in2, size_t num_elements);

    // CPU kernel for element-wise multiplication backward of int32_t tensors
    void plast_cpu_mul_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2, const int32_t* grad_out,
                                             const int32_t* in1, const int32_t* in2, size_t num_elements);

    // CPU kernel for matrix multiplication backward of float tensors
    void plast_cpu_matmul_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                                const float* in1, const float* in2,
                                                const int B, const int N, const int M, const int K);

    // CPU kernel for matrix multiplication backward of int32_t tensors
    void plast_cpu_matmul_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2, const int32_t* grad_out,
                                                const int32_t* in1, const int32_t* in2,
                                                const int B, const int N, const int M, const int K);

#ifdef __cplusplus
}
#endif