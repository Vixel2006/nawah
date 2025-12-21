#include "plast/kernels/cpu/binary_backward_kernels.h"

// CPU backward kernels for matmul

void plast_cpu_matmul_backward_kernel_float(float* grad_in1, float* grad_in2, const float* grad_out,
                                            const float* in1, const float* in2, const int B,
                                            const int N, const int M, const int K)
{
    // TODO: Implement
}

void plast_cpu_matmul_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2,
                                            const int32_t* grad_out, const int32_t* in1,
                                            const int32_t* in2, const int B, const int N,
                                            const int M, const int K)
{
    // TODO: Implement
}