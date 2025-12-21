#include "plast/kernels/cuda/binary_backward_kernels.h"

// CUDA backward kernels for matmul

extern "C" void plast_cuda_matmul_backward_kernel_float(float* grad_in1, float* grad_in2,
                                                        const float* grad_out, const float* in1,
                                                        const float* in2, const int B, const int N,
                                                        const int M, const int K)
{
    // TODO: Implement
}

extern "C" void plast_cuda_matmul_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2,
                                                        const int32_t* grad_out, const int32_t* in1,
                                                        const int32_t* in2, const int B,
                                                        const int N, const int M, const int K)
{
    // TODO: Implement
}