#include "plast/kernels/cuda/unary_backward_kernels.h"

// CUDA backward kernels for relu

extern "C" void plast_cuda_relu_backward_kernel_float(float* grad_in, const float* grad_out,
                                                      const float* in, size_t num_elements)
{
    // TODO: Implement
}

extern "C" void plast_cuda_relu_backward_kernel_int32(int32_t* grad_in, const int32_t* grad_out,
                                                      const int32_t* in, size_t num_elements)
{
    // TODO: Implement
}