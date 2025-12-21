#include "plast/kernels/cuda/init_backward_kernels.h"

// CUDA backward kernels for init

extern "C" void plast_cuda_full_backward_kernel_float(float* grad_value, const float* grad_out,
                                                      size_t num_elements)
{
    // TODO: Implement
}

extern "C" void plast_cuda_full_backward_kernel_int32(int32_t* grad_value, const int32_t* grad_out,
                                                      size_t num_elements)
{
    // TODO: Implement
}