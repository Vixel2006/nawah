#include "plast/kernels/cuda/binary_backward_kernels.h"

// CUDA backward kernels for add

extern "C" void plast_cuda_add_backward_kernel_float(float* grad_in1, float* grad_in2,
                                                     const float* grad_out, size_t num_elements)
{
    // TODO: Implement
}

extern "C" void plast_cuda_add_backward_kernel_int32(int32_t* grad_in1, int32_t* grad_in2,
                                                     const int32_t* grad_out, size_t num_elements)
{
    // TODO: Implement
}