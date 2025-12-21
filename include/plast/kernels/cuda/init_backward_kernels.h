#pragma once
// CUDA init backward kernels

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // If an initialization op like 'full' takes a scalar value,
    // its backward would sum the grad_out to get the gradient for that scalar.
    // For other initialization ops (zeros, ones, rand, etc.), there's typically no
    // input to backpropagate through, so their gradients would be zero.
    void plast_cuda_full_backward_kernel_float(float* grad_value, const float* grad_out, size_t num_elements);
    void plast_cuda_full_backward_kernel_int32(int32_t* grad_value, const int32_t* grad_out, size_t num_elements);

#ifdef __cplusplus
}
#endif