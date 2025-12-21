#include "plast/kernels/cuda/reduction_backward_kernels.h"

// CUDA backward kernels for mean

extern "C" void plast_cuda_mean_full_reduction_backward_float(float* grad_in, const float* grad_out,
                                                              const size_t* input_shape,
                                                              size_t input_ndim)
{
    // TODO: Implement
}

extern "C" void plast_cuda_mean_full_reduction_backward_int32(int32_t* grad_in,
                                                              const int32_t* grad_out,
                                                              const size_t* input_shape,
                                                              size_t input_ndim)
{
    // TODO: Implement
}

extern "C" void plast_cuda_mean_reduction_dim_backward_float(float* grad_in, const float* grad_out,
                                                             const size_t* input_shape,
                                                             size_t input_ndim,
                                                             const size_t* output_shape,
                                                             size_t output_ndim, int dim)
{
    // TODO: Implement
}

extern "C" void plast_cuda_mean_reduction_dim_backward_int32(
    int32_t* grad_in, const int32_t* grad_out, const size_t* input_shape, size_t input_ndim,
    const size_t* output_shape, size_t output_ndim, int dim)
{
    // TODO: Implement
}