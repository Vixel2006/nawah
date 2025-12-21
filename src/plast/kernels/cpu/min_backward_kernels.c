#include "plast/kernels/cpu/reduction_backward_kernels.h"

// CPU backward kernels for min

void plast_cpu_min_full_reduction_backward_float(float* grad_in, const float* grad_out,
                                                 const float* input_data, const float* output_data,
                                                 const size_t* input_shape, size_t input_ndim)
{
    // TODO: Implement
}

void plast_cpu_min_full_reduction_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                 const int32_t* input_data,
                                                 const int32_t* output_data,
                                                 const size_t* input_shape, size_t input_ndim)
{
    // TODO: Implement
}

void plast_cpu_min_reduction_dim_backward_float(float* grad_in, const float* grad_out,
                                                const float* input_data, const float* output_data,
                                                const size_t* input_shape, size_t input_ndim,
                                                const size_t* output_shape, size_t output_ndim,
                                                int dim)
{
    // TODO: Implement
}

void plast_cpu_min_reduction_dim_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                const int32_t* input_data,
                                                const int32_t* output_data,
                                                const size_t* input_shape, size_t input_ndim,
                                                const size_t* output_shape, size_t output_ndim,
                                                int dim)
{
    // TODO: Implement
}