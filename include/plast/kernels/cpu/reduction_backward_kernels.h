#pragma once
// CPU reduction backward kernels

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Full reduction backward kernels for float
    void plast_cpu_min_full_reduction_backward_float(float* grad_in, const float* grad_out,
                                                     const float* input_data, const float* output_data,
                                                     const size_t* input_shape, size_t input_ndim);

    void plast_cpu_mean_full_reduction_backward_float(float* grad_in, const float* grad_out,
                                                      const size_t* input_shape, size_t input_ndim);

    void plast_cpu_sum_full_reduction_backward_float(float* grad_in, const float* grad_out,
                                                     const size_t* input_shape, size_t input_ndim);

    void plast_cpu_max_full_reduction_backward_float(float* grad_in, const float* grad_out,
                                                     const float* input_data, const float* output_data,
                                                     const size_t* input_shape, size_t input_ndim);

    // Full reduction backward kernels for int32
    void plast_cpu_min_full_reduction_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                     const int32_t* input_data, const int32_t* output_data,
                                                     const size_t* input_shape, size_t input_ndim);

    void plast_cpu_mean_full_reduction_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                      const size_t* input_shape, size_t input_ndim);

    void plast_cpu_sum_full_reduction_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                     const size_t* input_shape, size_t input_ndim);

    void plast_cpu_max_full_reduction_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                     const int32_t* input_data, const int32_t* output_data,
                                                     const size_t* input_shape, size_t input_ndim);

    // Reduction along a dimension backward kernels for float
    void plast_cpu_min_reduction_dim_backward_float(float* grad_in, const float* grad_out,
                                                    const float* input_data, const float* output_data,
                                                    const size_t* input_shape, size_t input_ndim,
                                                    const size_t* output_shape, size_t output_ndim, int dim);

    void plast_cpu_mean_reduction_dim_backward_float(float* grad_in, const float* grad_out,
                                                     const size_t* input_shape, size_t input_ndim,
                                                     const size_t* output_shape, size_t output_ndim,
                                                     int dim);

    void plast_cpu_sum_reduction_dim_backward_float(float* grad_in, const float* grad_out,
                                                    const size_t* input_shape, size_t input_ndim,
                                                    const size_t* output_shape, size_t output_ndim, int dim);

    void plast_cpu_max_reduction_dim_backward_float(float* grad_in, const float* grad_out,
                                                    const float* input_data, const float* output_data,
                                                    const size_t* input_shape, size_t input_ndim,
                                                    const size_t* output_shape, size_t output_ndim, int dim);

    // Reduction along a dimension backward kernels for int32
    void plast_cpu_min_reduction_dim_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                    const int32_t* input_data, const int32_t* output_data,
                                                    const size_t* input_shape, size_t input_ndim,
                                                    const size_t* output_shape, size_t output_ndim, int dim);

    void plast_cpu_mean_reduction_dim_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                     const size_t* input_shape, size_t input_ndim,
                                                     const size_t* output_shape, size_t output_ndim,
                                                     int dim);

    void plast_cpu_sum_reduction_dim_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                    const size_t* input_shape, size_t input_ndim,
                                                    const size_t* output_shape, size_t output_ndim, int dim);

    void plast_cpu_max_reduction_dim_backward_int32(int32_t* grad_in, const int32_t* grad_out,
                                                    const int32_t* input_data, const int32_t* output_data,
                                                    const size_t* input_shape, size_t input_ndim,
                                                    const size_t* output_shape, size_t output_ndim, int dim);

#ifdef __cplusplus
}
#endif