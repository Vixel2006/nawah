#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void plast_cuda_sum_full_reduction_float(const float* input_data, float* output_data,
                                             const size_t* input_shape, size_t input_ndim);

    void plast_cuda_mean_full_reduction_float(const float* input_data, float* output_data,
                                              const size_t* input_shape, size_t input_ndim);

    void plast_cuda_max_full_reduction_float(const float* input_data, float* output_data,
                                             const size_t* input_shape, size_t input_ndim);

    void plast_cuda_min_full_reduction_float(const float* input_data, float* output_data,
                                             const size_t* input_shape, size_t input_ndim);
#ifdef __cplusplus
}
#endif
