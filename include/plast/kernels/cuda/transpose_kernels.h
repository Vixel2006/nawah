#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_transpose_forward_kernel(void* output_data, const void* input_data,
                                   const size_t* input_shape, const size_t* input_strides,
                                   size_t input_ndim,
                                   const size_t* output_shape, const size_t* output_strides,
                                   size_t output_ndim,
                                   const int* axes, // Original axes for transpose
                                   size_t item_size);

#ifdef __cplusplus
}
#endif
