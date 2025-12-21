#pragma once
// CPU movement backward kernels

#include "plast/core/types.h"
#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

// CPU kernel for broadcast backward
void cpu_broadcast_backward_kernel(void* grad_in_data,
                                   const void* grad_out_data,
                                   const size_t* grad_out_shape,
                                   const size_t* grad_out_strides,
                                   size_t grad_out_ndim,
                                   const size_t* input_shape, // Original input shape
                                   size_t input_ndim,
                                   size_t item_size);

// CPU kernel for expand backward
void cpu_expand_backward_kernel(void* grad_in_data,
                                const void* grad_out_data,
                                const size_t* grad_out_shape,
                                const size_t* grad_out_strides,
                                size_t grad_out_ndim,
                                const size_t* input_shape, // Original input shape
                                size_t input_ndim,
                                size_t item_size);

// Placeholder for transpose backward if it involves data movement
void cpu_transpose_backward_kernel(void* grad_in_data,
                                   const void* grad_out_data,
                                   const size_t* grad_out_shape,
                                   const size_t* grad_out_strides,
                                   size_t grad_out_ndim,
                                   const size_t* input_shape, // Original input shape
                                   size_t input_ndim,
                                   const int* axes, // Original axes for transpose
                                   size_t item_size);

// Placeholder for squeeze backward (essentially an unsqueeze)
void cpu_squeeze_backward_kernel(void* grad_in_data,
                                  const void* grad_out_data,
                                  const size_t* grad_out_shape,
                                  const size_t* grad_out_strides,
                                  size_t grad_out_ndim,
                                  const size_t* input_shape, // Original input shape
                                  size_t input_ndim,
                                  const int* squeeze_dims, // Dimensions that were squeezed
                                  size_t num_squeeze_dims,
                                  size_t item_size);

// Placeholder for unsqueeze backward (essentially a squeeze)
void cpu_unsqueeze_backward_kernel(void* grad_in_data,
                                    const void* grad_out_data,
                                    const size_t* grad_out_shape,
                                    const size_t* grad_out_strides,
                                    size_t grad_out_ndim,
                                    const size_t* input_shape, // Original input shape
                                    size_t input_ndim,
                                    const int* unsqueeze_dims, // Dimensions that were unsqueezed
                                    size_t num_unsqueeze_dims,
                                    size_t item_size);

// Placeholder for view backward (often a reshape)
void cpu_view_backward_kernel(void* grad_in_data,
                              const void* grad_out_data,
                              const size_t* grad_out_shape,
                              const size_t* grad_out_strides,
                              size_t grad_out_ndim,
                              const size_t* input_shape, // Original input shape
                              size_t input_ndim,
                              size_t item_size);


#ifdef __cplusplus
}
#endif