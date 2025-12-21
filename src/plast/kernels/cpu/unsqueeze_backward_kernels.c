#include "plast/kernels/cpu/movement_backward_kernels.h"

// CPU backward kernels for unsqueeze

void cpu_unsqueeze_backward_kernel(void* grad_in_data, const void* grad_out_data,
                                   const size_t* grad_out_shape, const size_t* grad_out_strides,
                                   size_t grad_out_ndim,
                                   const size_t* input_shape, // Original input shape
                                   size_t input_ndim,
                                   const int* unsqueeze_dims, // Dimensions that were unsqueezed
                                   size_t num_unsqueeze_dims, size_t item_size)
{
    // TODO: Implement
}