#include "plast/kernels/cpu/transpose_kernels.h"
#include "plast/core/shape_utils_c.h"
#include <string.h> // For memcpy

// CPU forward kernels for transpose

void cpu_transpose_forward_kernel(void* output_data, const void* input_data,
                                   const size_t* input_shape, const size_t* input_strides,
                                   size_t input_ndim,
                                   const size_t* output_shape, const size_t* output_strides,
                                   size_t output_ndim,
                                   const int* axes, // Original axes for transpose
                                   size_t item_size)
{
    size_t total_elements = get_total_elements(output_shape, output_ndim);

    if (total_elements == 0) {
        return;
    }

    // Create inverse_axes array
    int inverse_axes[output_ndim];
    for (size_t i = 0; i < output_ndim; ++i) {
        inverse_axes[axes[i]] = i;
    }

    size_t current_output_indices[output_ndim];
    for (size_t i = 0; i < output_ndim; ++i) {
        current_output_indices[i] = 0;
    }

    for (size_t i = 0; i < total_elements; ++i) {
        size_t output_linear_index = get_index(current_output_indices, output_strides, output_ndim);

        size_t input_indices[input_ndim];
        for (size_t j = 0; j < input_ndim; ++j) {
            input_indices[j] = current_output_indices[inverse_axes[j]];
        }
        size_t input_linear_index = get_index(input_indices, input_strides, input_ndim);

        memcpy((char*)output_data + output_linear_index * item_size,
               (const char*)input_data + input_linear_index * item_size,
               item_size);

        increment_indices(current_output_indices, output_shape, output_ndim);
    }
}