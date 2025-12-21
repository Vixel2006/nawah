#include "plast/kernels/cpu/reduction_kernels.h"
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

// Helper function to calculate total number of elements
static size_t calculate_total_elements(const size_t* shape, size_t ndim)
{
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        total *= shape[i];
    }
    return total;
}

// Helper function to calculate strides
// NOTE: This function assumes row-major order (C-style)
static void calculate_strides(const size_t* shape, size_t ndim, size_t* strides)
{
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// --- Full Reduction Kernels (float) ---

void plast_cpu_sum_full_reduction_float(const float* input_data, float* output_data,
                                        const size_t* input_shape, size_t input_ndim)
{
    size_t total_elements = calculate_total_elements(input_shape, input_ndim);
    if (total_elements == 0)
    {
        output_data[0] = 0.0f; // Sum of empty set is 0
        return;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < total_elements; ++i)
    {
        sum += input_data[i];
    }
    output_data[0] = sum;
}

// --- Full Reduction Kernels (int32) ---

void plast_cpu_sum_full_reduction_int32(const int32_t* input_data, int32_t* output_data,
                                        const size_t* input_shape, size_t input_ndim)
{
    size_t total_elements = calculate_total_elements(input_shape, input_ndim);
    if (total_elements == 0)
    {
        output_data[0] = 0;
        return;
    }

    int32_t sum = 0;
    for (size_t i = 0; i < total_elements; ++i)
    {
        sum += input_data[i];
    }
    output_data[0] = sum;
}

// --- Reduction along a dimension kernels (float) ---

void plast_cpu_sum_reduction_dim_float(const float* input_data, float* output_data,
                                       const size_t* input_shape, size_t input_ndim,
                                       const size_t* output_shape, size_t output_ndim, int dim)
{
    size_t reduction_dim_size = input_shape[dim];
    size_t outer_size = 1;
    for (size_t i = 0; i < dim; ++i)
    {
        outer_size *= input_shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < input_ndim; ++i)
    {
        inner_size *= input_shape[i];
    }

    for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
    {
        for (size_t inner_idx = 0; inner_idx < inner_size; ++inner_idx)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < reduction_dim_size; ++k)
            {
                size_t input_idx =
                    outer_idx * reduction_dim_size * inner_size + k * inner_size + inner_idx;
                sum += input_data[input_idx];
            }
            bool keepdim = (input_ndim == output_ndim);
            size_t output_idx;
            if (keepdim)
            {
                output_idx = outer_idx * 1 * inner_size + inner_idx;
            }
            else
            {
                output_idx = outer_idx * inner_size + inner_idx;
            }
            output_data[output_idx] = sum;
        }
    }
}

// --- Reduction along a dimension kernels (int32) ---

void plast_cpu_sum_reduction_dim_int32(const int32_t* input_data, int32_t* output_data,
                                       const size_t* input_shape, size_t input_ndim,
                                       const size_t* output_shape, size_t output_ndim, int dim)
{
    size_t reduction_dim_size = input_shape[dim];
    size_t outer_size = 1;
    for (size_t i = 0; i < dim; ++i)
    {
        outer_size *= input_shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < input_ndim; ++i)
    {
        inner_size *= input_shape[i];
    }

    for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
    {
        for (size_t inner_idx = 0; inner_idx < inner_size; ++inner_idx)
        {
            int32_t sum = 0;
            for (size_t k = 0; k < reduction_dim_size; ++k)
            {
                size_t input_idx =
                    outer_idx * reduction_dim_size * inner_size + k * inner_size + inner_idx;
                sum += input_data[input_idx];
            }
            bool keepdim = (input_ndim == output_ndim);
            size_t output_idx;
            if (keepdim)
            {
                output_idx = outer_idx * 1 * inner_size + inner_idx;
            }
            else
            {
                output_idx = outer_idx * inner_size + inner_idx;
            }
            output_data[output_idx] = sum;
        }
    }
}
