#include "plast/core/shape_utils_c.h"
#include "plast/core/shape_utils_cpp.h"
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace core
{

std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1,
                                     const std::vector<size_t>& shape2)
{
    size_t ndim1 = shape1.size();
    size_t ndim2 = shape2.size();
    size_t output_ndim = std::max(ndim1, ndim2);
    std::vector<size_t> output_shape(output_ndim);

    for (size_t i = 0; i < output_ndim; ++i)
    {
        size_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
        size_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
        {
            throw std::runtime_error("Shapes are not broadcastable.");
        }
        output_shape[output_ndim - 1 - i] = std::max(dim1, dim2);
    }
    return output_shape;
}

std::vector<size_t> calculate_strides(const std::vector<size_t>& shape)
{
    size_t ndim = shape.size();
    std::vector<size_t> strides(ndim);
    if (ndim == 0)
    {
        return strides;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<size_t> compute_strides(const std::vector<size_t>& original_shape,
                                    const std::vector<size_t>& target_shape)
{
    size_t original_ndim = original_shape.size();
    size_t target_ndim = target_shape.size();
    std::vector<size_t> strides(target_ndim);

    // Calculate strides for the original shape
    std::vector<size_t> original_strides = calculate_strides(original_shape);

    // Iterate through the dimensions of the target shape (from right to left)
    for (int i = target_ndim - 1; i >= 0; --i)
    {
        // Corresponding dimension in the original shape
        int original_dim_idx = original_ndim - 1 - (target_ndim - 1 - i);

        if (original_dim_idx >= 0)
        {
            // If the dimension exists in the original shape
            if (original_shape[original_dim_idx] == target_shape[i])
            {
                // If dimensions match, use the original stride
                strides[i] = original_strides[original_dim_idx];
            }
            else if (original_shape[original_dim_idx] == 1)
            {
                // If original dimension is 1, it's broadcasted, so stride is 0
                strides[i] = 0;
            }
            else
            {
                // This case should ideally not happen if broadcast_shapes was called first
                throw std::runtime_error("Error in compute_strides: non-broadcastable dimension.");
            }
        }
        else
        {
            // If the dimension does not exist in the original shape (prepended 1s), stride is 0
            strides[i] = 0;
        }
    }
    return strides;
}

} // namespace core
} // namespace plast

// C-compatible implementations
size_t get_index(const size_t* current_indices, const size_t* strides, size_t ndim)
{
    size_t index = 0;
    for (size_t i = 0; i < ndim; ++i)
    {
        // If stride is 0, it means this dimension was broadcasted from size 1.
        // In this case, the index for this dimension should effectively be 0,
        // so we don't multiply current_indices[i] by strides[i].
        // The element at index 0 of the original dimension is always accessed.
        if (strides[i] != 0)
        {
            index += current_indices[i] * strides[i];
        }
    }
    return index;
}

void increment_indices(size_t* current_indices, const size_t* shape, size_t ndim)
{
    for (int i = ndim - 1; i >= 0; --i)
    {
        current_indices[i]++;
        if (current_indices[i] < shape[i])
        {
            break;
        }
        else
        {
            current_indices[i] = 0;
        }
    }
}

size_t get_total_elements(const size_t* shape, size_t ndim)
{
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        total_elements *= shape[i];
    }
    return total_elements;
}

namespace plast
{
namespace core
{
std::vector<size_t> get_effective_broadcast_strides(
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& input_actual_strides,
    const std::vector<size_t>& target_shape)
{
    size_t input_ndim = input_shape.size();
    size_t target_ndim = target_shape.size();
    std::vector<size_t> result_strides(target_ndim);

    for (int i = target_ndim - 1; i >= 0; --i)
    {
        int input_idx = i - (target_ndim - input_ndim);

        if (input_idx < 0)
        {
            // Dimension was prepended, broadcast from 1 (stride 0)
            result_strides[i] = 0;
        }
        else
        {
            if (input_shape[input_idx] == target_shape[i])
            {
                // Dimension matches, use actual stride from the input tensor
                result_strides[i] = input_actual_strides[input_idx];
            }
            else if (input_shape[input_idx] == 1)
            {
                // Input dimension is 1, it's broadcasted, so stride is 0
                result_strides[i] = 0;
            }
            else
            {
                // This case should ideally not happen if broadcast_shapes was called first
                throw std::runtime_error(
                    "Error in get_effective_broadcast_strides: non-broadcastable dimension.");
            }
        }
    }
    return result_strides;
}

std::vector<size_t> get_reduced_shape(const std::vector<size_t>& grad_shape,
                                      const std::vector<size_t>& input_shape)
{
    size_t grad_ndim = grad_shape.size();
    size_t input_ndim = input_shape.size();
    std::vector<size_t> reduced_shape(grad_ndim);

    // Pad input_shape with 1s on the left to match grad_ndim
    std::vector<size_t> padded_input_shape(grad_ndim, 1);
    std::copy(input_shape.begin(), input_shape.end(),
              padded_input_shape.begin() + (grad_ndim - input_ndim));

    for (size_t i = 0; i < grad_ndim; ++i)
    {
        if (padded_input_shape[i] == grad_shape[i])
        {
            reduced_shape[i] = grad_shape[i];
        }
        else if (padded_input_shape[i] == 1)
        {
            // This dimension was broadcasted from 1 in input_shape to grad_shape[i]
            // We need to sum along this dimension, so its size in the reduced shape should be 1.
            reduced_shape[i] = 1;
        }
        else
        {
            // This case should not happen if shapes are broadcastable
            throw std::runtime_error("Error in get_reduced_shape: non-broadcastable dimension.");
        }
    }
    return reduced_shape;
}

std::vector<int> get_broadcasted_dims(const std::vector<size_t>& input_shape,
                                      const std::vector<size_t>& output_shape)
{
    std::vector<int> broadcasted_dims;
    size_t input_ndim = input_shape.size();
    size_t output_ndim = output_shape.size();

    // Iterate from the rightmost dimension
    for (int i = 0; i < output_ndim; ++i)
    {
        int input_dim_idx = i - (output_ndim - input_ndim);
        if (input_dim_idx < 0)
        {
            // Dimension was prepended, so it was broadcasted
            broadcasted_dims.push_back(i);
        }
        else if (input_shape[input_dim_idx] == 1 && output_shape[i] > 1)
        {
            // Input dimension was 1 and broadcasted to a larger output dimension
            broadcasted_dims.push_back(i);
        }
    }
    return broadcasted_dims;
}

} // namespace core
} // namespace plast
