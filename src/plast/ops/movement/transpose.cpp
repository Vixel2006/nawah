#include "plast/ops/movement/transpose.h"
#include <iostream>
#include <numeric>

namespace plast
{
namespace ops
{

tensor::Tensor
TransposeOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("TransposeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (N >= output_shape.size() || M >= output_shape.size())
    {
        throw std::runtime_error("Transpose dimensions out of bounds.");
    }

    std::swap(output_shape[N], output_shape[M]);
    std::swap(output_strides[N], output_strides[M]);

    return input_tensor->reshape(output_shape, output_strides);
}

tensor::Tensor
TransposeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("TransposeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (N >= output_shape.size() || M >= output_shape.size())
    {
        throw std::runtime_error("Transpose dimensions out of bounds.");
    }

    std::swap(output_shape[N], output_shape[M]);
    std::swap(output_strides[N], output_strides[M]);

    return input_tensor->reshape(output_shape, output_strides);
}

} // namespace ops
} // namespace plast
