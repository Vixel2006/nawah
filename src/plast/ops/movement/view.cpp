#include "plast/ops/movement/view.h"

#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor ViewOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("ViewOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    return input_tensor->reshape(new_shape_);
}

tensor::Tensor ViewOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    return execute_cpu(inputs);
}

void ViewOperation::backward(const tensor::Tensor& grad_output,
                             const tensor::Tensor& output,
                             std::vector<tensor::Tensor*>& inputs) const
{
    throw std::runtime_error("Not implemented");
}

} // namespace ops
} // namespace plast
