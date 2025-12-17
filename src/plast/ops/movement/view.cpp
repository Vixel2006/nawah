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

std::vector<tensor::Tensor>
ViewOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                            const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("View backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // For a view operation, the gradient simply needs to be reshaped back to the input's shape.
        // This assumes the view operation itself doesn't change the number of elements or their
        // order in a complex way. The grad_output needs to be reshaped to the original input's
        // shape.
        throw std::runtime_error(
            "View backward_cpu: Gradient for input not yet implemented (requires reshape).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
ViewOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                             const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("View backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // For a view operation, the gradient simply needs to be reshaped back to the input's shape.
        // This assumes the view operation itself doesn't change the number of elements or their
        // order in a complex way. The grad_output needs to be reshaped to the original input's
        // shape.
        throw std::runtime_error(
            "View backward_cuda: Gradient for input not yet implemented (requires reshape).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute View backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
