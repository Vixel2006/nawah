#include "plast/ops/movement/squeeze.h"

#include <numeric>

namespace plast
{
namespace ops
{

tensor::Tensor SqueezeOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("SqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (N >= output_shape.size())
    {
        throw std::runtime_error("Squeeze dimension out of bounds.");
    }

    if (output_shape[N] == 1)
    {
        output_shape.erase(output_shape.begin() + N);
        output_strides.erase(output_strides.begin() + N);
    }
    else
    {
        return input_tensor->view(input_tensor->shape(), input_tensor->strides());
    }

    return input_tensor->reshape(output_shape, output_strides);
}

tensor::Tensor
SqueezeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    return execute_cpu(inputs);
}

std::vector<tensor::Tensor>
SqueezeOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                               const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Squeeze backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The backward of squeeze is unsqueeze.
        // The grad_output needs to be unsqueezed back to the original input's shape.
        throw std::runtime_error(
            "Squeeze backward_cpu: Gradient for input not yet implemented (requires unsqueeze).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
SqueezeOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Squeeze backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The backward of squeeze is unsqueeze.
        // The grad_output needs to be unsqueezed back to the original input's shape.
        throw std::runtime_error(
            "Squeeze backward_cuda: Gradient for input not yet implemented (requires unsqueeze).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Squeeze backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
