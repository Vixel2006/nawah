#include "plast/ops/movement/unsqueeze.h"

#include <numeric>

namespace plast
{
namespace ops
{

tensor::Tensor
UnsqueezeOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("UnsqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    std::vector<size_t> output_shape = input_tensor->shape();
    std::vector<size_t> output_strides = input_tensor->strides();

    if (dim_ > output_shape.size())
    {
        throw std::runtime_error("Unsqueeze dimension out of bounds.");
    }

    output_shape.insert(output_shape.begin() + dim_, 1);

    // Calculate new stride for the inserted dimension
    size_t new_stride;
    if (dim_ < output_strides.size())
    {
        new_stride = output_strides[dim_];
    }
    else
    {
        // If inserting at the end, the new stride is 1 (for a dimension of size 1)
        new_stride = 1;
    }
    output_strides.insert(output_strides.begin() + dim_, new_stride);

    // Create a new Tensor that views the same data but with new shape and strides
    return input_tensor->reshape(output_shape, output_strides);
}

tensor::Tensor
UnsqueezeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    return execute_cpu(inputs);
}

std::vector<tensor::Tensor>
UnsqueezeOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                 const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Unsqueeze backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The backward of unsqueeze is squeeze.
        // The grad_output needs to be squeezed back to the original input's shape.
        throw std::runtime_error(
            "Unsqueeze backward_cpu: Gradient for input not yet implemented (requires squeeze).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
UnsqueezeOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                  const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Unsqueeze backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The backward of unsqueeze is squeeze.
        // The grad_output needs to be squeezed back to the original input's shape.
        throw std::runtime_error(
            "Unsqueeze backward_cuda: Gradient for input not yet implemented (requires squeeze).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Unsqueeze backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
