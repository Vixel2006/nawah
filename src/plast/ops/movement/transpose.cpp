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
    tensor::Tensor output = [&]() {
        std::vector<size_t> output_shape = input_tensor->shape();
        std::vector<size_t> output_strides = input_tensor->strides();

        if (N >= output_shape.size() || M >= output_shape.size())
        {
            throw std::runtime_error("Transpose dimensions out of bounds.");
        }

        std::swap(output_shape[N], output_shape[M]);
        std::swap(output_strides[N], output_strides[M]);

        return input_tensor->reshape(output_shape, output_strides);
    }();

    if (input_tensor->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
}

tensor::Tensor
TransposeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    return execute_cpu(inputs);
}

std::vector<tensor::Tensor>
TransposeOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                 const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Transpose backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The backward of transpose is transpose itself, with the same dimensions swapped.
        // grad_input = grad_output.transpose(N, M)
        throw std::runtime_error(
            "Transpose backward_cpu: Gradient for input not yet implemented (requires transpose).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
TransposeOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                  const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Transpose backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The backward of transpose is transpose itself, with the same dimensions swapped.
        // grad_input = grad_output.transpose(N, M)
        throw std::runtime_error("Transpose backward_cuda: Gradient for input not yet implemented "
                                 "(requires transpose).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Transpose backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
