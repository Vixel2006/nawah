#include "plast/ops/unary/leaky_relu.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h"
#include "plast/core/types.h"
#include "plast/kernels/cpu/unary_kernels.h"
#include "plast/kernels/cuda/unary_kernels.h"

namespace plast
{
namespace ops
{

tensor::Tensor
LeakyReluOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& input = *inputs[0];

    if (input.device() != core::DeviceType::CPU)
    {
        throw std::runtime_error("Input tensor must be on CPU for CPU execution.");
    }

    size_t num_elements = input.num_elements();
    core::DType dtype = input.dtype();

    // Allocate output tensor
    tensor::Tensor output(input.shape(), dtype, core::DeviceType::CPU);

    // Dispatch to type-specific C CPU kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cpu_leaky_relu_kernel_float(output.data_as<float>(), input.data_as<const float>(),
                                          num_elements, alpha_);
        break;
    case core::DType::INT32:
        plast_cpu_leaky_relu_kernel_int32(output.data_as<int32_t>(), input.data_as<const int32_t>(),
                                          num_elements, alpha_);
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for LeakyRelu operation on CPU.");
    }

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
}

tensor::Tensor
LeakyReluOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& input = *inputs[0];

    if (input.device() != core::DeviceType::CUDA)
    {
        throw std::runtime_error("Input tensor must be on CUDA for CUDA execution.");
    }

    size_t num_elements = input.num_elements();
    core::DType dtype = input.dtype();

    // Allocate output tensor on CUDA device
    tensor::Tensor output(input.shape(), dtype, core::DeviceType::CUDA);

    // Dispatch to type-specific CUDA kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cuda_leaky_relu_kernel_float(output.data_as<float>(), input.data_as<const float>(),
                                           num_elements, alpha_);
        break;
    case core::DType::INT32:
        // plast_cuda_leaky_relu_kernel_int32(output.data_as<int32_t>(), input.data_as<const
        // int32_t>(),
        //                                 num_elements, alpha_);
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for LeakyRelu operation on CUDA.");
    }

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute LeakyRelu operation on CUDA device.");
#endif
}

std::vector<tensor::Tensor>
LeakyReluOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                 const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("LeakyRelu backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // d(leaky_relu(x))/dx = 1 if x > 0, else alpha
        // grad_input = grad_output * (input > 0 ? 1 : alpha)
        throw std::runtime_error("LeakyRelu backward_cpu: Gradient for input not yet implemented "
                                 "(requires element-wise comparison).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
LeakyReluOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                  const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("LeakyRelu backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // d(leaky_relu(x))/dx = 1 if x > 0, else alpha
        // grad_input = grad_output * (input > 0 ? 1 : alpha)
        throw std::runtime_error("LeakyRelu backward_cuda: Gradient for input not yet implemented "
                                 "(requires element-wise comparison).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute LeakyRelu backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
