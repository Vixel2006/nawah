#include "plast/ops/reduction/max.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h" // Added for strided operations
#include "plast/core/types.h"
#include "plast/kernels/cpu/reduction_kernels.h"
#include "plast/kernels/cuda/reduction_kernels.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor MaxOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    bool input_contiguous = input.is_contiguous();

    if (!input_contiguous)
    {
        throw std::runtime_error(
            "Max operation on CPU does not yet support non-contiguous inputs.");
    }

    // Dispatch to type-specific C CPU kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cpu_max_full_reduction_float(input.data_as<const float>(),
                                               output.data_as<float>(), input.shape().data(),
                                               input.shape().size());
        }
        else
        {
            plast_cpu_max_reduction_dim_float(input.data_as<const float>(), output.data_as<float>(),
                                              input.shape().data(), input.shape().size(),
                                              output.shape().data(), output.shape().size(), dim_);
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            plast_cpu_max_full_reduction_int32(input.data_as<const int32_t>(),
                                               output.data_as<int32_t>(), input.shape().data(),
                                               input.shape().size());
        }
        else
        {
            plast_cpu_max_reduction_dim_int32(
                input.data_as<const int32_t>(), output.data_as<int32_t>(), input.shape().data(),
                input.shape().size(), output.shape().data(), output.shape().size(), dim_);
        }
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for Max operation on CPU.");
    }

    return output;
}

tensor::Tensor MaxOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CUDA);

    bool input_contiguous = input.is_contiguous();

    if (!input_contiguous)
    {
        throw std::runtime_error(
            "Max operation on CUDA does not yet support non-contiguous inputs.");
    }

    // Dispatch to type-specific C CUDA kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cuda_max_full_reduction_float(input.data_as<const float>(),
                                                output.data_as<float>(), input.shape().data(),
                                                input.shape().size());
        }
        else
        {
            throw std::runtime_error("CUDA Max reduction dim float operation not yet implemented "
                                     "for contiguous inputs.");
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            throw std::runtime_error("CUDA Max full reduction int32 operation not yet implemented "
                                     "for contiguous inputs.");
        }
        else
        {
            throw std::runtime_error("CUDA Max reduction dim int32 operation not yet implemented "
                                     "for contiguous inputs.");
        }
        break;
    default:
        throw std::runtime_error("Unsupported DType for Max operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Max operation on CUDA device.");
#endif
}

std::vector<tensor::Tensor>
MaxOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                           const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Max backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The gradient of max(x) with respect to x is 1 at the index of the maximum value, and 0
        // elsewhere. This requires knowing the indices of the maximum values from the forward pass.
        throw std::runtime_error("Max backward_cpu: Gradient for input not yet implemented "
                                 "(requires indices from forward pass).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
MaxOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                            const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Max backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // The gradient of max(x) with respect to x is 1 at the index of the maximum value, and 0
        // elsewhere. This requires knowing the indices of the maximum values from the forward pass.
        throw std::runtime_error("Max backward_cuda: Gradient for input not yet implemented "
                                 "(requires indices from forward pass).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Max backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
