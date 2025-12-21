#include "plast/ops/reduction/sum.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h" // Added for strided operations
#include "plast/core/types.h"
#include "plast/kernels/cpu/reduction_backward_kernels.h"
#include "plast/kernels/cpu/reduction_kernels.h"
#include "plast/kernels/cuda/reduction_backward_kernels.h"
#include "plast/kernels/cuda/reduction_kernels.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor SumOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    // Dispatch to type-specific C CPU kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cpu_sum_full_reduction_float(input.data_as<const float>(),
                                               output.data_as<float>(), input.shape().data(),
                                               input.shape().size());
        }
        else
        {
            plast_cpu_sum_reduction_dim_float(input.data_as<const float>(), output.data_as<float>(),
                                              input.shape().data(), input.shape().size(),
                                              output.shape().data(), output.shape().size(), dim_);
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            plast_cpu_sum_full_reduction_int32(input.data_as<const int32_t>(),
                                               output.data_as<int32_t>(), input.shape().data(),
                                               input.shape().size());
        }
        else
        {
            plast_cpu_sum_reduction_dim_int32(
                input.data_as<const int32_t>(), output.data_as<int32_t>(), input.shape().data(),
                input.shape().size(), output.shape().data(), output.shape().size(), dim_);
        }
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for Sum operation on CPU.");
    }

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
}

tensor::Tensor SumOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor on CUDA device
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CUDA);

    // Dispatch to type-specific CUDA kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cuda_sum_full_reduction_float(input.data_as<const float>(),
                                                output.data_as<float>(), input.shape().data(),
                                                input.shape().size());
        }
        else
        {
            throw std::runtime_error(
                "CUDA kernel for contiguous dim sum reduction (float) not implemented.");
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            throw std::runtime_error(
                "CUDA kernel for contiguous full sum reduction (int32) not implemented.");
        }
        else
        {
            throw std::runtime_error(
                "CUDA kernel for contiguous dim sum reduction (int32) not implemented.");
        }
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for Sum operation on CUDA.");
    }

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Sum operation on CUDA device.");
#endif
}

std::vector<tensor::Tensor>
SumOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                           const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Sum backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        tensor::Tensor grad_input(input->shape(), input->dtype(), input->device());
        std::memset(grad_input.data(), 0, grad_input.nbytes()); // Initialize to zeros

        // Dispatch to type-specific C CPU kernel
        switch (input->dtype())
        {
        case core::DType::FLOAT32:
            if (full_reduction_)
            {
                plast_cpu_sum_full_reduction_backward_float(
                    grad_input.data_as<float>(), grad_output.data_as<const float>(),
                    inputs[0]->shape().data(), inputs[0]->shape().size());
            }
            else
            {
                plast_cpu_sum_reduction_dim_backward_float(
                    grad_input.data_as<float>(), grad_output.data_as<const float>(),
                    inputs[0]->shape().data(), inputs[0]->shape().size(), output.shape().data(),
                    output.shape().size(), dim_);
            }
            break;
        case core::DType::INT32:
            if (full_reduction_)
            {
                plast_cpu_sum_full_reduction_backward_int32(
                    grad_input.data_as<int32_t>(), grad_output.data_as<const int32_t>(),
                    inputs[0]->shape().data(), inputs[0]->shape().size());
            }
            else
            {
                plast_cpu_sum_reduction_dim_backward_int32(
                    grad_input.data_as<int32_t>(), grad_output.data_as<const int32_t>(),
                    inputs[0]->shape().data(), inputs[0]->shape().size(), output.shape().data(),
                    output.shape().size(), dim_);
            }
            break;
        default:
            throw std::runtime_error("Unsupported DType for Sum backward on CPU.");
        }
        input_grads.push_back(std::move(grad_input));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(),
                                             input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
SumOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                            const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Sum backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        tensor::Tensor grad_input(input->shape(), input->dtype(), input->device());
        PLAST_CUDA_CHECK(
            cudaMemset(grad_input.data(), 0, grad_input.nbytes())); // Initialize to zeros

        // Dispatch to type-specific C CUDA kernel
        switch (input->dtype())
        {
        case core::DType::FLOAT32:
            if (full_reduction_)
            {
                plast_cuda_sum_full_reduction_backward_float(
                    grad_input.data_as<float>(), grad_output.data_as<const float>(),
                    inputs[0]->shape().data(), inputs[0]->shape().size());
            }
            else
            {
                throw std::runtime_error(
                    "CUDA Sum reduction dim float backward operation not yet implemented.");
            }
            break;
        case core::DType::INT32:
            if (full_reduction_)
            {
                plast_cuda_sum_full_reduction_backward_int32(
                    grad_input.data_as<int32_t>(), grad_output.data_as<const int32_t>(),
                    inputs[0]->shape().data(), inputs[0]->shape().size());
            }
            else
            {
                throw std::runtime_error(
                    "CUDA Sum reduction dim int32 backward operation not yet implemented.");
            }
            break;
        default:
            throw std::runtime_error("Unsupported DType for Sum backward on CUDA.");
        }
        input_grads.push_back(std::move(grad_input));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Sum backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
