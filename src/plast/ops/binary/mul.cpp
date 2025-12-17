#include "plast/ops/binary/mul.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h"
#include "plast/core/types.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"
#include "plast/ops/movement/broadcast.h" // Added include for BroadcastOperation

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{
tensor::Tensor MulOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Mul operation on CPU.");
    }

    core::DType dtype = lhs.dtype();

    // 1. Determine output shape based on broadcasting rules
    std::vector<size_t> output_shape_vec = core::broadcast_shapes(lhs.shape(), rhs.shape());

    // Handle broadcasting for lhs
    const tensor::Tensor* current_lhs = &lhs;
    std::unique_ptr<tensor::Tensor> lhs_broadcasted_ptr;
    if (lhs.shape() != output_shape_vec || !lhs.is_contiguous())
    {
        BroadcastOperation broadcast_op(output_shape_vec);
        lhs_broadcasted_ptr = std::make_unique<tensor::Tensor>(broadcast_op.execute_cpu({&lhs}));
        current_lhs = lhs_broadcasted_ptr.get();
    }

    // Handle broadcasting for rhs
    const tensor::Tensor* current_rhs = &rhs;
    std::unique_ptr<tensor::Tensor> rhs_broadcasted_ptr;
    if (rhs.shape() != output_shape_vec || !rhs.is_contiguous())
    {
        BroadcastOperation broadcast_op(output_shape_vec);
        rhs_broadcasted_ptr = std::make_unique<tensor::Tensor>(broadcast_op.execute_cpu({&rhs}));
        current_rhs = rhs_broadcasted_ptr.get();
    }

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    // Both inputs are now guaranteed to be contiguous and match the output shape
    size_t num_elements = output.num_elements();
    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cpu_mul_kernel_float(output.data_as<float>(), current_lhs->data_as<const float>(),
                                   current_rhs->data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cpu_mul_kernel_int32(output.data_as<int32_t>(), current_lhs->data_as<const int32_t>(),
                                   current_rhs->data_as<const int32_t>(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported DType for Mul operation on CPU.");
    }

    return output;
}

tensor::Tensor MulOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Add operation on CUDA.");
    }

    core::DType dtype = lhs.dtype();

    // 1. Determine output shape based on broadcasting rules
    std::vector<size_t> output_shape_vec = core::broadcast_shapes(lhs.shape(), rhs.shape());

    // Handle broadcasting for lhs
    const tensor::Tensor* current_lhs = &lhs;
    std::unique_ptr<tensor::Tensor> lhs_broadcasted_ptr;
    if (lhs.shape() != output_shape_vec || !lhs.is_contiguous())
    {
        BroadcastOperation broadcast_op(output_shape_vec);
        lhs_broadcasted_ptr = std::make_unique<tensor::Tensor>(broadcast_op.execute_cuda({&lhs}));
        current_lhs = lhs_broadcasted_ptr.get();
    }

    // Handle broadcasting for rhs
    const tensor::Tensor* current_rhs = &rhs;
    std::unique_ptr<tensor::Tensor> rhs_broadcasted_ptr;
    if (rhs.shape() != output_shape_vec || !rhs.is_contiguous())
    {
        BroadcastOperation broadcast_op(output_shape_vec);
        rhs_broadcasted_ptr = std::make_unique<tensor::Tensor>(broadcast_op.execute_cuda({&rhs}));
        current_rhs = rhs_broadcasted_ptr.get();
    }

    // Allocate output tensor on CUDA device
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CUDA);

    // Both inputs are now guaranteed to be contiguous and match the output shape
    size_t num_elements = output.num_elements();
    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cuda_mul_kernel_float(output.data_as<float>(), current_lhs->data_as<const float>(),
                                    current_rhs->data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cuda_mul_kernel_int32(output.data_as<int32_t>(),
                                    current_lhs->data_as<const int32_t>(),
                                    current_rhs->data_as<const int32_t>(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported DType for Mul operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Mul operation on CUDA device.");
#endif
}

std::vector<tensor::Tensor>
MulOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                           const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Mul backward expects 2 inputs.");
    }

    const tensor::Tensor* lhs_input = inputs[0];
    const tensor::Tensor* rhs_input = inputs[1];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Gradient for LHS (input[0])
    if (lhs_input->requires_grad())
    {
        // d(A*B)/dA = B, so grad_lhs = grad_output * rhs_input
        // This requires element-wise multiplication
        throw std::runtime_error(
            "Mul backward_cpu: Gradient for LHS not yet implemented (requires element-wise mul).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor(
            {}, lhs_input->dtype(), lhs_input->device())); // Empty tensor if no grad required
    }

    // Gradient for RHS (input[1])
    if (rhs_input->requires_grad())
    {
        // d(A*B)/dB = A, so grad_rhs = grad_output * lhs_input
        // This requires element-wise multiplication
        throw std::runtime_error(
            "Mul backward_cpu: Gradient for RHS not yet implemented (requires element-wise mul).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor(
            {}, rhs_input->dtype(), rhs_input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
MulOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                            const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Mul backward expects 2 inputs.");
    }

    const tensor::Tensor* lhs_input = inputs[0];
    const tensor::Tensor* rhs_input = inputs[1];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Gradient for LHS (input[0])
    if (lhs_input->requires_grad())
    {
        // d(A*B)/dA = B, so grad_lhs = grad_output * rhs_input
        throw std::runtime_error(
            "Mul backward_cuda: Gradient for LHS not yet implemented (requires element-wise mul).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, lhs_input->dtype(), lhs_input->device()));
    }

    // Gradient for RHS (input[1])
    if (rhs_input->requires_grad())
    {
        // d(A*B)/dB = A, so grad_rhs = grad_output * lhs_input
        throw std::runtime_error(
            "Mul backward_cuda: Gradient for RHS not yet implemented (requires element-wise mul).");
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, rhs_input->dtype(), rhs_input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Mul backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
