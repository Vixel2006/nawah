#include "plast/ops/binary/sub.h"
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

tensor::Tensor SubOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Sub operation on CPU.");
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
        plast_cpu_sub_kernel_float(output.data_as<float>(), current_lhs->data_as<const float>(),
                                   current_rhs->data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cpu_sub_kernel_int32(output.data_as<int32_t>(), current_lhs->data_as<const int32_t>(),
                                   current_rhs->data_as<const int32_t>(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported DType for Sub operation on CPU.");
    }

    return output;
}

tensor::Tensor SubOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Sub operation on CUDA.");
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
        plast_cuda_sub_kernel_float(output.data_as<float>(), current_lhs->data_as<const float>(),
                                    current_rhs->data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cuda_sub_kernel_int32(output.data_as<int32_t>(),
                                    current_lhs->data_as<const int32_t>(),
                                    current_rhs->data_as<const int32_t>(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported DType for Sub operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Sub operation on CUDA device.");
#endif
}

std::vector<tensor::Tensor>
SubOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                           const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Sub backward expects 2 inputs.");
    }

    const tensor::Tensor* lhs_input = inputs[0];
    const tensor::Tensor* rhs_input = inputs[1];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Gradient for LHS (input[0])
    if (lhs_input->requires_grad())
    {
        // TODO: Implement actual gradient calculation for LHS, considering broadcasting
        // For now, a simple copy if shapes match, otherwise a placeholder error
        if (lhs_input->shape() == grad_output.shape())
        {
            input_grads.push_back(grad_output.clone()); // Simple case: grad_output is the gradient
        }
        else
        {
            throw std::runtime_error(
                "Sub backward_cpu: Broadcasting gradient for LHS not yet implemented.");
        }
    }
    else
    {
        input_grads.push_back(tensor::Tensor(
            {}, lhs_input->dtype(), lhs_input->device())); // Empty tensor if no grad required
    }

    // Gradient for RHS (input[1])
    if (rhs_input->requires_grad())
    {
        // TODO: Implement actual gradient calculation for RHS, considering broadcasting
        // For now, a simple copy if shapes match, otherwise a placeholder error
        if (rhs_input->shape() == grad_output.shape())
        {
            // For subtraction, d(A-B)/dB = -1, so grad_rhs = -grad_output
            // This requires a unary negation operation or element-wise multiplication by -1
            throw std::runtime_error(
                "Sub backward_cpu: Gradient for RHS not yet implemented (requires negation).");
        }
        else
        {
            throw std::runtime_error(
                "Sub backward_cpu: Broadcasting gradient for RHS not yet implemented.");
        }
    }
    else
    {
        input_grads.push_back(tensor::Tensor(
            {}, rhs_input->dtype(), rhs_input->device())); // Empty tensor if no grad required
    }

    return input_grads;
}

std::vector<tensor::Tensor>
SubOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                            const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Sub backward expects 2 inputs.");
    }

    const tensor::Tensor* lhs_input = inputs[0];
    const tensor::Tensor* rhs_input = inputs[1];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Gradient for LHS (input[0])
    if (lhs_input->requires_grad())
    {
        // TODO: Implement actual gradient calculation for LHS on CUDA, considering broadcasting
        if (lhs_input->shape() == grad_output.shape())
        {
            input_grads.push_back(grad_output.clone());
        }
        else
        {
            throw std::runtime_error(
                "Sub backward_cuda: Broadcasting gradient for LHS not yet implemented.");
        }
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, lhs_input->dtype(), lhs_input->device()));
    }

    // Gradient for RHS (input[1])
    if (rhs_input->requires_grad())
    {
        // TODO: Implement actual gradient calculation for RHS on CUDA, considering broadcasting
        if (rhs_input->shape() == grad_output.shape())
        {
            // For subtraction, d(A-B)/dB = -1, so grad_rhs = -grad_output
            throw std::runtime_error(
                "Sub backward_cuda: Gradient for RHS not yet implemented (requires negation).");
        }
        else
        {
            throw std::runtime_error(
                "Sub backward_cuda: Broadcasting gradient for RHS not yet implemented.");
        }
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, rhs_input->dtype(), rhs_input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Sub backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
