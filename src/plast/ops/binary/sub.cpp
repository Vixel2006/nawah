#include "plast/ops/binary/sub.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h"
#include "plast/core/types.h"
#include "plast/kernels/cpu/binary_backward_kernels.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_backward_kernels.h"
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
    if (lhs.shape() != output_shape_vec)
    {
        BroadcastOperation broadcast_op(output_shape_vec);
        lhs_broadcasted_ptr = std::make_unique<tensor::Tensor>(broadcast_op.execute_cpu({&lhs}));
        current_lhs = lhs_broadcasted_ptr.get();
    }

    // Handle broadcasting for rhs
    const tensor::Tensor* current_rhs = &rhs;
    std::unique_ptr<tensor::Tensor> rhs_broadcasted_ptr;
    if (rhs.shape() != output_shape_vec)
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

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad() || inputs[1]->requires_grad())
    {
        output.set_requires_grad(true);
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
    if (lhs.shape() != output_shape_vec)
    {
        BroadcastOperation broadcast_op(output_shape_vec);
        lhs_broadcasted_ptr = std::make_unique<tensor::Tensor>(broadcast_op.execute_cuda({&lhs}));
        current_lhs = lhs_broadcasted_ptr.get();
    }

    // Handle broadcasting for rhs
    const tensor::Tensor* current_rhs = &rhs;
    std::unique_ptr<tensor::Tensor> rhs_broadcasted_ptr;
    if (rhs.shape() != output_shape_vec)
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

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad() || inputs[1]->requires_grad())
    {
        output.set_requires_grad(true);
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

    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Calculate gradient for LHS: grad_output
    if (lhs_input->requires_grad())
    {
        input_grads.push_back(std::move(grad_output.clone()));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, lhs_input->dtype(), lhs_input->device()));
    }

    // Calculate gradient for RHS: -grad_output
    if (rhs_input->requires_grad())
    {
        // Need a NegOperation for -grad_output
        // For now, we'll manually negate the data. A proper NegOperation should be used.
        // This is a placeholder.
        tensor::Tensor neg_grad_output = grad_output.clone();
        switch (neg_grad_output.dtype())
        {
        case core::DType::FLOAT32:
        {
            float* data = neg_grad_output.data_as<float>();
            for (size_t i = 0; i < neg_grad_output.num_elements(); ++i)
            {
                data[i] = -data[i];
            }
            break;
        }
        case core::DType::INT32:
        {
            int32_t* data = neg_grad_output.data_as<int32_t>();
            for (size_t i = 0; i < neg_grad_output.num_elements(); ++i)
            {
                data[i] = -data[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported DType for negation in Sub backward.");
        }
        input_grads.push_back(std::move(neg_grad_output));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, rhs_input->dtype(), rhs_input->device()));
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

    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Pointers to the data buffers for the gradients
    float* grad_lhs_data = nullptr;
    float* grad_rhs_data = nullptr;
    int32_t* grad_lhs_data_int = nullptr;
    int32_t* grad_rhs_data_int = nullptr;

    // Allocate grad_lhs if required
    std::unique_ptr<tensor::Tensor> grad_lhs_tensor_ptr;
    if (lhs_input->requires_grad())
    {
        grad_lhs_tensor_ptr = std::make_unique<tensor::Tensor>(
            lhs_input->shape(), lhs_input->dtype(), lhs_input->device());
        PLAST_CUDA_CHECK(cudaMemset(grad_lhs_tensor_ptr->data(), 0, grad_lhs_tensor_ptr->nbytes()));
        if (lhs_input->dtype() == core::DType::FLOAT32)
        {
            grad_lhs_data = grad_lhs_tensor_ptr->data_as<float>();
        }
        else if (lhs_input->dtype() == core::DType::INT32)
        {
            grad_lhs_data_int = grad_lhs_tensor_ptr->data_as<int32_t>();
        }
    }

    // Allocate grad_rhs if required
    std::unique_ptr<tensor::Tensor> grad_rhs_tensor_ptr;
    if (rhs_input->requires_grad())
    {
        grad_rhs_tensor_ptr = std::make_unique<tensor::Tensor>(
            rhs_input->shape(), rhs_input->dtype(), rhs_input->device());
        PLAST_CUDA_CHECK(cudaMemset(grad_rhs_tensor_ptr->data(), 0, grad_rhs_tensor_ptr->nbytes()));
        if (rhs_input->dtype() == core::DType::FLOAT32)
        {
            grad_rhs_data = grad_rhs_tensor_ptr->data_as<float>();
        }
        else if (rhs_input->dtype() == core::DType::INT32)
        {
            grad_rhs_data_int = grad_rhs_tensor_ptr->data_as<int32_t>();
        }
    }

    // Call the backward kernel only if at least one input requires grad
    if (lhs_input->requires_grad() || rhs_input->requires_grad())
    {
        switch (grad_output.dtype())
        {
        case core::DType::FLOAT32:
            plast_cuda_sub_backward_kernel_float(grad_lhs_data, grad_rhs_data,
                                                 grad_output.data_as<const float>(),
                                                 grad_output.num_elements());
            break;
        case core::DType::INT32:
            plast_cuda_sub_backward_kernel_int32(grad_lhs_data_int, grad_rhs_data_int,
                                                 grad_output.data_as<const int32_t>(),
                                                 grad_output.num_elements());
            break;
        default:
            throw std::runtime_error("Unsupported DType for Sub backward on CUDA.");
        }
    }

    // Explicitly negate grad_rhs if it requires grad, for consistency with CPU backward
    if (rhs_input->requires_grad())
    {
        switch (rhs_input->dtype())
        {
        case core::DType::FLOAT32:
        {
            float* data = grad_rhs_tensor_ptr->data_as<float>();
            for (size_t i = 0; i < grad_rhs_tensor_ptr->num_elements(); ++i)
            {
                data[i] = -data[i];
            }
            break;
        }
        case core::DType::INT32:
        {
            int32_t* data = grad_rhs_tensor_ptr->data_as<int32_t>();
            for (size_t i = 0; i < grad_rhs_tensor_ptr->num_elements(); ++i)
            {
                data[i] = -data[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported DType for negation in Sub backward (CUDA).");
        }
    }

    // Push the resulting gradient tensors
    if (lhs_input->requires_grad())
    {
        input_grads.push_back(std::move(*grad_lhs_tensor_ptr));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, lhs_input->dtype(), lhs_input->device()));
    }

    if (rhs_input->requires_grad())
    {
        input_grads.push_back(std::move(*grad_rhs_tensor_ptr));
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
