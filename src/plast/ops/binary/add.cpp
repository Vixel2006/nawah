#include "plast/ops/binary/add.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h"
#include "plast/core/types.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"
#include "plast/ops/movement/broadcast.h" // Added include for BroadcastOperation

#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
// Kernels are declared in binary_kernels.h
#endif

namespace plast
{
namespace ops
{

tensor::Tensor AddOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Add operation on CPU.");
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
        plast_cpu_add_kernel_float(output.data_as<float>(), current_lhs->data_as<const float>(),
                                   current_rhs->data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cpu_add_kernel_int32(output.data_as<int32_t>(), current_lhs->data_as<const int32_t>(),
                                   current_rhs->data_as<const int32_t>(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported DType for Add operation on CPU.");
    }

    return output;
}

tensor::Tensor AddOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
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
        plast_cuda_add_kernel_float(output.data_as<float>(), current_lhs->data_as<const float>(),
                                    current_rhs->data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cuda_add_kernel_int32(output.data_as<int32_t>(), current_lhs->data_as<const int32_t>(),
                                    current_rhs->data_as<const int32_t>(), num_elements);
        break;
    default:
        throw std::runtime_error("Unsupported DType for Add operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Add operation on CUDA device.");
#endif
}

void AddOperation::backward(const tensor::Tensor& grad_output,
                            const tensor::Tensor& output,
                            std::vector<tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Add backward expects 2 inputs.");
    }

    tensor::Tensor* lhs = inputs[0];
    tensor::Tensor* rhs = inputs[1];

    if (lhs->requires_grad())
    {
        if (lhs->grad() == nullptr)
        {
            lhs->set_grad(std::make_shared<tensor::Tensor>(lhs->shape(), lhs->dtype(), lhs->device()));
        }

        if (lhs->shape() == grad_output.shape())
        {
            // Simple case: no broadcasting
            for (size_t i = 0; i < lhs->num_elements(); ++i)
            {
                lhs->grad()->data_as<float>()[i] += grad_output.data_as<const float>()[i];
            }
        }
        else
        {
            // TODO: Handle broadcasting
        }
    }

    if (rhs->requires_grad())
    {
        if (rhs->grad() == nullptr)
        {
            rhs->set_grad(std::make_shared<tensor::Tensor>(rhs->shape(), rhs->dtype(), rhs->device()));
        }

        if (rhs->shape() == grad_output.shape())
        {
            // Simple case: no broadcasting
            for (size_t i = 0; i < rhs->num_elements(); ++i)
            {
                rhs->grad()->data_as<float>()[i] += grad_output.data_as<const float>()[i];
            }
        }
        else
        {
            // TODO: Handle broadcasting
        }
    }
}

} // namespace ops
} // namespace plast
