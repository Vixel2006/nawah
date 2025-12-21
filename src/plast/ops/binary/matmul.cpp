#include "plast/ops/binary/matmul.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"
#include "plast/kernels/cpu/binary_backward_kernels.h"
#include "plast/kernels/cuda/binary_backward_kernels.h"
#include "plast/ops/movement/transpose.h" // Added for TransposeOperation

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{

namespace ops
{

tensor::Tensor MatmulOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Matmul operation on cpu");
    }

    core::DType dtype = lhs.dtype();

    std::vector<std::vector<size_t>> input_shapes_vec = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape_vec = infer_output_shape(input_shapes_vec);

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    // Replicate effective shape logic from infer_output_shape to get B, N, M, K
    std::vector<size_t> current_lhs_shape = lhs.shape();
    std::vector<size_t> current_rhs_shape = rhs.shape();

    size_t current_lhs_ndim = current_lhs_shape.size();
    size_t current_rhs_ndim = current_rhs_shape.size();

    if (current_lhs_ndim == 1)
    {
        current_lhs_shape.insert(current_lhs_shape.begin(), 1); // (D) -> (1, D)
        current_lhs_ndim++;
    }
    if (current_rhs_ndim == 1)
    {
        current_rhs_shape.push_back(1); // (D) -> (D, 1)
        current_rhs_ndim++;
    }

    std::vector<size_t> lhs_batch_shape(current_lhs_shape.begin(), current_lhs_shape.end() - 2);
    std::vector<size_t> rhs_batch_shape(current_rhs_shape.begin(), current_rhs_shape.end() - 2);

    std::vector<size_t> output_batch_shape =
        core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

    int B_dim = 1;
    for (size_t dim_size : output_batch_shape)
    {
        B_dim *= dim_size;
    }

    int N_dim = current_lhs_shape[current_lhs_ndim - 2];
    int K_dim_val = current_lhs_shape[current_lhs_ndim - 1];
    int M_dim = current_rhs_shape[current_rhs_ndim - 1];

    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cpu_matmul_kernel_float(output.data_as<float>(), lhs.data_as<const float>(),
                                      rhs.data_as<const float>(), B_dim, N_dim, M_dim, K_dim_val);
        break;
    case core::DType::INT32:
        plast_cpu_matmul_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                      rhs.data_as<const int32_t>(), B_dim, N_dim, M_dim, K_dim_val);
        break;
    default:
        throw std::runtime_error("Unsupoorted DType for Matmul operation on CPU.");
    }

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad() || inputs[1]->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
}

tensor::Tensor MatmulOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Matmul operation on CUDA");
    }

    core::DType dtype = lhs.dtype();

    std::vector<std::vector<size_t>> input_shapes_vec = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape_vec = infer_output_shape(input_shapes_vec);

    // Allocate output tensor on CUDA device
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CUDA);

    switch (dtype)
    {
    case core::DType::FLOAT32:
    {
        // Replicate effective shape logic from infer_output_shape to get B, N, M, K
        std::vector<size_t> current_lhs_shape = lhs.shape();
        std::vector<size_t> current_rhs_shape = rhs.shape();

        size_t current_lhs_ndim = current_lhs_shape.size();
        size_t current_rhs_ndim = current_rhs_shape.size();

        if (current_lhs_ndim == 1)
        {
            current_lhs_shape.insert(current_lhs_shape.begin(), 1); // (D) -> (1, D)
            current_lhs_ndim++;
        }
        if (current_rhs_ndim == 1)
        {
            current_rhs_shape.push_back(1); // (D) -> (D, 1)
            current_rhs_ndim++;
        }

        std::vector<size_t> lhs_batch_shape(current_lhs_shape.begin(), current_lhs_shape.end() - 2);
        std::vector<size_t> rhs_batch_shape(current_rhs_shape.begin(), current_rhs_shape.end() - 2);

        std::vector<size_t> output_batch_shape =
            core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

        int B_dim = 1;
        for (size_t dim_size : output_batch_shape)
        {
            B_dim *= dim_size;
        }

        int N_dim = current_lhs_shape[current_lhs_ndim - 2];
        int K_dim_val = current_lhs_shape[current_lhs_ndim - 1];
        int M_dim = current_rhs_shape[current_rhs_ndim - 1];

        plast_cuda_matmul_kernel_float(output.data_as<float>(), lhs.data_as<const float>(),
                                       rhs.data_as<const float>(), B_dim, N_dim, M_dim, K_dim_val);
    }
    break;
    case core::DType::INT32:
    {
        // Replicate effective shape logic from infer_output_shape to get B, N, M, K
        std::vector<size_t> current_lhs_shape = lhs.shape();
        std::vector<size_t> current_rhs_shape = rhs.shape();

        size_t current_lhs_ndim = current_lhs_shape.size();
        size_t current_rhs_ndim = current_rhs_shape.size();

        if (current_lhs_ndim == 1)
        {
            current_lhs_shape.insert(current_lhs_shape.begin(), 1); // (D) -> (1, D)
            current_lhs_ndim++;
        }
        if (current_rhs_ndim == 1)
        {
            current_rhs_shape.push_back(1); // (D) -> (D, 1)
            current_rhs_ndim++;
        }

        std::vector<size_t> lhs_batch_shape(current_lhs_shape.begin(), current_lhs_shape.end() - 2);
        std::vector<size_t> rhs_batch_shape(current_rhs_shape.begin(), current_rhs_shape.end() - 2);

        std::vector<size_t> output_batch_shape =
            core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

        int B_dim = 1;
        for (size_t dim_size : output_batch_shape)
        {
            B_dim *= dim_size;
        }

        int N_dim = current_lhs_shape[current_lhs_ndim - 2];
        int K_dim_val = current_lhs_shape[current_lhs_ndim - 1];
        int M_dim = current_rhs_shape[current_rhs_ndim - 1];

        plast_cuda_matmul_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                       rhs.data_as<const int32_t>(), B_dim, N_dim, M_dim,
                                       K_dim_val);
    }
    break;
    default:
        throw std::runtime_error("Unsupported DType for Matmul operation on CUDA.");
    }

    // If any input requires grad, the output also requires grad.
    if (inputs[0]->requires_grad() || inputs[1]->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Matmul operation on CUDA device.");
#endif
}

std::vector<tensor::Tensor>
MatmulOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                              const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Matmul backward expects 2 inputs.");
    }

    const tensor::Tensor* lhs_input = inputs[0];
    const tensor::Tensor* rhs_input = inputs[1];

    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(2);

    // Calculate gradient for LHS: grad_output @ rhs_input.T
    if (lhs_input->requires_grad())
    {
        MatmulOperation matmul_op;
        TransposeOperation transpose_rhs_op(rhs_input->ndim() - 2, rhs_input->ndim() - 1);
        tensor::Tensor rhs_transposed = transpose_rhs_op.execute_cpu({rhs_input});
        tensor::Tensor grad_for_lhs = matmul_op.execute_cpu({&grad_output, &rhs_transposed});
        input_grads.push_back(std::move(grad_for_lhs));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, lhs_input->dtype(), lhs_input->device()));
    }

    // Calculate gradient for RHS: lhs_input.T @ grad_output
    if (rhs_input->requires_grad())
    {
        MatmulOperation matmul_op;
        TransposeOperation transpose_lhs_op(lhs_input->ndim() - 2, lhs_input->ndim() - 1);
        tensor::Tensor lhs_transposed = transpose_lhs_op.execute_cpu({lhs_input});
        tensor::Tensor grad_for_rhs = matmul_op.execute_cpu({&lhs_transposed, &grad_output});
        input_grads.push_back(std::move(grad_for_rhs));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, rhs_input->dtype(), rhs_input->device()));
    }

    return input_grads;
}

std::vector<tensor::Tensor>
MatmulOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                               const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 2)
    {
        throw std::runtime_error("Matmul backward expects 2 inputs.");
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
        grad_lhs_tensor_ptr = std::make_unique<tensor::Tensor>(lhs_input->shape(), lhs_input->dtype(), lhs_input->device());
        PLAST_CUDA_CHECK(cudaMemset(grad_lhs_tensor_ptr->data(), 0, grad_lhs_tensor_ptr->nbytes()));
        if (lhs_input->dtype() == core::DType::FLOAT32) {
            grad_lhs_data = grad_lhs_tensor_ptr->data_as<float>();
        } else if (lhs_input->dtype() == core::DType::INT32) {
            grad_lhs_data_int = grad_lhs_tensor_ptr->data_as<int32_t>();
        }
    }

    // Allocate grad_rhs if required
    std::unique_ptr<tensor::Tensor> grad_rhs_tensor_ptr;
    if (rhs_input->requires_grad())
    {
        grad_rhs_tensor_ptr = std::make_unique<tensor::Tensor>(rhs_input->shape(), rhs_input->dtype(), rhs_input->device());
        PLAST_CUDA_CHECK(cudaMemset(grad_rhs_tensor_ptr->data(), 0, grad_rhs_tensor_ptr->nbytes()));
        if (rhs_input->dtype() == core::DType::FLOAT32) {
            grad_rhs_data = grad_rhs_tensor_ptr->data_as<float>();
        } else if (rhs_input->dtype() == core::DType::INT32) {
            grad_rhs_data_int = grad_rhs_tensor_ptr->data_as<int32_t>();
        }
    }

    // Call the backward kernel only if at least one input requires grad
    if (lhs_input->requires_grad() || rhs_input->requires_grad())
    {
        // Replicate effective shape logic from infer_output_shape to get B, N, M, K
        std::vector<size_t> current_lhs_shape = lhs_input->shape();
        std::vector<size_t> current_rhs_shape = rhs_input->shape();

        size_t current_lhs_ndim = current_lhs_shape.size();
        size_t current_rhs_ndim = current_rhs_shape.size();

        if (current_lhs_ndim == 1)
        {
            current_lhs_shape.insert(current_lhs_shape.begin(), 1); // (D) -> (1, D)
            current_lhs_ndim++;
        }
        if (current_rhs_ndim == 1)
        {
            current_rhs_shape.push_back(1); // (D) -> (D, 1)
            current_rhs_ndim++;
        }

        std::vector<size_t> lhs_batch_shape(current_lhs_shape.begin(), current_lhs_shape.end() - 2);
        std::vector<size_t> rhs_batch_shape(current_rhs_shape.begin(), current_rhs_shape.end() - 2);

        std::vector<size_t> output_batch_shape =
            core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

        int B_dim = 1;
        for (size_t dim_size : output_batch_shape)
        {
            B_dim *= dim_size;
        }

        int N_dim = current_lhs_shape[current_lhs_ndim - 2];
        int K_dim_val = current_lhs_shape[current_lhs_ndim - 1];
        int M_dim = current_rhs_shape[current_rhs_ndim - 1];

        switch (grad_output.dtype())
        {
        case core::DType::FLOAT32:
            plast_cuda_matmul_backward_kernel_float(grad_lhs_data, grad_rhs_data,
                                                    grad_output.data_as<const float>(),
                                                    lhs_input->data_as<const float>(),
                                                    rhs_input->data_as<const float>(),
                                                    B_dim, N_dim, M_dim, K_dim_val);
            break;
        case core::DType::INT32:
            plast_cuda_matmul_backward_kernel_int32(grad_lhs_data_int, grad_rhs_data_int,
                                                    grad_output.data_as<const int32_t>(),
                                                    lhs_input->data_as<const int32_t>(),
                                                    rhs_input->data_as<const int32_t>(),
                                                    B_dim, N_dim, M_dim, K_dim_val);
            break;
        default:
            throw std::runtime_error("Unsupported DType for Matmul backward on CUDA.");
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
        "CUDA is not enabled. Cannot execute Matmul backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
