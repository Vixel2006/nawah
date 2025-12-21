#include "plast/ops/movement/broadcast.h"
#include "plast/kernels/cpu/broadcast_kernels.h"  // New include
#include "plast/kernels/cuda/broadcast_kernels.h" // New include
#include "plast/kernels/cpu/movement_backward_kernels.h"
#include "plast/kernels/cuda/movement_backward_kernels.h"
#include "plast/tensor/tensor.h"                  // For get_dtype_size

#include <algorithm>
#include <numeric>
#include <stdexcept>

// Forward declaration for get_dtype_size (defined in tensor.cpp)
namespace plast
{
namespace tensor
{
size_t get_dtype_size(core::DType dtype);
}
} // namespace plast

namespace plast
{
namespace ops
{

tensor::Tensor
BroadcastOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("BroadcastOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();

    // Use the infer_output_shape logic to get the actual output shape and validate broadcastability
    std::vector<size_t> output_shape = infer_output_shape({input_shape});

    // Create a new contiguous output tensor
    tensor::Tensor output_tensor(output_shape, input_tensor->dtype(), input_tensor->device());

    // Get raw pointers and sizes for the kernel
    const void* input_data = input_tensor->data();
    void* output_data = output_tensor.data();
    size_t item_size = plast::tensor::get_dtype_size(input_tensor->dtype());

    // Convert std::vector to C-style arrays for kernel
    std::vector<size_t> input_shape_vec = input_tensor->shape();
    std::vector<size_t> input_strides_vec = input_tensor->strides();
    std::vector<size_t> output_shape_vec = output_tensor.shape();

    cpu_broadcast_kernel(input_data, input_shape_vec.data(), input_strides_vec.data(),
                         input_shape_vec.size(), output_data, output_shape_vec.data(),
                         output_shape_vec.size(), item_size);

    if (input_tensor->requires_grad())
    {
        output_tensor.set_requires_grad(true);
    }

    return output_tensor;
}

tensor::Tensor
BroadcastOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("BroadcastOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();

    // Use the infer_output_shape logic to get the actual output shape and validate broadcastability
    std::vector<size_t> output_shape = infer_output_shape({input_shape});

    // Create a new contiguous output tensor
    tensor::Tensor output_tensor(output_shape, input_tensor->dtype(), input_tensor->device());

    // Get raw pointers and sizes for the kernel
    const void* input_data = input_tensor->data();
    void* output_data = output_tensor.data();
    size_t item_size = plast::tensor::get_dtype_size(input_tensor->dtype());

    // Convert std::vector to C-style arrays for kernel
    std::vector<size_t> input_shape_vec = input_tensor->shape();
    std::vector<size_t> input_strides_vec = input_tensor->strides();
    std::vector<size_t> output_shape_vec = output_tensor.shape();

    cuda_broadcast_kernel(input_data, input_shape_vec.data(), input_strides_vec.data(),
                          input_shape_vec.size(), output_data, output_shape_vec.data(),
                          output_shape_vec.size(), item_size);

    if (input_tensor->requires_grad())
    {
        output_tensor.set_requires_grad(true);
    }

    return output_tensor;
}

std::vector<tensor::Tensor>
BroadcastOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                 const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Broadcast backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // Create a new tensor for the gradient of the input
        tensor::Tensor grad_input(input->shape(), input->dtype(), input->device());
        std::memset(grad_input.data(), 0, grad_input.nbytes()); // Initialize to zeros

        // Get raw pointers and sizes for the kernel
        void* grad_in_data = grad_input.data();
        const void* grad_out_data = grad_output.data();
        size_t item_size = plast::tensor::get_dtype_size(input->dtype());

        // Convert std::vector to C-style arrays for kernel
        std::vector<size_t> grad_out_shape_vec = grad_output.shape();
        std::vector<size_t> grad_out_strides_vec = grad_output.strides();
        std::vector<size_t> input_shape_vec = input->shape();

        cpu_broadcast_backward_kernel(grad_in_data, grad_out_data, grad_out_shape_vec.data(),
                                      grad_out_strides_vec.data(), grad_out_shape_vec.size(),
                                      input_shape_vec.data(), input_shape_vec.size(), item_size);

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
BroadcastOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                  const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Broadcast backward expects 1 input.");
    }

    const tensor::Tensor* input = inputs[0];

    // Initialize gradients for inputs
    std::vector<tensor::Tensor> input_grads;
    input_grads.reserve(1);

    // Gradient for input
    if (input->requires_grad())
    {
        // Create a new tensor for the gradient of the input
        tensor::Tensor grad_input(input->shape(), input->dtype(), input->device());
        PLAST_CUDA_CHECK(cudaMemset(grad_input.data(), 0, grad_input.nbytes())); // Initialize to zeros

        // Get raw pointers and sizes for the kernel
        void* grad_in_data = grad_input.data();
        const void* grad_out_data = grad_output.data();
        size_t item_size = plast::tensor::get_dtype_size(input->dtype());

        // Convert std::vector to C-style arrays for kernel
        std::vector<size_t> grad_out_shape_vec = grad_output.shape();
        std::vector<size_t> grad_out_strides_vec = grad_output.strides();
        std::vector<size_t> input_shape_vec = input->shape();

        cuda_broadcast_backward_kernel(grad_in_data, grad_out_data, grad_out_shape_vec.data(),
                                       grad_out_strides_vec.data(), grad_out_shape_vec.size(),
                                       input_shape_vec.data(), input_shape_vec.size(), item_size);

        input_grads.push_back(std::move(grad_input));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Broadcast backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
