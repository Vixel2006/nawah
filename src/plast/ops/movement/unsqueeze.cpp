#include "plast/ops/movement/unsqueeze.h"
#include "plast/kernels/cpu/movement_backward_kernels.h"
#include "plast/kernels/cuda/movement_backward_kernels.h"

#include <numeric>

namespace plast
{
namespace ops
{

tensor::Tensor
UnsqueezeOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("UnsqueezeOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    tensor::Tensor output = [&]() {
        std::vector<size_t> output_shape = input_tensor->shape();
        std::vector<size_t> output_strides = input_tensor->strides();

        if (dim_ > output_shape.size())
        {
            throw std::runtime_error("Unsqueeze dimension out of bounds.");
        }

        output_shape.insert(output_shape.begin() + dim_, 1);

        // Calculate new stride for the inserted dimension
        size_t new_stride;
        if (dim_ < output_strides.size())
        {
            new_stride = output_strides[dim_];
        }
        else
        {
            // If inserting at the end, the new stride is 1 (for a dimension of size 1)
            new_stride = 1;
        }
        output_strides.insert(output_strides.begin() + dim_, new_stride);

        // Create a new Tensor that views the same data but with new shape and strides
        return input_tensor->reshape(output_shape, output_strides);
    }();

    if (input_tensor->requires_grad())
    {
        output.set_requires_grad(true);
    }

    return output;
}

tensor::Tensor
UnsqueezeOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    return execute_cpu(inputs);
}

std::vector<tensor::Tensor>
UnsqueezeOperation::backward_cpu(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                 const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Unsqueeze backward expects 1 input.");
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

        int squeeze_dims_arr[] = {(int)dim_}; // The dimension that was unsqueezed
        size_t num_squeeze_dims = 1;

        cpu_squeeze_backward_kernel(grad_in_data, grad_out_data, grad_out_shape_vec.data(),
                                    grad_out_strides_vec.data(), grad_out_shape_vec.size(),
                                    input_shape_vec.data(), input_shape_vec.size(),
                                    squeeze_dims_arr, num_squeeze_dims, item_size);

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
UnsqueezeOperation::backward_cuda(const tensor::Tensor& grad_output, const tensor::Tensor& output,
                                  const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Unsqueeze backward expects 1 input.");
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

        int squeeze_dims_arr[] = {(int)dim_}; // The dimension that was unsqueezed
        size_t num_squeeze_dims = 1;

        cuda_squeeze_backward_kernel(grad_in_data, grad_out_data, grad_out_shape_vec.data(),
                                     grad_out_strides_vec.data(), grad_out_shape_vec.size(),
                                     input_shape_vec.data(), input_shape_vec.size(),
                                     squeeze_dims_arr, num_squeeze_dims, item_size);

        input_grads.push_back(std::move(grad_input));
    }
    else
    {
        input_grads.push_back(tensor::Tensor({}, input->dtype(), input->device()));
    }

    return input_grads;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Unsqueeze backward operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
