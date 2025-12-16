#include "plast/ops/movement/broadcast.h"
#include "plast/kernels/cpu/broadcast_kernels.h" // New include
#include "plast/kernels/cuda/broadcast_kernels.h" // New include
#include "plast/tensor/tensor.h"                 // For get_dtype_size

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

    return output_tensor;
}

void BroadcastOperation::backward(const tensor::Tensor& grad_output,
                                  const tensor::Tensor& output,
                                  std::vector<tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Broadcast backward expects 1 input.");
    }

    tensor::Tensor* input = inputs[0];

    if (input->requires_grad())
    {
        if (input->grad() == nullptr)
        {
            input->set_grad(std::make_shared<tensor::Tensor>(input->shape(), input->dtype(), input->device()));
        }

        if (input->device() == core::DeviceType::CPU)
        {
            // Sum the gradients along the broadcasted dimensions
            std::vector<size_t> reduction_axes;
            int N = grad_output.ndim() - input->ndim();
            for (int i = 0; i < N; ++i)
            {
                reduction_axes.push_back(i);
            }
            for (int i = 0; i < input->ndim(); ++i)
            {
                if (input->shape()[i] == 1 && grad_output.shape()[N + i] > 1)
                {
                    reduction_axes.push_back(N + i);
                }
            }

            // TODO: Implement a reduction kernel
            // For now, we will do a naive sum
            if (reduction_axes.empty())
            {
                for (size_t i = 0; i < input->num_elements(); ++i)
                {
                    input->grad()->data_as<float>()[i] += grad_output.data_as<const float>()[i];
                }
            }
            else
            {
                // This is a naive implementation and will be slow.
                // A proper reduction kernel should be used here.
                for (size_t i = 0; i < grad_output.num_elements(); ++i)
                {
                    size_t input_index = 0;
                    size_t grad_output_index = i;
                    for (int j = 0; j < grad_output.ndim(); ++j)
                    {
                        size_t coord = (grad_output_index / grad_output.strides()[j]) % grad_output.shape()[j];
                        if (j >= N && input->shape()[j - N] != 1)
                        {
                            input_index += coord * input->strides()[j - N];
                        }
                    }
                    input->grad()->data_as<float>()[input_index] += grad_output.data_as<const float>()[i];
                }
            }
        }
        else if (input->device() == core::DeviceType::CUDA)
        {
            throw std::runtime_error("Broadcast backward for CUDA is not implemented.");
        }
    }
}

} // namespace ops
} // namespace plast
