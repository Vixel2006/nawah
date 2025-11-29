#include "plast/ops/binary/matmul.h"
#include "plast/core/device_management.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#endif

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

    std::vector<std::vector<size_t>> input_shapes = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape = infer_output_shape(input_shapes);

    int B = 1;
    for (int i = 0; i < output_shape.size() - 2; ++i)
    {
        B *= output_shape[i];
    }

    int N = output_shape[output_shape.size() - 2];
    int M = output_shape[output_shape.size() - 1];
    int K = lhs.shape()[output_shape.size() - 1];

    tensor::Tensor output(output_shape, dtype, core::DeviceType::CPU);

    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cpu_matmul_kernel_float(output.data_as<float>(), lhs.data_as<const float>(),
                                      rhs.data_as<const float>(), B, N, M, K);
        break;
    case core::DType::INT32:
        plast_cpu_matmul_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                      rhs.data_as<const int32_t>(), B, N, M, K);
        break;
    default:
        throw std::runtime_error("Unsupoorted DType for Matmul operation on CPU.");
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
        throw std::runtime_error("DType mismatch for Matmul operation on cpu");
    }

    core::DType dtype = lhs.dtype();

    std::vector<std::vector<size_t>> input_shapes = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape = infer_output_shape(input_shapes);

    int B = 1;
    for (int i = 0; i < output_shape.size() - 2; ++i)
    {
        B *= output_shape[i];
    }

    int N = output_shape[output_shape.size() - 2];
    int M = output_shape[output_shape.size() - 1];
    int K = lhs.shape()[output_shape.size() - 1];

    tensor::Tensor output(output_shape, lhs.dtype(), core::DeviceType::CPU);
    return output;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Matmul operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
