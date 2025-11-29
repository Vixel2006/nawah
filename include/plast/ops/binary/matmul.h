#pragma once

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace plast
{
namespace ops
{

class MatmulOperation : public BaseOperation
{
  public:
    const std::string& name() const override
    {
        static std::string op_name = "matmul";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        if (input_shapes.size() != 2)
        {
            throw std::runtime_error("Matmul operation requires exactly two input tensors.");
        }

        std::vector<size_t> lhs_shape = input_shapes[0];
        std::vector<size_t> rhs_shape = input_shapes[1];

        size_t lhs_ndim = lhs_shape.size();
        size_t rhs_ndim = rhs_shape.size();

        if (lhs_ndim < 2 || lhs_ndim < 2)
        {
            throw std::runtime_error("Matmul operation can't be done on a tensor with ndim < 2.");
        }

        if (lhs_ndim != rhs_ndim)
        {
            throw std::runtime_error(
                "Matmul operation can't be done on tensors with different ndim.");
        }

        std::vector<size_t> output_shape(lhs_ndim);

        for (size_t i = 0; i < lhs_ndim - 2; ++i)
        {
            if (lhs_shape[i] != rhs_shape[i])
            {
                throw std::runtime_error(
                    "Matmul operation can't be done on tensors with different batch dims");
            }

            output_shape[i] = lhs_shape[i];
        }

        size_t N = lhs_shape[lhs_ndim - 2];
        size_t K1 = lhs_shape[lhs_ndim - 1];
        size_t K2 = rhs_shape[rhs_ndim - 2];
        size_t M = rhs_shape[rhs_ndim - 1];

        if (K1 != K2)
        {
            throw std::runtime_error("Matmul operation can't be done on tensors with different K.");
        }

        output_shape[lhs_ndim - 2] = N;
        output_shape[lhs_ndim - 1] = M;

        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;
};

} // namespace ops
} // namespace plast
