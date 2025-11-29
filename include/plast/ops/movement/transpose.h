#pragma once

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

#include <string>
#include <vector>

namespace plast
{
namespace ops
{

class TransposeOperation : public BaseOperation
{
  public:
    TransposeOperation(size_t N, size_t M) : N(N), M(M) {}

    const std::string& name() const override
    {
        static std::string op_name = "transpose";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        std::vector<size_t> output_shape = input_shapes[0];
        size_t temp = output_shape[N];
        output_shape[N] = output_shape[M];
        output_shape[M] = temp;

        return output_shape;
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

  private:
    size_t N, M;
};

} // namespace ops
} // namespace plast
