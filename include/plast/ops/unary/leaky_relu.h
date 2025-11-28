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

class LeakyReluOperation : public BaseOperation
{
  private:
    float alpha_;

  public:
    explicit LeakyReluOperation(float alpha = 0.01f) : alpha_(alpha) {}

    const std::string& name() const override
    {
        static const std::string op_name = "LeakyRelu";
        return op_name;
    }

    std::vector<size_t>
    infer_output_shape(const std::vector<std::vector<size_t>>& input_shapes) const override
    {
        return input_shapes[0];
    }

    tensor::Tensor execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const override;
    tensor::Tensor execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const override;

    float alpha() const { return alpha_; }
};

} // namespace ops
} // namespace plast
