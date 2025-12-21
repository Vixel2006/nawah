#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "plast/core/types.h"
#include "plast/ops/base_op.h"
#include "plast/tensor/tensor.h"

namespace plast
{
namespace graph
{

class Node
{
  public:
    // Constructor for operation nodes
    Node(std::shared_ptr<ops::BaseOperation> op, const std::vector<std::shared_ptr<Node>>& inputs);
    // Constructor for leaf nodes (input tensors with actual values)
    Node(std::shared_ptr<tensor::Tensor> value);

    bool is_leaf() const { return op_ == nullptr; }
    const std::shared_ptr<ops::BaseOperation> operation() const { return op_; }
    const std::vector<std::shared_ptr<Node>>& inputs() const { return inputs_; }
    const std::vector<size_t>& shape() const;

    // For caching results during execution
    void set_output_tensor(std::shared_ptr<tensor::Tensor> value);
    std::shared_ptr<tensor::Tensor> get_output_tensor() const;
    bool has_output_tensor() const;
    void clear_output_tensor();

    std::vector<const tensor::Tensor*> get_inputs_as_raw_pointers() const;

    // Gradient tensor management
    void clear_grad_tensor();
    void set_grad_tensor(std::shared_ptr<tensor::Tensor> grad);
    std::shared_ptr<tensor::Tensor> get_grad_tensor() const;
    bool has_grad_tensor() const;

    // For autograd
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    // Backward pass
    void backward(std::shared_ptr<tensor::Tensor> grad_output);

  private:
    std::shared_ptr<ops::BaseOperation> op_;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::shared_ptr<tensor::Tensor> output_tensor_;
    std::shared_ptr<tensor::Tensor> grad_;
    bool requires_grad_ = false;
    bool is_leaf_;
};

} // namespace graph
} // namespace plast
