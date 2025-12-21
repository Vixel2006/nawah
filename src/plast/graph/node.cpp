#include "plast/graph/node.h"
#include <stdexcept>

namespace plast
{
namespace graph
{

// Constructor for operation nodes
Node::Node(std::shared_ptr<ops::BaseOperation> op, const std::vector<std::shared_ptr<Node>>& inputs)
    : op_(std::move(op)), inputs_(inputs), output_tensor_(nullptr)
{
    if (!op_)
    {
        throw std::runtime_error("Operation cannot be null for an operation node.");
    }
}

// Constructor for leaf nodes (input tensors with actual values)
Node::Node(std::shared_ptr<tensor::Tensor> value)
    : op_(nullptr), inputs_({}), output_tensor_(std::move(value))
{
    if (!output_tensor_)
    {
        throw std::runtime_error("Leaf node cannot be initialized with a null tensor.");
    }
}

const std::vector<size_t>& Node::shape() const
{
    if (!output_tensor_)
    {
        throw std::runtime_error("Attempted to access shape from a node with no output tensor. "
                                 "Node has not been executed or is not a leaf node.");
    }
    return output_tensor_->shape();
}

void Node::set_output_tensor(std::shared_ptr<tensor::Tensor> value)
{
    output_tensor_ = std::move(value);
}

std::shared_ptr<tensor::Tensor> Node::get_output_tensor() const
{
    if (!output_tensor_)
    {
        throw std::runtime_error(
            "Attempted to get output tensor from a node that has not been computed.");
    }
    return output_tensor_;
}

bool Node::has_output_tensor() const { return output_tensor_ != nullptr; }

bool Node::requires_grad() const
{
    if (!output_tensor_)
    {
        return false; // Or throw an error, but returning false is safer.
    }
    return output_tensor_->requires_grad();
}

void Node::clear_output_tensor() { output_tensor_.reset(); }

std::vector<const tensor::Tensor*> Node::get_inputs_as_raw_pointers() const
{
    std::vector<const tensor::Tensor*> input_tensors;
    input_tensors.reserve(inputs_.size());
    for (const auto& input_node : inputs_)
    {
        input_tensors.push_back(input_node->get_output_tensor().get());
    }
    return input_tensors;
}

} // namespace graph
} // namespace plast
