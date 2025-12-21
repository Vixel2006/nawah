#include "plast/graph/node.h"
#include <stdexcept>

namespace plast
{
namespace graph
{

// Constructor for operation nodes
Node::Node(std::shared_ptr<ops::BaseOperation> op, const std::vector<std::shared_ptr<Node>>& inputs)
    : op_(std::move(op)), inputs_(inputs),
      output_tensor_(nullptr), requires_grad_(false), grad_(nullptr) // Initialize requires_grad_ and grad_
{
    if (!op_)
    {
        throw std::runtime_error("Operation cannot be null for an operation node.");
    }
    // If any input requires grad, then this node also requires grad
    for (const auto& input : inputs_)
    {
        if (input->requires_grad())
        {
            requires_grad_ = true;
            break;
        }
    }
}

// Constructor for leaf nodes (input tensors with actual values)
Node::Node(std::shared_ptr<tensor::Tensor> value)
    : op_(nullptr), inputs_({}), output_tensor_(std::move(value)),
      requires_grad_(output_tensor_->requires_grad()), grad_(nullptr) // Initialize requires_grad_ and grad_
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

void Node::clear_grad_tensor()
{
    grad_.reset();
}

void Node::set_grad_tensor(std::shared_ptr<tensor::Tensor> grad)
{
    grad_ = grad;
}

std::shared_ptr<tensor::Tensor> Node::get_grad_tensor() const
{
    return grad_;
}

bool Node::has_grad_tensor() const
{
    return grad_ != nullptr;
}



void Node::backward(std::shared_ptr<tensor::Tensor> grad_output)
{
    if (!requires_grad_)
    {
        return;
    }

    // This method is now primarily responsible for computing input gradients
    // and propagating them. Accumulation into Node::grad_ and Tensor::grad_
    // is handled by ExecutionEngine::compute_gradients.

    if (is_leaf())
    {
        // For leaf nodes, the gradient is directly accumulated into Tensor::grad_
        // by ExecutionEngine::compute_gradients. This method does nothing for leaf nodes.
        return;
    }
    else
    {
        // Get the output tensor of this node (from forward pass)
        std::shared_ptr<tensor::Tensor> output = get_output_tensor();
        if (!output)
        {
            throw std::runtime_error("Node output tensor not available for backward pass.");
        }

        // Get raw pointers to input tensors for the backward operation
        std::vector<const tensor::Tensor*> raw_inputs;
        for (const auto& input_node : inputs_)
        {
            if (!input_node->has_output_tensor())
            {
                throw std::runtime_error("Input node value not computed for backward pass.");
            }
            raw_inputs.push_back(input_node->get_output_tensor().get());
        }

        // Determine device type for the operation
        core::DeviceType device = output->device();

        std::vector<tensor::Tensor> input_grads;
        if (device == core::DeviceType::CPU)
        {
            input_grads = op_->backward_cpu(*grad_output, *output, raw_inputs);
        }
        else if (device == core::DeviceType::CUDA)
        {
            input_grads = op_->backward_cuda(*grad_output, *output, raw_inputs);
        }
        else
        {
            throw std::runtime_error("Unsupported device type for backward pass.");
        }

        // Propagate gradients to input nodes
        for (size_t i = 0; i < inputs_.size(); ++i)
        {
            if (inputs_[i]->requires_grad())
            {
                // The ExecutionEngine::compute_gradients will accumulate these
                // into the input_node's grad_tensor.
                // Here, we just need to ensure the input_node's grad_tensor is set
                // with the computed gradient for this path.
                // For now, we'll just set it, and rely on ExecutionEngine to accumulate.
                inputs_[i]->set_grad_tensor(std::make_shared<tensor::Tensor>(std::move(input_grads[i])));
            }
        }
    }
}

} // namespace graph
} // namespace plast