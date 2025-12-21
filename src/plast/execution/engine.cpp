#include "plast/execution/engine.h"
#include "plast/ops/binary/add.h" // Added include for AddOperation
#include "plast/ops/init/init_ops.h"
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace plast
{
namespace execution
{

ExecutionEngine::ExecutionEngine() {}

ExecutionEngine::~ExecutionEngine() {}

void ExecutionEngine::visit(std::shared_ptr<graph::Node> node,
                            std::unordered_map<std::shared_ptr<graph::Node>, bool>& visited,
                            std::unordered_map<std::shared_ptr<graph::Node>, bool>& in_stack,
                            std::vector<std::shared_ptr<graph::Node>>& sorted_nodes)
{
    visited[node] = true;
    in_stack[node] = true;

    for (const auto& input_node : node->inputs())
    {
        if (!visited[input_node])
        {
            visit(input_node, visited, in_stack, sorted_nodes);
        }
        else if (in_stack[input_node])
        {
            throw std::runtime_error("Cycle detected in computation graph!");
        }
    }

    in_stack[node] = false;
    sorted_nodes.push_back(node);
}

std::vector<std::shared_ptr<graph::Node>>
ExecutionEngine::topological_sort(std::shared_ptr<graph::Node> root_node)
{
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes;
    std::unordered_map<std::shared_ptr<graph::Node>, bool> visited;
    
    std::unordered_map<std::shared_ptr<graph::Node>, bool> in_stack;

    visit(root_node, visited, in_stack, sorted_nodes);
    return sorted_nodes;
}

std::shared_ptr<plast::tensor::Tensor>
ExecutionEngine::execute(std::shared_ptr<graph::Node> root_node)
{
    if (!root_node)
    {
        throw std::runtime_error("Cannot execute a null graph node.");
    }

    std::vector<std::shared_ptr<graph::Node>> sorted_nodes = topological_sort(root_node);

    // Clear output tensors for all NON-LEAF nodes in the current graph before execution
    for (const auto& node : sorted_nodes)
    {
        if (!node->is_leaf())
        {
            node->clear_output_tensor();
        }
    }

    for (const auto& node : sorted_nodes)
    {
        if (node->is_leaf())
        {
            if (!node->has_output_tensor())
            {
                throw std::runtime_error(
                    "Leaf node without output tensor encountered during execution.");
            }
            continue;
        }

        // Collect inputs for the current operation
        std::vector<const tensor::Tensor*> inputs_for_op;
        for (const auto& input_node : node->inputs())
        {
            if (!input_node->has_output_tensor())
            {
                throw std::runtime_error(
                    "Input node value not computed before its dependent operation.");
            }
            inputs_for_op.push_back(input_node->get_output_tensor().get());
        }

        // Execute the operation
        if (inputs_for_op.empty())
        {
            throw std::runtime_error("Operation with no tensor inputs not yet supported.");
        }

        // Determine target device for execution
        core::DeviceType target_device =
            inputs_for_op[0]->device(); // Simple heuristic: use first input's device
        for (size_t i = 1; i < inputs_for_op.size(); ++i)
        {
            if (inputs_for_op[i]->device() != target_device)
            {
                throw std::runtime_error("Inputs to an operation are on different devices. "
                                         "Automatic transfer not yet implemented.");
            }
        }

        if (target_device == core::DeviceType::CPU)
        {
            tensor::Tensor output_tensor = node->operation()->execute_cpu(inputs_for_op);
            node->set_output_tensor(std::make_shared<plast::tensor::Tensor>(
                std::move(output_tensor))); // Cache the result
        }
        else if (target_device == core::DeviceType::CUDA)
        {
            tensor::Tensor output_tensor = node->operation()->execute_cuda(inputs_for_op);
            node->set_output_tensor(std::make_shared<plast::tensor::Tensor>(
                std::move(output_tensor))); // Cache the result
        }
        else
        {
            throw std::runtime_error("Unsupported device type for operation execution.");
        }
    }

    // The result of the root node is the final output
    if (!root_node->has_output_tensor())
    {
        throw std::runtime_error("Root node value not computed after graph execution.");
    }
    return root_node->get_output_tensor(); // Return the shared_ptr
}

void ExecutionEngine::backward(std::shared_ptr<graph::Node> root_node,
                               std::shared_ptr<tensor::Tensor> grad_output)
{
    if (!root_node)
    {
        throw std::runtime_error("Cannot perform backward pass on a null graph node.");
    }

    // Clear gradients for all nodes in the graph
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes = topological_sort(root_node);
    for (const auto& node : sorted_nodes)
    {
        node->clear_grad_tensor();
        if (node->is_leaf() && node->has_output_tensor()) {
            node->get_output_tensor()->set_grad(nullptr); // Clear grad for leaf tensors
        }
    }

    compute_gradients(sorted_nodes, grad_output);
}

void ExecutionEngine::compute_gradients(std::vector<std::shared_ptr<graph::Node>>& sorted_nodes,
                                        std::shared_ptr<tensor::Tensor> grad_output)
{
    // Initialize gradients for the root node
    std::shared_ptr<graph::Node> root_node = sorted_nodes.back();
    if (!root_node->requires_grad())
    {
        // If the root node does not require gradients, there's nothing to do.
        return;
    }

    if (grad_output)
    {
        root_node->set_grad_tensor(grad_output);
    }
    else
    {
        // Default grad_output is a tensor of ones with the same shape as the root_node's output
        if (!root_node->has_output_tensor())
        {
            throw std::runtime_error(
                "Root node output tensor not available to initialize grad_output.");
        }
        root_node->set_grad_tensor(plast::ops::init::ones(
            root_node->get_output_tensor()->shape(), root_node->get_output_tensor()->dtype(),
            root_node->get_output_tensor()->device()));
    }

    // Iterate through nodes in reverse topological order
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());

    for (const auto& node : sorted_nodes)
    {
        if (!node->requires_grad())
        {
            continue; // Skip if no gradient is required
        }

        // Ensure the node has a gradient tensor (it should have been set for the root, or propagated)
        if (!node->has_grad_tensor())
        {
            throw std::runtime_error("Gradient tensor not available for node during backward pass.");
        }

        if (node->is_leaf())
        {
            std::shared_ptr<tensor::Tensor> leaf_tensor = node->get_output_tensor();
            std::shared_ptr<tensor::Tensor> grad_to_accumulate = node->get_grad_tensor();

            if (leaf_tensor->grad() == nullptr)
            {
                leaf_tensor->set_grad(grad_to_accumulate);
            }
            else
            {
                // Accumulate gradients
                ops::AddOperation add_op;
                std::vector<const tensor::Tensor*> add_inputs = {leaf_tensor->grad(), grad_to_accumulate.get()};
                tensor::Tensor accumulated_grad = add_op.execute_cpu(add_inputs);
                leaf_tensor->set_grad(std::make_shared<tensor::Tensor>(std::move(accumulated_grad)));
            }
        }
        else // Non-leaf (operation) nodes
        {
            // Get the output tensor of this node (from forward pass)
            std::shared_ptr<tensor::Tensor> output = node->get_output_tensor();
            if (!output)
            {
                throw std::runtime_error("Node output tensor not available for backward pass.");
            }

            // Get raw pointers to input tensors for the backward operation
            std::vector<const tensor::Tensor*> raw_inputs;
            for (const auto& input_node : node->inputs())
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
                input_grads = node->operation()->backward_cpu(*node->get_grad_tensor(), *output, raw_inputs);
            }
            else if (device == core::DeviceType::CUDA)
            {
                input_grads = node->operation()->backward_cuda(*node->get_grad_tensor(), *output, raw_inputs);
            }
            else
            {
                throw std::runtime_error("Unsupported device type for backward pass.");
            }

            // Propagate gradients to input nodes
            for (size_t i = 0; i < node->inputs().size(); ++i)
            {
                std::shared_ptr<graph::Node> input_node = node->inputs()[i];
                if (input_node->requires_grad())
                {
                    std::shared_ptr<tensor::Tensor> grad_to_propagate = std::make_shared<tensor::Tensor>(std::move(input_grads[i]));
                    
                    if (input_node->get_grad_tensor() == nullptr)
                    {
                        input_node->set_grad_tensor(grad_to_propagate);
                        input_node->get_output_tensor()->set_grad(grad_to_propagate); // Set on the Tensor as well
                    }
                    else
                    {
                        // Accumulate gradients
                        ops::AddOperation add_op;
                        std::vector<const tensor::Tensor*> add_inputs = {input_node->get_grad_tensor().get(), grad_to_propagate.get()};
                        tensor::Tensor accumulated_grad = add_op.execute_cpu(add_inputs);
                        std::shared_ptr<tensor::Tensor> accumulated_grad_ptr = std::make_shared<tensor::Tensor>(std::move(accumulated_grad));
                        input_node->set_grad_tensor(accumulated_grad_ptr);
                        input_node->get_output_tensor()->set_grad(accumulated_grad_ptr); // Set on the Tensor as well
                    }
                }
            }
        }
    }
}

} // namespace execution
} // namespace plast
