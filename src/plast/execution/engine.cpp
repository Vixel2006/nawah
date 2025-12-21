#include "plast/execution/engine.h"
#include "plast/ops/binary/add.h"
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
        core::DeviceType target_device = inputs_for_op[0]->device();
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

std::shared_ptr<plast::tensor::Tensor>
ExecutionEngine::backward(std::shared_ptr<graph::Node> root_node)
{
    if (!root_node)
    {
        throw std::runtime_error("Cannot execute a null graph node.");
    }

    // Ensure the forward pass has been executed, as backward pass needs the output tensors.
    if (!root_node->has_output_tensor())
    {
        execute(root_node);
    }

    // Topologically sort the graph to process nodes in a valid order.
    std::vector<std::shared_ptr<graph::Node>> sorted_nodes = topological_sort(root_node);
    // Reverse the order for the backward pass (from output to inputs).
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());

    // Map to store and accumulate gradients for each node.
    std::unordered_map<std::shared_ptr<graph::Node>, std::shared_ptr<tensor::Tensor>> gradient_map;

    // The gradient of the final output is initialized to 1.
    gradient_map[root_node] =
        ops::init::ones(root_node->shape(), root_node->get_output_tensor()->dtype(),
                        root_node->get_output_tensor()->device());

    ops::AddOperation add_op; // For accumulating gradients.

    for (const auto& node : sorted_nodes)
    {
        if (node->is_leaf())
        {
            continue; // Leaf nodes are inputs, their gradients are accumulated.
        }

        // Get the gradient flowing backwards from the output of this node.
        auto it = gradient_map.find(node);
        if (it == gradient_map.end())
        {
            // This node does not contribute to the final output, so we can skip it.
            continue;
        }
        const auto& grad_output = *(it->second);

        // Get the tensors that were inputs to this node's forward operation.
        std::vector<const tensor::Tensor*> inputs_for_op = node->get_inputs_as_raw_pointers();
        core::DeviceType target_device = node->get_output_tensor()->device();

        // Compute the gradients with respect to the inputs of the operation.
        std::vector<tensor::Tensor> grad_inputs;
        if (target_device == core::DeviceType::CPU)
        {
            grad_inputs = node->operation()->backward_cpu(grad_output, *node->get_output_tensor(),
                                                          inputs_for_op);
        }
        else if (target_device == core::DeviceType::CUDA)
        {
            grad_inputs = node->operation()->backward_cuda(grad_output, *node->get_output_tensor(),
                                                           inputs_for_op);
        }
        else
        {
            throw std::runtime_error("Unsupported device type for backward operation execution.");
        }

        // Distribute and accumulate the computed gradients to the nodes that produced the inputs.
        for (size_t i = 0; i < node->inputs().size(); ++i)
        {
            auto input_node = node->inputs()[i];
            if (!input_node->requires_grad())
            {
                continue; // Skip nodes that don't require gradients.
            }

            tensor::Tensor grad_input = std::move(grad_inputs[i]);
            auto& current_grad_for_input = gradient_map[input_node];

            if (!current_grad_for_input)
            {
                // This is the first gradient for this node, just store it.
                current_grad_for_input = std::make_shared<tensor::Tensor>(std::move(grad_input));
            }
            else
            {
                // Gradient already exists, so we accumulate by adding the new gradient.
                std::shared_ptr<tensor::Tensor> new_accumulated_grad;
                if (target_device == core::DeviceType::CPU)
                {
                    new_accumulated_grad = std::make_shared<tensor::Tensor>(
                        add_op.execute_cpu({current_grad_for_input.get(), &grad_input}));
                }
                else if (target_device == core::DeviceType::CUDA)
                {
                    new_accumulated_grad = std::make_shared<tensor::Tensor>(
                        add_op.execute_cuda({current_grad_for_input.get(), &grad_input}));
                }
                else
                {
                    throw std::runtime_error("Unsupported device type for gradient accumulation.");
                }
                current_grad_for_input = new_accumulated_grad;
            }
        }
    }

    // After the backward pass, set the .grad attribute on each tensor.
    for (const auto& pair : gradient_map)
    {
        if (pair.first->has_output_tensor())
        {
            pair.first->get_output_tensor()->set_grad(pair.second);
        }
    }

    // Return the gradient of the root node.
    return root_node->get_output_tensor()->grad()
               ? root_node->get_output_tensor()->grad_shared_ptr()
               : nullptr;
}

} // namespace execution
} // namespace plast
