# Plast Documentation Guide

Welcome to the Plast technical documentation. This sitemap indexes the architecture, design patterns, and programming interfaces of the Plast deep learning engine.

---

## 1. Documentation Index

* **[Getting Started](./getting_started.md)**: Environment setup, compiling the Python bindings, and writing a model training script.
* **[Engine Architecture](./architecture.md)**: Deep dive into the custom memory Arenas (persistent vs. transient), strided tensor layouts, DAG autograd traversal, and the graph scheduler.
* **[Python API Reference](./python_api.md)**: Classes, methods, optimizer options, dataloaders, and experiment tracking APIs.
* **[C API Guide](./c_api.md)**: Working with memory pools, creating tensors, creating nodes, running schedule passes, and compiling/linking standalone binaries in C.
* **[Adding Custom Kernels & Fusions](./custom_kernels.md)**: Developer guide for registering custom CPU (AVX) and CUDA (sm_80+) kernels, and writing manual operator fusions in the scheduler.

---

## 2. Core Concepts Reference

If you are new to Plast, these are the core differences to keep in mind compared to other frameworks (such as PyTorch or JAX):

### Arena Allocation Pattern
To avoid system allocation overhead, Plast pre-allocates memory arenas. 
```python
# Initialize metadata & data memory pools
import plast as p
p.init_arenas(device=p.Device.CUDA)

# ... training step ...

# Reset intermediate activations & gradients
p.reset_transient_arenas()
```
* **Persistent Arena**: Model weights, optimizers, and biases.
* **Transient Arena**: Activation outputs and gradient updates. This is cleared at the end of each training loop iteration using `reset_transient_arenas()`.

### Caching JIT Graph Scheduler
The execution paths are compiled and cached on their first sweep. Tensors are realized dynamically via:
```python
# Explicit execution
p.forward(tensor)

# Implicit execution (triggered automatically)
values = tensor.numpy()
```
Decorating step functions with `@p.jit` enables structural hashing, caching the topological ordering and scheduling fusions for subsequent passes.
