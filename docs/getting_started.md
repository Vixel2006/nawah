# Getting Started

Welcome to Plast! This guide will help you install the library, configure its custom memory arenas, and train your first model in Python.

---

## 1. What is Plast?

Plast is a lightweight deep learning engine built in C and CUDA. It is designed to be highly efficient, transparent, and completely free of large runtime framework dependencies. 

### Key Features
* **Zero Runtime Overhead**: C/CUDA core avoids slow Python dispatch loops during tensor sweeps.
* **Deterministic Allocation**: Custom memory arenas pre-allocate space for weights and activations, removing dynamic allocations from the execution path.
* **Familiar Syntax**: Exposes PyTorch-style layers, optimizers, and dataloaders for rapid integration.
* **Smart Scheduling**: Traverses the autograd graph, caches topological execution paths, and dispatches patterns directly to optimized fused kernels.

---

## 2. Installation

Plast compiles its core library and packages it with Python bindings via [pybind11](https://github.com/pybind/pybind11).

### Prerequisites
* **C Compiler**: `gcc` or `clang` with OpenMP support.
* **CUDA Toolkit**: Required for GPU acceleration (requires `nvcc` and an NVIDIA Ampere or newer GPU with sm_80+ support).
* **Python**: Python 3.8 or newer.
* **uv** or **pip** for package management.

### Build and Install

1. Clone the repository:
   ```bash
   git clone https://github.com/Vixel2006/plast.git
   cd plast
   ```

2. Compile and install in editable mode (which compiles the C extension automatically):
   ```bash
   pip install -e . --no-build-isolation
   ```
   *If you are using the modern, faster `uv` packet manager:*
   ```bash
   uv pip install -e . --no-build-isolation
   ```

3. (Optional) Run the test suite to confirm compile checks and hardware capabilities:
   ```bash
   pytest tests/ -v
   ```

---

## 3. Your First Training Step

Training a model in Python with Plast uses a familiar workflow. However, because Plast uses a custom arena memory model, there are a few minor differences to keep in mind.

### End-to-End XOR Classifier

Here is a complete, working script that initializes memory arenas, builds a Multi-Layer Perceptron (MLP) model, runs a training loop, and resets the transient graph memory at each step.

```python
import plast as p
import numpy as np

def train_xor():
    # 1. Initialize Arenas
    # We pre-allocate a fast metadata pool and a tensor data pool.
    # intermediate activations and gradients are stored in a transient arena.
    p.init_arenas(device=p.Device.CPU)

    # 2. Define the Model
    # Sequential containers manage parameters, weight initialization, and graph linkages.
    model = p.nn.Sequential(
        p.nn.Linear(2, 8),
        p.nn.ReLU(),
        p.nn.Linear(8, 1)
    )

    loss_fn = p.nn.MSELoss()
    optimizer = p.optim.Adam(model.parameters(), lr=0.01)

    # 3. Prepare Tensors
    # Tensors are mapped to memory locations inside our active arenas.
    x_data = p.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=p.Device.CPU)
    y_data = p.tensor([[0], [1], [1], [0]], device=p.Device.CPU)

    # 4. Define the training step
    # We decorate the function with @p.jit. This fingerprints the model's graph structure 
    # and caches the topological schedule to bypass graph analysis on subsequent epochs.
    @p.jit
    def train_step(x, y):
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(x)
        loss = loss_fn(preds, y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer updates
        optimizer.step()
        return loss

    # 5. Training loop
    print("Training MLP XOR model...")
    for epoch in range(1001):
        loss = train_step(x_data, y_data)
        
        # CRITICAL: Reset transient memory pools to free activation and gradient buffers
        p.reset_transient_arenas()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    # 6. Evaluate Model Predictions
    test_preds = model(x_data)
    
    # Force realization of forward operations prior to accessing values
    p.forward(test_preds)
    print("\nFinal Model Predictions:")
    print(test_preds.numpy())

if __name__ == "__main__":
    train_xor()
```

---

## 4. Key Execution Differences vs. PyTorch

When moving from PyTorch to Plast, keep the following execution characteristics in mind:

### Graph Realization
In PyTorch, operations execute eagerly. In Plast, operations build a DAG of nodes. A tensor's data is only calculated when:
1. `p.forward(tensor)` (or `tensor.forward()`) is explicitly called.
2. `tensor.numpy()` or `tensor.item()` is called (which automatically triggers `.forward()`).
3. `tensor.backward()` is called (which automatically realizes the forward pass first).

### Arena Memory Model
In PyTorch, memory is dynamically allocated and garbage-collected. Plast avoids heap allocation overhead during training by utilizing static Arenas:
* **Persistent Arena**: Retains model parameters and weights.
* **Transient Arena**: Caches dynamic inputs, intermediate activations, and backpropagation gradients.
* **Garbage Collection**: To prevent memory overflow, you must call `p.reset_transient_arenas()` at the end of each training batch or step.
