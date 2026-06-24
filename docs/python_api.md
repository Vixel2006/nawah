# Python API Reference

This document provides a comprehensive reference of the high-level Python API exposed by the `plast` package.

---

## 1. Tensors (`plast.Tensor`)

The `Tensor` class is the primary data structure in Plast. It wraps the compiled C-level tensor and provides mathematical operations, shape transformations, and autograd linkages.

### Factory Functions
* **`plast.tensor(data, *, device=Device.CPU, requires_grad=False)`**: Creates a tensor initialized with standard Python lists or NumPy arrays.
* **`plast.zeros(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor of the specified shape filled with zeros.
* **`plast.ones(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor of the specified shape filled with ones.
* **`plast.randn(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor filled with random normal values.
* **`plast.rand(shape, *, device=Device.CPU, requires_grad=False)`**: Returns a tensor filled with random uniform values.

### Shape Transformations
* **`tensor.view(*shape)`** / **`tensor.reshape(*shape)`**: Returns a tensor with the same data viewed under a new shape. A single dimension can be set to `-1` to be inferred.
* **`tensor.transpose(dim0, dim1)`**: Swaps two dimensions.
* **`tensor.permute(*dims)`**: Permutes tensor dimensions in arbitrary order.
* **`tensor.squeeze(dim=None)`**: Removes dimensions of size 1.
* **`tensor.unsqueeze(dim)`**: Inserts a dimension of size 1 at the specified position.
* **`tensor.flatten(start_dim=0, end_dim=-1)`**: Flattens a range of dimensions into a single vector.

### Graph & Realization
* **`tensor.forward()`** / **`tensor.realize()`**: Executes the forward DAG path to populate the tensor's data buffer.
* **`tensor.backward()`**: Triggers the backward pass to calculate gradients for all ancestors. This can only be called on scalar tensors (single-element).
* **`tensor.numpy()`**: Realizes the tensor and returns its values as a standard NumPy array. (Shares memory on CPU).
* **`tensor.item()`**: Realizes the tensor and returns its value as a Python scalar.
* **`tensor.to(device)`**: Moves the tensor data to the target device (`plast.Device.CPU` or `plast.Device.CUDA`).

---

## 2. Neural Network Modules (`plast.nn`)

The `plast.nn` subpackage contains neural network layers, containers, activation functions, and losses.

### Base Class & Container
* **`plast.nn.Module`**: The base class for all neural network layers. Handles tracking submodules and parameters.
* **`plast.nn.Sequential(*layers)`**: A sequential container. Modules are appended and called in the order they are passed.

### Standard Layers
* **`plast.nn.Linear(in_features, out_features, bias=True, device=Device.CPU)`**: A fully-connected layer. 
* **`plast.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device=Device.CPU)`**: 2D Convolutional layer.
* **`plast.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, device=Device.CPU)`**: Batch Normalization layer.
* **`plast.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, device=Device.CPU)`**: Layer Normalization layer.
* **`plast.nn.Dropout(p=0.5)`**: Standard dropout layer (zeros inputs randomly during training).

### Activations
* **`plast.nn.ReLU()`** / **`plast.nn.functional.relu(x)`**: Rectified Linear Unit ($y = \max(0, x)$).
* **`plast.nn.LeakyReLU(negative_slope=0.01)`**: Leaky ReLU activation.
* **`plast.nn.Sigmoid()`** / **`plast.nn.functional.sigmoid(x)`**: Sigmoid activation.
* **`plast.nn.Tanh()`** / **`plast.nn.functional.tanh(x)`**: Hyperbolic tangent activation.
* **`plast.nn.functional.softmax(x, dim=-1)`**: Softmax normalization.

### Loss Functions
* **`plast.nn.MSELoss()`**: Mean Squared Error loss.
* **`plast.nn.CrossEntropyLoss()`**: Softmax Cross Entropy loss for classification.

---

## 3. Optimizers & Schedulers (`plast.optim`)

Optimizers update model parameters during training using calculated gradients.

### Optimizers
* **`plast.optim.SGD(params, lr=0.01)`**: Stochastic Gradient Descent.
* **`plast.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`**: Adam optimizer.
* **`plast.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`**: Adam with decoupled weight decay.

### Learning Rate Schedulers
* **`StepLR(optimizer, step_size, gamma=0.1)`**: Decays the learning rate by `gamma` every `step_size` epochs.
* **`MultiStepLR(optimizer, milestones, gamma=0.1)`**: Decays the learning rate by `gamma` when the epoch count reaches milestone marks.
* **`ExponentialLR(optimizer, gamma)`**: Decays the learning rate exponentially.
* **`CosineAnnealingLR(optimizer, T_max, eta_min=0)`**: Applies cosine annealing.
* **`ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)`**: Decays learning rate when performance metrics plateau.

---

## 4. Data Loading Utilities (`plast.data`)

Plast provides optimized dataloading interfaces designed to load datasets with minimal latency.

* **`plast.data.TensorDataset(x_tensor, y_tensor)`**: Wraps inputs and targets into a single dataset.
* **`plast.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, device=Device.CPU)`**: 
  An iterator that batches and shuffles datasets, and automatically stages batch tensors directly onto target execution hardware (e.g. staging CPU arrays to GPU memory).

---

## 5. Experiment Tracking (`plast.experiment`)

The experiment framework logs run states, configurations, metrics, and best checkpoints automatically.

### Usage Example
```python
from plast.experiment import ExperimentConfig, ExperimentTracker

# Define tracking configuration
config = ExperimentConfig(
    name="mnist_mlp",
    model={"hidden_size": 128, "activation": "relu"},
    training={"lr": 0.01, "epochs": 10},
    device="cuda"
)

# Start tracking run
tracker = ExperimentTracker(config)

for epoch in range(10):
    loss = train_epoch(...)
    
    # Logs values and saves checkpoints automatically if performance improves
    tracker.log_epoch(epoch, {"train_loss": loss}, model=model)

# Finalize the run
tracker.finish()
```

Logs are written under `experiments/{experiment_name}/{run_id}/` (e.g., `experiments/mnist_mlp/run_001/`), including a frozen `config.yaml`, per-epoch `metrics.yaml`, and weight parameter checkpoints (`checkpoints/best_model.npz`).
