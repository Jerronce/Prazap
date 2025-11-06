# Prazap

**Neural Network Acceleration Framework**

Prazap is a high-performance framework for neural network optimization and acceleration, designed to make AI models faster and more efficient for deployment on edge devices.

## Features

- **Model Quantization**: Dynamic and static quantization for reduced model size
- **Model Pruning**: Structured and unstructured pruning to remove redundant parameters
- **Performance Benchmarking**: Built-in tools to measure inference speed
- **Easy Integration**: Simple API for quick model optimization

## Installation

```bash
pip install torch numpy
git clone https://github.com/Jerronce/Prazap.git
cd Prazap
```

## Quick Start

```python
import torch
import torch.nn as nn
from prazap import OptimizationPipeline

# Your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Optimize
pipeline = OptimizationPipeline(model)
optimized_model = pipeline.optimize(prune_amount=0.4, use_quantization=True)

# Benchmark
test_input = torch.randn(1, 784)
results = pipeline.benchmark(test_input)
print(f"Performance: {results}")
```

## Core Components

### ModelQuantizer
Quantizes models using PyTorch's quantization toolkit.

### ModelPruner  
Removes redundant weights to reduce model size.

### OptimizationPipeline
Combines pruning and quantization for maximum efficiency.

## Use Cases

- Mobile AI applications
- IoT device deployment
- Real-time inference systems
- Resource-constrained environments

## License

MIT License

---

*Created with dedication by Jerronce | PraeTech*
