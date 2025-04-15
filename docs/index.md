# TorchPrint Documentation

TorchPrint is an enhanced PyTorch model analysis tool that provides functionality similar to torchinfo but with custom formatting and additional features.

## Installation

```bash
pip install torchprint
```

## Main Features

- **Detailed Model Structure Visualization**: Display model hierarchy in a tree structure
- **Parameter Statistics**: Calculate parameter counts for each layer and the entire model
- **Computational Complexity Analysis**: Automatically calculate MACs and FLOPs
- **Memory Usage Estimation**: Estimate memory usage during model execution
- **Colorized Output**: Colorized terminal output for improved readability
- **Flexible Input Specifications**: Support for various input formats and data types

## Quick Start

```python
import torch
import torch.nn as nn
from torchprint import analyze_model

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc(x)
        return x

# Create model instance
model = SimpleModel()

# Analyze model
summary = analyze_model(
    model,
    model_name="SimpleModel",
    input_dims=(3, 32, 32),  # Input dimensions (channels, height, width)
    batch_size=64,  # Batch size
)

# Print analysis results
print(summary)
```

## More Documentation

- [Usage Guide](usage.md)
- [API Reference](api.md)

[中文](index_zh.md)
