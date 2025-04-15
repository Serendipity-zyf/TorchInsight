# TorchPrint

TorchPrint is an enhanced PyTorch model analysis tool that provides functionality similar to torchinfo but with custom formatting and additional features.

## Features

- Detailed model structure visualization
- Automatic FLOPS calculation with appropriate unit selection (K, M, G)
- Support for input dimension specification without batch dimension
- Support for long dtype specification for specific inputs
- Analysis of various model architectures (CNN, Attention, Recommendation systems, etc.)
- Colorized output for improved readability

## Installation

```bash
pip install torchprint
```

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

## Advanced Usage

TorchPrint supports multiple input formats and data types:

```python
# Analyze model with multiple inputs
summary = analyze_model(
    model,
    model_name="ComplexModel",
    input_dims=[(13,), (5,)],  # Two inputs with dimensions (13,) and (5,)
    long_indices=[1],  # Second input (index 1) should be torch.long
    batch_size=128,  # Batch size
)
```

For more examples, see the `examples` directory.

## Documentation

For complete documentation, visit:
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)

## Contributing

Contributions are welcome! Feel free to submit Pull Requests or create Issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[中文](README_zh.md)
