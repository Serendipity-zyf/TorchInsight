# API Reference

This document provides detailed information about the TorchInsight API.

## Main Functions

### analyze_model

```python
def analyze_model(
    model: nn.Module,
    model_name: Optional[str] = None,
    input_data: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]] = None,
    input_size: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    input_dims: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    long_indices: Optional[List[int]] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    max_depth: int = 3,
) -> str:
```

Analyzes a PyTorch model and returns a formatted summary string.

**Parameters:**

- **model** (`nn.Module`): The PyTorch model to analyze.
- **model_name** (`Optional[str]`): Optional name for the model. Defaults to model's class name.
- **input_data** (`Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]]`): Input data for the model. If provided, a forward pass will be performed.
- **input_size** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): Input size for the model. If provided and input_data is None, random tensors of this size will be created.
- **input_dims** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): Input dimensions excluding batch dimension. If provided and input_size is None, random tensors will be created with the specified batch size and these dimensions.
- **dtypes** (`Optional[List[torch.dtype]]`): Data types for the input tensors.
- **long_indices** (`Optional[List[int]]`): List of indices for inputs that should have torch.long dtype.
- **batch_size** (`int`): Batch size to use when creating random tensors. Defaults to 1.
- **device** (`Optional[torch.device]`): Device to run the analysis on. Defaults to CUDA if available, else CPU.
- **max_depth** (`int`): Maximum depth of modules to display. Defaults to 3.

**Returns:**

- `str`: Formatted summary string.

**Example:**

```python
import torch.nn as nn
from torchinsight import analyze_model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 30 * 30, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16 * 30 * 30)
        x = self.fc(x)
        return x

model = SimpleModel()
summary = analyze_model(model, input_dims=(3, 32, 32), batch_size=64)
print(summary)
```

## Classes

### ModelAnalyzer

```python
class ModelAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        input_dtypes: Optional[List[torch.dtype]] = None,
    ):
```

Analyzer for PyTorch models that provides detailed information about model structure, parameters, memory usage, and computational complexity.

**Parameters:**

- **model** (`nn.Module`): The PyTorch model to analyze.
- **model_name** (`Optional[str]`): Optional name for the model. Defaults to model's class name.
- **device** (`Optional[torch.device]`): Device to run the analysis on. Defaults to CUDA if available, else CPU.
- **input_dtypes** (`Optional[List[torch.dtype]]`): List of dtypes for input tensors. Defaults to torch.float32.

**Methods:**

#### analyze

```python
def analyze(
    self,
    input_data: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]] = None,
    input_size: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    input_dims: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    long_indices: Optional[List[int]] = None,
    batch_size: int = 1,
) -> None:
```

Analyzes the model structure, parameters, and computational complexity.

**Parameters:**

- **input_data** (`Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]]`): Input data for the model. If provided, a forward pass will be performed.
- **input_size** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): Input size for the model. If provided and input_data is None, random tensors of this size will be created.
- **input_dims** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): Input dimensions excluding batch dimension. If provided and input_size is None, random tensors will be created with the specified batch size and these dimensions.
- **dtypes** (`Optional[List[torch.dtype]]`): Data types for the input tensors.
- **long_indices** (`Optional[List[int]]`): List of indices for inputs that should have torch.long dtype.
- **batch_size** (`int`): Batch size to use when creating random tensors. Defaults to 1.

#### summary

```python
def summary(self, max_depth: int = 3) -> str:
```

Generates a summary of the model.

**Parameters:**

- **max_depth** (`int`): Maximum depth of modules to display. Defaults to 3.

**Returns:**

- `str`: Formatted summary string.

**Example:**

```python
from torchinsight import ModelAnalyzer

analyzer = ModelAnalyzer(model, model_name="MyModel")
analyzer.analyze(input_dims=(3, 224, 224), batch_size=32)
summary = analyzer.summary(max_depth=2)
print(summary)
```

## Utility Functions

TorchInsight also provides several utility functions that are used internally but can be useful for custom analysis:

- `count_parameters(module)`: Count the number of parameters in a module.
- `estimate_memory_usage(module, input_size, output_size)`: Estimate the memory usage of a module in bytes.
- `calculate_macs(module, input_size, output_size)`: Calculate the number of multiply-accumulate operations (MACs) for a module.
- `calculate_flops(macs)`: Calculate the number of floating point operations (FLOPs) from MACs.
- `format_size(size)`: Format tensor size as a string.
- `format_bytes(num_bytes)`: Format bytes as a human-readable string with appropriate units.
- `format_flops(flops)`: Format FLOPs as a human-readable string with appropriate units.
- `format_param_count(count)`: Format parameter count as a string.

[中文](api_zh.md)
