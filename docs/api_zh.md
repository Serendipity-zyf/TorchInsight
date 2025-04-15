# API 参考

本文档提供了 TorchPrint API 的详细信息。

## 主要函数

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

分析 PyTorch 模型并返回格式化的摘要字符串。

**参数：**

- **model** (`nn.Module`): 要分析的 PyTorch 模型。
- **model_name** (`Optional[str]`): 模型的可选名称。默认为模型的类名。
- **input_data** (`Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]]`): 模型的输入数据。如果提供，将执行前向传播。
- **input_size** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): 模型的输入大小。如果提供且 input_data 为 None，将创建此大小的随机张量。
- **input_dims** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): 不包括批次维度的输入维度。如果提供且 input_size 为 None，将创建具有指定批次大小和这些维度的随机张量。
- **dtypes** (`Optional[List[torch.dtype]]`): 输入张量的数据类型。
- **long_indices** (`Optional[List[int]]`): 应具有 torch.long 数据类型的输入索引列表。
- **batch_size** (`int`): 创建随机张量时使用的批次大小。默认为 1。
- **device** (`Optional[torch.device]`): 运行分析的设备。默认为 CUDA（如果可用），否则为 CPU。
- **max_depth** (`int`): 要显示的模块的最大深度。默认为 3。

**返回：**

- `str`: 格式化的摘要字符串。

**示例：**

```python
import torch.nn as nn
from torchprint import analyze_model

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

## 类

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

PyTorch 模型分析器，提供有关模型结构、参数、内存使用和计算复杂度的详细信息。

**参数：**

- **model** (`nn.Module`): 要分析的 PyTorch 模型。
- **model_name** (`Optional[str]`): 模型的可选名称。默认为模型的类名。
- **device** (`Optional[torch.device]`): 运行分析的设备。默认为 CUDA（如果可用），否则为 CPU。
- **input_dtypes** (`Optional[List[torch.dtype]]`): 输入张量的数据类型列表。默认为 torch.float32。

**方法：**

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

分析模型结构、参数和计算复杂度。

**参数：**

- **input_data** (`Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]]`): 模型的输入数据。如果提供，将执行前向传播。
- **input_size** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): 模型的输入大小。如果提供且 input_data 为 None，将创建此大小的随机张量。
- **input_dims** (`Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]]`): 不包括批次维度的输入维度。如果提供且 input_size 为 None，将创建具有指定批次大小和这些维度的随机张量。
- **dtypes** (`Optional[List[torch.dtype]]`): 输入张量的数据类型。
- **long_indices** (`Optional[List[int]]`): 应具有 torch.long 数据类型的输入索引列表。
- **batch_size** (`int`): 创建随机张量时使用的批次大小。默认为 1。

#### summary

```python
def summary(self, max_depth: int = 3) -> str:
```

生成模型的摘要。

**参数：**

- **max_depth** (`int`): 要显示的模块的最大深度。默认为 3。

**返回：**

- `str`: 格式化的摘要字符串。

**示例：**

```python
from torchprint import ModelAnalyzer

analyzer = ModelAnalyzer(model, model_name="MyModel")
analyzer.analyze(input_dims=(3, 224, 224), batch_size=32)
summary = analyzer.summary(max_depth=2)
print(summary)
```

## 实用函数

TorchPrint 还提供了几个内部使用但对自定义分析有用的实用函数：

- `count_parameters(module)`: 计算模块中的参数数量。
- `estimate_memory_usage(module, input_size, output_size)`: 估计模块的内存使用量（以字节为单位）。
- `calculate_macs(module, input_size, output_size)`: 计算模块的乘加运算（MACs）数量。
- `calculate_flops(macs)`: 从 MACs 计算浮点运算（FLOPs）数量。
- `format_size(size)`: 将张量大小格式化为字符串。
- `format_bytes(num_bytes)`: 将字节格式化为具有适当单位的人类可读字符串。
- `format_flops(flops)`: 将 FLOPs 格式化为具有适当单位的人类可读字符串。
- `format_param_count(count)`: 将参数计数格式化为字符串。

[English](api.md)
