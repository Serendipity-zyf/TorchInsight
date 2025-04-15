# TorchInsight 文档

TorchInsight 是一个增强型 PyTorch 模型分析工具，提供类似于 torchinfo 的功能，但具有自定义格式和额外特性。

## 安装

```bash
pip install torchinsight
```

## 主要功能

- **详细的模型结构可视化**：以树状结构展示模型层次
- **参数统计**：计算每层和整个模型的参数数量
- **计算复杂度分析**：自动计算 MACs 和 FLOPs
- **内存使用估计**：估计模型运行时的内存占用
- **彩色输出**：提高可读性的彩色终端输出
- **灵活的输入规格**：支持多种输入格式和数据类型

## 快速开始

```python
import torch
import torch.nn as nn
from torchinsight import analyze_model

# 创建一个简单的模型
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

# 创建模型实例
model = SimpleModel()

# 分析模型
summary = analyze_model(
    model,
    model_name="SimpleModel",
    input_dims=(3, 32, 32),  # 输入维度 (通道, 高度, 宽度)
    batch_size=64,  # 批次大小
)

# 打印分析结果
print(summary)
```

## 更多文档

- [使用指南](usage_zh.md)
- [API 参考](api_zh.md)

[English](index.md)
