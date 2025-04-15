# TorchInsight

TorchInsight 是一个增强型 PyTorch 模型分析工具，提供类似于 torchinfo 的功能，但具有自定义格式和额外特性。

## 特点

- 详细的模型结构可视化
- 自动计算 FLOPS 并选择合适的单位 (K, M, G)
- 支持不包含批次维度的输入维度规格
- 支持为特定输入指定 long 数据类型
- 支持分析各种模型架构 (CNN, Attention, 推荐系统等)
- 彩色输出，提高可读性

## 安装

```bash
pip install torchinsight
```

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

## 高级用法

TorchInsight 支持多种输入格式和数据类型：

```python
# 分析具有多个输入的模型
summary = analyze_model(
    model,
    model_name="ComplexModel",
    input_dims=[(13,), (5,)],  # 两个输入，维度分别为 (13,) 和 (5,)
    long_indices=[1],  # 第二个输入 (索引 1) 应为 torch.long 类型
    batch_size=128,  # 批次大小
)
```

更多示例请参见 `examples` 目录。

## 文档

完整文档请访问：
- [使用指南](docs/usage_zh.md)
- [API 参考](docs/api_zh.md)

## 贡献

欢迎贡献！请随时提交 Pull Request 或创建 Issue。

## 许可证

本项目采用 MIT 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

[English](README.md)
