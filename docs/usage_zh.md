# 使用指南

本文档提供了 TorchInsight 的详细使用说明和示例。

## 基本用法

TorchInsight 提供了两种主要的使用方式：

1. 使用 `analyze_model` 函数进行快速分析
2. 使用 `ModelAnalyzer` 类进行更详细的分析和自定义

### 使用 analyze_model 函数

`analyze_model` 函数是最简单的使用方式，适合大多数场景：

```python
from torchinsight import analyze_model

summary = analyze_model(
    model,                    # PyTorch 模型
    model_name="MyModel",     # 可选：模型名称
    input_dims=(3, 224, 224), # 输入维度（不包括批次维度）
    batch_size=32,            # 批次大小
)
print(summary)
```

### 使用 ModelAnalyzer 类

`ModelAnalyzer` 类提供了更多的控制和自定义选项：

```python
from torchinsight import ModelAnalyzer

# 创建分析器实例
analyzer = ModelAnalyzer(
    model,
    model_name="MyModel",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# 分析模型
analyzer.analyze(
    input_dims=(3, 224, 224),
    batch_size=32,
)

# 获取分析结果
summary = analyzer.summary(max_depth=3)  # 控制显示的最大深度
print(summary)
```

## 高级用法

### 多输入模型

TorchInsight 支持分析具有多个输入的模型：

```python
summary = analyze_model(
    model,
    input_dims=[(10,), (20,)],  # 两个输入，维度分别为 (10,) 和 (20,)
    batch_size=64,
)
```

### 指定输入数据类型

对于需要特定数据类型的输入（如 `torch.long` 类型的索引或标记），可以使用 `long_indices` 参数：

```python
summary = analyze_model(
    model,
    input_dims=[(13,), (5,)],  # 两个输入
    long_indices=[1],          # 第二个输入（索引 1）应为 torch.long 类型
    batch_size=128,
)
```

### 使用真实数据进行分析

如果你想使用真实数据而不是随机生成的数据进行分析，可以直接提供输入数据：

```python
# 准备输入数据
input_data = torch.randn(32, 3, 224, 224)  # 批次大小为 32 的图像数据

# 使用真实数据分析
summary = analyze_model(
    model,
    input_data=input_data,
)
```

## 示例

### CNN 模型分析

```python
import torch.nn as nn
from torchinsight import analyze_model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

model = CNN()
summary = analyze_model(model, input_dims=(3, 32, 32), batch_size=64)
print(summary)
```

### Transformer 模型分析

```python
import torch.nn as nn
from torchinsight import analyze_model

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x

model = TransformerModel()
summary = analyze_model(
    model,
    input_dims=(50,),  # 序列长度为 50
    long_indices=[0],  # 输入应为 torch.long 类型（词元 ID）
    batch_size=32,
)
print(summary)
```

更多示例请参见 `examples` 目录。

[English](usage.md)
