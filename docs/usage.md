# Usage Guide

This document provides detailed usage instructions and examples for TorchInsight.

## Basic Usage

TorchInsight offers two main ways to use it:

1. Use the `analyze_model` function for quick analysis
2. Use the `ModelAnalyzer` class for more detailed analysis and customization

### Using the analyze_model Function

The `analyze_model` function is the simplest way to use TorchInsight and is suitable for most scenarios:

```python
from torchinsight import analyze_model

summary = analyze_model(
    model,                    # PyTorch model
    model_name="MyModel",     # Optional: model name
    input_dims=(3, 224, 224), # Input dimensions (excluding batch dimension)
    batch_size=32,            # Batch size
)
print(summary)
```

### Using the ModelAnalyzer Class

The `ModelAnalyzer` class provides more control and customization options:

```python
from torchinsight import ModelAnalyzer

# Create analyzer instance
analyzer = ModelAnalyzer(
    model,
    model_name="MyModel",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Analyze model
analyzer.analyze(
    input_dims=(3, 224, 224),
    batch_size=32,
)

# Get analysis results
summary = analyzer.summary(max_depth=3)  # Control maximum display depth
print(summary)
```

## Advanced Usage

### Multiple Input Models

TorchInsight supports analyzing models with multiple inputs:

```python
summary = analyze_model(
    model,
    input_dims=[(10,), (20,)],  # Two inputs with dimensions (10,) and (20,)
    batch_size=64,
)
```

### Specifying Input Data Types

For inputs that require specific data types (such as `torch.long` for indices or tokens), you can use the `long_indices` parameter:

```python
summary = analyze_model(
    model,
    input_dims=[(13,), (5,)],  # Two inputs
    long_indices=[1],          # Second input (index 1) should be torch.long
    batch_size=128,
)
```

### Using Real Data for Analysis

If you want to use real data instead of randomly generated data for analysis, you can provide the input data directly:

```python
# Prepare input data
input_data = torch.randn(32, 3, 224, 224)  # Batch size 32 image data

# Analyze with real data
summary = analyze_model(
    model,
    input_data=input_data,
)
```

## Examples

### CNN Model Analysis

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

### Transformer Model Analysis

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
    input_dims=(50,),  # Sequence length 50
    long_indices=[0],  # Input should be torch.long (token IDs)
    batch_size=32,
)
print(summary)
```

For more examples, see the `examples` directory.

[中文](usage_zh.md)
