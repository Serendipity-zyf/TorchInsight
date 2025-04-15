"""
Simple example of using torchinsight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinsight import analyze_model


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc(x)
        return x


def main():
    # Create model
    model = SimpleModel()

    # Analyze model
    summary = analyze_model(
        model,
        model_name="SimpleModel",
        input_dims=(3, 32, 32),  # Input dimensions (channels, height, width)
        batch_size=64,  # Batch size
    )

    # Print summary
    print(summary)


if __name__ == "__main__":
    main()
