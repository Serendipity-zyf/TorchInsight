"""
Unit tests for torchinsight.
"""

import unittest
import torch
import torch.nn as nn
from torchinsight import analyze_model, ModelAnalyzer


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


class TestTorchInsight(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()

    def test_analyze_model(self):
        """Test that analyze_model runs without errors."""
        summary = analyze_model(
            self.model,
            model_name="SimpleModel",
            input_dims=(3, 32, 32),
            batch_size=1,
        )
        self.assertIsInstance(summary, str)
        self.assertIn("SimpleModel", summary)

    def test_model_analyzer(self):
        """Test that ModelAnalyzer class works correctly."""
        analyzer = ModelAnalyzer(self.model, model_name="SimpleModel")
        analyzer.analyze(input_dims=(3, 32, 32), batch_size=1)
        summary = analyzer.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("SimpleModel", summary)

    def test_parameter_count(self):
        """Test that parameter counting works correctly."""
        analyzer = ModelAnalyzer(self.model)
        analyzer.analyze(input_dims=(3, 32, 32), batch_size=1)
        # SimpleModel should have parameters in conv1 and fc layers
        self.assertGreater(analyzer.total_params, 0)

    def test_multiple_inputs(self):
        """Test with multiple input tensors."""

        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(20, 5)
                self.fc3 = nn.Linear(10, 1)

            def forward(self, x1, x2):
                x1 = self.fc1(x1)
                x2 = self.fc2(x2)
                x = torch.cat([x1, x2], dim=1)
                return self.fc3(x)

        model = MultiInputModel()
        summary = analyze_model(
            model,
            input_dims=[(10,), (20,)],
            batch_size=1,
        )
        self.assertIsInstance(summary, str)


if __name__ == "__main__":
    unittest.main()
