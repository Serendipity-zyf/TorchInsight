"""
Complete demo script for the torchinsight module.

This script demonstrates all the features of the torchinsight module:
1. FLOPS calculation with automatic unit selection (K, M, G)
2. Input dimensions specification without batch dimension
3. Long dtype specification for specific inputs
4. Analysis of various model architectures (CNN, Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinsight import analyze_model


# Simple logger for demo purposes
class ColorLogger:
    def __init__(self, name):
        self.name = name

    def info(self, message):
        print(f"[INFO] {self.name}: {message}")


# Simple FMCTR model for demonstration
class FMCTR(nn.Module):
    def __init__(self, feature_dims, dense_feature_dim, embed_dim):
        super().__init__()
        self.feature_dims = feature_dims
        self.dense_feature_dim = dense_feature_dim
        self.embed_dim = embed_dim

        # Embedding layers for sparse features
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in feature_dims])

        # FM part
        self.fm_first_order = nn.Linear(dense_feature_dim, 1)
        self.fm_second_order_size = len(feature_dims) * embed_dim + dense_feature_dim

        # Deep part
        self.deep = nn.Sequential(
            nn.Linear(self.fm_second_order_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, dense_input, sparse_indices):
        # Process dense features
        first_order = self.fm_first_order(dense_input)

        # Process sparse features
        embeddings = [
            self.embeddings[i](sparse_indices[:, i]) for i in range(len(self.feature_dims))
        ]

        # Concatenate all features for deep part
        concat_features = [dense_input]
        for emb in embeddings:
            concat_features.append(emb)

        all_features = torch.cat([feat.view(feat.size(0), -1) for feat in concat_features], dim=1)

        # Deep part
        deep_out = self.deep(all_features)

        # Final output
        output = first_order + deep_out
        return output


logger = ColorLogger(name="CompleteDemo")


def create_fmctr_model():
    """Create an FMCTR model for demonstration."""
    feature_dims = [10, 20, 30, 40, 50]
    dense_feature_dim = 13
    embed_dim = 8
    return FMCTR(feature_dims, dense_feature_dim, embed_dim)


def create_cnn_model():
    """Create a CNN model for demonstration."""

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

            # Pooling layer
            self.pool = nn.MaxPool2d(2, 2)

            # Fully connected layers
            self.fc1 = nn.Linear(64 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)

            # Dropout
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            # Convolutional layers with ReLU and pooling
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            # Flatten
            x = x.view(-1, 64 * 4 * 4)

            # Fully connected layers with ReLU and dropout
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)

            return x

    return CNN()


def create_attention_model():
    """Create a self-attention model for demonstration."""

    class SelfAttention(nn.Module):
        def __init__(self, embed_dim=64, num_heads=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.layer_norm1 = nn.LayerNorm(embed_dim)
            self.layer_norm2 = nn.LayerNorm(embed_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim)
            )

        def forward(self, x):
            # Self-attention with residual connection and layer normalization
            attn_output, _ = self.attention(x, x, x)
            x = self.layer_norm1(x + attn_output)

            # Feed-forward with residual connection and layer normalization
            ff_output = self.feed_forward(x)
            x = self.layer_norm2(x + ff_output)

            return x

    class AttentionModel(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=64, num_heads=8, num_layers=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.position_encoding = nn.Parameter(torch.zeros(1, 100, embed_dim))

            # Stack of self-attention layers
            self.attention_layers = nn.ModuleList(
                [SelfAttention(embed_dim, num_heads) for _ in range(num_layers)]
            )

            # Output layer
            self.output_layer = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            # Get sequence length
            seq_length = x.size(1)

            # Embedding with positional encoding
            x = self.embedding(x) + self.position_encoding[:, :seq_length, :]

            # Pass through attention layers
            for layer in self.attention_layers:
                x = layer(x)

            # Output projection
            x = self.output_layer(x)

            return x

    return AttentionModel()


def demo_complete():
    """Demonstrate all features of the torchinsight module."""
    logger.info("Starting complete demo for torchinsight module")

    print("\n" + "=" * 70)
    print("TORCHINSIGHT MODULE DEMO WITH MULTIPLE ARCHITECTURES")
    print("=" * 70)

    # Create and analyze FMCTR model
    print("\n" + "=" * 50)
    print("1. ANALYZING FMCTR MODEL (RECOMMENDATION)")
    print("=" * 50)
    fmctr_model = create_fmctr_model()
    summary = analyze_model(
        fmctr_model,
        model_name="FMCTR",
        input_dims=[(13,), (5,)],  # Two inputs with dimensions (13,) and (5,)
        long_indices=[1],  # Second input (index 1) should be torch.long
        batch_size=128,  # Specify batch size
    )
    print(summary)

    # Create and analyze CNN model
    print("\n" + "=" * 50)
    print("2. ANALYZING CNN MODEL (COMPUTER VISION)")
    print("=" * 50)
    cnn_model = create_cnn_model()
    summary = analyze_model(
        cnn_model,
        model_name="CNN",
        input_dims=(3, 32, 32),  # Input dimensions for image (channels, height, width)
        batch_size=64,  # Specify batch size
    )
    print(summary)

    # Create and analyze Attention model
    print("\n" + "=" * 50)
    print("3. ANALYZING ATTENTION MODEL (NLP)")
    print("=" * 50)
    attention_model = create_attention_model()
    summary = analyze_model(
        attention_model,
        model_name="Attention",
        input_dims=(50,),  # Sequence length
        long_indices=[0],  # Input should be torch.long (token IDs)
        batch_size=32,  # Specify batch size
    )
    print(summary)

    print("\n" + "=" * 70)
    print("DEMO COMPLETED!")
    print("=" * 70)
    print("\nFeatures demonstrated:")
    print("1. FLOPS calculation with automatic unit selection (K, M, G)")
    print("2. Input dimensions without batch dimension specification")
    print("3. Long dtype specification for token inputs")
    print("4. Analysis of various model architectures (CNN, Attention)")


if __name__ == "__main__":
    demo_complete()
