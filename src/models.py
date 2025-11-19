"""
Neural network models for keyword spotting.

This module implements:
1. MLP (Multi-Layer Perceptron) for MFCC summary vectors
2. CNN (Convolutional Neural Network) for full MFCC spectrograms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MLPClassifier(nn.Module):
    """
    Simple MLP classifier for keyword spotting.

    This model works with MFCC summary vectors (mean + std statistics).
    It uses fully-connected layers with ReLU activations and dropout
    for regularization.

    Architecture:
        Input -> FC -> ReLU -> Dropout -> FC -> ReLU -> Dropout -> ... -> Output
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 num_classes: int,
                 dropout_rate: float = 0.3):
        """
        Initialize the MLP classifier.

        Args:
            input_dim: Dimension of input features (e.g., 2 * N_MFCC = 80)
            hidden_dims: List of hidden layer dimensions (e.g., [256, 128, 64])
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Build the network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization for training stability
            layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU(inplace=True))
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation - will use CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, num_classes))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


class CNNClassifier(nn.Module):
    """
    CNN classifier for keyword spotting.

    This model works with full MFCC matrices (2D spectrograms).
    It uses convolutional layers to learn spatial patterns in the
    time-frequency representation.

    Architecture:
        Input (1, n_mfcc, time_frames)
        -> Conv2D -> ReLU -> MaxPool
        -> Conv2D -> ReLU -> MaxPool
        -> Conv2D -> ReLU -> MaxPool
        -> Flatten -> FC -> Dropout -> Output
    """

    def __init__(self,
                 input_channels: int = 1,
                 conv_channels: List[int] = [32, 64, 128],
                 num_classes: int = 6,
                 dropout_rate: float = 0.3):
        """
        Initialize the CNN classifier.

        Args:
            input_channels: Number of input channels (1 for single MFCC matrix)
            conv_channels: List of output channels for each conv layer
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(CNNClassifier, self).__init__()

        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels

        for channels in conv_channels:
            conv_block = nn.Sequential(
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout_rate * 0.5)  # Spatial dropout
            )
            self.conv_layers.append(conv_block)
            prev_channels = channels

        # Calculate the size after convolutions (depends on input size)
        # For n_mfcc=40, time_frames≈62, after 3 maxpool(2): 40/8=5, 62/8≈7
        # This is an approximation - will be computed dynamically in forward pass
        self.fc_input_dim = None

        # Fully connected layers
        self.fc = None  # Will be initialized in first forward pass
        self.dropout = nn.Dropout(dropout_rate)

    def _initialize_fc(self, flattened_size: int):
        """
        Initialize fully connected layers based on actual input size.

        Args:
            flattened_size: Size of flattened features from conv layers
        """
        self.fc_input_dim = flattened_size
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout.p),
            nn.Linear(256, self.num_classes)
        ).to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, n_mfcc, time_frames)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Initialize FC layers on first forward pass
        if self.fc is None:
            self._initialize_fc(x.size(1))

        # Pass through fully connected layers
        x = self.fc(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor of shape (batch_size, 1, n_mfcc, time_frames)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor of shape (batch_size, 1, n_mfcc, time_frames)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def create_model(model_type: str,
                input_dim: int,
                num_classes: int,
                **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('mlp' or 'cnn')
        input_dim: Input dimension (for MLP) or channels (for CNN)
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    if model_type == 'mlp':
        from src.config import HIDDEN_DIMS, DROPOUT_RATE
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', HIDDEN_DIMS),
            num_classes=num_classes,
            dropout_rate=kwargs.get('dropout_rate', DROPOUT_RATE)
        )
    elif model_type == 'cnn':
        from src.config import CNN_CHANNELS, DROPOUT_RATE
        model = CNNClassifier(
            input_channels=input_dim,
            conv_channels=kwargs.get('conv_channels', CNN_CHANNELS),
            num_classes=num_classes,
            dropout_rate=kwargs.get('dropout_rate', DROPOUT_RATE)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    from src.config import N_MFCC

    # Test MLP
    print("Testing MLP Classifier...")
    mlp = MLPClassifier(
        input_dim=2 * N_MFCC,  # 80 (mean + std for 40 MFCCs)
        hidden_dims=[256, 128, 64],
        num_classes=6,
        dropout_rate=0.3
    )

    # Create dummy input
    batch_size = 16
    dummy_input = torch.randn(batch_size, 2 * N_MFCC)

    # Forward pass
    output = mlp(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in mlp.parameters()):,}")

    # Test CNN
    print("\nTesting CNN Classifier...")
    cnn = CNNClassifier(
        input_channels=1,
        conv_channels=[32, 64, 128],
        num_classes=6,
        dropout_rate=0.3
    )

    # Create dummy input (batch_size, channels, height, width)
    # Approximating MFCC dimensions: (1, 40, 62)
    dummy_input_cnn = torch.randn(batch_size, 1, 40, 62)

    # Forward pass
    output_cnn = cnn(dummy_input_cnn)
    print(f"Input shape: {dummy_input_cnn.shape}")
    print(f"Output shape: {output_cnn.shape}")
    print(f"Model parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    # Test predictions
    predictions = mlp.predict(dummy_input)
    probabilities = mlp.predict_proba(dummy_input)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    print("\nModel tests passed!")
