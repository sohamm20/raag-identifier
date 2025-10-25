"""
CNN models for Raag classification.
Implements lightweight and ResNet-based architectures for CQT/spectrogram input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple lightweight CNN for quick experiments.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 1,
        input_height: int = 84,  # CQT bins
        dropout: float = 0.3,
    ):
        """
        Args:
            n_classes: Number of output classes
            n_channels: Number of input channels (1 for single spectrogram)
            input_height: Height of input spectrogram (frequency bins)
            dropout: Dropout rate
        """
        super(SimpleCNN, self).__init__()

        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor with shape [batch, channels, freq, time]

        Returns:
            Output logits with shape [batch, n_classes]
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet-style architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            downsample: Downsample layer for skip connection
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCNN(nn.Module):
    """
    ResNet-inspired CNN for Raag classification.
    More powerful model for final training.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 1,
        layers: list = [2, 2, 2, 2],
        base_channels: int = 64,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_classes: Number of output classes
            n_channels: Number of input channels
            layers: Number of residual blocks per stage
            base_channels: Base number of channels
            dropout: Dropout rate
        """
        super(ResNetCNN, self).__init__()

        self.in_channels = base_channels

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_layer(base_channels, layers[0])
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, n_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1):
        """
        Create a stage of residual blocks.

        Args:
            out_channels: Number of output channels
            blocks: Number of blocks
            stride: Stride for first block

        Returns:
            Sequential module of residual blocks
        """
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor with shape [batch, channels, freq, time]

        Returns:
            Output logits with shape [batch, n_classes]
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_model(
    model_type: str = 'simple',
    n_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('simple' or 'resnet')
        n_classes: Number of output classes
        **kwargs: Additional arguments for model

    Returns:
        Model instance
    """
    if model_type == 'simple':
        return SimpleCNN(n_classes=n_classes, **kwargs)
    elif model_type == 'resnet':
        return ResNetCNN(n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
