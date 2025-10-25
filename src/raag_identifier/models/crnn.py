"""
CRNN (Convolutional Recurrent Neural Network) for Raag classification.
Combines CNN for spatial features with RNN for temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class CRNN(nn.Module):
    """
    CRNN model: CNN to extract spatial features + BiGRU/LSTM for temporal patterns.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 1,
        cnn_channels: list = [32, 64, 128, 256],
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: Literal['lstm', 'gru'] = 'gru',
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_classes: Number of output classes
            n_channels: Number of input channels
            cnn_channels: List of CNN channel sizes
            rnn_hidden_size: RNN hidden size
            rnn_num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm' or 'gru')
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate
        """
        super(CRNN, self).__init__()

        self.n_classes = n_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional

        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = n_channels

        for i, out_channels in enumerate(cnn_channels):
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout * 0.5),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output feature size
        # After 4 pooling layers of stride 2, spatial dimensions are reduced by 2^4 = 16
        self.cnn_output_channels = cnn_channels[-1]

        # RNN layers for temporal modeling
        rnn_input_size = self.cnn_output_channels

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if rnn_num_layers > 1 else 0,
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if rnn_num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Classifier
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, 128),
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
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # CNN feature extraction
        # Input: [batch, channels, freq, time]
        x = self.cnn(x)
        # Output: [batch, cnn_channels, freq', time']

        # Reshape for RNN: collapse frequency dimension
        # Global average pooling over frequency dimension
        x = torch.mean(x, dim=2)  # [batch, cnn_channels, time']

        # Transpose for RNN: [batch, time, features]
        x = x.permute(0, 2, 1)

        # RNN temporal modeling
        x, _ = self.rnn(x)  # [batch, time, rnn_hidden * directions]

        # Aggregate temporal information
        # Option 1: Take last hidden state
        # x = x[:, -1, :]

        # Option 2: Average pooling over time (more robust)
        x = torch.mean(x, dim=1)  # [batch, rnn_hidden * directions]

        # Classification
        x = self.classifier(x)

        return x


class AttentionCRNN(nn.Module):
    """
    CRNN with attention mechanism for temporal aggregation.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 1,
        cnn_channels: list = [32, 64, 128, 256],
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: Literal['lstm', 'gru'] = 'gru',
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_classes: Number of output classes
            n_channels: Number of input channels
            cnn_channels: List of CNN channel sizes
            rnn_hidden_size: RNN hidden size
            rnn_num_layers: Number of RNN layers
            rnn_type: Type of RNN
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate
        """
        super(AttentionCRNN, self).__init__()

        self.n_classes = n_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional

        # CNN layers
        cnn_layers = []
        in_channels = n_channels

        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout * 0.5),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_output_channels = cnn_channels[-1]

        # RNN layers
        rnn_input_size = self.cnn_output_channels

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if rnn_num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if rnn_num_layers > 1 else 0,
            )

        # Attention mechanism
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """
        Forward pass with attention.

        Args:
            x: Input tensor with shape [batch, channels, freq, time]

        Returns:
            Output logits with shape [batch, n_classes]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.cnn(x)
        x = torch.mean(x, dim=2)
        x = x.permute(0, 2, 1)

        # RNN
        x, _ = self.rnn(x)  # [batch, time, features]

        # Attention weights
        attention_weights = self.attention(x)  # [batch, time, 1]
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum
        x = torch.sum(x * attention_weights, dim=1)  # [batch, features]

        # Classification
        x = self.classifier(x)

        return x


def create_crnn_model(
    model_type: str = 'crnn',
    n_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CRNN models.

    Args:
        model_type: Type of model ('crnn' or 'attention_crnn')
        n_classes: Number of output classes
        **kwargs: Additional arguments

    Returns:
        Model instance
    """
    if model_type == 'crnn':
        return CRNN(n_classes=n_classes, **kwargs)
    elif model_type == 'attention_crnn':
        return AttentionCRNN(n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown CRNN model type: {model_type}")
