"""
Unit tests for model architectures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from raag_identifier.models import (
    SimpleCNN,
    ResNetCNN,
    CRNN,
    AttentionCRNN,
    create_model,
    create_crnn_model,
)


@pytest.fixture
def sample_input():
    """Generate sample input tensor."""
    # Batch of 2, 84 frequency bins, 100 time frames
    return torch.randn(2, 84, 100)


def test_simple_cnn_forward(sample_input):
    """Test SimpleCNN forward pass."""
    model = SimpleCNN(n_classes=3)
    output = model(sample_input)

    assert output.shape == (2, 3)  # Batch size 2, 3 classes


def test_simple_cnn_training():
    """Test SimpleCNN in training mode."""
    model = SimpleCNN(n_classes=3)
    model.train()

    sample_input = torch.randn(4, 84, 100)
    output = model(sample_input)

    assert output.shape == (4, 3)


def test_resnet_cnn_forward(sample_input):
    """Test ResNetCNN forward pass."""
    model = ResNetCNN(n_classes=3)
    output = model(sample_input)

    assert output.shape == (2, 3)


def test_resnet_cnn_custom_layers():
    """Test ResNetCNN with custom layer configuration."""
    model = ResNetCNN(n_classes=3, layers=[1, 1, 1, 1], base_channels=32)

    sample_input = torch.randn(2, 84, 100)
    output = model(sample_input)

    assert output.shape == (2, 3)


def test_crnn_forward(sample_input):
    """Test CRNN forward pass."""
    model = CRNN(n_classes=3)
    output = model(sample_input)

    assert output.shape == (2, 3)


def test_crnn_lstm():
    """Test CRNN with LSTM."""
    model = CRNN(n_classes=3, rnn_type='lstm')

    sample_input = torch.randn(2, 84, 100)
    output = model(sample_input)

    assert output.shape == (2, 3)


def test_crnn_unidirectional():
    """Test CRNN with unidirectional RNN."""
    model = CRNN(n_classes=3, bidirectional=False)

    sample_input = torch.randn(2, 84, 100)
    output = model(sample_input)

    assert output.shape == (2, 3)


def test_attention_crnn_forward(sample_input):
    """Test AttentionCRNN forward pass."""
    model = AttentionCRNN(n_classes=3)
    output = model(sample_input)

    assert output.shape == (2, 3)


def test_create_model_simple():
    """Test model factory for simple CNN."""
    model = create_model(model_type='simple', n_classes=3)

    assert isinstance(model, SimpleCNN)


def test_create_model_resnet():
    """Test model factory for ResNet."""
    model = create_model(model_type='resnet', n_classes=3)

    assert isinstance(model, ResNetCNN)


def test_create_crnn_model():
    """Test CRNN model factory."""
    model = create_crnn_model(model_type='crnn', n_classes=3)

    assert isinstance(model, CRNN)


def test_create_attention_crnn_model():
    """Test Attention CRNN model factory."""
    model = create_crnn_model(model_type='attention_crnn', n_classes=3)

    assert isinstance(model, AttentionCRNN)


def test_model_gradient_flow():
    """Test gradient flow through model."""
    model = SimpleCNN(n_classes=3)
    optimizer = torch.optim.Adam(model.parameters())

    # Forward pass
    input_tensor = torch.randn(2, 84, 100)
    output = model(input_tensor)

    # Backward pass
    loss = output.mean()
    loss.backward()

    # Check gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_model_parameter_count():
    """Test parameter counting."""
    model = SimpleCNN(n_classes=3)
    param_count = sum(p.numel() for p in model.parameters())

    assert param_count > 0
    print(f"SimpleCNN parameters: {param_count:,}")


def test_model_output_range():
    """Test model output range (logits)."""
    model = SimpleCNN(n_classes=3)
    model.eval()

    with torch.no_grad():
        input_tensor = torch.randn(2, 84, 100)
        output = model(input_tensor)

        # Logits can be any real number
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_model_with_different_input_sizes():
    """Test model with varying input sizes."""
    model = SimpleCNN(n_classes=3)
    model.eval()

    # Different time dimensions
    with torch.no_grad():
        for time_frames in [50, 100, 200]:
            input_tensor = torch.randn(2, 84, time_frames)
            output = model(input_tensor)
            assert output.shape == (2, 3)
