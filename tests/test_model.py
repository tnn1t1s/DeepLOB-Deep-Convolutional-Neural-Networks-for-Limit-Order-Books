import pytest
import torch
from src.model import deeplob

def test_deeplob_forward_pass():
    # Parameters
    batch_size = 16
    T = 100
    y_len = 3

    # Instantiate the model
    model = deeplob(y_len=y_len)
    
    # Create a dummy input
    # Shape: (batch_size, 1, T, 40)
    # Here 1 is the channel dimension and 40 is the number of features.
    x = torch.randn(batch_size, 1, T, 40)

    # Forward pass
    output = model(x)

    # Check output shape: should be (batch_size, y_len)
    assert output.shape == (batch_size, y_len), f"Expected output shape {(batch_size, y_len)}, got {output.shape}"

    # Check output values are valid probabilities (after softmax)
    # Values should be between 0 and 1, and sum to 1 along dim=1
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output probabilities should be between 0 and 1."
    sum_along_classes = output.sum(dim=1)
    # Due to floating point arithmetic, use a tolerance
    assert torch.allclose(sum_along_classes, torch.ones(batch_size), atol=1e-5), "Probabilities should sum to approximately 1."

def test_deeplob_on_gpu_if_available():
    if torch.cuda.is_available():
        batch_size = 8
        T = 50
        y_len = 3
        device = 'cuda'
        
        model = deeplob(y_len=y_len).to(device)
        x = torch.randn(batch_size, 1, T, 40, device=device)

        output = model(x)
        # Just check that it runs without error and remains on GPU
        assert output.is_cuda, "Output should be on the GPU if input is on GPU."
        assert output.shape == (batch_size, y_len)

