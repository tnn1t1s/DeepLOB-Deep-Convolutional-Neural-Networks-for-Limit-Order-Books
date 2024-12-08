import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from src.training import batch_gd

class DummyModel(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        # Just a single linear layer to produce outputs for 3 classes
        self.fc = nn.Linear(40, y_len)

    def forward(self, x):
        # x: (B, 1, T, 40) - we will just take the last time step for simplicity
        x = x[:, 0, -1, :]  # shape (B, 40)
        return self.fc(x)   # shape (B, y_len)

class MockDataLoader:
    def __init__(self, num_batches=2, batch_size=8, T=100, num_classes=3):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.T = T
        self.num_classes = num_classes

    def __iter__(self):
        for _ in range(self.num_batches):
            inputs = torch.randn(self.batch_size, 1, self.T, 40)
            targets = torch.randint(0, self.num_classes, (self.batch_size,))
            yield inputs, targets

    def __len__(self):
        return self.num_batches


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batch_gd_mock(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test.")

    # Create mock train and test loaders
    train_loader = MockDataLoader(num_batches=2, batch_size=8, T=100, num_classes=3)
    test_loader = MockDataLoader(num_batches=1, batch_size=8, T=100, num_classes=3)

    # Create dummy model, criterion, optimizer
    model = DummyModel(y_len=3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1
    train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs, device=torch.device(device))

    # Check that we got training and test losses for the single epoch
    assert len(train_losses) == epochs, f"Expected {epochs} training loss values, got {len(train_losses)}"
    assert len(test_losses) == epochs, f"Expected {epochs} test loss values, got {len(test_losses)}"

    # Check that the loss arrays are floats
    assert isinstance(train_losses[0], float), "Train loss should be a float."
    assert isinstance(test_losses[0], float), "Test loss should be a float."

