import os
import pytest
import torch

from src.dataset import Dataset
from src.utils import load_txt_file

@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data")

def test_train_file_exists(data_dir):
    train_file = os.path.join(data_dir, 'Train_Dst_NoAuction_DecPre_CF_7.txt')
    assert os.path.isfile(train_file), "Train file does not exist."

def test_test_file_exists(data_dir):
    test_file = os.path.join(data_dir, 'Test_Dst_NoAuction_DecPre_CF_7.txt')
    assert os.path.isfile(test_file), "Test file does not exist."

def test_dataset_construction(data_dir):
    train_file = os.path.join(data_dir, 'Train_Dst_NoAuction_DecPre_CF_7.txt')
    data = load_txt_file(train_file)
    # data shape: (features, time), e.g. (149, 203800)

    # Basic checks
    assert data.ndim == 2, f"Expected 2D data, got {data.ndim}D"
    # At least 45 rows: The original FI-2010 format: first 40 for features, last 5 for labels.
    assert data.shape[0] >= 45, f"Expected at least 45 rows (40 features + 5 labels), got {data.shape[0]}"

    T = 100
    k = 4
    num_classes = 3
    # Ensure we have enough time steps for T
    time_steps = data.shape[1]
    if time_steps <= T:
        pytest.skip("Not enough data to form a sequence of length T.")

    dataset = Dataset(data=data, k=k, num_classes=num_classes, T=T)
    expected_length = time_steps - T + 1
    assert len(dataset) == expected_length, f"Expected {expected_length} sequences, got {len(dataset)}"

    x, y = dataset[0]
    # x: (1, T, 40)
    assert x.dim() == 3, "x should be a 3D tensor (1, T, 40)"
    assert x.shape[0] == 1, f"Channel dimension should be 1, got {x.shape[0]}"
    assert x.shape[1] == T, f"Sequence length should be {T}, got {x.shape[1]}"
    assert x.shape[2] == 40, f"Expected 40 features, got {x.shape[2]}"

    # y: scalar
    assert y.dim() == 0, "y should be a scalar label."
    assert 0 <= y.item() < num_classes, f"Label should be in [0, {num_classes-1}], got {y.item()}"

