import pytest
import torch
import numpy as np

# Assuming these functions come from the original code snippet:
# They should be defined similarly to what the user provided or imported from your source.
def prepare_x(data):
    # Example: data is (features=40, time=N), we transpose
    df1 = data[:40, :].T
    return np.array(df1)  # shape (N, 40)

def get_label(data):
    # Example: last 5 rows correspond to label info
    lob = data[-5:, :].T
    return lob  # shape (N, 5) if original label dimension is 5

def data_classification(X, Y, T):
    # Given (N, D) for X and (N, C) for Y, produce sequences of length T
    N, D = X.shape
    dataY = Y[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX, dataY

class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)   # (N, D) after transpose
        y = get_label(data)   # (N, 5) for example
        x, y = data_classification(x, y, self.T)  # x: (N-T+1, T, D), y: (N-T+1, 5)
        y = y[:, self.k] - 1  # select column k from y and shift
        self.length = len(x)

        x = torch.from_numpy(x)       # (N-T+1, T, D)
        self.x = torch.unsqueeze(x, 1)  # (N-T+1, 1, T, D)
        self.y = torch.from_numpy(y)    # (N-T+1,)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


@pytest.fixture
def dummy_data():
    """
    Create a dummy data array similar to what the original code expects.
    Suppose we have D=40 features and 5 label rows, total 45 rows and N columns (time steps).
    data shape is (45, N).
    """
    N = 200  # number of time steps
    D = 40   # features
    # top 40 rows are features
    features = np.random.randn(D, N)
    # last 5 rows as labels
    labels = np.random.randint(1,4,size=(5, N))  # e.g., values in {1,2,3}
    data = np.vstack([features, labels])  # shape (45, N)
    return data


def test_dataset_shapes(dummy_data):
    T = 20
    k = 2
    num_classes = 3
    dataset = Dataset(data=dummy_data, k=k, num_classes=num_classes, T=T)

    length = len(dataset)
    # Expected length = N - T + 1
    N = dummy_data.shape[1]
    expected_length = N - T + 1
    assert length == expected_length, f"Expected length {expected_length}, got {length}"

    x, y = dataset[0]
    # x shape: (1, T, D) from the code, since unsqueeze is along dimension 1
    # Our x: (N-T+1, 1, T, D)
    # After indexing a single sample: (1, T, D)
    assert x.shape[0] == 1, f"Expected first dim = 1, got {x.shape[0]}"
    assert x.shape[1] == T, f"Expected sequence length = {T}, got {x.shape[1]}"
    D = 40
    assert x.shape[2] == D, f"Expected D = {D}, got {x.shape[2]}"

    # y shape: scalar
    assert y.dim() == 0, "y should be a scalar tensor."
    assert y.item() in range(num_classes), f"y should be in [0, {num_classes-1}]"


def test_dataset_label_range(dummy_data):
    T = 30
    k = 1
    num_classes = 3
    dataset = Dataset(data=dummy_data, k=k, num_classes=num_classes, T=T)

    # Check that all labels are in [0, num_classes-1]
    all_labels = dataset.y
    assert all_labels.min().item() >= 0, "Labels should be >= 0"
    assert all_labels.max().item() < num_classes, f"Labels should be < {num_classes}"

def test_dataset_indexing(dummy_data):
    T = 10
    k = 0
    num_classes = 3
    dataset = Dataset(data=dummy_data, k=k, num_classes=num_classes, T=T)

    # Fetch two consecutive samples just to ensure indexing works as expected
    x0, y0 = dataset[0]
    x1, y1 = dataset[1]

    # Check shapes again to ensure consistency
    assert x0.shape == x1.shape, "Shapes of consecutive samples should match."
    # Instead of checking label difference, just verify no errors and shapes:
    assert y0.shape == y1.shape, "Labels should be scalar tensors with identical shape."
    # Remove the assertion that they differ, since it's not guaranteed.

def test_dataset_various_T(dummy_data):
    # Test with multiple T values
    for T in [5, 10, 50]:
        k = 0
        num_classes = 3
        dataset = Dataset(data=dummy_data, k=k, num_classes=num_classes, T=T)
        N = dummy_data.shape[1]
        expected_length = N - T + 1
        assert len(dataset) == expected_length, f"For T={T}, expected length {expected_length}, got {len(dataset)}"
        x, y = dataset[0]
        assert x.shape[1] == T, f"Sequence length should be {T} for T={T}."

