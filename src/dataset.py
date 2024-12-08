import numpy as np
import torch
from torch.utils import data

def prepare_x(data):
    """
    Extract the first 40 rows (features) and transpose:
    data: (features, time)
    returns: (time, 40)
    """
    # first 40 rows are features
    df1 = data[:40, :].T  # shape: (time, 40)
    return np.array(df1)

def get_label(data):
    """
    Extract the last 5 rows (labels) and transpose:
    data: (features, time)
    returns: (time, 5)
    """
    lob = data[-5:, :].T  # shape: (time, 5)
    return lob

def data_classification(X, Y, T):
    """
    Build sequences of length T from X and Y.

    X: (time, D)
    Y: (time, 5)
    T: int, sequence length

    Returns:
      dataX: (time - T + 1, T, D)
      dataY: (time - T + 1, 5)
    """
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization""" 
        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]

