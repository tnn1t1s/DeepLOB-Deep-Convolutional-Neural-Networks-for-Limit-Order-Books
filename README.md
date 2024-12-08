# DeepLOB Project

This repository provides code and instructions for training and evaluating a Deep Convolutional Neural Network (DeepLOB) model on Limit Order Book (LOB) data. The DeepLOB architecture is designed for financial time series, specifically top-level LOB features, and aims to predict future price movements.

## Contents

- `src/`: Source code directory  
  - `dataset.py`: Defines a PyTorch `Dataset` class that transforms raw LOB data into `(inputs, targets)` suitable for the DeepLOB model.  
  - `model.py`: Contains the DeepLOB model implementation (convolution + LSTM layers).  
  - `training.py`: Provides the `batch_gd` function to run batch gradient descent training loops.  
  - `utils.py`: Utility functions for data loading and other helper routines.

- `tests/`: Contains test scripts to verify dataset loading, the model’s forward pass, and the training loop.

- `DATA.md`: Details on the data format, how sequences are constructed, and how labeling is performed.

- `README.md`: This file, explaining the project structure and how to get started.

## Getting Started

1. **Install Dependencies**  
   Ensure you have Python and pip installed. Run:
   ```bash
   pip install -r requirements.txt

Also, ensure PyTorch is installed. For GPU support on NVIDIA hardware, follow PyTorch’s instructions to install a CUDA-enabled version.

2. Prepare Data
Place your LOB .txt files into the data/ directory. The dataset should have a (features, time) shape, where the top 40 rows represent LOB features and the last 5 rows represent label horizons.

3,. .Configure the Dataset
In dataset.py, adjust parameters like the sequence length T and the chosen label horizon k. The Dataset class will create (N-T+1, 1, T, 40) input tensors with corresponding integer labels.

4. Training
Example usage:

from src.training import batch_gd
# Assume model, optimizer, criterion, train_loader, val_loader are defined
train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, val_loader, epochs=50)


Evaluation
After training, evaluate the model on a test set. Check tests/test_model.py and tests/test_data_loading.py for example validation steps and to ensure the pipeline works as intended.


