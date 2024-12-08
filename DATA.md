# DATA Description

This document describes how the FI-2010 or similarly structured LOB (Limit Order Book) dataset is processed and prepared for the DeepLOB model.

## Dataset Structure

The dataset is originally provided in a format where:
- The first dimension (rows) contains features plus label rows.
- The second dimension (columns) represents time steps.

For example, you might have a `(features, time)` shaped array:
- The **first 40 rows** represent the top 10 levels of ask/bid prices and volumes (LOB features).
- The **last 5 rows** represent future price movement labels at different prediction horizons.

In total, this might look like `(45, N)` or `(149, N)`, depending on preprocessing. The key assumption is that the top 40 rows are features and the last 5 rows are labels, with `N` time steps.

## Transformations

1. **Extract Features and Labels**:
   - Features: Take the first 40 rows and transpose to get `(time, 40)`.
   - Labels: Take the last 5 rows and transpose to get `(time, 5)`.

   After this step:
   - `X` has shape `(time, 40)`.
   - `Y` has shape `(time, 5)`.

2. **Sequence Construction (Length T)**:
   We construct fixed-length sequences of length `T` from `X`. Each sequence covers `T` consecutive time steps:
   
   - Input sequences: `(N - T + 1, T, 40)`, where `N` is the number of original time steps.
   - Labels are aligned so that the label corresponding to the last time step of each `T`-length sequence is used. Thus, `Y` also becomes `(N - T + 1, 5)`.

3. **Select Label Horizon**:
   Since we have 5 label horizons, we pick one using index `k` (0-based). We then shift the labels from {1, 2, 3} to {0, 1, 2} for zero-based class indices:
   - `y = (y[:, k] - 1)`, resulting in a `(N - T + 1,)` array of integer labels.

4. **Convert to PyTorch Tensors**:
   - `X` is converted to a tensor of shape `(N - T + 1, T, 40)`.
   - A channel dimension is added: `(N - T + 1, 1, T, 40)`.
   - `y` is converted into a `(N - T + 1,)` tensor of integer labels.

## Final Dataset

The final PyTorch `Dataset` provides samples as `(x, y)`:
- `x`: A `(1, T, 40)` float32 tensor (one sequence of input features).
- `y`: A scalar (0, 1, or 2), the class label.

This structure aligns with the DeepLOB model’s expected input format and makes it straightforward to load data into PyTorch `DataLoader`s for training and evaluation.

