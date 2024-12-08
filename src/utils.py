import numpy as np

def load_txt_file(path):
    """
    Loads a text file containing FI-2010 LOB data (or a similarly structured dataset).
    The data is expected to have shape (features+labels, time), for example (149, 203800),
    where the first dimension contains features and label rows, and the second dimension
    is the time dimension.

    In the original DeepLOB code and FI-2010 data:
    - The first 40 rows are LOB features.
    - The last 5 rows are label info.
    - Additional rows may be present if the dataset has been preprocessed further.

    We do not reshape or transpose here. We just load the data as is.
    Downstream code (prepare_x, get_label in dataset.py) will handle slicing and transposing.
    """

    data = np.loadtxt(path)
    # Just return as loaded. The code that uses this data expects (features, time).
    # No checks for columns = 45 are needed since the paper's snippet shows more than 45 rows.
    # It's possible that their provided dataset includes precomputed features/transformations.

    return data

