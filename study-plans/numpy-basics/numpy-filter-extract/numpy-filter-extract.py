import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    """
    Returns matching values in row-major order as a 1D float64 array.
    """
    data = np.array(data, np.float64)
    row_mask = data[row_start:row_stop, :]
    return row_mask[row_mask > threshold]
