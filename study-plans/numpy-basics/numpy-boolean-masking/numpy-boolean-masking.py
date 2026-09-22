import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    pass
    data = np.array(data, dtype=float)

    layer_1 = np.where(data > threshold, 1.0, 0.0)

    layer_2 = data.copy()
    condition_2 = np.any(layer_2 > threshold, axis=1)
    layer_2[~condition_2] = 0

    layer_3 = data.copy()
    condition_3 = np.all(layer_3 > threshold, axis=1)
    layer_3[~condition_3] = 0

    return np.array([layer_1, layer_2, layer_3])
