import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    arr = np.array(data, dtype="float64")
    
    if operation == "flatten":
        return arr.flatten()

    if operation == "transpose":
        return arr.T

    if operation == "add_batch":
        return np.expand_dims(arr, axis=0)

    raise ValueError("Not a valid operation")
