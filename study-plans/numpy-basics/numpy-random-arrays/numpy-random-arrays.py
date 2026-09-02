import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.default_rng(seed)
    
    if kind == "normal":
        return rng.standard_normal(size=shape)

    if kind == "uniform":
        return rng.uniform(size=shape)


    raise ValueError("Not a valid kind for the array")
