import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    if len(x) != len(y):
        raise ValueError()
    
    total = 0
    for i in range(len(x)):
        total += (x[i] * y[i])

    return total