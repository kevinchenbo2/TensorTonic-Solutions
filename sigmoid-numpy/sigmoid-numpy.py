import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here

    def _eval(val):
        return 1/(1 + (np.exp(-val)))

    if isinstance(x, (int, float, np.integer, np.floating)):
        return _eval(x)
    
    ans = []
    for val in x:
        if isinstance(val, (int, float, np.integer, np.floating)):
            ans.append(_eval(val))
            
        else:
            temp = []
            for num in val:
                temp.append(_eval(num))
            ans.append(temp)
            
    return np.array(ans)