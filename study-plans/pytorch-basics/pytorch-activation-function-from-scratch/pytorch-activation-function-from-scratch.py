import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    alpha = 0.01
    x = torch.tensor(x)
    
    if method == "relu":
        result = torch.clamp(x, min=0)
        return result.tolist()

    if method == "sigmoid":
        result = 1 / (1 + torch.pow(torch.e, -x))
        return result.tolist()

    if method == "tanh":
        result = (torch.pow(torch.e, x) - torch.pow(torch.e, -x)) / (torch.pow(torch.e, x) + torch.pow(torch.e, -x))
        
        return result.tolist()

    if method == "leaky_relu":
        result = torch.where(x > 0, x, alpha * x)
        
        return result.tolist()

    raise ValueError("Not a supported method")