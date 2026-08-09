import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)
    
    if op == "flatten":
        return x.view(-1)
        # return torch.flatten(x)
    elif op == "squeeze":
        return x.squeeze()

    elif op == "transpose":
        return x.transpose(0,1)

    raise ValueError("not a valid operation")