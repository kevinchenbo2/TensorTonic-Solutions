import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    if op == "add":
        res = torch.add(tx, ty)
    elif op == "multiply":
        res = torch.mul(tx, ty)
    elif op == "matmul":
        res = torch.matmul(tx, ty)
    elif op == "power":
        # res = torch.pow(tx, ty)
        res = tx ** ty
    elif op == "max":
        res = torch.max(tx, ty)
    else:
        raise ValueError("Not a valid method")

    return res.tolist()