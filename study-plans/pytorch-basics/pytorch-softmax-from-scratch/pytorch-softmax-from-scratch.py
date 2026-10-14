import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    m = torch.max(logits, dim=1, keepdim=True).values
    num = torch.exp(logits - m)
    den = torch.sum(torch.exp(logits - m), dim=1, keepdim=True)

    return num/den
