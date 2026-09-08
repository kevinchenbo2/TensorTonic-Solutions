import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    n = X.shape[0]
    mu = torch.sum(X, dim=0) / n
    var = torch.sum(torch.pow(X - mu, 2), dim=0) / n
    X = (X - mu)/torch.sqrt(var + eps)
    Y = gamma * X + beta
    return Y
