import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    n = len(pred)
    pred = torch.tensor(pred, dtype=torch.float)
    if method == "mse":
        target = torch.tensor(target, dtype=torch.float)
        loss = torch.sum(torch.pow(pred - target, 2)) / n
        # loss = torch.sum(torch.pow(pred - target, 2))
        
        return float(loss)

    if method == "cross_entropy":
        max_logit = torch.max(pred[torch.arange(pred.shape[0])])
        log_sum = torch.log(torch.sum(torch.exp(pred - max_logit), dim = 1))
        correct_class = pred[torch.arange(pred.shape[0]), target]
    
        loss = torch.sum(max_logit + log_sum - correct_class) / n
        
        return float(loss)

    if method == "huber":
        target = torch.tensor(target, dtype=torch.float)
        a = torch.abs(pred - target)
        loss = torch.sum(torch.where(a <= delta, (a**2/2), delta*(a - delta/2)))/n
        
        return float(loss)

    raise ValueError("Not a valid method")
