import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """

        if self.training:
            if self.p == 1:
                return torch.zeros(x.shape)
            
            mask = (torch.rand_like(x) > self.p).float()
            
            return x * mask / (1 - self.p)

        return x