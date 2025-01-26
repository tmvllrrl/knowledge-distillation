import torch
import torch.nn as nn

def kd_softmax(x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    '''
    Calculates softmax over a tensor using a temperature value

    Args:
        x (torch.Tensor): the tensor to compute softmax on
        T (float): the temperature value to soften the softmax probabilities

    Returns:
        torch.Tensor: the computed softmax tensor
    '''
    return torch.exp(x / T) / torch.sum(torch.exp(x / T), dim=0)


# Module wrapper for knowledge distillation softmax
class KnowledgeDistilSoftmax(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = kd_softmax(x, self.T)
        return output


def kd_cross_entropy_loss(x: torch.Tensor, target: torch.Tensor) -> float:
    '''
    Calculates cross entropy for two probability distributions

    Args:
        x (torch.Tensor): the predicted probability distribution
        target (torch.Tensor): the target probability distribution

    Returns
        float: the average (assuming batches in x and target) cross entropy loss 
    '''

    return torch.mean(-torch.sum(target * torch.log(x), 1))


# Module wrapper for soft cross entropy loss
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Computes cross entropy loss between x (predicted) and y (target) probabilities
        return kd_cross_entropy_loss(x, y)