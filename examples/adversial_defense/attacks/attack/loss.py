import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyAdvLoss(nn.Module):
    """Adversarial loss using cross entropy"""
    def __init__(self):
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, target):
        """Forward pass
        Args:
            logits: model output before softmax
            target: attack target label
        """
        return -F.cross_entropy(logits, target)

class LogitsAdvLoss(nn.Module):
    """Adversarial loss using logits"""
    def __init__(self, kappa=0.):
        super(LogitsAdvLoss, self).__init__()
        self.kappa = kappa

    def forward(self, logits, target):
        """Forward pass
        Args:
            logits: model output before softmax
            target: attack target label
        """
        target_one_hot = F.one_hot(target, num_classes=logits.shape[-1])
        real = torch.sum(target_one_hot * logits, dim=1)
        other = torch.max((1 - target_one_hot) * logits - target_one_hot * 10000, dim=1)[0]
        loss = torch.clamp(other - real + self.kappa, min=0.)
        return loss 