import torch
import torch.nn as nn

def chamfer_distance(x, y, batch_avg=True, weights=None):
    """Compute chamfer distance between two point clouds
    Args:
        x: point cloud 1, [B, N, 3]
        y: point cloud 2, [B, M, 3]
        batch_avg: whether to average over batch
        weights: weights for weighted chamfer distance
    """
    x = x.unsqueeze(2)  # [B, N, 1, 3]
    y = y.unsqueeze(1)  # [B, 1, M, 3]
    
    dist = torch.sum((x - y) ** 2, dim=-1)  # [B, N, M]
    
    dist_xy = torch.min(dist, dim=2)[0]  # [B, N]
    dist_yx = torch.min(dist, dim=1)[0]  # [B, M]
    
    if weights is not None:
        dist_xy = weights.view(-1, 1) * dist_xy
        dist_yx = weights.view(-1, 1) * dist_yx
    
    loss = torch.mean(dist_xy, dim=-1) + torch.mean(dist_yx, dim=-1)  # [B]
    
    if batch_avg:
        loss = loss.mean()
    return loss

def hausdorff_distance(x, y, batch_avg=True, weights=None):
    """Compute Hausdorff distance between two point clouds
    Args:
        x: point cloud 1, [B, N, 3]
        y: point cloud 2, [B, M, 3]
        batch_avg: whether to average over batch
        weights: weights for weighted Hausdorff distance
    """
    x = x.unsqueeze(2)  # [B, N, 1, 3]
    y = y.unsqueeze(1)  # [B, 1, M, 3]
    
    dist = torch.sum((x - y) ** 2, dim=-1)  # [B, N, M]
    
    dist_xy = torch.min(dist, dim=2)[0]  # [B, N]
    dist_yx = torch.min(dist, dim=1)[0]  # [B, M]
    
    if weights is not None:
        dist_xy = weights.view(-1, 1) * dist_xy
        dist_yx = weights.view(-1, 1) * dist_yx
    
    loss = torch.max(torch.max(dist_xy, dim=1)[0], torch.max(dist_yx, dim=1)[0])  # [B]
    
    if batch_avg:
        loss = loss.mean()
    return loss

class ChamferDist(nn.Module):
    """Chamfer distance"""
    def __init__(self, method='adv2ori'):
        super(ChamferDist, self).__init__()
        self.method = method

    def forward(self, adv, ori, batch_avg=True, weights=None):
        """Forward pass
        Args:
            adv: adversarial point cloud, [B, N, 3]
            ori: original point cloud, [B, M, 3]
            batch_avg: whether to average over batch
            weights: weights for weighted chamfer distance
        """
        return chamfer_distance(adv, ori, batch_avg, weights)

class HausdorffDist(nn.Module):
    """Hausdorff distance"""
    def __init__(self, method='adv2ori'):
        super(HausdorffDist, self).__init__()
        self.method = method

    def forward(self, adv, ori, batch_avg=True, weights=None):
        """Forward pass
        Args:
            adv: adversarial point cloud, [B, N, 3]
            ori: original point cloud, [B, M, 3]
            batch_avg: whether to average over batch
            weights: weights for weighted Hausdorff distance
        """
        return hausdorff_distance(adv, ori, batch_avg, weights)

class L2Dist(nn.Module):
    """L2 distance"""
    def __init__(self, method='adv2ori'):
        super(L2Dist, self).__init__()
        self.method = method

    def forward(self, adv, ori, batch_avg=True, weights=None):
        """Forward pass
        Args:
            adv: adversarial point cloud, [B, N, 3]
            ori: original point cloud, [B, M, 3]
            batch_avg: whether to average over batch
            weights: weights for weighted L2 distance
        """
        # 计算对应点之间的L2距离
        dist = torch.sum((adv - ori) ** 2, dim=-1)  # [B, N]
        
        if weights is not None:
            dist = weights.view(-1, 1) * dist
        
        loss = torch.mean(dist, dim=-1)  # [B]
        
        if batch_avg:
            loss = loss.mean()
        return loss