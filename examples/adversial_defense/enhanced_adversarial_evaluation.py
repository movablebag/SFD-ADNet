import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
from collections import defaultdict
from easydict import EasyDict 
import json
import time
from sklearn.cluster import DBSCAN
import copy

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# 添加attacks文件夹路径
attacks_path = os.path.join(os.path.dirname(__file__), 'attacks')
sys.path.append(attacks_path)

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.utils import load_checkpoint, setup_logger_dist
from openpoints.transforms import build_transforms_from_cfg

# 删除重复的类定义，直接使用attacks模块中的实现
try:
    from attack.loss import CrossEntropyAdvLoss, LogitsAdvLoss
    from attack.distance import ChamferDist, HausdorffDist, chamfer_distance
    from attack.CW.Perturb import CWPerturb
    from attack.CW.Add import CWAdd, get_critical_points
    from attack.CW.Add_Cluster import CWAddClusters
    from attack.CW.kNN import CWKNN
except ImportError:
    # 如果attacks模块不存在，使用本地实现
    class CrossEntropyAdvLoss(nn.Module):
        """Adversarial loss using cross entropy"""
        def __init__(self):
            super(CrossEntropyAdvLoss, self).__init__()

        def forward(self, logits, target):
            return -F.cross_entropy(logits, target)

    class LogitsAdvLoss(nn.Module):
        """Adversarial loss using logits"""
        def __init__(self, kappa=0.):
            super(LogitsAdvLoss, self).__init__()
            self.kappa = kappa

        def forward(self, logits, target):
            target_one_hot = F.one_hot(target, num_classes=logits.shape[-1])
            real = torch.sum(target_one_hot * logits, dim=1)
            other = torch.max((1 - target_one_hot) * logits - target_one_hot * 10000, dim=1)[0]
            loss = torch.clamp(other - real + self.kappa, min=0.)
            return loss

    def chamfer_distance(x, y, batch_avg=True):
        """Compute chamfer distance between two point clouds"""
        x = x.unsqueeze(2)  # [B, N, 1, 3]
        y = y.unsqueeze(1)  # [B, 1, M, 3]
        
        dist = torch.sum((x - y) ** 2, dim=-1)  # [B, N, M]
        
        dist_xy = torch.min(dist, dim=2)[0]  # [B, N]
        dist_yx = torch.min(dist, dim=1)[0]  # [B, M]
        
        loss = torch.mean(dist_xy, dim=-1) + torch.mean(dist_yx, dim=-1)  # [B]
        
        if batch_avg:
            loss = loss.mean()
        return loss

    class ChamferDist(nn.Module):
        """Chamfer distance"""
        def __init__(self):
            super(ChamferDist, self).__init__()

        def forward(self, adv, ori, batch_avg=True, weights=None):
            if weights is not None:
                # 如果有权重参数，需要特殊处理
                dist = chamfer_distance(adv, ori, batch_avg=False)
                if isinstance(weights, torch.Tensor):
                    dist = dist * weights
                else:
                    dist = dist * torch.from_numpy(weights).to(dist.device)
                return dist.mean() if batch_avg else dist
            return chamfer_distance(adv, ori, batch_avg)

# 裁剪函数
def clip_by_tensor(t, t_min, t_max):
    """Clip tensor values"""
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

# 修复的kNN裁剪函数
def knn_clip(adv_data, ori_data, normal=None, k=5):
    """kNN based clipping function with bounds checking"""
    B, C, N = adv_data.shape
    clipped_data = adv_data.clone()
    
    # 确保k值不超过点数
    k = min(k, N - 1)
    if k <= 0:
        return clipped_data
    
    for b in range(B):
        try:
            # 计算原始点云的kNN
            ori_points = ori_data[b, :3, :].transpose(0, 1)  # [N, 3] 只使用前3个通道
            adv_points = adv_data[b, :3, :].transpose(0, 1)  # [N, 3]
            
            # 计算距离矩阵
            dist_matrix = torch.cdist(ori_points, ori_points)
            _, knn_indices = torch.topk(dist_matrix, k+1, dim=1, largest=False)
            knn_indices = knn_indices[:, 1:]  # 排除自己
            
            # 对每个点进行约束
            for i in range(N):
                if len(knn_indices[i]) > 0:
                    neighbors = ori_points[knn_indices[i]]
                    center = neighbors.mean(dim=0)
                    max_dist = torch.max(torch.norm(neighbors - center, dim=1))
                    
                    # 约束对抗点在邻域范围内
                    adv_point = adv_points[i]
                    dist_to_center = torch.norm(adv_point - center)
                    if dist_to_center > max_dist * 1.5:  # 允许一定的扩展
                        direction = (adv_point - center) / (dist_to_center + 1e-8)
                        new_point = center + direction * max_dist * 1.5
                        clipped_data[b, :3, i] = new_point
        except Exception as e:
            print(f"kNN clipping failed for batch {b}: {e}")
            continue
    
    return clipped_data

# 修复的Add-CD攻击实现
class AddCDAttack:
    """Add-CD Attack - 添加聚类点攻击 (修复版)"""
    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 init_weight=5., max_weight=30., binary_step=3,
                 num_iter=200, num_add=3, cl_num_p=32):
        self.model = model.cuda()
        self.model.eval()
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.num_add = num_add
        self.cl_num_p = cl_num_p
    
    def _get_cluster_points(self, pc, label, num_add):
        """获取聚类点 - 修复索引越界问题"""
        B = pc.shape[0]
        cluster_points = []
        
        for b in range(B):
            try:
                points = pc[b, :3, :].transpose(0, 1)  # [N, 3] 只使用前3个通道
                N = points.shape[0]
                
                # 确保聚类参数合理
                actual_cl_num_p = min(self.cl_num_p, N // 2, 64)  # 限制聚类点数
                if actual_cl_num_p < 3:
                    # 如果点数太少，直接随机选择
                    selected_indices = torch.randperm(N)[:min(num_add, N)]
                    selected_points = points[selected_indices]
                else:
                    # 使用DBSCAN进行聚类
                    points_np = points.detach().cpu().numpy()
                    clustering = DBSCAN(eps=0.1, min_samples=3).fit(points_np)
                    labels = clustering.labels_
                    
                    # 获取最大的聚类
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 1 and -1 in unique_labels:
                        # 移除噪声标签
                        unique_labels = unique_labels[unique_labels != -1]
                    
                    if len(unique_labels) > 0:
                        # 选择最大的聚类
                        cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
                        largest_cluster_label = max(cluster_sizes, key=lambda x: x[1])[0]
                        cluster_indices = np.where(labels == largest_cluster_label)[0]
                        
                        # 安全地选择聚类中的点
                        num_cluster_points = min(actual_cl_num_p, len(cluster_indices))
                        if num_cluster_points > 0:
                            selected_indices = np.random.choice(cluster_indices, num_cluster_points, replace=False)
                            selected_points = points[selected_indices]
                        else:
                            # 如果聚类失败，随机选择
                            selected_indices = torch.randperm(N)[:min(num_add, N)]
                            selected_points = points[selected_indices]
                    else:
                        # 如果没有有效聚类，随机选择
                        selected_indices = torch.randperm(N)[:min(num_add, N)]
                        selected_points = points[selected_indices]
                
                # 生成新的聚类点
                if len(selected_points) >= num_add:
                    cluster_center = selected_points.mean(dim=0)
                    noise = torch.randn(num_add, 3, device=pc.device) * 0.01
                    new_points = cluster_center.unsqueeze(0).repeat(num_add, 1) + noise
                else:
                    # 如果选择的点不够，重复采样
                    repeat_times = (num_add // len(selected_points)) + 1
                    repeated_points = selected_points.repeat(repeat_times, 1)[:num_add]
                    noise = torch.randn(num_add, 3, device=pc.device) * 0.01
                    new_points = repeated_points + noise
                
                cluster_points.append(new_points.transpose(0, 1))  # [3, num_add]
                
            except Exception as e:
                print(f"Cluster generation failed for batch {b}: {e}")
                # 生成随机点作为备选
                random_points = torch.randn(3, num_add, device=pc.device) * 0.1
                cluster_points.append(random_points)
        
        return torch.stack(cluster_points, dim=0)  # [B, 3, num_add]
    
    def attack(self, data, target):
        """执行Add-CD攻击"""
        B, C, N = data.shape
        data = data.float().cuda().detach()
        
        # 确保数据格式正确
        if C not in [3, 4]:
            if data.shape[1] > data.shape[2]:
                data = data.transpose(1, 2).contiguous()
                B, C, N = data.shape
        
        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()
        
        # 权重因子
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight
        
        # 记录最佳结果
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, C, self.num_add))
        
        # 获取聚类点
        cluster_points = self._get_cluster_points(ori_data, target, self.num_add)
        
        # 如果原始数据有4通道，为新点添加第4通道
        if C == 4:
            heights = torch.zeros(B, 1, self.num_add, device=data.device)
            cluster_points = torch.cat([cluster_points, heights], dim=1)
        
        # 二分搜索
        for binary_step in range(self.binary_step):
            adv_data = cluster_points + torch.randn((B, C, self.num_add)).cuda() * 1e-7
            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)
            
            for iteration in range(self.num_iter):
                try:
                    # 前向传播
                    cat_data = torch.cat([ori_data, adv_data], dim=-1)
                    logits = self.model(cat_data)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    # 计算损失
                    adv_loss = self.adv_func(logits, target).mean()
                    adv_pos = adv_data[:, :3, :].transpose(1, 2).contiguous()
                    ori_pos = ori_data[:, :3, :].transpose(1, 2).contiguous()
                    dist_loss = self.dist_func(adv_pos, ori_pos, batch_avg=True) * torch.from_numpy(current_weight).to(adv_data.device).mean()
                    loss = adv_loss + dist_loss
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    # 更新最佳结果
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=-1)
                        dist_val = self.dist_func(adv_pos, ori_pos, batch_avg=False).detach().cpu().numpy()
                        pred_val = pred.detach().cpu().numpy()
                        input_val = adv_data.detach().cpu().numpy()
                        
                        for e, (dist, pred, label, ii) in enumerate(zip(dist_val, pred_val, label_val, input_val)):
                            if dist < bestdist[e] and pred != label:
                                bestdist[e] = dist
                                bestscore[e] = pred
                            if dist < o_bestdist[e] and pred != label:
                                o_bestdist[e] = dist
                                o_bestscore[e] = pred
                                o_bestattack[e] = ii
                                
                except Exception as e:
                    print(f"Add-CD iteration {iteration} failed: {e}")
                    break
            
            # 调整权重因子
            for e, label in enumerate(label_val):
                if bestscore[e] != label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
        
        # 处理失败的样本
        fail_idx = (lower_bound == 0.)
        if np.any(fail_idx):
            o_bestattack[fail_idx] = cluster_points[fail_idx].detach().cpu().numpy()
        
        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))
        
        # 拼接原始点和添加的点
        ori_data = ori_data.detach().cpu().numpy()
        o_bestattack = np.concatenate([ori_data, o_bestattack], axis=-1)
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num

# 修复的kNN攻击实现
class KNNAttack:
    """kNN Attack - 修复版"""
    def __init__(self, model, adv_func, dist_func, clip_func, attack_lr=1e-4, num_iter=1000, k=5):
        self.model = model.cuda()
        self.model.eval()
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.clip_func = clip_func
        self.attack_lr = attack_lr
        self.num_iter = num_iter
        self.k = k
    
    def attack(self, data, target):
        """执行kNN攻击"""
        B, C, N = data.shape
        data = data.float().cuda().detach()
        
        # 确保k值合理
        actual_k = min(self.k, N - 1, 10)  # 限制k值
        if actual_k <= 0:
            print(f"Warning: k value too small ({actual_k}), using original data")
            return data.detach().cpu().numpy().transpose((0, 2, 1)), 0
        
        ori_data = data.clone().detach()
        target = target.long().cuda().detach()
        
        # 初始化对抗样本
        adv_data = data.clone()
        adv_data.requires_grad_()
        
        opt = optim.Adam([adv_data], lr=self.attack_lr)
        
        success_count = 0
        
        for iteration in range(self.num_iter):
            try:
                # 前向传播
                logits = self.model(adv_data)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # 计算损失
                adv_loss = self.adv_func(logits, target).mean()
                
                opt.zero_grad()
                adv_loss.backward()
                opt.step()
                
                # 应用kNN约束
                with torch.no_grad():
                    adv_data.data = self.clip_func(adv_data.data, ori_data, k=actual_k)
                
                # 检查攻击成功率
                if iteration % 100 == 0:
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=-1)
                        success_count = (pred != target).sum().item()
                        
            except Exception as e:
                print(f"kNN attack iteration {iteration} failed: {e}")
                break
        
        return adv_data.detach().cpu().numpy().transpose((0, 2, 1)), success_count

# 修复的Add-HD攻击实现
class AddHDAttack:
    """Add-HD Attack - 添加高密度点攻击 (修复版)"""
    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 init_weight=5e3, max_weight=4e4, binary_step=5,
                 num_iter=300, num_add=100):
        self.model = model.cuda()
        self.model.eval()
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.num_add = num_add
    
    def _get_high_density_points(self, pc, label, num_add):
        """获取高密度区域的点 - 修复通道数问题"""
        B = pc.shape[0]
        C = pc.shape[1]
        high_density_points = []
        
        for b in range(B):
            try:
                # 只使用前3个通道进行密度计算
                points = pc[b, :3, :].transpose(0, 1)  # [N, 3]
                N = points.shape[0]
                
                if N < 16:  # 如果点数太少
                    # 直接复制现有点并添加噪声
                    selected_points = points[:min(num_add, N)]
                    if len(selected_points) < num_add:
                        repeat_times = (num_add // len(selected_points)) + 1
                        selected_points = selected_points.repeat(repeat_times, 1)[:num_add]
                else:
                    # 计算每个点的局部密度
                    distances = torch.cdist(points, points)
                    k = min(16, N - 1)
                    knn_distances, _ = torch.topk(distances, k, dim=1, largest=False)
                    local_density = 1.0 / (knn_distances.mean(dim=1) + 1e-8)
                    
                    # 选择高密度区域的点
                    num_high_density = min(num_add, N)
                    _, high_density_indices = torch.topk(local_density, num_high_density, largest=True)
                    selected_points = points[high_density_indices]
                
                # 在高密度点附近生成新点
                noise = torch.randn(num_add, 3, device=pc.device) * 0.01
                if len(selected_points) < num_add:
                    repeat_times = (num_add // len(selected_points)) + 1
                    repeated_points = selected_points.repeat(repeat_times, 1)[:num_add]
                    new_points = repeated_points + noise
                else:
                    new_points = selected_points[:num_add] + noise
                
                # 构建完整的点（包括所有通道）
                if C == 4:
                    # 为新点添加第4通道（高度信息）
                    heights = new_points[:, 2:3] - new_points[:, 2:3].min(dim=0, keepdim=True)[0]
                    new_points_full = torch.cat([new_points, heights], dim=1)  # [num_add, 4]
                else:
                    new_points_full = new_points  # [num_add, 3]
                
                high_density_points.append(new_points_full.transpose(0, 1))  # [C, num_add]
                
            except Exception as e:
                print(f"High density point generation failed for batch {b}: {e}")
                # 生成随机点作为备选
                if C == 4:
                    random_points = torch.randn(4, num_add, device=pc.device) * 0.1
                else:
                    random_points = torch.randn(3, num_add, device=pc.device) * 0.1
                high_density_points.append(random_points)
        
        return torch.stack(high_density_points, dim=0)  # [B, C, num_add]
    
    def attack(self, data, target):
        """执行Add-HD攻击"""
        B, C, N = data.shape
        data = data.float().cuda().detach()
        
        # 确保数据格式正确
        if C not in [3, 4]:
            if data.shape[1] > data.shape[2]:
                data = data.transpose(1, 2).contiguous()
                B, C, N = data.shape
        
        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()
        
        # 权重因子
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight
        
        # 记录最佳结果
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, C, self.num_add))
        
        # 获取高密度点
        hd_points = self._get_high_density_points(ori_data, target, self.num_add)
        
        # 二分搜索
        for binary_step in range(self.binary_step):
            adv_data = hd_points + torch.randn((B, C, self.num_add)).cuda() * 1e-7
            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)
            
            for iteration in range(self.num_iter):
                try:
                    # 前向传播
                    cat_data = torch.cat([ori_data, adv_data], dim=-1)
                    logits = self.model(cat_data)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    # 计算损失
                    adv_loss = self.adv_func(logits, target).mean()
                    # 只使用前3个通道计算距离
                    adv_pos = adv_data[:, :3, :].transpose(1, 2).contiguous()
                    ori_pos = ori_data[:, :3, :].transpose(1, 2).contiguous()
                    dist_loss = self.dist_func(adv_pos, ori_pos, batch_avg=True) * torch.from_numpy(current_weight).to(adv_data.device).mean()
                    loss = adv_loss + dist_loss
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
                    # 更新最佳结果
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=-1)
                        dist_val = self.dist_func(adv_pos, ori_pos, batch_avg=False).detach().cpu().numpy()
                        pred_val = pred.detach().cpu().numpy()
                        input_val = adv_data.detach().cpu().numpy()
                        
                        for e, (dist, pred, label, ii) in enumerate(zip(dist_val, pred_val, label_val, input_val)):
                            if dist < bestdist[e] and pred != label:
                                bestdist[e] = dist
                                bestscore[e] = pred
                            if dist < o_bestdist[e] and pred != label:
                                o_bestdist[e] = dist
                                o_bestscore[e] = pred
                                o_bestattack[e] = ii
                                
                except Exception as e:
                    print(f"Add-HD iteration {iteration} failed: {e}")
                    break
            
            # 调整权重因子
            for e, label in enumerate(label_val):
                if bestscore[e] != label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
        
        # 处理失败的样本
        fail_idx = (lower_bound == 0.)
        if np.any(fail_idx):
            o_bestattack[fail_idx] = hd_points[fail_idx].detach().cpu().numpy()
        
        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))
        
        # 拼接原始点和添加的点
        ori_data = ori_data.detach().cpu().numpy()
        o_bestattack = np.concatenate([ori_data, o_bestattack], axis=-1)
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num

# Drop攻击实现
class DropAttack:
    """Drop Attack - 丢弃点攻击"""
    def __init__(self, drop_num=100):
        self.drop_num = drop_num
    
    def attack(self, data):
        """执行Drop攻击"""
        if data.dim() == 3 and data.shape[1] in [3, 4]:  # [B, C, N] 格式
            B, C, N = data.shape
            num_keep_points = max(N - self.drop_num, N // 2)  # 至少保留一半的点
            
            attacked_points = torch.zeros(B, C, num_keep_points, device=data.device)
            
            for b in range(B):
                # 随机选择要保留的点
                keep_indices = torch.randperm(N)[:num_keep_points]
                attacked_points[b] = data[b, :, keep_indices]
            
            return attacked_points
        else:
            # 处理 [B, N, C] 格式
            B, N, C = data.shape
            num_keep_points = max(N - self.drop_num, N // 2)
            
            attacked_points = torch.zeros(B, num_keep_points, C, device=data.device)
            
            for b in range(B):
                keep_indices = torch.randperm(N)[:num_keep_points]
                attacked_points[b] = data[b, keep_indices]
            
            # 转换为 [B, C, N] 格式
            return attacked_points.transpose(1, 2).contiguous()

class AdversarialEvaluator:
    """对抗鲁棒性评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 模型性能基准数据
        self.baseline_performance = {
            'DGCNN+AdaptPointMamba': {'clean_acc': 93.5, 'params': '2.1M'},
            'PointNet+AdaptPointMamba': {'clean_acc': 90.1, 'params': '3.8M'},
            'PointNext+AdaptPointMamba': {'clean_acc': 94.8, 'params': '1.7M'}
        }
    
    def load_model(self, config_path, checkpoint_path):
        """加载模型"""
        try:
            with open(config_path, 'r') as f:
                cfg = EasyDict(yaml.safe_load(f))
            
            model = build_model_from_cfg(cfg.model)
            load_checkpoint(model, checkpoint_path)
            model = model.to(self.device)
            model.eval()
            
            return model, cfg
        except Exception as e:
            print(f"Error loading model from {config_path}: {e}")
            return None, None
    
    def _get_data_and_target(self, batch_data):
        """统一的数据提取方法"""
        # 优先使用 'x' 字段（包含坐标+高度信息），否则使用 'pos'
        if 'x' in batch_data:
            data = batch_data['x'].to(self.device)
        else:
            data = batch_data['pos'].to(self.device)
        target = batch_data['y'].to(self.device)
        
        # 统一的维度处理
        if data.dim() == 3:
            if data.shape[1] in [3, 4]:  # [B, C, N] 格式
                pass  # 已经是正确格式
            elif data.shape[2] in [3, 4]:  # [B, N, C] 格式
                data = data.transpose(1, 2).contiguous()  # 转换为 [B, C, N]
            else:
                # 处理特殊情况：如果第二维度很大，可能是点数
                if data.shape[1] > data.shape[2]:
                    data = data.transpose(1, 2).contiguous()
        
        return data, target
    
    def _get_num_classes(self, model, cfg=None):
        """动态获取类别数量"""
        if hasattr(model, 'num_classes'):
            return model.num_classes
        elif cfg and hasattr(cfg, 'model') and hasattr(cfg.model, 'num_classes'):
            return cfg.model.num_classes
        else:
            return 15  # ScanObjectNN默认类别数
    
    def evaluate_robustness(self, model, dataloader, model_name, num_samples=100):
        """评估模型鲁棒性"""
        results = {}
        
        # 初始化攻击方法
        adv_loss = LogitsAdvLoss(kappa=0.)
        dist_func = ChamferDist()
        clip_func = knn_clip
        
        # 获取类别数量
        num_classes = self._get_num_classes(model)
        
        # 使用修复后的攻击实现
        attacks = {
            'Add-CD': AddCDAttack(model, adv_loss, dist_func, 
                                attack_lr=1e-3, init_weight=5., max_weight=30., 
                                binary_step=3, num_iter=200, num_add=3, cl_num_p=32),
            'kNN': KNNAttack(model, adv_loss, dist_func, clip_func, 
                           attack_lr=1e-4, num_iter=500, k=5),
            'Add-HD': AddHDAttack(model, adv_loss, dist_func, 
                                attack_lr=1e-3, init_weight=5e3, max_weight=4e4, 
                                binary_step=3, num_iter=200, num_add=100),
        }
        
        print(f"\n=== 评估 {model_name} 的鲁棒性 ===")
        
        # 首先测试clean accuracy
        print("\n--- Clean Accuracy ---")
        clean_acc = self.test_clean_accuracy(model, dataloader, num_samples)
        results['Clean'] = clean_acc
        print(f"Clean Accuracy: {clean_acc:.3f}")
        
        # 测试各种攻击
        for attack_name, attack in attacks.items():
            print(f"\n--- {attack_name} 攻击 ---")
            
            correct = 0
            total = 0
            sample_count = 0
            
            for i, batch_data in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                try:
                    data, target = self._get_data_and_target(batch_data)
                    
                    # 生成对抗样本
                    if attack_name in ['Add-CD', 'Add-HD']:
                        # 需要目标标签的攻击
                        target_labels = torch.randint(0, num_classes, target.shape, device=self.device)
                        mask = target_labels == target
                        target_labels[mask] = (target_labels[mask] + 1) % num_classes
                        
                        _, adv_data, _ = attack.attack(data, target_labels)
                    elif attack_name == 'kNN':
                        target_labels = torch.randint(0, num_classes, target.shape, device=self.device)
                        mask = target_labels == target
                        target_labels[mask] = (target_labels[mask] + 1) % num_classes
                        
                        adv_data, _ = attack.attack(data, target_labels)
                    
                    # 转换为tensor
                    if isinstance(adv_data, np.ndarray):
                        adv_data = torch.from_numpy(adv_data).float().to(self.device)
                    
                    # 调整输入格式
                    if adv_data.dim() == 3:
                        if adv_data.shape[1] in [3, 4]:  # [B, C, N]
                            pass
                        else:  # [B, N, C]
                            adv_data = adv_data.transpose(1, 2).contiguous()
                    
                    # 确保对抗样本有正确的通道数
                    if adv_data.shape[1] != data.shape[1]:
                        if adv_data.shape[1] == 3 and data.shape[1] == 4:
                            heights = adv_data[:, 2:3, :] - adv_data[:, 2:3, :].min(dim=2, keepdim=True)[0]
                            adv_data = torch.cat([adv_data, heights], dim=1)
                        elif adv_data.shape[1] == 4 and data.shape[1] == 3:
                            adv_data = adv_data[:, :3, :]
                    
                    # 模型预测
                    with torch.no_grad():
                        logits = model(adv_data)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        
                        pred = torch.argmax(logits, dim=1)
                        correct += (pred == target).sum().item()
                        total += target.size(0)
                        sample_count += target.size(0)
                        
                except RuntimeError as e:
                    print(f"攻击 {attack_name} 在批次 {i} 失败 (RuntimeError): {e}")
                    continue
                except Exception as e:
                    print(f"攻击 {attack_name} 在批次 {i} 失败 (其他错误): {e}")
                    continue
                finally:
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            accuracy = correct / total if total > 0 else 0
            results[attack_name] = accuracy
            print(f"{attack_name} 准确率: {accuracy:.3f}")
        
        return results
    
    def test_clean_accuracy(self, model, dataloader, num_samples=100):
        """测试clean accuracy"""
        model.eval()
        correct = 0
        total = 0
        sample_count = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                if sample_count >= num_samples:
                    break
                
                try:
                    data, target = self._get_data_and_target(batch_data)
                    
                    logits = model(data)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    sample_count += target.size(0)
                except Exception as e:
                    print(f"Clean accuracy测试批次失败: {e}")
                    continue
        
        return correct / total if total > 0 else 0
    
    def print_results_table(self, all_results):
        """打印结果表格"""
        print("\n" + "="*120)
        print("AdaptPointMamba模型鲁棒性评估结果")
        print("="*120)
        
        # 打印基准性能
        print("\n基准性能:")
        print("-" * 60)
        for model_name, perf in self.baseline_performance.items():
            print(f"{model_name:<30} Clean: {perf['clean_acc']:.1f}% ({perf['params']})")
        
        # 打印鲁棒性结果
        if all_results:
            print("\n鲁棒性测试结果:")
            print("-" * 120)
            
            # 表头
            header = "Model".ljust(30)
            for attack in ['Clean', 'Perturb', 'Add', 'Add-CD', 'Add-HD', 'kNN', 'Drop-100', 'Drop-200']:
                header += f"{attack}".ljust(12)
            print(header)
            print("-" * 150)
            
            # 数据行
            for model_name, results in all_results.items():
                row = model_name.ljust(30)
                for attack in ['Clean', 'Perturb', 'Add', 'Add-CD', 'Add-HD', 'kNN', 'Drop-100', 'Drop-200']:
                    if attack in results:
                        acc = results[attack]
                        row += f"{acc:.3f}".ljust(12)
                    else:
                        row += "N/A".ljust(12)
                print(row)
        
        print("\n" + "="*120)

def main():
    parser = argparse.ArgumentParser(description='AdaptPointMamba模型对抗鲁棒性评估')
    parser.add_argument('--dgcnn_config', type=str, 
                       default='cfgs/scanobjectnn/dgcnn_adaptpoint_generator_component_12.yaml',
                       help='DGCNN+AdaptPointMamba配置文件路径')
    parser.add_argument('--dgcnn_checkpoint', type=str, required=True,
                       help='DGCNN+AdaptPointMamba权重文件路径')
    parser.add_argument('--pointnet_config', type=str,
                       default='cfgs/scanobjectnn/pointnet2_adaptpoint_generator_component_12.yaml', 
                       help='PointNet+AdaptPointMamba配置文件路径')
    parser.add_argument('--pointnet_checkpoint', type=str, required=True,
                       help='PointNet+AdaptPointMamba权重文件路径')
    parser.add_argument('--pointnext_config', type=str,
                       default='cfgs/scanobjectnn/mamba3d_adaptpoint_generator_component_12.yaml',
                       help='PointNext+AdaptPointMamba配置文件路径')
    parser.add_argument('--pointnext_checkpoint', type=str, required=True,
                       help='PointNext+AdaptPointMamba权重文件路径')
    parser.add_argument('--data_path', type=str, default='/media/shangli211/4TB_SSD/program_file/Data/ScanObjectNN/h5_files/main_split',
                       help='scanobjectnn数据集路径')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='测试样本数量')
    parser.add_argument('--show_performance_only', action='store_true',
                       help='仅显示性能对比，不运行攻击测试')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = AdversarialEvaluator()
    
    if args.show_performance_only:
        # 仅显示性能对比
        evaluator.print_results_table({})
        return
    
    # 加载数据
    print("加载数据集...")
    try:
        # 使用第一个配置文件来构建数据加载器
        with open(args.dgcnn_config, 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))
        
        # 确保配置对象的正确性
        if not hasattr(cfg, 'dataset'):
            cfg.dataset = EasyDict()
        if not hasattr(cfg.dataset, 'common'):
            cfg.dataset.common = EasyDict()
        if not hasattr(cfg, 'dataloader'):
            cfg.dataloader = EasyDict()
        if not hasattr(cfg.dataloader, 'test'):
            cfg.dataloader.test = EasyDict()
        
        # 设置数据集配置
        cfg.dataset.common.data_root = args.data_path
        cfg.dataloader.test.batch_size = args.batch_size
        cfg.dataloader.test.num_workers = args.num_workers
        
        # 正确的调用方式：第一个参数是batch_size
        test_loader = build_dataloader_from_cfg(
            batch_size=args.batch_size,
            dataset_cfg=cfg.dataset,
            dataloader_cfg=cfg.dataloader.test,
            datatransforms_cfg=cfg.get('datatransforms', None),
            split='test',
            distributed=False  # 单机测试，不使用分布式
        )
        print(f"数据集加载完成，测试样本数: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 模型配置
    model_configs = [
        (args.dgcnn_config, args.dgcnn_checkpoint, 'DGCNN+AdaptPointMamba'),
        (args.pointnet_config, args.pointnet_checkpoint, 'PointNet+AdaptPointMamba'),
        (args.pointnext_config, args.pointnext_checkpoint, 'PointNext+AdaptPointMamba')
    ]
    
    all_results = {}
    
    # 评估每个模型
    for config_path, checkpoint_path, model_name in model_configs:
        print(f"\n加载模型: {model_name}")
        model, cfg = evaluator.load_model(config_path, checkpoint_path)
        
        if model is None:
            print(f"跳过 {model_name}")
            continue
        
        # 评估鲁棒性
        results = evaluator.evaluate_robustness(model, test_loader, model_name, args.num_samples)
        all_results[model_name] = results
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
    
    # 打印最终结果
    evaluator.print_results_table(all_results)
    
    # 保存结果
    output_file = 'adaptpoint_mamba_robustness_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()