#!/usr/bin/env python
# 新增：Mamba3D特征提取器和自适应频域增强权重预测网络
# 修改：将SAComponent替换为Mamba3DFeatureExtractor
# 修改：采用generator_component4_15 copy.py中的参数预测和掩码生成方式

'''
    pred probs for PointWOLF with Mamba3D, no offset         ---wangjie
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.layers import furthest_point_sample, three_nn, three_interpolate
from ..models.layers.group import ball_query
from .build import ADAPTMODELS
from ..online_aug.frequency_enhance import SpectralAugmentor
from ..online_aug.spectral_wavelet_transform_update import GraphWaveletTransform

# 导入Mamba模块（需要确保已安装mamba_ssm库）
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not found. Please install it for Mamba3D functionality.")
    # 提供一个简单的替代实现
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            return self.linear(x)

# 添加低频处理模块
class LowFreqProcessor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 低频特征增强网络
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 低频特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(feature_dim + in_channels, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(feature_dim, in_channels, 1),
            nn.BatchNorm1d(in_channels)
        )
    
    def forward(self, low_freq):
        """处理低频特征
        输入:
            low_freq: [B, N, 3] - 低频点云
        输出:
            enhanced_low_freq: [B, N, 3] - 增强的低频特征
            low_freq_features: [B, N, feature_dim] - 低频特征向量
        """
        B, N, C = low_freq.shape
        
        # 提取特征
        x = low_freq.transpose(1, 2)  # [B, 3, N]
        features = self.feature_extractor(x)  # [B, feature_dim, N]
        
        # 融合特征
        combined = torch.cat([x, features], dim=1)  # [B, 3+feature_dim, N]
        enhanced = self.feature_fusion(combined)  # [B, 3, N]
        
        # 返回增强的低频和特征
        return enhanced.transpose(1, 2), features.transpose(1, 2)

# 添加局部几何聚合模块
class LocalGeometryAggregation(nn.Module):
    """局部几何聚合模块，借鉴Mamba3D的K_Norm设计"""
    def __init__(self, in_dim, out_dim, k_neighbors=8):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )
        
        # 几何编码层
        self.geo_encoder = nn.Sequential(
            nn.Conv1d(3, out_dim // 2, 1),
            nn.BatchNorm1d(out_dim // 2),
            nn.SiLU(),
            nn.Conv1d(out_dim // 2, out_dim, 1)
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )
        
        # 可学习的仿射参数
        self.affine_alpha = nn.Parameter(torch.ones(1, 1, out_dim))
        self.affine_beta = nn.Parameter(torch.zeros(1, 1, out_dim))
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: [B, N, 3] 点坐标
            features: [B, N, C] 点特征
        Returns:
            enhanced_features: [B, N, out_dim] 增强后的特征
        """
        B, N, _ = xyz.shape
        
        # 使用KNN找到邻居
        knn_idx = knn_point(self.k_neighbors, xyz, xyz)  # [B, N, k]
        
        # 获取邻居的坐标和特征
        neighbor_xyz = index_points(xyz, knn_idx)  # [B, N, k, 3]
        neighbor_feat = index_points(features, knn_idx)  # [B, N, k, C]
        
        # 计算相对位置
        relative_pos = neighbor_xyz - xyz.unsqueeze(2)  # [B, N, k, 3]
        
        # 几何编码
        geo_encoding = self.geo_encoder(relative_pos.permute(0, 3, 1, 2).reshape(B, 3, -1))
        geo_encoding = geo_encoding.reshape(B, self.out_dim, N, self.k_neighbors).permute(0, 2, 3, 1)
        
        # 特征变换
        transformed_feat = self.feature_transform(neighbor_feat)  # [B, N, k, out_dim]
        
        # 特征融合
        combined_feat = torch.cat([transformed_feat, geo_encoding], dim=-1)  # [B, N, k, 2*out_dim]
        fused_feat = self.fusion(combined_feat)  # [B, N, k, out_dim]
        
        # 仿射变换
        fused_feat = self.affine_alpha * fused_feat + self.affine_beta
        
        # 聚合（使用加权平均）
        weights = F.softmax(fused_feat.sum(dim=-1), dim=-1)  # [B, N, k]
        aggregated_feat = (weights.unsqueeze(-1) * fused_feat).sum(dim=2)  # [B, N, out_dim]
        
        return aggregated_feat

# 改进的Mamba块
class EnhancedMambaBlock(nn.Module):
    """增强的Mamba块，支持双向处理和更好的残差连接"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bimamba_type="v2"):
        super().__init__()
        self.d_model = d_model
        self.bimamba_type = bimamba_type
        
        # 双向Mamba
        if bimamba_type == "v2":
            self.mamba_forward = Mamba(d_model, d_state, d_conv, expand)
            self.mamba_backward = Mamba(d_model, d_state, d_conv, expand)
            self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            self.mamba = Mamba(d_model, d_state, d_conv, expand)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, d_model]
        Returns:
            output: [B, N, d_model]
        """
        # Mamba处理
        residual = x
        x = self.norm1(x)
        
        if self.bimamba_type == "v2":
            # 双向处理
            x_forward = self.mamba_forward(x)
            x_backward = self.mamba_backward(torch.flip(x, dims=[1]))
            x_backward = torch.flip(x_backward, dims=[1])
            x = self.fusion(torch.cat([x_forward, x_backward], dim=-1))
        else:
            x = self.mamba(x)
        
        x = x + residual
        
        # MLP处理
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x

# 升级后的Mamba3DFeatureExtractor（采用第二种参数预测和掩码生成方式）
class Mamba3DFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_channel=6, 
                 embed_dim=64, 
                 depth=4, 
                 num_anchors=4, 
                 k_neighbors=8,
                 num_groups=64,
                 group_size=32,
                 bimamba_type="v2",
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_groups = num_groups
        self.group_size = group_size
        self.k_neighbors = k_neighbors
        
        # 点云分组模块（借鉴Mamba3D的Group设计）
        if self.num_groups > 0:
            self.group_divider = Group(num_group=num_groups, group_size=group_size)
        
        # 局部编码器（借鉴Mamba3D的Encoder设计）
        self.local_encoder = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, embed_dim, 1)
        )
        
        # 全局特征嵌入
        self.global_embedding = nn.Linear(in_channel, embed_dim)
        
        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, embed_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 局部几何聚合模块
        self.lga_layers = nn.ModuleList([
            LocalGeometryAggregation(embed_dim, embed_dim, k_neighbors)
            for _ in range(depth // 2)
        ])
        
        # 增强的Mamba块
        self.mamba_blocks = nn.ModuleList([
            EnhancedMambaBlock(
                d_model=embed_dim,
                bimamba_type=bimamba_type
            ) for _ in range(depth)
        ])
        
        # 层归一化
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(depth)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 参数预测头（采用第二种方式：Conv1d + BatchNorm）
        self.global_layer = nn.Sequential(
            nn.Conv1d(3, embed_dim, 1, bias=False),
            nn.BatchNorm1d(embed_dim)
        )
        
        self.prob_head = nn.Sequential(
            nn.Conv1d(embed_dim * 2, 3 * 3, 1, bias=False),
            nn.BatchNorm1d(3 * 3)
        )
        
        self.anchor_selfattention = Anchor_selfattention(dim=embed_dim, head_num=4)
        
        # 掩码生成网络（采用第二种方式：局部+全局特征融合）
        self.localfeat_mask_selfattention = Anchor_selfattention(dim=embed_dim, head_num=4)
        self.extract_local_feat_masking = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm1d(3),
        )
        self.extract_global_feat_masking = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm1d(3),
        )
        self.fuse_masking = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=2, kernel_size=1, bias=False),
            nn.BatchNorm1d(2),
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, a_index):
        """
        Args:
            x: [B, N, C] 输入点云
            a_index: [B, M] 锚点索引
        Returns:
            prob: [B, M, 9] 变换参数
            masking: [B, N, 2] 掩码
        """
        B, N, C = x.shape
        xyz = x[:, :, :3]  # 提取坐标
        
        # 获取锚点
        a_points = index_points(x, a_index)  # [B, M, 3]
        
        # 初始化centers变量
        centers = None
        
        # 点云分组和局部编码
        if self.num_groups > 0:
            try:
                neighborhood, centers = self.group_divider(xyz)  # [B, G, K, 3], [B, G, 3]
                # 局部特征编码
                local_features = []
                for i in range(self.num_groups):
                    group_points = neighborhood[:, i]  # [B, K, 3]
                    group_feat = self.local_encoder(group_points.transpose(1, 2))  # [B, embed_dim, K]
                    group_feat = torch.max(group_feat, dim=2)[0]  # [B, embed_dim]
                    local_features.append(group_feat)
                local_features = torch.stack(local_features, dim=1)  # [B, G, embed_dim]
                
                # 位置编码
                pos_encoding = self.pos_encoder(centers)  # [B, G, embed_dim]
                
                # 添加CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                cls_pos = self.cls_pos.expand(B, -1, -1)
                
                # 拼接特征
                features = torch.cat([cls_tokens, local_features], dim=1)  # [B, G+1, embed_dim]
                pos_features = torch.cat([cls_pos, pos_encoding], dim=1)  # [B, G+1, embed_dim]
                
            except Exception as e:
                print(f"Warning: Group division failed: {e}")
                # 如果分组失败，使用全局处理
                centers = None
                features = self.global_embedding(x)  # [B, N, embed_dim]
                pos_features = self.pos_encoder(xyz)  # [B, N, embed_dim]
                
                # 添加CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                cls_pos = self.cls_pos.expand(B, -1, -1)
                features = torch.cat([cls_tokens, features], dim=1)
                pos_features = torch.cat([cls_pos, pos_features], dim=1)
        else:
            # 全局处理
            features = self.global_embedding(x)  # [B, N, embed_dim]
            pos_features = self.pos_encoder(xyz)  # [B, N, embed_dim]
            
            # 添加CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_pos = self.cls_pos.expand(B, -1, -1)
            features = torch.cat([cls_tokens, features], dim=1)
            pos_features = torch.cat([cls_pos, pos_features], dim=1)
        
        # 特征处理流水线
        x_feat = features + pos_features
        
        # 交替使用LGA和Mamba块
        for i, (mamba_block, norm_layer) in enumerate(zip(self.mamba_blocks, self.norm_layers)):
            # 残差连接
            residual = x_feat
            
            # 层归一化
            x_feat = norm_layer(x_feat)
            
            # Mamba处理
            x_feat = mamba_block(x_feat)
            
            # 残差连接
            x_feat = x_feat + residual
            
            # 每隔一层应用LGA
            if i < len(self.lga_layers) and i % 2 == 0:
                # 只对非CLS token应用LGA
                if x_feat.shape[1] > 1:
                    cls_feat = x_feat[:, :1]  # CLS token
                    point_feat = x_feat[:, 1:]  # 点特征 Fm
                    
                    # 需要坐标信息进行LGA
                    if centers is not None:
                        enhanced_feat = self.lga_layers[i//2](centers, point_feat)
                    else:
                        # 使用原始坐标
                        if point_feat.shape[1] == xyz.shape[1]:
                            enhanced_feat = self.lga_layers[i//2](xyz, point_feat)
                        else:
                            enhanced_feat = point_feat  # 跳过LGA
                    
                    x_feat = torch.cat([cls_feat, enhanced_feat], dim=1) #f'
        
        # 最终归一化
        x_feat = self.final_norm(x_feat)
        
        # 提取锚点特征
        if x_feat.shape[1] > 1:
            # 如果有分组，需要映射到原始点
            if centers is not None:
                # 使用最近邻插值将分组特征映射到原始点
                point_features = x_feat[:, 1:]  # [B, G, embed_dim] #f
                # 简单的最近邻映射
                expanded_features = point_features.repeat_interleave(N // self.num_groups + 1, dim=1)[:, :N]
            else:
                expanded_features = x_feat[:, 1:]  # [B, N, embed_dim]
        else:
            # 如果只有CLS token，复制到所有点
            expanded_features = x_feat.repeat(1, N, 1) #f'
        
        # 获取锚点特征
        anchor_features = index_points(expanded_features, a_index)  # [B, M, embed_dim]
        
        # 参数预测（采用第二种方式）
        _, num_anchor, _ = a_points.shape
        
        # 自注意力增强锚点特征
        local_feat_res = self.anchor_selfattention(x=anchor_features, xyz=a_points)
        local_feat = anchor_features + local_feat_res
        
        # 提取全局特征
        global_feat = self.global_layer(a_points.permute(0, 2, 1)).permute(0, 2, 1)  # [B, M, embed_dim]
        global_feat = torch.max(global_feat, dim=1, keepdim=True)[0]  # [B, 1, embed_dim]
        
        # 拼接局部和全局特征
        feat = torch.cat([local_feat, global_feat.repeat(1, num_anchor, 1)], dim=-1)  # [B, M, 2*embed_dim]
        prob = self.prob_head(feat.permute(0, 2, 1)).permute(0, 2, 1)  # [B, M, 9]
        
        # 掩码生成（采用第二种方式）
        # 局部特征处理
        mask_localfeat = self.localfeat_mask_selfattention(x=expanded_features, xyz=xyz)
        mask_localfeat = mask_localfeat + expanded_features
        masking_local = self.extract_local_feat_masking(mask_localfeat.permute(0, 2, 1))  # [B, 3, N]
        
        # 全局特征处理
        global_feat_for_mask = x_feat[:, 0:1].permute(0, 2, 1)  # [B, embed_dim, 1] CLS token
        masking_global = self.extract_global_feat_masking(global_feat_for_mask)  # [B, 3, 1]
        masking_global = masking_global.repeat(1, 1, N)  # [B, 3, N]
        
        # 融合局部和全局掩码特征 
        masking = torch.cat([masking_local, masking_global], dim=1)  # [B, 6, N]
        masking = self.fuse_masking(masking).permute(0, 2, 1)  # [B, N, 2]
        masking = F.gumbel_softmax(masking, tau=0.1, hard=True, eps=1e-10, dim=-1)  # [B, N, 2]
        
        return prob, masking

# 改进的Group类（如果原始导入失败）
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
    
    def forward(self, xyz):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            neighborhood: [B, G, M, 3]
            center: [B, G, 3]
        """
        B, N, _ = xyz.shape
        
        # 使用FPS采样中心点
        try:
            center_idx = furthest_point_sample(xyz.contiguous(), self.num_group).long()
            centers = index_points(xyz, center_idx)  # [B, G, 3]
        except:
            # 如果FPS失败，使用随机采样
            center_idx = torch.randperm(N)[:self.num_group].unsqueeze(0).expand(B, -1).to(xyz.device)
            centers = index_points(xyz, center_idx)
        
        # 使用KNN找邻居
        knn_idx = knn_point(self.group_size, xyz, centers)  # [B, G, M]
        neighborhood = index_points(xyz, knn_idx)  # [B, G, M, 3]
        
        # 归一化：相对距离
        neighborhood = neighborhood - centers.unsqueeze(2)
        
        return neighborhood, centers

# 修改后的Producefactor模块
class Producefactor(nn.Module):
    def __init__(self, kneighbors, out_channels):
        super().__init__()
        self.keighbors = kneighbors
        self.out_channels = out_channels

        # 全局特征处理层
        self.global_layer = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        # 变换参数预测头
        self.prob_head = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 9),  # 9个变换参数
            nn.Tanh()  # 限制输出范围
        )

        self.anchor_selfattention = Anchor_selfattention(dim=out_channels, head_num=4)

    def forward(self, a_points, a_feat, global_feat):
        """
        Input:
            a_points: [B, M, 3] - 锚点坐标
            a_feat: [B, M, C] - 锚点特征
            global_feat: [B, C] - 全局特征
        Output:
            prob: [B, M, 9] - 变换参数概率
        """
        B, M, C = a_feat.shape
        
        # 自注意力增强锚点特征
        local_feat_res = self.anchor_selfattention(x=a_feat, xyz=a_points)
        local_feat = a_feat + local_feat_res
        
        # 处理全局特征
        global_feat_processed = self.global_layer(global_feat)  # [B, C]
        global_feat_expanded = global_feat_processed.unsqueeze(1).expand(-1, M, -1)  # [B, M, C]
        
        # 拼接局部和全局特征
        feat = torch.cat([local_feat, global_feat_expanded], dim=-1)  # [B, M, 2C]
        
        # 预测变换参数
        prob = self.prob_head(feat)  # [B, M, 9]
        
        return prob

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class Anchor_selfattention(nn.Module):
    def __init__(self, dim, head_num=4):
        super().__init__()
        self.head_num = head_num
        self.dim = dim
        self.head_dim = dim // head_num
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, xyz=None):
        B, N, C = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

@ADAPTMODELS.register_module()
class PointWOLF_Mamba3D(nn.Module):
    def __init__(self, 
                 num_anchors=4, 
                 out_channels=64, 
                 kneighbors=512, 
                 use_spectral_norm=False,
                 embed_dim=64,
                 depth=4,
                 num_groups=64,
                 group_size=32,
                 bimamba_type="v2",
                 **kwargs):
        super().__init__()
        self.num_anchors = num_anchors
        self.out_channels = out_channels
        self.kneighbors = kneighbors
        
        # 使用Mamba3D特征提取器替代原来的SAComponent
        self.sa_component = Mamba3DFeatureExtractor(
            in_channel=6,
            embed_dim=embed_dim,
            depth=depth,
            num_anchors=num_anchors,
            num_groups=num_groups,
            group_size=group_size,
            bimamba_type=bimamba_type
        )
        
        注释掉频域增强模块
        self.spectral_augmentor = SpectralAugmentor(
            num_points=1024,
            num_anchors=num_anchors,
            embed_dim=embed_dim
        )
        
        # 图小波变换模块
        self.graph_wavelet = GraphWaveletTransform(
            num_points=1024,
            num_scales=3,
            embed_dim=embed_dim
        )
        
        # 低频处理模块
        self.low_freq_processor = LowFreqProcessor(
            in_channels=3,
            feature_dim=32
        )
        
        # 自适应权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),  # 3个频段的权重
            nn.Softmax(dim=-1)
        )
        
        # 最终特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, N, 6] 输入点云（坐标+法向量）
        Returns:
            prob: [B, M, 9] 变换参数
            masking: [B, N, 2] 掩码
        """
        B, N, C = x.shape
        xyz = x[:, :, :3]
        
        # 使用FPS采样锚点
        anchor_idx = farthest_point_sample(xyz, self.num_anchors)  # [B, M]
        
        # 图小波变换分解
        low_freq, high_freq = self.graph_wavelet(xyz)
        
        # 低频特征处理
        enhanced_low_freq, low_freq_features = self.low_freq_processor(low_freq)
        
        # 移除频域增强步骤
        # enhanced_features = self.spectral_augmentor(x, anchor_idx)
        
        # 直接使用Mamba3D特征提取器
        prob, masking = self.sa_component(x, anchor_idx)
        
        # 简化的权重预测（仅使用低频特征）
        anchor_low_freq_feat = index_points(low_freq_features, anchor_idx)  # [B, M, 32]
        # 使用原始特征替代增强特征
        anchor_original_feat = index_points(x[:, :, :self.sa_component.embed_dim] if x.shape[-1] >= self.sa_component.embed_dim else 
                                          F.pad(x, (0, self.sa_component.embed_dim - x.shape[-1])), anchor_idx)  # [B, M, embed_dim]
        
        combined_feat = torch.cat([anchor_original_feat, anchor_low_freq_feat], dim=-1)
        adaptive_weights = self.weight_predictor(combined_feat)  # [B, M, 3]
        
        # 应用自适应权重到变换参数
        prob = prob * adaptive_weights.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(B, self.num_anchors, 9)
        
        return prob, masking

# ... existing code ...

# 添加工具函数
def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class ConvBN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(ConvBN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)

# 添加Self_attention类
class Self_attention(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = int(self.dim // self.head_num)
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)

        self.pos_embedding = nn.Sequential(nn.Conv1d(3, self.dim, 1),
                                           nn.BatchNorm1d(self.dim)
                                           )
        self.res = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1),
                                 nn.BatchNorm1d(self.dim))

    def forward(self, x, xyz=None):
        '''
        Input:
            x: [B, M, C]
            xyz: [B, M, 3]
        Output:
            v: [B, M, C]
        '''
        B, M, C = x.shape

        gravity_center = torch.mean(xyz, dim=1, keepdim=True)       #   [B, 1, 3]
        relative_xyz = xyz - gravity_center                         #   [B, M, 3]
        rxyz_embedding = self.pos_embedding(relative_xyz.permute(0, 2, 1)).permute(0, 2, 1)     #   [B, M, C]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)                   #   q,k,v:  [B, M, C]
        q = q + rxyz_embedding
        k = k + rxyz_embedding
        v = v + rxyz_embedding
        q = q.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        q = q.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        k = k.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        k = k.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        v = v.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        v = v.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        attn = q @ k.transpose(-2, -1)                              #   [B, head_num, M, M]
        attn /= self.head_dim ** 0.5                                #   [B, head_num, M, M]
        attn = attn.softmax(dim=-1)                                 #   [B, head_num, M, M]
        v = attn @ v                                                #   [B, head_num, M, C']
        v = v.permute(0, 2, 1, 3)                                   #   [B, M, head_num, C']
        v = v.reshape(B, M, -1)                                     #   [B, M, C]
        v = self.res(v.permute(0, 2, 1)).permute(0, 2, 1)           #   [B, M, C]

        return v

@ADAPTMODELS.register_module()
class AdaptPoint_Augmentor(nn.Module):
    """AdaptPoint增强模块，基于Mamba3D特征提取器和频域增强"""
    
    def __init__(self, 
                 spectral_chunk_size=1,  # 减小chunk_size
                 spectral_k=4,  # 减小k值
                 w_num_anchor=4, w_sigma=0.5, w_R_range=10, w_S_range=3, w_T_range=0.25,
                 use_spectral=True, 
                 spectral_scales=2,  # 减小scales
                 alpha=0.8,  # 增大alpha，减少频域增强的影响
                 adaptive_spectral=False,  # 默认设为False，避免初始化问题
                 mix_ratio=0.5,  # 添加混合比例参数
                 wavelet_levels=3,  # 小波变换层级
                 low_freq_weight=0.7,  # 低频引导权重
                 low_freq_feature_dim=32):  # 低频特征维度
        super().__init__()
        self.num_anchor = w_num_anchor
        self.sigma = w_sigma
        self.R_range = (-abs(w_R_range), abs(w_R_range))
        self.S_range = (1., w_S_range)
        self.T_range = (-abs(w_T_range), abs(w_T_range))
        self.w_R_range = w_R_range
        self.w_S_range = w_S_range
        self.w_T_range = w_T_range
        self.alpha = alpha
        self.spectral_k = spectral_k
        self.predict_prob_layer = Mamba3DFeatureExtractor(
            in_channel=6,
            embed_dim=64,
            depth=4,
            num_anchors=w_num_anchor
        )
        self.mix_ratio = mix_ratio  # 混合比例
        self.low_freq_weight = low_freq_weight  # 低频引导权重
        
        # 频域增强相关参数
        self.use_spectral = use_spectral
        self.adaptive_spectral = adaptive_spectral
        self.wavelet_levels = wavelet_levels
        
        if self.use_spectral:
            # 使用图小波变换替代原来的频谱增强器
            self.wavelet_transform = GraphWaveletTransform(
                num_scales=spectral_scales,
                k_neighbors=spectral_k,
                wavelet_levels=wavelet_levels
            )
            
            # 添加低频处理模块
            self.low_freq_processor = LowFreqProcessor(
                in_channels=3,
                feature_dim=low_freq_feature_dim
            )
            
            # 低频特征融合层 - 用于变换参数预测
            self.low_freq_fusion = nn.Sequential(
                nn.Linear(low_freq_feature_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 9),  # 9个变换参数
                nn.Tanh()  # 限制输出范围
            )

    def forward(self, xyz):
        """前向传播函数
        输入:
            xyz: [B, N, 3] - 输入点云
        输出:
            xyz: [B, N, 3] - 原始点云
            xyz_new: [B, N, 3] - 变换后的点云
        """
        M = self.num_anchor
        B, N, _ = xyz.shape
        fps_idx = furthest_point_sample(xyz.contiguous(), self.num_anchor).long()  # [B, M]
        xyz_anchor = self.index_points(xyz, fps_idx)                            #   [B, M, 3]

        xyz_repeat = xyz.unsqueeze(dim=1).repeat(1, self.num_anchor, 1, 1)      #   [B, M, N, 3]

        # 移动到标准空间
        xyz_normalize = xyz_repeat - xyz_anchor.unsqueeze(dim=-2)               #   [B, M, N, 3]

        # 获取变换参数的概率和掩码
        probs, masking = self.predict_prob_layer(xyz, fps_idx)     # probs: [B, M, 9]   masking: [B, N, 2]

        # 提取低频特征并创建低频数据集
        if self.use_spectral:
            # 提取原始点云的低频特征
            enhanced_xyz, xyz_low_freq = self.wavelet_transform(xyz)
            
            # 将wavelet_coeffs转换为正确形状 [B, wavelet_levels, N, C] -> [B, N, C]
            # 方法1: 取第一个层级（低频分量）
            xyz_low_freq_processed = xyz_low_freq[:, 0, :, :]  # [B, N, C]
            
            # 处理低频特征
            enhanced_low_freq, low_freq_features = self.low_freq_processor(xyz_low_freq_processed)
            
            # 使用低频特征指导变换参数
            low_freq_anchor_features = self.index_points(low_freq_features, fps_idx)  # [B, M, feature_dim]
            low_freq_transform_params = self.low_freq_fusion(low_freq_anchor_features)  # [B, M, 9]
            
            # 混合原始预测和低频引导的变换参数
            mixed_probs = (1 - self.low_freq_weight) * probs + self.low_freq_weight * low_freq_transform_params
            
            # 对标准化点云也提取低频特征
            xyz_normalize_reshaped = xyz_normalize.view(B * M, N, 3)
            enhanced_xyz_normalize, xyz_normalize_low_freq = self.wavelet_transform(xyz_normalize_reshaped)
            
            # 同样处理标准化点云的低频特征
            xyz_normalize_low_freq_processed = xyz_normalize_low_freq[:, 0, :, :]  # [B*M, N, C]
            
            # 将低频特征与原始特征混合
            xyz_mixed = torch.cat([xyz_normalize_reshaped, xyz_normalize_low_freq_processed], dim=0)
            
            # 重塑回原始维度
            xyz_normalize = xyz_mixed.view(B*2, M, N, 3)  # 现在批次大小翻倍

            # 局部变换 - 使用混合的变换参数
            xyz_transformed = self.local_transformaton(xyz_normalize, mixed_probs.repeat(2, 1, 1))  # (B*2,M,N,3)

            # 移回原始空间
            xyz_anchor_expanded = xyz_anchor.repeat(2, 1, 1)  # [B*2, M, 3]
            xyz_transformed = xyz_transformed + xyz_anchor_expanded.reshape(B*2, M, 1, 3)  # (B*2,M,N,3)
            
            # 分离原始和低频变换结果
            xyz_transformed_orig = xyz_transformed[:B]  # (B,M,N,3)
            xyz_transformed_low = xyz_transformed[B:]   # (B,M,N,3)
            
            # 对原始和低频结果分别应用核回归
            xyz_new_orig = self.kernel_regression(xyz, xyz_anchor, xyz_transformed_orig)  # [B, N, 3]
            xyz_new_low = self.kernel_regression(xyz, xyz_anchor, xyz_transformed_low)    # [B, N, 3]
            
            # 混合原始和低频结果
            xyz_new = self.mix_ratio * xyz_new_orig + (1 - self.mix_ratio) * xyz_new_low
        else:
            # 原始处理逻辑
            xyz_transformed = self.local_transformaton(xyz_normalize, probs)  # (B,M,N,3)
            xyz_transformed = xyz_transformed + xyz_anchor.reshape(B, M, 1, 3)  # (B,M,N,3)
            xyz_new = self.kernel_regression(xyz, xyz_anchor, xyz_transformed)  # [B, N, 3]
        
        # 标准化和掩码应用
        xyz_new = self.normalize(xyz_new)
        xyz_new = xyz_new * masking[:,:,0].unsqueeze(dim=-1)

        return xyz, xyz_new

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([B,N,3])
            pos_anchor([B,M,3])
            pos_transformed([B,M,N,3])

        output :
            pos_new([B,N,3]) : Pointcloud after weighted local transformation
        """
        B, M, N, _ = pos_transformed.shape

        # Distance between anchor points & entire points
        sub = pos_anchor.unsqueeze(dim=-2).repeat(1, 1, N, 1) - \
              pos.unsqueeze(dim=1).repeat(1, M, 1, 1)  # (B, M, N, 3)

        project_axis = self.get_random_axis(B, 1).to(pos.device)   #   [B, 1, 3]

        projection = project_axis.unsqueeze(dim=-2) * torch.eye(3).to(pos.device)   #   [B, 1, 3, 3]

        # Project distance
        sub = sub @ projection  # (B, M, N, 3)
        sub = torch.sqrt(((sub) ** 2).sum(dim=-1))  # (B, M, N)

        # Kernel regression
        weight = torch.exp(-0.5 * (sub ** 2) / (self.sigma ** 2)).to(pos.device)  # (B, M, N)

        pos_new = (weight.unsqueeze(dim=-1).repeat(1, 1, 1, 3) * pos_transformed).sum(dim=1)  # (B, N, 3)
        pos_new = (pos_new / weight.sum(dim=1, keepdims=True).transpose(1,2).contiguous())  # normalize by weight   [B, N, 3]

        return pos_new

    def local_transformaton(self, pos_normalize, prob):
        """
        input :
            pos_normalize([B,M,N,3])

        output :
            pos_normalize([B,M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        B, M, N, _ = pos_normalize.shape
        a = torch.Tensor(B, M, 3).uniform_(0, 1)
        transformation_dropout = torch.bernoulli(a).to(pos_normalize.device)     #   [B, M, 3]
        transformation_axis = self.get_random_axis(B, M).to(pos_normalize.device)  # [B, M, 3]

        prob_R = prob[:,:,0:3]                        #   [B, M, 3]
        prob_R = torch.tanh(prob_R)                       #   [B, M, 3]
        prob_R = prob_R * self.w_R_range              #   [B, M, 3]
        degree = torch.tensor(math.pi).to(pos_normalize.device) * prob_R.to(pos_normalize.device) / 180.0 \
                 * transformation_dropout[:, :, 0:1]    #   [B, M, 3], sampling from (-R_range, R_range)

        prob_S = prob[:, :, 3:6]                      #  [B, M, 3]
        prob_S = torch.sigmoid(prob_S)                    #  [B, M, 3]
        prob_S = prob_S * (self.w_S_range-1) + 1      #  [B, M, 3]
        scale = prob_S.to(pos_normalize.device) * transformation_dropout[:, :, 1:2]  # [B, M, 3], sampling from (1, S_range)

        scale = scale * transformation_axis
        scale = scale + 1 * (scale == 0)  # Scaling factor must be larger than 1

        prob_T = prob[:, :, 6:9]                    #   [B, M, 3]
        prob_T = torch.tanh(prob_T)                     #   [B, M, 3]
        prob_T = prob_T * self.w_T_range            #   [B, M, 3]
        trl = prob_T.to(pos_normalize.device) * transformation_dropout[:, :, 2:3]    # [B, M, 3], sampling from (1, S_range)

        trl *= transformation_axis

        # Scaling Matrix
        S = scale.unsqueeze(dim=-2) * torch.eye(3).to(pos_normalize.device)  # scailing factor to diagonal matrix (M,3) -> (M,3,3)

        # Rotation Matrix
        sin = torch.sin(degree).unsqueeze(dim=-1)
        cos = torch.cos(degree).unsqueeze(dim=-1)
        sx, sy, sz = sin[:, :, 0], sin[:, :, 1], sin[:, :, 2]
        cx, cy, cz = cos[:, :, 0], cos[:, :, 1], cos[:, :, 2]

        R = torch.cat([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                      sz * cy, sz * sy * sx + cz * cy, sz * sy * cx - cz * sx,
                      -sy, cy * sx, cy * cx], dim=-1).reshape(B, M, 3, 3)
        pos_normalize = pos_normalize @ R @ S + trl.reshape(B, M, 1, 3)

        return pos_normalize

    def get_random_axis(self, batch, n_axis):
        """
        input :
            batch(int)
            n_axis(int)

        output :
            axis([batch, n_axis,3]) : projection axis
        """
        axis = torch.randint(1, 8, (batch, n_axis))  # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz
        m = 3
        axis = (((axis[:, :, None] & (1 << torch.arange(m)))) > 0).int()
        return axis

    def normalize(self, pos):
        """
        input :
            pos([B, N, 3])

        output :
            pos([B, N, 3]) : normalized Pointcloud
        """
        B, N, C = pos.shape
        pos = pos - pos.mean(axis=-2, keepdims=True)    #   [B, N, 3]
        scale = (1 / torch.sqrt((pos ** 2).sum(dim=-1)).max(dim=-1)[0]) * 0.999999   #   [B, 1]
        pos = scale.reshape(B, 1, 1).repeat(1, N, C) * pos
        return pos

        return prob, masking