import torch
import torch.nn as nn
import logging
from typing import List, Type
from knn_cuda import KNN
from ..build import MODELS
from ..layers import create_convblock1d, create_linearblock, create_grouper
from ..layers import trunc_normal_, DropPath, fps, SubsampleGroup
from openpoints.utils.logger import *
from openpoints.utils.ckpt_util import get_missing_parameters_message, get_unexpected_parameters_message

# Mamba3D的GroupFeature用于局部特征增强
class GroupFeature(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # 第一个是点本身
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            输入: 
                xyz: B N 3
                feat: B N C
            ---------------------------
            输出: 
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape # B N 3
        C = feat.shape[-1]

        center = xyz
        # knn获取邻域
        _, idx = self.knn(xyz, xyz) # B N K : 为每个中心点获取K个索引
        assert idx.size(1) == num_points # N个中心点
        assert idx.size(2) == self.group_size # K个knn组
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :] 
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous()
        # 归一化
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, neighborhood_feat

# K_Norm用于局部几何聚合
class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha=1.0, beta=0.0):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        # 获取knn的xyz和特征
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x) # B G K 3, B G K C

        # 归一化x(特征)和xyz(坐标)
        mean_x = lc_x.unsqueeze(dim=-2) # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz) # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5) # B G K 3

        B, G, K, C = knn_x.shape

        # 特征扩展
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1) # B G K 2C

        # 仿射变换
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat 
        
        # 几何特征提取
        knn_x_w = knn_x.permute(0, 3, 1, 2) # B 2C G K

        return knn_x_w

# Mamba3D Block用于替代transformer
class Mamba3DBlock(nn.Module):
    def __init__(self, dim, k_group_size=8, drop_path=0., num_group=128, num_heads=6, bimamba_type="v2"):
        super().__init__()
        self.dim = dim
        self.k_group_size = k_group_size
        self.num_group = num_group
        self.num_heads = num_heads
        
        # 局部特征增强
        self.k_norm = K_Norm(dim, k_group_size=k_group_size, alpha=1.0, beta=0.0)
        
        # 特征变换层
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # 随机深度的dropout路径
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP块
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, center, x):
        # 应用局部特征增强
        B, N, C = x.shape  # B G+1 C
        
        # 跳过cls token进行局部特征增强
        x_no_cls = x[:, 1:, :]  # B G C
        center_xyz = center  # B G 3
        
        # 应用归一化
        x_norm = self.norm1(x)
        x_cls, x_no_cls_norm = x_norm[:, 0:1, :], x_norm[:, 1:, :]
        
        # 如果点数足够，应用局部特征增强
        if N-1 >= self.k_group_size:
            # 获取增强特征
            enhanced_features = self.k_norm(center_xyz, x_no_cls_norm)  # B 2C G K
            
            # 处理增强特征
            enhanced_features = torch.mean(enhanced_features, dim=-1)  # B 2C G
            enhanced_features = enhanced_features.permute(0, 2, 1).contiguous()  # B G 2C
            
            # 应用注意力机制
            attn_out = self.attn(enhanced_features)  # B G C
            
            # 与cls token组合
            attn_out = torch.cat([x_cls, attn_out], dim=1)  # B G+1 C
        else:
            # 点数较少时的回退方案
            attn_out = x_norm
        
        # 应用第一个残差连接
        x = x + self.drop_path(attn_out)
        
        # 应用MLP块和第二个残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

# Mamba3D编码器
class Mamba3DEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6, k_group_size=8, bimamba_type="v2"):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        
        # 创建块
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                dim=embed_dim,
                k_group_size=self.k_group_size,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
            )
            for i in range(depth)
        ])

    def forward(self, center, x, pos=None):
        '''
        输入:
            x: 分块点云并编码, B G+1 C
            pos: 位置编码, B G+1 C (可选)
            center: 中心点, B G 3
        输出:
            x: 经过transformer块处理后的x, B G+1 C
        '''
        # 添加位置编码（如果提供）
        if pos is not None:
            x = x + pos
        
        # 应用块
        for _, block in enumerate(self.blocks):
            x = block(center, x)
            
        return x

# 增强型编码器
class EnhancedEncoder(nn.Module):
    def __init__(self, encoder_channel, k_group_size=8):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.k_group_size = k_group_size
        
        # 第一个卷积层
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        # 第二个卷积层
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        
        # 局部特征增强
        self.k_norm = K_Norm(self.encoder_channel, k_group_size=self.k_group_size, alpha=1.0, beta=0.0)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(self.encoder_channel * 2, self.encoder_channel, 1),
            nn.BatchNorm1d(self.encoder_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        
        # 基本编码器
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG encoder_channel n
        
        # 重塑以进行局部特征增强
        feature_reshaped = feature.transpose(1, 2).contiguous()  # BG n encoder_channel
        
        # 如果点数足够，应用局部特征增强
        if n >= self.k_group_size:
            # 局部特征增强
            enhanced_features = self.k_norm(point_groups, feature_reshaped)  # B 2C G K
            
            # 处理增强特征
            enhanced_features = torch.mean(enhanced_features, dim=-1)  # B 2C G
            enhanced_features = enhanced_features.permute(0, 2, 1).contiguous()  # BG 2C
            
            # 重塑原始特征以进行融合
            feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
            
            # 融合原始和增强特征
            fused_features = self.fusion_layer(
                torch.cat([feature_global.unsqueeze(2), enhanced_features.unsqueeze(2)], dim=1)
            ).squeeze(2)  # BG encoder_channel
            
            return fused_features.reshape(bs, g, self.encoder_channel)
        else:
            # 点数较少时的回退方案
            feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
            return feature_global.reshape(bs, g, self.encoder_channel)

# 从SetAbstraction和PointNextEncoder中提取的关键组件
from ..layers.local_aggregation import LocalAggregation
from ..layers.set_abstraction import SetAbstraction

@MODELS.register_module()
class Mamba3DEncoder(nn.Module):
    """基于Mamba3D的编码器，用于替代PointNextEncoder
    """
    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 k_group_size: int = 8,
                 bimamba_type: str = "v2",
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        self.k_group_size = k_group_size
        self.bimamba_type = bimamba_type
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # 下采样后宽度加倍
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param可以是: radius, nsample
        param_list = []
        if isinstance(param, List):
            # 使param成为完整列表
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius是标量（在这种情况下，只提供初始半径），然后创建一个列表（每个块的半径）
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        
        # 使用SetAbstraction进行下采样和特征提取
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args 
                                     ))
        self.in_channels = channels
        
        # 对于每个块，添加Mamba3D块而不是原始块
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            
            # 使用Mamba3D块替代原始块
            # 注意：这里我们需要适配接口，因为Mamba3D块的输入输出与原始块不同
            # 我们使用一个包装器来适配接口
            layers.append(Mamba3DBlockAdapter(
                channels, 
                k_group_size=self.k_group_size,
                drop_path=0.0,
                group_args=group_args,
                norm_args=self.norm_args,
                act_args=self.act_args
            ))
            
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)

# Mamba3D块适配器，用于适配Mamba3D块到PointNext框架
class Mamba3DBlockAdapter(nn.Module):
    def __init__(self, dim, k_group_size=8, drop_path=0., group_args=None, norm_args=None, act_args=None):
        super().__init__()
        self.dim = dim
        self.k_group_size = k_group_size
        self.group_args = group_args
        
        # 创建Mamba3D块
        self.mamba_block = Mamba3DBlock(
            dim=dim,
            k_group_size=k_group_size,
            drop_path=drop_path
        )
        
        # 创建局部聚合层，用于获取局部特征
        self.local_aggregation = LocalAggregation(
            channels=dim,
            norm_args=norm_args,
            act_args=act_args,
            group_args=group_args
        )

    def forward(self, inputs):
        p, f = inputs
        # 获取局部特征
        f_local = self.local_aggregation([p, f])[1]
        
        # 将特征重塑为Mamba3D块期望的格式
        B, C, N = f.shape
        f_reshaped = f.transpose(1, 2).contiguous()  # B N C
        
        # 应用Mamba3D块
        f_enhanced = self.mamba_block(p, f_reshaped)
        
        # 将特征转换回原始格式
        f_out = f_enhanced.transpose(1, 2).contiguous()  # B C N
        
        return [p, f_out]