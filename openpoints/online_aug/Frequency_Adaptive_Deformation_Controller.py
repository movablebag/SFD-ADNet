import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from scipy import sparse
# from sklearn.neighbors import kneighbors_graph
#06 08 4

class AdvancedGraphWaveletTransform(nn.Module):
    """高级图谱小波变换模块，集成最新研究成果提升点云分类鲁棒性"""
    
    def __init__(self, num_scales=4, k_neighbors=16, wavelet_levels=3, 
                 adaptive_threshold=True, spectral_norm=True, 
                 multi_hop=True, max_hops=3, sparse_ratio=0.5):
        super().__init__()
        self.num_scales = num_scales
        self.k_neighbors = k_neighbors
        self.wavelet_levels = wavelet_levels
        self.adaptive_threshold = adaptive_threshold
        self.multi_hop = multi_hop
        self.max_hops = max_hops
        self.sparse_ratio = sparse_ratio
        
        # 多跳邻域聚合模块
        if multi_hop:
            self.multi_hop_aggregator = MultiHopAggregator(
                input_dim=3, hidden_dim=64, max_hops=max_hops
            )
        
        # 自适应稀疏图卷积
        self.sparse_graph_conv = AdaptiveSparseGraphConv(
            in_channels=3, out_channels=64, k_neighbors=k_neighbors,
            sparse_ratio=sparse_ratio
        )
        
        # 动态图结构学习
        self.graph_structure_learner = DynamicGraphStructureLearner(
            feature_dim=64, k_neighbors=k_neighbors
        )
        
        # 多尺度特征融合（升级版）
        self.enhanced_fusion = EnhancedMultiScaleFusion(
            input_dim=3 * wavelet_levels, 
            hidden_dims=[128, 64, 32],
            output_dim=3,
            spectral_norm=spectral_norm
        )
        
        # 低频特征增强层（带残差连接和注意力）
        self.low_freq_enhancer = ResidualSpectralEnhancer(
            in_channels=3, hidden_channels=64, 
            spectral_norm=spectral_norm
        )
        
        # 频域对抗训练模块
        self.adversarial_trainer = SpectralAdversarialTrainer(
            feature_dim=64, epsilon=0.1
        )
        
        # 自适应高频抑制参数（升级版）
        # 在 __init__ 方法中
        if self.adaptive_threshold:
            self.adaptive_suppressor = AdaptiveFrequencySupressor(
                input_dim=3, 
                wavelet_levels=self.wavelet_levels,  # 使用类的 wavelet_levels 参数
                context_aware=True,
                context_dim=64
            )
        else:
            self.high_freq_suppressor = nn.Parameter(
                torch.ones(wavelet_levels) * 0.8
            )
        
        # 可学习小波核函数（多尺度）
        self.wavelet_kernels = nn.ParameterList([
            nn.Parameter(torch.linspace(0.1, 1.0, wavelet_levels)) 
            for _ in range(num_scales)
        ])
        
        # 图结构正则化层（升级版）
        self.graph_regularizer = EnhancedGraphRegularizer(
            k_neighbors=k_neighbors, feature_dim=64
        )
        
        # 频域注意力机制（多头注意力）
        # 修复：embed_dim应该是wavelet_levels*C，这里C=3（xyz坐标）
        self.spectral_attention = MultiHeadSpectralAttention(
            embed_dim=wavelet_levels * 3, num_heads=3  # 3是xyz坐标的维度
        )
        
        # 鲁棒性损失计算
        self.robustness_loss = RobustnessLoss()
        
    def build_adaptive_graph_laplacian(self, xyz, features=None):
        """构建自适应图拉普拉斯矩阵（结合几何和语义信息）"""
        B, N, _ = xyz.shape
        
        if features is not None:
            # 结合几何距离和特征相似度
            geo_dist = torch.cdist(xyz, xyz)  # [B, N, N]
            feat_sim = torch.bmm(features, features.transpose(1, 2))  # [B, N, N]
            
            # 自适应权重融合
            alpha = torch.sigmoid(self.graph_structure_learner(features))
            combined_dist = alpha * geo_dist + (1 - alpha) * (1 - feat_sim)
        else:
            combined_dist = torch.cdist(xyz, xyz)
        
        # 构建自适应KNN图
        _, topk_indices = torch.topk(
            combined_dist, self.k_neighbors, dim=-1, largest=False
        )
        
        # 稀疏化处理
        if self.sparse_ratio < 1.0:
            sparse_k = int(self.k_neighbors * self.sparse_ratio)
            topk_indices = topk_indices[:, :, :sparse_k]
        
        # 构建邻接矩阵
        adj_matrix = torch.zeros_like(combined_dist)
        batch_indices = torch.arange(B).view(B, 1, 1).repeat(
            1, N, topk_indices.size(-1)
        )
        row_indices = torch.arange(N).view(1, N, 1).repeat(
            B, 1, topk_indices.size(-1)
        )
        adj_matrix[batch_indices, row_indices, topk_indices] = 1
        
        # 对称化
        adj_matrix = torch.min(
            adj_matrix + adj_matrix.transpose(1, 2), 
            torch.ones_like(adj_matrix)
        )
        
        # 构建归一化拉普拉斯矩阵
        degree = torch.sum(adj_matrix, dim=-1)
        degree_inv_sqrt = torch.diag_embed(
            torch.pow(degree + 1e-5, -0.5)
        )
        laplacian = torch.eye(N).unsqueeze(0).repeat(B, 1, 1).to(xyz.device)
        laplacian -= torch.matmul(
            torch.matmul(degree_inv_sqrt, adj_matrix), degree_inv_sqrt
        )
        
        return laplacian, adj_matrix
    
    def enhanced_chebyshev_polynomials(self, laplacian, order=4):
        """增强的切比雪夫多项式近似（支持高阶和自适应阶数）"""
        B, N, _ = laplacian.shape
        
        # 自适应确定多项式阶数
        lambda_max = torch.max(torch.linalg.eigvalsh(laplacian), dim=1)[0]
        adaptive_order = torch.clamp(
            (lambda_max * order).int(), min=2, max=order
        )
        
        # 预处理拉普拉斯矩阵
        scaled_laplacian = (
            2.0 / lambda_max.view(B, 1, 1)
        ) * laplacian - torch.eye(N).unsqueeze(0).to(laplacian.device)
        
        # 初始化切比雪夫多项式序列
        T_k = []
        T_0 = torch.eye(N).unsqueeze(0).repeat(B, 1, 1).to(laplacian.device)
        T_k.append(T_0)
        
        if order >= 1:
            T_1 = scaled_laplacian
            T_k.append(T_1)
        
        # 递推计算高阶切比雪夫多项式
        for k in range(2, order + 1):
            T_k.append(
                2 * torch.matmul(scaled_laplacian, T_k[-1]) - T_k[-2]
            )
        
        return T_k
    
    def multi_scale_wavelet_transform(self, xyz, laplacian):
        """多尺度图小波变换（支持不同尺度的小波核）"""
        B, N, C = xyz.shape
        
        # 计算增强的切比雪夫多项式基
        T_k = self.enhanced_chebyshev_polynomials(laplacian, order=4)
        
        all_scale_coeffs = []
        
        # 多尺度小波变换
        for scale in range(self.num_scales):
            scale_coeffs = []
            
            for level in range(self.wavelet_levels):
                # 尺度相关的小波核函数
                t = self.wavelet_kernels[scale][level]
                g = torch.exp(
                    -t * torch.arange(len(T_k)).float().to(xyz.device)
                )
                
                # 计算小波系数
                coeff = torch.zeros_like(xyz)
                for k, T in enumerate(T_k):
                    coeff += g[k] * torch.matmul(T, xyz)
                
                scale_coeffs.append(coeff)
            
            all_scale_coeffs.append(torch.stack(scale_coeffs, dim=1))
        
        # 融合不同尺度的系数
        fused_coeffs = torch.stack(all_scale_coeffs, dim=1).mean(dim=1)
        
        return fused_coeffs
    
    def forward(self, xyz, training=True):
        """前向传播"""
        B, N, C = xyz.shape
        
        # 多跳邻域特征聚合
        if self.multi_hop:
            multi_hop_features = self.multi_hop_aggregator(xyz)
        else:
            multi_hop_features = xyz
        
        # 稀疏图卷积特征提取
        graph_features = self.sparse_graph_conv(xyz)
        
        # 构建自适应图拉普拉斯矩阵
        laplacian, adj_matrix = self.build_adaptive_graph_laplacian(
            xyz, graph_features
        )
        
        # 多尺度图小波变换
        wavelet_coeffs = self.multi_scale_wavelet_transform(xyz, laplacian)
        
        # 分离频率成分
        low_freq = wavelet_coeffs[:, 0, :, :]
        mid_freq = wavelet_coeffs[:, 1, :, :] if self.wavelet_levels > 1 else low_freq
        high_freq = wavelet_coeffs[:, 2:, :, :].mean(dim=1) if self.wavelet_levels > 2 else mid_freq
        
        # 多头频域注意力
        # 确保wavelet_coeffs的形状正确
        if len(wavelet_coeffs.shape) == 5:  # [B, num_scales, wavelet_levels, N, C]
            # 如果有多个尺度，需要先处理维度
            B, num_scales, wavelet_levels, N, C = wavelet_coeffs.shape
            # 重塑为 [B, wavelet_levels, N, C]
            wavelet_coeffs = wavelet_coeffs.mean(dim=1)  # 平均多个尺度
        elif len(wavelet_coeffs.shape) == 4:  # [B, wavelet_levels, N, C]
            B, wavelet_levels, N, C = wavelet_coeffs.shape
        else:
            raise ValueError(f"Unexpected wavelet_coeffs shape: {wavelet_coeffs.shape}")
        
        # 现在可以安全地进行view操作 - 修复view操作
        attention_input = wavelet_coeffs.permute(0, 2, 1, 3).contiguous()  # [B, N, wavelet_levels, C]
        attention_input = attention_input.reshape(B, N, -1)  # [B, N, wavelet_levels*C]
        
        attention_weights = self.spectral_attention(attention_input)  # [B, N, wavelet_levels*C]
        attention_weights = attention_weights.reshape(B, N, wavelet_levels, C)  # [B, N, wavelet_levels, C]
        attention_weights = attention_weights.permute(0, 2, 1, 3)  # [B, wavelet_levels, N, C]
        
        # 增强低频成分（带残差连接）
        enhanced_low_freq = self.low_freq_enhancer(low_freq)
        
        # 自适应高频抑制
        if self.adaptive_threshold:
            suppress_weights_raw = self.adaptive_suppressor(
                xyz, context=graph_features
            )  # [B, N, wavelet_levels]
            # 重塑为与attention_weights匹配的形状 [B, wavelet_levels, N, 1]
            suppress_weights = suppress_weights_raw.permute(0, 2, 1).unsqueeze(-1)  # [B, wavelet_levels, N, 1]
        else:
            # 确保形状一致
            suppress_weights_raw = self.high_freq_suppressor.unsqueeze(0).unsqueeze(0)  # [1, 1, wavelet_levels]
            suppress_weights = suppress_weights_raw.permute(0, 2, 1).unsqueeze(-1).expand(B, -1, N, -1)  # [B, wavelet_levels, N, 1]
        
        # 应用频域注意力和抑制
        suppressed_high_freq = high_freq * (
            suppress_weights[:, -1:, :, :] * attention_weights[:, -1:, :, :]
        ).squeeze(1)  # 移除wavelet_levels维度
        
        suppressed_mid_freq = mid_freq * (
            suppress_weights[:, 1:2, :, :] * attention_weights[:, 1:2, :, :]
        ).squeeze(1) if self.wavelet_levels > 1 else mid_freq
        
        # 增强的多尺度特征融合
        fused_features = torch.cat([
            enhanced_low_freq,
            suppressed_mid_freq,
            suppressed_high_freq
        ], dim=-1)
        
        enhanced_xyz = self.enhanced_fusion(fused_features)
        
        # 图结构正则化
        enhanced_xyz = self.graph_regularizer(
            enhanced_xyz, laplacian, graph_features
        )
        
        # 频域对抗训练（仅在训练时）
        if training and hasattr(self, 'adversarial_trainer'):
            enhanced_xyz = self.adversarial_trainer(
                enhanced_xyz, graph_features
            )
        
        # 计算鲁棒性损失
        robustness_loss = self.robustness_loss(
            enhanced_xyz, xyz, wavelet_coeffs
        )
        
        return enhanced_xyz, wavelet_coeffs, robustness_loss


class MultiHopAggregator(nn.Module):
    """多跳邻域聚合模块"""
    
    def __init__(self, input_dim, hidden_dim, max_hops):
        super().__init__()
        self.max_hops = max_hops
        self.hop_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(max_hops)
        ])
        # 修改 num_heads 从 4 改为 3，确保能够整除 input_dim=3
        self.hop_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=3, batch_first=True
        )
    
    def forward(self, xyz):
        B, N, C = xyz.shape
        
        # 计算多跳邻域特征
        hop_features = [xyz]
        current_features = xyz
        
        for hop in range(self.max_hops):
            # 简化的邻域聚合（可以替换为更复杂的图卷积）
            dist_matrix = torch.cdist(current_features, current_features)
            _, topk_indices = torch.topk(dist_matrix, k=8, dim=-1, largest=False)
            
            # 聚合邻域特征
            neighbor_features = torch.gather(
                current_features.unsqueeze(2).repeat(1, 1, 8, 1),
                1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 1, C)
            ).mean(dim=2)
            
            # 编码当前跳的特征
            encoded_features = self.hop_encoders[hop](neighbor_features)
            hop_features.append(encoded_features)
            current_features = encoded_features
        
        # 使用注意力融合多跳特征
        stacked_features = torch.stack(hop_features, dim=2)  # [B, N, hops+1, C]
        attended_features, _ = self.hop_attention(
            stacked_features.view(B*N, -1, C),
            stacked_features.view(B*N, -1, C),
            stacked_features.view(B*N, -1, C)
        )
        
        return attended_features.view(B, N, -1, C).mean(dim=2)


class AdaptiveSparseGraphConv(nn.Module):
    """自适应稀疏图卷积"""
    
    def __init__(self, in_channels, out_channels, k_neighbors, sparse_ratio):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sparse_ratio = sparse_ratio
        
        self.feature_transform = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
        self.edge_conv = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, xyz):
        B, N, C = xyz.shape
        
        # 特征变换
        features = self.feature_transform(
            xyz.view(-1, C)
        ).view(B, N, -1)
        
        # 构建稀疏邻接图
        dist_matrix = torch.cdist(xyz, xyz)
        sparse_k = int(self.k_neighbors * self.sparse_ratio)
        _, topk_indices = torch.topk(
            dist_matrix, sparse_k, dim=-1, largest=False
        )
        
        # 边卷积
        edge_features = []
        for i in range(sparse_k):
            neighbor_features = torch.gather(
                features, 1, 
                topk_indices[:, :, i:i+1].repeat(1, 1, features.size(-1))
            )
            edge_feat = torch.cat([features, neighbor_features], dim=-1)
            edge_features.append(edge_feat)
        
        edge_features = torch.stack(edge_features, dim=2)  # [B, N, k, 2*C]
        
        # 应用边卷积
        conv_features = self.edge_conv(
            edge_features.view(-1, edge_features.size(-1))
        ).view(B, N, sparse_k, -1)
        
        # 聚合邻域特征
        aggregated_features = conv_features.max(dim=2)[0]
        
        return aggregated_features


class DynamicGraphStructureLearner(nn.Module):
    """动态图结构学习"""
    
    def __init__(self, feature_dim, k_neighbors):
        super().__init__()
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        
        self.structure_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        # 预测图结构权重
        structure_weights = self.structure_predictor(features)
        return structure_weights


class EnhancedMultiScaleFusion(nn.Module):
    """增强的多尺度特征融合"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, spectral_norm=True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        output_linear = nn.Linear(prev_dim, output_dim)
        if spectral_norm:
            output_linear = nn.utils.spectral_norm(output_linear)
        layers.append(output_linear)
        
        self.fusion_network = nn.Sequential(*layers)
    
    def forward(self, fused_features):
        B, N, C = fused_features.shape
        output = self.fusion_network(
            fused_features.view(-1, C)
        ).view(B, N, -1)
        return output


class ResidualSpectralEnhancer(nn.Module):
    """残差谱增强器"""
    
    def __init__(self, in_channels, hidden_channels, spectral_norm=True):
        super().__init__()
        def make_layer(in_dim, out_dim):
            linear = nn.Linear(in_dim, out_dim)
            if spectral_norm and out_dim > 0:  # 添加维度检查
                linear = nn.utils.spectral_norm(linear)
            return linear
        
        self.enhancer = nn.Sequential(
            make_layer(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            make_layer(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.2),
            make_layer(hidden_channels, in_channels),
            nn.BatchNorm1d(in_channels)
        )
        
        # 注意力机制
        attention_dim = max(1, in_channels // 4)
        self.attention = nn.Sequential(
            make_layer(in_channels, attention_dim),
            nn.ReLU(inplace=True),
            make_layer(attention_dim, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, low_freq):
        B, N, C = low_freq.shape
        
        # 特征增强 - 修复view操作
        enhanced = self.enhancer(
            low_freq.contiguous().view(-1, C)
        ).view(B, N, C)
        
        # 注意力权重 - 修复view操作
        attention_weights = self.attention(
            low_freq.contiguous().view(-1, C)
        ).view(B, N, C)
        
        # 残差连接 + 注意力
        output = low_freq + enhanced * attention_weights
        
        return output


class SpectralAdversarialTrainer(nn.Module):
    """频域对抗训练模块"""
    
    def __init__(self, feature_dim, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.feature_dim = feature_dim
        
        # 修复：adversarial_generator应该输出3维（xyz坐标）
        self.adversarial_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 3),  # 输出3维，匹配xyz坐标
            nn.Tanh()
        )
    
    def forward(self, enhanced_xyz, graph_features):
        if not self.training:
            return enhanced_xyz
        
        B, N, C = enhanced_xyz.shape
        
        # 修复：确保graph_features的形状正确
        # graph_features形状: [B, N, feature_dim]
        graph_features_reshaped = graph_features.contiguous().view(B*N, self.feature_dim)
        
        # 生成对抗扰动
        noise = self.adversarial_generator(graph_features_reshaped)
        noise = noise.view(B, N, 3)  # 重塑为[B, N, 3]
        
        # 限制扰动幅度
        noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        
        # 应用对抗扰动
        adversarial_xyz = enhanced_xyz + noise
        
        return adversarial_xyz


class AdaptiveFrequencySupressor(nn.Module):
    def __init__(self, input_dim, wavelet_levels, context_aware=True, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.wavelet_levels = wavelet_levels
        self.context_aware = context_aware
        
        # 基础抑制权重生成器 - 确保输出维度正确
        self.suppressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, wavelet_levels),  # 关键修复：确保输出是wavelet_levels而不是固定值
            nn.Sigmoid()
        )
        
        # 上下文编码器（如果启用上下文感知）
        if self.context_aware:
            context_input_dim = context_dim if context_dim is not None else input_dim
            self.context_encoder = nn.Sequential(
                nn.Linear(context_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, wavelet_levels),  # 关键修复：确保输出是wavelet_levels
                nn.Tanh()
            )
    
    def forward(self, xyz, context=None):
        B, N, C = xyz.shape
        
        # 基础抑制权重 - 修复view操作
        base_weights = self.suppressor(
            xyz.contiguous().view(-1, C)
        ).reshape(B, N, self.wavelet_levels)
        
        if self.context_aware and context is not None:
            # 上下文调制 - 修复view操作
            context_weights = self.context_encoder(
                context.contiguous().view(-1, context.size(-1))
            ).reshape(B, N, self.wavelet_levels)
            
            # 融合基础权重和上下文权重
            final_weights = base_weights * torch.sigmoid(context_weights)
        else:
            final_weights = base_weights
        
        # 确保返回抑制权重
        return final_weights


class MultiHeadSpectralAttention(nn.Module):
    """多头频域注意力机制"""
    
    def __init__(self, embed_dim, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 确保embed_dim能被num_heads整除
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: [B, N, embed_dim]
        B, N, C = x.shape
        
        # 如果输入是3D，需要重塑为2D进行注意力计算
        if len(x.shape) == 3:
            x_2d = x.view(B*N, -1).unsqueeze(1)  # [B*N, 1, C]
            attended, _ = self.attention(x_2d, x_2d, x_2d)
            attended = attended.squeeze(1).view(B, N, C)
        else:
            attended, _ = self.attention(x, x, x)
        
        # 残差连接和层归一化
        output = self.norm(x + attended)
        
        return output


class EnhancedGraphRegularizer(nn.Module):
    """增强图正则化器"""
    
    def __init__(self, k_neighbors, feature_dim):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.feature_dim = feature_dim
        
        self.regularizer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 3),  # 输出xyz坐标
            nn.Tanh()
        )
    
    def forward(self, enhanced_xyz, laplacian, graph_features):
        B, N, C = enhanced_xyz.shape
        
        # 图正则化
        regularized = self.regularizer(
            graph_features.contiguous().view(-1, self.feature_dim)
        ).view(B, N, 3)
        
        # 与原始坐标融合
        output = enhanced_xyz + 0.1 * regularized
        
        return output


class RobustnessLoss(nn.Module):
    """鲁棒性损失计算"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, enhanced_xyz, original_xyz, wavelet_coeffs):
        # 计算重构损失
        reconstruction_loss = F.mse_loss(enhanced_xyz, original_xyz)
        
        # 计算频域正则化损失
        spectral_loss = torch.mean(torch.abs(wavelet_coeffs))
        
        # 总损失
        total_loss = reconstruction_loss + 0.01 * spectral_loss
        
        return total_loss


# 为了兼容性，保留原有的类名
class GraphWaveletTransform(AdvancedGraphWaveletTransform):
    """兼容性包装器"""
    def __init__(self, num_scales=4, k_neighbors=16, wavelet_levels=3):
        super().__init__(
            num_scales=num_scales,
            k_neighbors=k_neighbors, 
            wavelet_levels=wavelet_levels,
            adaptive_threshold=True,
            spectral_norm=True,
            multi_hop=True
        )
    
    def forward(self, xyz):
        enhanced_xyz, wavelet_coeffs, _ = super().forward(xyz, training=self.training)
        return enhanced_xyz, wavelet_coeffs


# 内存高效版本
class MemoryEfficientGraphWaveletTransform(nn.Module):
    """内存高效的图小波变换（适用于大规模点云）"""
    
    def __init__(self, num_scales=2, k_neighbors=8, wavelet_levels=2):
        super().__init__()
        # 使用较小的参数以节省内存
        self.base_transform = AdvancedGraphWaveletTransform(
            num_scales=num_scales,
            k_neighbors=k_neighbors,
            wavelet_levels=wavelet_levels,
            adaptive_threshold=False,  # 简化计算
            spectral_norm=False,       # 减少内存使用
            multi_hop=False,           # 禁用多跳
            sparse_ratio=0.3           # 更稀疏的图
        )
    
    def forward(self, xyz):
        return self.base_transform(xyz, training=self.training)[:2]