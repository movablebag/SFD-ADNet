import torch
import torch.nn as nn
# from knn_cuda import KNN
from knn_cuda import KNN
from ..build import MODELS
from openpoints.utils.logger import *
from ..layers import trunc_normal_, DropPath, fps, SubsampleGroup, TransformerEncoder
from openpoints.utils.ckpt_util import get_missing_parameters_message, get_unexpected_parameters_message


# Mamba3D's GroupFeature for local feature enhancement
class GroupFeature(nn.Module):
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input: 
                xyz: B N 3
                feat: B N C
            ---------------------------
            output: 
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape # B N 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz) # B N K : get K idx for every center
        assert idx.size(1) == num_points # N center
        assert idx.size(2) == self.group_size # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :] 
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, neighborhood_feat


# K_Norm for local geometry aggregation from Mamba3D
class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha=1.0, beta=0.0):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        #get knn xyz and feature 
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x) # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2) # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz) # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5) # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1) # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat 
        
        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2) # B 2C G K

        return knn_x_w


# Enhanced Encoder with local feature enhancement
class EnhancedEncoder(nn.Module):
    def __init__(self, encoder_channel, k_group_size=8):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.k_group_size = k_group_size
        
        # First convolution layer
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        
        # Local feature enhancement
        self.k_norm = K_Norm(self.encoder_channel, k_group_size=self.k_group_size, alpha=1.0, beta=0.0)
        
        # Feature fusion layer
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
        
        # Basic encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG encoder_channel n
        
        # Reshape for local feature enhancement
        feature_reshaped = feature.transpose(1, 2).contiguous()  # BG n encoder_channel
        
        # Apply local feature enhancement if point count is sufficient
        if n >= self.k_group_size:
            # Local feature enhancement
            enhanced_features = self.k_norm(point_groups, feature_reshaped)  # B 2C G K
            
            # Process enhanced features
            enhanced_features = torch.mean(enhanced_features, dim=-1)  # B 2C G
            enhanced_features = enhanced_features.permute(0, 2, 1).contiguous()  # BG 2C
            
            # Reshape original features for fusion
            feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
            
            # Fuse original and enhanced features
            fused_features = self.fusion_layer(
                torch.cat([feature_global.unsqueeze(2), enhanced_features.unsqueeze(2)], dim=1)
            ).squeeze(2)  # BG encoder_channel
            
            return fused_features.reshape(bs, g, self.encoder_channel)
        else:
            # Fallback to original method for small point groups
            feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
            return feature_global.reshape(bs, g, self.encoder_channel)


# Mamba3D Block for transformer replacement
class Mamba3DBlock(nn.Module):
    def __init__(self, dim, k_group_size=8, drop_path=0., num_group=128, num_heads=6, bimamba_type="v2"):
        super().__init__()
        self.dim = dim
        self.k_group_size = k_group_size
        self.num_group = num_group
        self.num_heads = num_heads
        
        # Local feature enhancement
        self.k_norm = K_Norm(dim, k_group_size=k_group_size, alpha=1.0, beta=0.0)
        
        # Feature transformation layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Dropout path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP block
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, center, x):
        # Apply local feature enhancement
        B, N, C = x.shape  # B G+1 C
        
        # Skip the cls token for local feature enhancement
        x_no_cls = x[:, 1:, :]  # B G C
        center_xyz = center  # B G 3
        
        # Apply normalization
        x_norm = self.norm1(x)
        x_cls, x_no_cls_norm = x_norm[:, 0:1, :], x_norm[:, 1:, :]
        
        # Apply local feature enhancement if we have enough points
        if N-1 >= self.k_group_size:
            # Get enhanced features
            enhanced_features = self.k_norm(center_xyz, x_no_cls_norm)  # B 2C G K
            
            # Process enhanced features
            enhanced_features = torch.mean(enhanced_features, dim=-1)  # B 2C G
            enhanced_features = enhanced_features.permute(0, 2, 1).contiguous()  # B G 2C
            
            # Apply attention mechanism
            attn_out = self.attn(enhanced_features)  # B G C
            
            # Combine with cls token
            attn_out = torch.cat([x_cls, attn_out], dim=1)  # B G+1 C
        else:
            # Fallback for small point counts
            attn_out = x_norm
        
        # Apply first residual connection
        x = x + self.drop_path(attn_out)
        
        # Apply MLP block with second residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


# Mamba3D Encoder to replace TransformerEncoder
class Mamba3DEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6, k_group_size=8, bimamba_type="v2"):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        
        # Create blocks
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

    def forward(self, center, x, pos):
        '''
        INPUT:
            x: patched point cloud and encoded, B G+1 C
            pos: positional encoding, B G+1 C
            center: center points, B G 3
        OUTPUT:
            x: x after transformer block, B G+1 C
        '''
        # Add positional encoding
        x = x + pos
        
        # Apply blocks
        for _, block in enumerate(self.blocks):
            x = block(center, x)
            
        return x


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, 
                 in_chans=3, num_classes=40,
                 embed_dim=768, depth=12,
                 num_heads=12, 
                #  mlp_ratio=4., qkv_bias=False,
                #  drop_rate=0., attn_drop_rate=0., 
                 
                 drop_path_rate=0.,

                 num_groups=256, group_size=32,
                 subsample='fps',  # random, FPS
                 group='ballquery', 
                 radius=0.1,

                 norm_args=None,
                 act_args=None,
                 k_group_size=8,  # For local feature enhancement
                 bimamba_type="v2",  # Mamba3D parameter
                 ):
        super().__init__()
        # Define encoder dimensions
        self.encoder_dims = 1024  # Same as original
        
        # Grouper
        self.group_divider = SubsampleGroup(num_groups, group_size, subsample, group, radius)
        
        # Replace original encoder with enhanced encoder
        self.encoder = EnhancedEncoder(encoder_channel=self.encoder_dims, k_group_size=k_group_size)
        
        # Bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, embed_dim)

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )  

        # Create drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Replace TransformerEncoder with Mamba3DEncoder
        self.blocks = Mamba3DEncoder(
            embed_dim=embed_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_group=num_groups,
            num_heads=num_heads,
            k_group_size=k_group_size,
            bimamba_type=bimamba_type
        )
        
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        ckpt = ckpt['base_model'] if hasattr(ckpt, 'base_model') else ckpt['model']
        
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]
        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                
            )
        logging.info(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        # Divide the point cloud in the same form
        neighborhood, center = self.group_divider(pts)
        
        # Encode the input cloud blocks with enhanced encoder
        group_input_tokens = self.encoder(neighborhood)  # B G C
        group_input_tokens = self.reduce_dim(group_input_tokens)  # B G embed_dim
        
        # Prepare cls token
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        
        # Add positional embedding
        pos = self.pos_embed(center)  # B G embed_dim
        
        # Final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # B G+1 embed_dim
        pos = torch.cat((cls_pos, pos), dim=1)  # B G+1 embed_dim
        
        # Apply Mamba3D blocks
        x = self.blocks(center, x, pos)
        x = self.norm(x)
        
        # Global feature aggregation
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)
        
        # Classification
        ret = self.cls_head_finetune(concat_f)
        return ret
