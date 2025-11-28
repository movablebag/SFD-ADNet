"""Targeted point perturbation attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import importlib
import sys
sys.path.append('../')

from config import BEST_WEIGHTS
from config import MAX_PERTURB_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
# 添加ScanObjectNN数据集导入
from dataset.scanobjectnn import ScanObjectNNHardest
# 修复模型导入 - 直接从各自的文件导入
from DGCNN_cls import DGCNN
from pointnet_cls import get_model as PointNetCls
from pointnet2_cls_ssg import get_model as PointNet2ClsSsg
from util.utils import str2bool, set_seed
from attack import CWPerturb
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import L2Dist
# 在文件顶部导入部分添加
import yaml
from easydict import EasyDict
from openpoints.models import build_model_from_cfg
from openpoints.utils import load_checkpoint

# 创建适配ScanObjectNN的攻击数据集类
class ScanObjectNNAttack(ScanObjectNNHardest):
    def __init__(self, data_dir, num_points=1024, normalize=True):
        super().__init__(data_dir, split='test', num_points=num_points, uniform_sample=True)
        self.normalize = normalize
        
    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        
        if self.normalize:
            # 归一化到单位球
            current_points = current_points - np.mean(current_points, axis=0)
            current_points = current_points / np.max(np.linalg.norm(current_points, axis=1))
        
        # 生成目标标签（随机选择不同于真实标签的类别）
        target_label = np.random.randint(0, 15)
        while target_label == label:
            target_label = np.random.randint(0, 15)
            
        return current_points.astype(np.float32), label, target_label

def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    num = 0
    for pc, label, target in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)
            target_label = target.long().cuda(non_blocking=True)

        # attack!
        _, best_pc, success_num = attacker.attack(pc, target_label)

        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(target_label.detach().cpu().numpy())

    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, all_target_lbl, num


def load_model(model_name, num_class, model_path=None):
    """加载指定的模型"""
    if model_name.lower() == 'dgcnn':
        model = DGCNN(k=args.k, emb_dims=args.emb_dims, dropout_p=0.5, output_channels=num_class).cuda()
    elif model_name.lower() == 'pointnet':
        model = PointNetCls(num_class=num_class, normal_channel=False)
    elif model_name.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_class, normal_channel=False)
    elif model_name.lower() == 'pointconv':
        from pointconv import PointConvDensityClsSsg
        model = PointConvDensityClsSsg(num_class, normal_channel=False)
    elif model_name.lower() == 'pointnext':
        # 加载PointNext模型
        if args.cfg:
            cfg_path = args.cfg
        else:
            # 根据数据集选择默认配置文件
            if args.dataset == 'scanobjectnn':
                cfg_path = 'cfgs/scanobjectnn/pointnext-s.yaml'
            else:
                cfg_path = 'cfgs/modelnet40/pointnext-s.yaml'
        
        # 加载配置文件
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)
        
        # 构建模型
        model = build_model_from_cfg(cfg.model)
        
        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            load_checkpoint(model, model_path)
            print(f"Loaded PointNext weights from: {model_path}")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 加载预训练权重
    # 加载预训练权重
    if model_path and os.path.exists(model_path) and model_name.lower() != 'pointnext':
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Checkpoint keys: {checkpoint.keys()}")
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded using 'model_state_dict' key")
            elif 'model' in checkpoint:
                # 如果checkpoint包含'model'键，说明是通过save_checkpoint保存的
                model.load_state_dict(checkpoint['model'])
                print("Loaded using 'model' key")
            else:
                # 尝试直接加载
                model.load_state_dict(checkpoint, strict=False)
                print("Loaded directly with strict=False")
                
            print(f"Successfully loaded model weights from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # 尝试使用非严格模式加载
            if 'model' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model'], strict=False)
                    print("Loaded using 'model' key with strict=False")
                except Exception as e2:
                    print(f"Failed to load with strict=False: {e2}")
    
    return model.cuda()
