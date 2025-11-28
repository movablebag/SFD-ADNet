import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from collections import defaultdict
from easydict import EasyDict 


# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.utils import load_checkpoint, setup_logger_dist
from openpoints.transforms import build_transforms_from_cfg

class AdversarialAttacks:
    """对抗攻击方法实现"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def point_perturbation_attack(self, points, epsilon=0.01, attack_ratio=0.1):
        """点扰动攻击
        Args:
            points: 输入点云 [B, N, 3]
            epsilon: 扰动强度
            attack_ratio: 攻击点的比例
        """
        B, N, C = points.shape
        perturbed_points = points.clone()
        
        # 随机选择要攻击的点
        num_attack_points = int(N * attack_ratio)
        for b in range(B):
            attack_indices = torch.randperm(N)[:num_attack_points]
            # 添加高斯噪声
            noise = torch.randn(num_attack_points, C, device=points.device) * epsilon
            perturbed_points[b, attack_indices] += noise
        
        return perturbed_points
    
    def individual_point_adding_attack(self, points, num_add_points=50):
        """单个点添加攻击
        Args:
            points: 输入点云 [B, N, 3]
            num_add_points: 添加的点数量
        """
        B, N, C = points.shape
        
        # 计算点云的边界框
        min_vals = torch.min(points, dim=1, keepdim=True)[0]  # [B, 1, 3]
        max_vals = torch.max(points, dim=1, keepdim=True)[0]  # [B, 1, 3]
        
        # 在边界框内随机生成新点
        random_points = torch.rand(B, num_add_points, C, device=points.device)
        random_points = random_points * (max_vals - min_vals) + min_vals
        
        # 拼接原始点和新增点
        attacked_points = torch.cat([points, random_points], dim=1)
        
        return attacked_points
    
    def knn_attack(self, points, k=16, epsilon=0.02):
        """kNN攻击
        Args:
            points: 输入点云 [B, N, 3]
            k: kNN的邻居数
            epsilon: 扰动强度
        """
        B, N, C = points.shape
        attacked_points = points.clone()
        
        for b in range(B):
            # 计算距离矩阵
            dist_matrix = torch.cdist(points[b], points[b])  # [N, N]
            
            # 找到每个点的k个最近邻
            _, knn_indices = torch.topk(dist_matrix, k+1, dim=1, largest=False)
            knn_indices = knn_indices[:, 1:]  # 排除自己
            
            # 对每个点及其邻居添加相关扰动
            for i in range(N):
                neighbors = knn_indices[i]
                # 生成相关的扰动
                base_noise = torch.randn(C, device=points.device) * epsilon
                attacked_points[b, i] += base_noise
                
                # 对邻居添加相似的扰动
                for neighbor_idx in neighbors:
                    neighbor_noise = base_noise + torch.randn(C, device=points.device) * epsilon * 0.5
                    attacked_points[b, neighbor_idx] += neighbor_noise
        
        return attacked_points
    
    def point_dropping_attack(self, points, drop_ratio=0.1):
        """点丢弃攻击
        Args:
            points: 输入点云 [B, N, 3]
            drop_ratio: 丢弃点的比例
        """
        B, N, C = points.shape
        num_keep_points = int(N * (1 - drop_ratio))
        
        attacked_points = torch.zeros(B, num_keep_points, C, device=points.device)
        
        for b in range(B):
            # 随机选择要保留的点
            keep_indices = torch.randperm(N)[:num_keep_points]
            attacked_points[b] = points[b, keep_indices]
        
        return attacked_points

def evaluate_model_robustness(model, dataloader, attacks, device='cuda'):
    """评估模型对各种攻击的鲁棒性"""
    model.eval()
    results = defaultdict(list)
    
    attack_methods = {
        'clean': lambda x: x,
        'point_perturbation': attacks.point_perturbation_attack,
        'point_adding': attacks.individual_point_adding_attack,
        'knn_attack': attacks.knn_attack,
        'point_dropping': attacks.point_dropping_attack
    }
    
    with torch.no_grad():
        for attack_name, attack_func in attack_methods.items():
            correct = 0
            total = 0
            
            print(f"\n评估 {attack_name} 攻击...")
            
            for data in tqdm(dataloader):
                points, labels = data['pos'], data['y']
                points, labels = points.to(device), labels.to(device)
                
                # 应用攻击
                if attack_name != 'clean':
                    attacked_points = attack_func(points)
                else:
                    attacked_points = points
                
                # 预测
                if attack_name == 'point_adding':
                    # 对于点添加攻击，需要处理不同的点数量
                    data_attacked = {'pos': attacked_points, 'x': attacked_points}
                else:
                    data_attacked = {'pos': attacked_points, 'x': attacked_points}
                
                # 1. 首先确保数据是正确的维度顺序 [B, C, N]
                if data_attacked['x'].dim() == 3 and data_attacked['x'].shape[1] > data_attacked['x'].shape[2]:
                    data_attacked['x'] = data_attacked['x'].transpose(1, 2)
                
                # 2. 然后处理通道数量问题，从3个通道扩展到4个通道
                if data_attacked['x'].shape[1] == 3:  # 如果只有3个通道
                    B, C, N = data_attacked['x'].shape
                    # 创建一个额外的特征通道（可以是零或计算的特征）
                    # 这里简单地使用点的高度（y坐标）作为第四个通道
                    extra_channel = data_attacked['x'][:, 1:2, :]  # 使用y坐标作为额外特征
                    # 或者使用全零通道
                    # extra_channel = torch.zeros((B, 1, N), device=data_attacked['x'].device)
                    
                    # 拼接到原始数据
                    data_attacked['x'] = torch.cat([data_attacked['x'], extra_channel], dim=1)
                
                try:
                    outputs = model(data_attacked)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0
            results[attack_name] = accuracy
            print(f"{attack_name} 准确率: {accuracy:.4f}")
    
    return results





def validate_config(cfg):
    """验证配置文件的完整性"""
    required_sections = ['model', 'datatransforms_scanobjectnn_c']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f'配置文件缺少必要的部分: {section}')
    
    # 验证model配置
    model_cfg = cfg['model']
    if 'encoder_args' not in model_cfg:
        raise ValueError('model配置缺少encoder_args部分')
    
    encoder_args = model_cfg['encoder_args']
    if encoder_args['NAME'] == 'PointNextEncoder':
        required_params = ['blocks', 'strides', 'width', 'in_channels', 'sa_layers', 'radius']
        missing_params = [param for param in required_params if param not in encoder_args]
        if missing_params:
            raise ValueError(f'PointNextEncoder缺少必要的参数: {missing_params}')

def main():
    parser = argparse.ArgumentParser('AdversarialEvaluation')
    parser.add_argument('--cfg', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--dataset', type=str, choices=['modelnet40', 'scanobjectnn'], 
                       default='scanobjectnn', help='数据集选择')
    parser.add_argument('--model_type', type=str, choices=['pointnet++', 'pointnext'], 
                       default='pointnext', help='模型类型选择')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--device', type=str, default='cuda', help='设备选择')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger_dist()
    
    # 加载配置
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    
    try:
        validate_config(cfg)
        print("配置验证通过")
    except Exception as e:
        print(f"配置验证失败: {e}")
        return
    
   
 
    
    # 根据数据集调整配置
    if args.dataset == 'modelnet40':
        cfg['dataset']['common']['NAME'] = 'ModelNet'
        cfg['dataset']['common']['data_root'] = './data/ModelNet40/modelnet40_normal_resampled/'
        cfg['model']['cls_args']['num_classes'] = 40
    elif args.dataset == 'scanobjectnn':
        cfg.setdefault('dataset', {}).setdefault('common', {})
        cfg['dataset']['common']['NAME'] = 'ScanObjectNNHardest'
        # cfg['dataset']['common']['data_root'] = './data/ScanObjectNN/h5_files/main_split/'
        cfg['model']['cls_args']['num_classes'] = 15
    
    # 构建模型
    def build_model(cfg):
        # 将字典配置转换为EasyDict对象
        from easydict import EasyDict
        model_cfg = EasyDict(cfg['model'])
        
        # 确保encoder_args是EasyDict类型
        model_cfg.encoder_args = EasyDict(model_cfg.encoder_args)
        
        # 构建模型
        model = build_model_from_cfg(model_cfg)
        return model
    
    # 构建模型
    try:
        model = build_model(cfg).to(args.device)
        print("成功构建模型")
    except Exception as e:
        print(f"构建模型失败: {e}")
        return
    
    # 加载检查点
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, pretrained_path=args.checkpoint)
        print(f"成功加载模型检查点: {args.checkpoint}")
    else:
        print(f"警告: 检查点文件不存在: {args.checkpoint}")
        print("将使用随机初始化的模型进行评估")
    
    # 构建数据加载器
    try:
        transforms_cfg = cfg.get('datatransforms_scanobjectnn_c', {})
        if not transforms_cfg:
            print("警告: 未找到datatransforms_scanobjectnn_c配置，使用默认转换")
            transforms_cfg = {
                'train': ['PointsToTensor', 'PointCloudCenterAndNormalize'],
                'val': ['PointsToTensor', 'PointCloudCenterAndNormalize'],
                'kwargs': {'gravity_dim': 1}
            }
        
        val_transforms = build_transforms_from_cfg('val', transforms_cfg)
        
        # 确保dataset_cfg是一个完整的EasyDict对象
        dataset_cfg = EasyDict()
        
        data_root = '/media/shangli211/4TB_SSD/program_file/Data/ScanObjectNN/h5_files/main_split'
        common_cfg = {
            'NAME': 'ScanObjectNNHardest',
            'num_points': 1024,
            'num_classes': 15,
            'data_root': data_root,
            'data_dir': data_root,  # 添加data_dir参数
            'subset': cfg.get('dataset', {}).get('common', {}).get('subset', 'main_split')
        }
        dataset_cfg.common = EasyDict(common_cfg)
        
        # 创建val属性
        dataset_cfg.val = EasyDict({
            'split': 'test',
            'transform': val_transforms
        })
        
        # 数据加载器配置
        dataloader_cfg = EasyDict({
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'persistent_workers': True
        })
        
        
        val_loader = build_dataloader_from_cfg(
            dataloader_cfg.batch_size,  # 第一个参数应该是batch_size
            dataset_cfg,                # 第二个参数是完整的dataset_cfg
            dataloader_cfg,             # 第三个参数是dataloader_cfg
            split='val',                 # 指定split为'val'
            distributed=False
        )
        print(f"成功构建数据加载器，数据集: {args.dataset}")
    except Exception as e:
        print(f"构建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return
    
    # 初始化攻击方法
    attacks = AdversarialAttacks(device=args.device)
    
    # 评估模型鲁棒性
    print(f"\n开始评估 {args.model_type} 模型在 {args.dataset} 数据集上的对抗鲁棒性...")
    results = evaluate_model_robustness(model, val_loader, attacks, args.device)
    
    # 打印结果
    print("\n=== 对抗攻击评估结果 ===")
    print(f"模型: {args.model_type}")
    print(f"数据集: {args.dataset}")
    print("-" * 40)
    
    for attack_name, accuracy in results.items():
        print(f"{attack_name:20s}: {accuracy:.4f}")
    
    # 计算鲁棒性指标
    clean_acc = results.get('clean', 0)
    attack_accs = [acc for name, acc in results.items() if name != 'clean']
    avg_attack_acc = np.mean(attack_accs) if attack_accs else 0
    robustness_drop = clean_acc - avg_attack_acc
    
    print("-" * 40)
    print(f"清洁数据准确率: {clean_acc:.4f}")
    print(f"平均攻击准确率: {avg_attack_acc:.4f}")
    print(f"鲁棒性下降: {robustness_drop:.4f}")
    print(f"相对鲁棒性: {avg_attack_acc/clean_acc:.4f}" if clean_acc > 0 else "N/A")

if __name__ == '__main__':
    main()