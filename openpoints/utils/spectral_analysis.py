import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class SpectralAnalyzer:
    """用于分析和可视化频域增强效果的工具"""
    
    def __init__(self, save_dir='./spectral_analysis'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_spectral_weights(self, weights, point_clouds, labels=None, step=0):
        """可视化频域增强权重分布"""
        B, N = weights.shape[:2]
        weights = weights.detach().cpu().numpy()
        
        # 计算每个点云的平均权重
        avg_weights = np.mean(weights, axis=1)
        
        # 绘制权重分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(avg_weights, bins=20)
        plt.title('Distribution of Spectral Enhancement Weights')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.savefig(f'{self.save_dir}/weight_dist_step{step}.png')
        plt.close()
        
        # 如果有标签，分析不同类别的权重分布
        if labels is not None:
            labels = labels.detach().cpu().numpy()
            unique_labels = np.unique(labels)
            
            plt.figure(figsize=(12, 8))
            for label in unique_labels:
                label_weights = avg_weights[labels == label]
                plt.hist(label_weights, bins=10, alpha=0.5, label=f'Class {label}')
            
            plt.title('Spectral Weights by Class')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{self.save_dir}/weight_by_class_step{step}.png')
            plt.close()
    
    def visualize_point_cloud_with_weights(self, point_cloud, weights, idx=0, step=0):
        """可视化单个点云的权重分布"""
        pc = point_cloud[idx].detach().cpu().numpy()
        w = weights[idx].detach().cpu().numpy()
        
        # 使用t-SNE将3D点云投影到2D
        tsne = TSNE(n_components=2, random_state=42)
        pc_2d = tsne.fit_transform(pc)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pc_2d[:, 0], pc_2d[:, 1], c=w, cmap='viridis', s=5)
        plt.colorbar(scatter, label='Spectral Weight')
        plt.title('Point Cloud with Spectral Enhancement Weights')
        plt.savefig(f'{self.save_dir}/pc_weights_idx{idx}_step{step}.png')
        plt.close()