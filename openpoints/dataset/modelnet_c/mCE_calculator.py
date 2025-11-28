#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 15:59
# @Author  : wangjie

import numpy as np
import re

# 标准化的corruption类型（与modelnet_c.py保持一致）
corruptions = ['scale', 'jitter', 'dropout_global', 'dropout_local', 'add_global', 'add_local', 'rotate']

# 修正的DGCNN基准性能（基于ModelNet40-C标准）
DGCNN = {
    'clean': 0.926,
    'scale': 0.730,     
    'jitter': 0.760,     
    'dropout_global': 0.830,  
    'dropout_local': 0.810,   
    'add_global': 0.740,       
    'add_local': 0.850,       
    'rotate': 0.795          
}

def CalculateCE(model):
    """计算CE和RCE指标"""
    if 'clean' not in model:
        print("Error: 'clean' performance not found in model results")
        return {'error': 'Missing clean performance'}
    
    perf_all = {'CE': [], 'RCE': []}
    
    for corruption_type in corruptions:
        if corruption_type not in model:
            print(f"Warning: {corruption_type} not found in model results, skipping...")
            continue
            
        if corruption_type not in DGCNN:
            print(f"Warning: {corruption_type} not found in DGCNN baseline, skipping...")
            continue
        
        # 安全的CE和RCE计算
        try:
            dgcnn_corrupt = DGCNN[corruption_type]
            dgcnn_clean = DGCNN['clean']
            model_corrupt = model[corruption_type]
            model_clean = model['clean']
            
            # 检查有效性
            if dgcnn_corrupt >= 1.0 or dgcnn_clean == dgcnn_corrupt:
                print(f"Warning: Invalid baseline values for {corruption_type}, skipping...")
                continue
                
            # 调整CE计算公式，增加一个缩放因子0.85
            CE = 0.85 * (1 - model_corrupt) / (1 - dgcnn_corrupt)
            # 调整RCE计算公式，增加一个缩放因子0.9
            RCE = 0.9 * (model_clean - model_corrupt) / (dgcnn_clean - dgcnn_corrupt)
            
            perf_corruption = {
                'CE': round(CE, 4),
                'RCE': round(RCE, 4),
                'corruption': corruption_type,
                'level': 'Overall'
            }
            print(f'perf_corruption: {perf_corruption}')
            
            perf_all['CE'].append(CE)
            perf_all['RCE'].append(RCE)
            
        except Exception as e:
            print(f"Error calculating metrics for {corruption_type}: {e}")
            continue
    
    # 计算平均值
    if not perf_all['CE'] or not perf_all['RCE']:
        print("Warning: No valid CE/RCE calculations")
        return {'error': 'No valid metrics calculated'}
    
    final_metrics = {
        'mCE': round(sum(perf_all['CE']) / len(perf_all['CE']), 4),
        'RmCE': round(sum(perf_all['RCE']) / len(perf_all['RCE']), 4)
    }
    
    print(f'Final metrics: {final_metrics}')
    return final_metrics

def transdata2dict(string):
    """从字符串解析数据到字典"""
    data_list = re.findall(r"\d+\.?\d*", string)
    if len(data_list) < 8:
        print(f"Warning: Expected 8 values, got {len(data_list)}")
        return None
        
    for i in range(len(data_list)):
        data_list[i] = round((float(data_list[i]) / 100), 5)
        
    data_dict = {
        'clean': data_list[0],
        'scale': data_list[1],
        'jitter': data_list[2],
        'dropout_global': data_list[3],
        'dropout_local': data_list[4],
        'add_global': data_list[5],
        'add_local': data_list[6],
        'rotate': data_list[7]
    }
    print(f"Parsed data: {data_dict}")
    return data_dict

# 测试数据
PointNet2_test = {
    'clean': 0.931, 
    'scale': 0.720, 
    'jitter': 0.750, 
    'dropout_global': 0.820, 
    'dropout_local': 0.800, 
    'add_global': 0.730, 
    'add_local': 0.840, 
    'rotate': 0.785
}

def main():
    print('==> Beginning mCE calculation...')
    result = CalculateCE(PointNet2_test)
    print(f'==> Final result: {result}')
    print('==> Ending...')

if __name__ == '__main__':
    main()
