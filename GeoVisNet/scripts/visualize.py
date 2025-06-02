#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet可视化脚本

提供各种可视化功能，包括注意力图、结果分析等
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geovisnet.models import GeoVisNet
from geovisnet.data import UAVVisLocDataset, get_val_transforms
from geovisnet.utils import load_model_weights
from geovisnet.visualization import (
    visualize_attention, plot_attention_heatmap,
    visualize_dataset_sample, plot_coordinate_distribution,
    plot_error_distribution, plot_region_comparison
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GeoVisNet可视化脚本')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    
    # 可视化类型
    parser.add_argument('--vis_type', type=str, default='all',
                       choices=['all', 'attention', 'dataset', 'results'],
                       help='可视化类型')
    
    # 模型参数（用于注意力可视化）
    parser.add_argument('--model_path', type=str, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'efficientnet_b2'], help='骨干网络')
    parser.add_argument('--emb_size', type=int, default=256, help='嵌入维度')
    
    # 数据参数
    parser.add_argument('--regions', type=str, default='01,07,08', help='要可视化的区域')
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--num_samples', type=int, default=10, help='可视化样本数量')
    
    # 结果分析参数（如果有测试结果）
    parser.add_argument('--results_file', type=str, help='测试结果文件路径')
    
    return parser.parse_args()


def visualize_attention_maps(model, data_loader, device, output_dir, num_samples=5):
    """可视化注意力图"""
    print("生成注意力可视化...")
    
    model.eval()
    attention_dir = os.path.join(output_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if sample_count >= num_samples:
                break
                
            drone_imgs = batch['drone_img'].to(device)
            sat_imgs = batch['sat_img'].to(device)
            regions = batch['region']
            
            # 获取注意力图
            attention_maps = model.get_attention_maps(drone_imgs, sat_imgs)
            
            # 为每个样本生成可视化
            batch_size = drone_imgs.size(0)
            for i in range(min(batch_size, num_samples - sample_count)):
                # 提取单个样本
                drone_img = drone_imgs[i]
                sat_img = sat_imgs[i]
                region = regions[i]
                
                # 提取对应的注意力图
                sample_attention = {}
                for key, value in attention_maps.items():
                    sample_attention[key] = value[i:i+1]  # 保持批次维度
                
                # 生成可视化
                save_path = os.path.join(attention_dir, f'attention_sample_{sample_count+1}_region_{region}.png')
                visualize_attention(drone_img, sat_img, sample_attention, save_path)
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
    
    print(f"注意力可视化完成，保存到: {attention_dir}")


def visualize_dataset_samples(data_loader, sat_coords_df, output_dir, num_samples=10):
    """可视化数据集样本"""
    print("生成数据集样本可视化...")
    
    dataset_dir = os.path.join(output_dir, 'dataset_samples')
    os.makedirs(dataset_dir, exist_ok=True)
    
    sample_count = 0
    all_coords = []
    all_regions = []
    
    for batch_idx, batch in enumerate(data_loader):
        if sample_count >= num_samples:
            break
            
        drone_imgs = batch['drone_img']
        sat_imgs = batch['sat_img']
        drone_lats = batch['drone_lat']
        drone_lons = batch['drone_lon']
        regions = batch['region']
        
        batch_size = drone_imgs.size(0)
        for i in range(min(batch_size, num_samples - sample_count)):
            # 转换图像为numpy格式
            drone_img = drone_imgs[i].permute(1, 2, 0).numpy()
            sat_img = sat_imgs[i].permute(1, 2, 0).numpy()
            
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            drone_img = drone_img * std + mean
            sat_img = sat_img * std + mean
            drone_img = np.clip(drone_img, 0, 1)
            sat_img = np.clip(sat_img, 0, 1)
            
            # 获取坐标信息
            drone_coords = (drone_lats[i].item(), drone_lons[i].item())
            region = regions[i]
            
            # 获取卫星图像坐标范围
            sat_name = f'satellite{region}.tif'
            sat_info = sat_coords_df[sat_coords_df['mapname'] == sat_name]
            
            if len(sat_info) > 0:
                sat_coords = {
                    'LT_lat_map': sat_info['LT_lat_map'].values[0],
                    'LT_lon_map': sat_info['LT_lon_map'].values[0],
                    'RB_lat_map': sat_info['RB_lat_map'].values[0],
                    'RB_lon_map': sat_info['RB_lon_map'].values[0]
                }
            else:
                sat_coords = {}
            
            # 生成可视化
            save_path = os.path.join(dataset_dir, f'sample_{sample_count+1}_region_{region}.png')
            visualize_dataset_sample(drone_img, sat_img, drone_coords, sat_coords, region, save_path)
            
            # 收集坐标用于分布图
            all_coords.append(drone_coords)
            all_regions.append(region)
            
            sample_count += 1
            if sample_count >= num_samples:
                break
    
    # 生成坐标分布图
    if all_coords:
        coord_save_path = os.path.join(dataset_dir, 'coordinate_distribution.png')
        plot_coordinate_distribution(all_coords, all_regions, save_path=coord_save_path)
    
    print(f"数据集样本可视化完成，保存到: {dataset_dir}")


def visualize_results(results_file, output_dir):
    """可视化测试结果"""
    print("生成结果可视化...")
    
    results_dir = os.path.join(output_dir, 'results_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载结果文件
    if results_file.endswith('.json'):
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 如果有详细结果CSV文件，也加载它
        csv_file = results_file.replace('.json', '_detailed.csv')
        if not os.path.exists(csv_file):
            csv_file = os.path.join(os.path.dirname(results_file), 'detailed_results.csv')
        
        if os.path.exists(csv_file):
            detailed_df = pd.read_csv(csv_file)
            
            # 提取误差数据
            geo_errors = detailed_df['geo_error_m'].tolist()
            regions = detailed_df['region'].tolist()
            
            # 生成误差分布图
            error_dist_path = os.path.join(results_dir, 'error_distribution.png')
            plot_error_distribution(geo_errors, save_path=error_dist_path)
            
            # 按区域分组误差
            errors_by_region = {}
            for region in set(regions):
                region_errors = [geo_errors[i] for i, r in enumerate(regions) if r == region]
                errors_by_region[region] = region_errors
            
            # 生成区域比较图
            region_comp_path = os.path.join(results_dir, 'region_comparison.png')
            plot_region_comparison(errors_by_region, save_path=region_comp_path)
            
            print(f"结果可视化完成，保存到: {results_dir}")
        else:
            print(f"未找到详细结果文件: {csv_file}")
    else:
        print(f"不支持的结果文件格式: {results_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载卫星坐标信息
    sat_coords_path = os.path.join(args.data_root, 'satellite_coordinates_range.csv')
    if not os.path.exists(sat_coords_path):
        sat_coords_path = os.path.join(args.data_root, 'satellite_ coordinates_range.csv')
    
    if os.path.exists(sat_coords_path):
        sat_coords_df = pd.read_csv(sat_coords_path)
    else:
        print(f"警告: 未找到卫星坐标文件: {sat_coords_path}")
        sat_coords_df = pd.DataFrame()
    
    # 解析区域
    regions = args.regions.split(',')
    
    # 创建数据集和数据加载器
    if args.vis_type in ['all', 'attention', 'dataset']:
        transform = get_val_transforms(args.img_size)
        dataset = UAVVisLocDataset(
            data_root=args.data_root,
            regions=regions,
            img_size=args.img_size,
            transform=transform,
            augment=False,
            mode='test',
            cache_satellite=True
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"数据集大小: {len(dataset)}")
    
    # 注意力可视化
    if args.vis_type in ['all', 'attention']:
        if args.model_path and os.path.exists(args.model_path):
            # 创建模型
            model = GeoVisNet(
                emb_size=args.emb_size,
                backbone=args.backbone,
                dropout=0.5
            )
            model = model.to(device)
            
            # 加载模型权重
            if load_model_weights(model, args.model_path, map_location=device):
                visualize_attention_maps(model, data_loader, device, args.output_dir, args.num_samples)
            else:
                print(f"无法加载模型权重: {args.model_path}")
        else:
            print("跳过注意力可视化：未提供有效的模型路径")
    
    # 数据集可视化
    if args.vis_type in ['all', 'dataset']:
        visualize_dataset_samples(data_loader, sat_coords_df, args.output_dir, args.num_samples)
    
    # 结果可视化
    if args.vis_type in ['all', 'results']:
        if args.results_file and os.path.exists(args.results_file):
            visualize_results(args.results_file, args.output_dir)
        else:
            print("跳过结果可视化：未提供有效的结果文件路径")
    
    print(f"\n可视化完成！所有结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
