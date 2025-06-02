#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet评估脚本

对训练好的模型进行详细评估和分析
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geovisnet.models import GeoVisNet
from geovisnet.data import UAVVisLocDataset, get_val_transforms
from geovisnet.utils import (
    calculate_geo_error_with_region_calibration,
    calculate_distance_statistics,
    calculate_accuracy_at_thresholds,
    load_model_weights
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GeoVisNet评估脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--eval_regions', type=str, default='07,08,10,11', help='评估区域')
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--sat_patch_scale', type=float, default=3.0, help='卫星图像缩放比例')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                       choices=['efficientnet_b0', 'efficientnet_b2'], help='骨干网络')
    parser.add_argument('--emb_size', type=int, default=256, help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作线程数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True, help='结果输出目录')
    parser.add_argument('--generate_plots', action='store_true', help='生成可视化图表')
    parser.add_argument('--save_attention', action='store_true', help='保存注意力图')
    
    return parser.parse_args()


def evaluate_model(model, data_loader, device, sat_coords_df, args):
    """评估模型性能"""
    model.eval()
    
    results = {
        'predictions': [],
        'targets': [],
        'geo_errors': [],
        'regions': [],
        'indices': [],
        'attention_maps': [] if args.save_attention else None
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f'评估进度: {batch_idx+1}/{len(data_loader)}', end='\r')
            
            # 数据移动到设备
            drone_imgs = batch['drone_img'].to(device, non_blocking=True)
            sat_imgs = batch['sat_img'].to(device, non_blocking=True)
            norm_lat = batch['norm_lat'].to(device, non_blocking=True)
            norm_lon = batch['norm_lon'].to(device, non_blocking=True)
            regions = batch['region']
            indices = batch['idx']
            
            # 前向传播
            _, _, geo_preds = model(drone_imgs, sat_imgs)
            
            # 保存注意力图（如果需要）
            if args.save_attention and batch_idx < 5:  # 只保存前5个批次的注意力图
                attention_maps = model.get_attention_maps(drone_imgs, sat_imgs)
                results['attention_maps'].append(attention_maps)
            
            # 转换为CPU numpy数组
            pred_coords = geo_preds.cpu().numpy()
            true_coords = torch.stack([norm_lat, norm_lon], dim=1).cpu().numpy()
            
            # 计算地理误差
            for i in range(len(pred_coords)):
                pred_lat, pred_lon = pred_coords[i]
                true_lat, true_lon = true_coords[i]
                region = regions[i]
                
                # 获取卫星图像信息
                sat_name = f'satellite{region}.tif'
                sat_info = sat_coords_df[sat_coords_df['mapname'] == sat_name]
                
                if len(sat_info) > 0:
                    sat_info_dict = {
                        'LT_lat_map': sat_info['LT_lat_map'].values[0],
                        'LT_lon_map': sat_info['LT_lon_map'].values[0],
                        'RB_lat_map': sat_info['RB_lat_map'].values[0],
                        'RB_lon_map': sat_info['RB_lon_map'].values[0]
                    }
                    
                    # 计算地理误差
                    geo_error, _, _ = calculate_geo_error_with_region_calibration(
                        pred_lat, pred_lon, true_lat, true_lon,
                        sat_info_dict, region, args.img_size
                    )
                    
                    results['geo_errors'].append(geo_error)
                    results['predictions'].append([pred_lat, pred_lon])
                    results['targets'].append([true_lat, true_lon])
                    results['regions'].append(region)
                    results['indices'].append(indices[i].item())
    
    print('\n评估完成!')
    return results


def analyze_results(results):
    """分析评估结果"""
    geo_errors = results['geo_errors']
    regions = results['regions']
    
    # 总体统计
    overall_stats = calculate_distance_statistics(geo_errors)
    overall_acc = calculate_accuracy_at_thresholds(geo_errors)
    
    analysis = {
        'overall': {
            **overall_stats,
            **overall_acc,
            'num_samples': len(geo_errors)
        },
        'by_region': {},
        'error_percentiles': {}
    }
    
    # 按区域分析
    unique_regions = sorted(set(regions))
    for region in unique_regions:
        region_errors = [geo_errors[i] for i, r in enumerate(regions) if r == region]
        if region_errors:
            region_stats = calculate_distance_statistics(region_errors)
            region_acc = calculate_accuracy_at_thresholds(region_errors)
            analysis['by_region'][region] = {
                **region_stats,
                **region_acc,
                'num_samples': len(region_errors)
            }
    
    # 误差百分位数分析
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        analysis['error_percentiles'][f'p{p}'] = np.percentile(geo_errors, p)
    
    return analysis


def generate_plots(results, analysis, output_dir):
    """生成可视化图表"""
    geo_errors = results['geo_errors']
    regions = results['regions']
    
    # 创建图表目录
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. 误差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(geo_errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('地理误差 (米)')
    plt.ylabel('频次')
    plt.title('地理误差分布')
    plt.axvline(np.mean(geo_errors), color='red', linestyle='--', label=f'平均值: {np.mean(geo_errors):.2f}m')
    plt.axvline(np.median(geo_errors), color='green', linestyle='--', label=f'中位数: {np.median(geo_errors):.2f}m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 按区域的误差箱线图
    unique_regions = sorted(set(regions))
    region_errors = [
        [geo_errors[i] for i, r in enumerate(regions) if r == region]
        for region in unique_regions
    ]
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(region_errors, labels=unique_regions)
    plt.xlabel('区域')
    plt.ylabel('地理误差 (米)')
    plt.title('各区域地理误差分布')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'error_by_region.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 累积分布函数
    sorted_errors = np.sort(geo_errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_errors, cumulative_prob * 100)
    plt.xlabel('地理误差 (米)')
    plt.ylabel('累积概率 (%)')
    plt.title('地理误差累积分布函数')
    plt.grid(True, alpha=0.3)
    
    # 添加关键阈值线
    thresholds = [1, 5, 10, 25, 50, 100]
    for threshold in thresholds:
        prob = np.mean(np.array(geo_errors) <= threshold) * 100
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.5)
        plt.text(threshold, prob + 5, f'{threshold}m\n{prob:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    plt.savefig(os.path.join(plots_dir, 'cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 区域性能比较
    region_means = [np.mean([geo_errors[i] for i, r in enumerate(regions) if r == region]) 
                   for region in unique_regions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_regions, region_means, alpha=0.7)
    plt.xlabel('区域')
    plt.ylabel('平均地理误差 (米)')
    plt.title('各区域平均地理误差')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, mean_error in zip(bars, region_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mean_error:.1f}m', ha='center', va='bottom')
    
    plt.savefig(os.path.join(plots_dir, 'region_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {plots_dir}")


def save_results(results, analysis, args):
    """保存评估结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存分析结果
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # 保存详细结果
    detailed_results = []
    for i in range(len(results['predictions'])):
        detailed_results.append({
            'index': results['indices'][i],
            'region': results['regions'][i],
            'pred_lat': results['predictions'][i][0],
            'pred_lon': results['predictions'][i][1],
            'true_lat': results['targets'][i][0],
            'true_lon': results['targets'][i][1],
            'geo_error_m': results['geo_errors'][i]
        })
    
    df = pd.DataFrame(detailed_results)
    df.to_csv(os.path.join(args.output_dir, 'detailed_evaluation.csv'), index=False)
    
    # 保存注意力图（如果有）
    if args.save_attention and results['attention_maps']:
        attention_dir = os.path.join(args.output_dir, 'attention_maps')
        os.makedirs(attention_dir, exist_ok=True)
        torch.save(results['attention_maps'], os.path.join(attention_dir, 'attention_maps.pth'))
    
    print(f"评估结果已保存到: {args.output_dir}")


def print_analysis(analysis):
    """打印分析结果"""
    print("\n" + "="*80)
    print("GeoVisNet模型评估结果")
    print("="*80)
    
    # 总体结果
    overall = analysis['overall']
    print(f"\n总体性能 (样本数: {overall['num_samples']}):")
    print(f"  平均误差: {overall['mean']:.2f} ± {overall['std']:.2f} m")
    print(f"  中位误差: {overall['median']:.2f} m")
    print(f"  最大误差: {overall['max']:.2f} m")
    print(f"  最小误差: {overall['min']:.2f} m")
    
    print(f"\n准确率指标:")
    accuracy_keys = [k for k in overall.keys() if k.startswith('acc@')]
    for key in sorted(accuracy_keys):
        threshold = key.split('@')[1]
        print(f"  {threshold}: {overall[key]:.2f}%")
    
    # 按区域结果
    if analysis['by_region']:
        print(f"\n按区域性能:")
        for region, data in sorted(analysis['by_region'].items()):
            print(f"  区域 {region} (样本数: {data['num_samples']}):")
            print(f"    平均误差: {data['mean']:.2f} m")
            print(f"    中位误差: {data['median']:.2f} m")
            print(f"    准确率@25m: {data.get('acc@25m', 0):.2f}%")
    
    # 误差百分位数
    print(f"\n误差百分位数:")
    for key, value in analysis['error_percentiles'].items():
        percentile = key[1:]  # 去掉'p'前缀
        print(f"  {percentile}%: {value:.2f} m")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载卫星图像坐标信息
    sat_coords_path = os.path.join(args.data_root, 'satellite_coordinates_range.csv')
    if not os.path.exists(sat_coords_path):
        sat_coords_path = os.path.join(args.data_root, 'satellite_ coordinates_range.csv')
    sat_coords_df = pd.read_csv(sat_coords_path)
    
    # 创建数据变换
    eval_transform = get_val_transforms(args.img_size)
    
    # 解析评估区域
    eval_regions = args.eval_regions.split(',')
    
    # 创建评估数据集
    eval_dataset = UAVVisLocDataset(
        data_root=args.data_root,
        regions=eval_regions,
        img_size=args.img_size,
        transform=eval_transform,
        augment=False,
        mode='test',
        cache_satellite=True,
        sat_patch_scale=args.sat_patch_scale
    )
    
    # 创建数据加载器
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = GeoVisNet(
        emb_size=args.emb_size,
        backbone=args.backbone,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 加载模型权重
    if not load_model_weights(model, args.model_path, map_location=device):
        print(f"无法加载模型权重: {args.model_path}")
        return
    
    print(f"成功加载模型: {args.model_path}")
    print(f"评估数据集大小: {len(eval_dataset)}")
    print(f"评估区域: {eval_regions}")
    
    # 评估模型
    results = evaluate_model(model, eval_loader, device, sat_coords_df, args)
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 打印结果
    print_analysis(analysis)
    
    # 保存结果
    save_results(results, analysis, args)
    
    # 生成可视化图表
    if args.generate_plots:
        generate_plots(results, analysis, args.output_dir)


if __name__ == '__main__':
    main()
