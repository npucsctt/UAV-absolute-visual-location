#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet测试脚本

使用训练好的模型进行测试和评估
"""

import os
import sys
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description='GeoVisNet测试脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--test_regions', type=str, default='08,10,11', help='测试区域')
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--sat_patch_scale', type=float, default=3.0, help='卫星图像缩放比例')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                       choices=['efficientnet_b0', 'efficientnet_b2'], help='骨干网络')
    parser.add_argument('--emb_size', type=int, default=256, help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作线程数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True, help='结果输出目录')
    parser.add_argument('--save_predictions', action='store_true', help='保存预测结果')
    parser.add_argument('--save_detailed_results', action='store_true', help='保存详细结果')
    
    return parser.parse_args()


def load_satellite_info(data_root):
    """加载卫星图像坐标信息"""
    sat_coords_path = os.path.join(data_root, 'satellite_coordinates_range.csv')
    if not os.path.exists(sat_coords_path):
        sat_coords_path = os.path.join(data_root, 'satellite_ coordinates_range.csv')
    
    return pd.read_csv(sat_coords_path)


def test_model(model, data_loader, device, sat_coords_df, args):
    """测试模型"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_geo_errors = []
    all_regions = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='测试中'):
            # 数据移动到设备
            drone_imgs = batch['drone_img'].to(device, non_blocking=True)
            sat_imgs = batch['sat_img'].to(device, non_blocking=True)
            norm_lat = batch['norm_lat'].to(device, non_blocking=True)
            norm_lon = batch['norm_lon'].to(device, non_blocking=True)
            regions = batch['region']
            indices = batch['idx']
            
            # 前向传播
            _, _, geo_preds = model(drone_imgs, sat_imgs)
            
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
                    
                    all_geo_errors.append(geo_error)
                    all_predictions.append([pred_lat, pred_lon])
                    all_targets.append([true_lat, true_lon])
                    all_regions.append(region)
                    all_indices.append(indices[i].item())
    
    return all_predictions, all_targets, all_geo_errors, all_regions, all_indices


def calculate_metrics(geo_errors, regions=None):
    """计算评估指标"""
    # 总体统计
    overall_stats = calculate_distance_statistics(geo_errors)
    overall_acc = calculate_accuracy_at_thresholds(geo_errors)
    
    results = {
        'overall': {
            **overall_stats,
            **overall_acc,
            'num_samples': len(geo_errors)
        }
    }
    
    # 按区域统计
    if regions is not None:
        unique_regions = sorted(set(regions))
        for region in unique_regions:
            region_errors = [geo_errors[i] for i, r in enumerate(regions) if r == region]
            if region_errors:
                region_stats = calculate_distance_statistics(region_errors)
                region_acc = calculate_accuracy_at_thresholds(region_errors)
                results[f'region_{region}'] = {
                    **region_stats,
                    **region_acc,
                    'num_samples': len(region_errors)
                }
    
    return results


def save_results(results, predictions, targets, geo_errors, regions, indices, args):
    """保存测试结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存统计结果
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存预测结果
    if args.save_predictions:
        predictions_data = {
            'predictions': predictions,
            'targets': targets,
            'geo_errors': geo_errors,
            'regions': regions,
            'indices': indices
        }
        
        torch.save(predictions_data, os.path.join(args.output_dir, 'predictions.pth'))
    
    # 保存详细结果CSV
    if args.save_detailed_results:
        detailed_results = []
        for i in range(len(predictions)):
            detailed_results.append({
                'index': indices[i],
                'region': regions[i],
                'pred_lat': predictions[i][0],
                'pred_lon': predictions[i][1],
                'true_lat': targets[i][0],
                'true_lon': targets[i][1],
                'geo_error_m': geo_errors[i]
            })
        
        df = pd.DataFrame(detailed_results)
        df.to_csv(os.path.join(args.output_dir, 'detailed_results.csv'), index=False)


def print_results(results):
    """打印测试结果"""
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    
    # 总体结果
    overall = results['overall']
    print(f"\n总体结果 (样本数: {overall['num_samples']}):")
    print(f"  平均误差: {overall['mean']:.2f} m")
    print(f"  中位误差: {overall['median']:.2f} m")
    print(f"  标准差: {overall['std']:.2f} m")
    print(f"  最大误差: {overall['max']:.2f} m")
    print(f"  最小误差: {overall['min']:.2f} m")
    
    print(f"\n准确率:")
    for key, value in overall.items():
        if key.startswith('acc@'):
            threshold = key.split('@')[1]
            print(f"  {threshold}: {value:.2f}%")
    
    # 按区域结果
    region_keys = [k for k in results.keys() if k.startswith('region_')]
    if region_keys:
        print(f"\n按区域结果:")
        for region_key in sorted(region_keys):
            region_data = results[region_key]
            region_name = region_key.split('_')[1]
            print(f"\n  区域 {region_name} (样本数: {region_data['num_samples']}):")
            print(f"    平均误差: {region_data['mean']:.2f} m")
            print(f"    中位误差: {region_data['median']:.2f} m")
            print(f"    准确率@25m: {region_data.get('acc@25m', 0):.2f}%")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载卫星图像坐标信息
    sat_coords_df = load_satellite_info(args.data_root)
    
    # 创建数据变换
    test_transform = get_val_transforms(args.img_size)
    
    # 解析测试区域
    test_regions = args.test_regions.split(',')
    
    # 创建测试数据集
    test_dataset = UAVVisLocDataset(
        data_root=args.data_root,
        regions=test_regions,
        img_size=args.img_size,
        transform=test_transform,
        augment=False,
        mode='test',
        cache_satellite=True,
        sat_patch_scale=args.sat_patch_scale
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
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
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 测试模型
    predictions, targets, geo_errors, regions, indices = test_model(
        model, test_loader, device, sat_coords_df, args
    )
    
    # 计算指标
    results = calculate_metrics(geo_errors, regions)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    save_results(results, predictions, targets, geo_errors, regions, indices, args)
    
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
