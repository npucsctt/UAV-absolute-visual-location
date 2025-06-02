#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进的可视化脚本
解决黑色边框问题，生成更好的对比图像
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geovisnet.data.dataset import UAVVisLocDataset
from geovisnet.models.geovisnet import GeoVisNet
from geovisnet.utils.model_utils import load_model_weights
from geovisnet.visualization.attention_maps import visualize_attention
from geovisnet.visualization.dataset_viz import visualize_dataset_sample
from geovisnet.visualization.results import plot_coordinate_distribution


def get_valid_satellite_regions(sat_path, img_size, sat_patch_scale, min_valid_ratio=0.8):
    """
    获取卫星图像中有效的区域（避免黑色边框）
    
    Args:
        sat_path (str): 卫星图像路径
        img_size (int): 目标图像大小
        sat_patch_scale (float): 卫星图像裁剪缩放比例
        min_valid_ratio (float): 最小有效像素比例
        
    Returns:
        list: 有效区域的归一化坐标列表 [(norm_lat, norm_lon), ...]
    """
    try:
        # 读取卫星图像
        satellite_img = cv2.imread(sat_path)
        if satellite_img is None:
            img = Image.open(sat_path)
            satellite_img = np.array(img)
            if len(satellite_img.shape) == 2:
                satellite_img = np.stack([satellite_img] * 3, axis=2)
            elif satellite_img.shape[2] > 3:
                satellite_img = satellite_img[:, :, :3]
        
        if satellite_img.shape[2] == 3 and satellite_img.dtype == np.uint8:
            satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        
        h, w = satellite_img.shape[:2]
        initial_patch_size = int(img_size * sat_patch_scale)
        half_size = initial_patch_size // 2
        
        valid_regions = []
        
        # 在图像中心区域寻找有效区域
        # 避免边缘区域以防止黑色边框
        margin = half_size + 50  # 额外的边距
        
        # 生成候选区域
        num_candidates = 100
        for _ in range(num_candidates):
            # 在安全区域内随机选择中心点
            center_y = random.randint(margin, h - margin)
            center_x = random.randint(margin, w - margin)
            
            # 计算裁剪区域
            y1 = center_y - half_size
            x1 = center_x - half_size
            y2 = center_y + half_size
            x2 = center_x + half_size
            
            # 确保区域完全在图像内
            if y1 >= 0 and x1 >= 0 and y2 <= h and x2 <= w:
                # 提取图像块
                patch = satellite_img[y1:y2, x1:x2]
                
                # 检查图像块的有效性
                # 计算非零像素的比例
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                valid_pixels = np.sum(gray_patch > 10)  # 排除接近黑色的像素
                total_pixels = gray_patch.size
                valid_ratio = valid_pixels / total_pixels
                
                # 检查图像块的方差（避免纯色区域）
                variance = np.var(gray_patch)
                
                if valid_ratio >= min_valid_ratio and variance > 100:
                    # 转换为归一化坐标
                    norm_lat = center_y / h
                    norm_lon = center_x / w
                    
                    # 计算图像质量分数
                    quality_score = valid_ratio * np.log(variance + 1)
                    
                    valid_regions.append({
                        'norm_lat': norm_lat,
                        'norm_lon': norm_lon,
                        'quality_score': quality_score,
                        'center_y': center_y,
                        'center_x': center_x
                    })
        
        # 按质量分数排序
        valid_regions.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return valid_regions[:20]  # 返回前20个最佳区域
        
    except Exception as e:
        print(f"处理卫星图像时出错 {sat_path}: {e}")
        return []


def create_improved_satellite_patch(sat_path, norm_lat, norm_lon, img_size, sat_patch_scale):
    """
    创建改进的卫星图像块，避免黑色边框
    
    Args:
        sat_path (str): 卫星图像路径
        norm_lat (float): 归一化纬度
        norm_lon (float): 归一化经度
        img_size (int): 目标图像大小
        sat_patch_scale (float): 缩放比例
        
    Returns:
        np.ndarray: 处理后的卫星图像块
    """
    try:
        # 读取卫星图像
        satellite_img = cv2.imread(sat_path)
        if satellite_img is None:
            img = Image.open(sat_path)
            satellite_img = np.array(img)
            if len(satellite_img.shape) == 2:
                satellite_img = np.stack([satellite_img] * 3, axis=2)
            elif satellite_img.shape[2] > 3:
                satellite_img = satellite_img[:, :, :3]
        
        if satellite_img.shape[2] == 3 and satellite_img.dtype == np.uint8:
            satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        
        h, w = satellite_img.shape[:2]
        initial_patch_size = int(img_size * sat_patch_scale)
        
        # 计算中心点坐标
        center_y = int(norm_lat * h)
        center_x = int(norm_lon * w)
        
        # 计算裁剪区域
        half_size = initial_patch_size // 2
        y1 = center_y - half_size
        x1 = center_x - half_size
        y2 = center_y + half_size
        x2 = center_x + half_size
        
        # 确保裁剪区域在图像边界内
        y1 = max(0, min(y1, h - initial_patch_size))
        x1 = max(0, min(x1, w - initial_patch_size))
        y2 = y1 + initial_patch_size
        x2 = x1 + initial_patch_size
        
        # 如果调整后的区域超出边界，进一步调整
        if y2 > h:
            y2 = h
            y1 = max(0, y2 - initial_patch_size)
        if x2 > w:
            x2 = w
            x1 = max(0, x2 - initial_patch_size)
        
        # 提取图像块
        patch = satellite_img[y1:y2, x1:x2].copy()
        
        # 如果提取的图像块大小不足，使用边缘填充而不是零填充
        if patch.shape[0] < initial_patch_size or patch.shape[1] < initial_patch_size:
            # 使用边缘像素进行填充
            patch = cv2.copyMakeBorder(
                patch,
                max(0, (initial_patch_size - patch.shape[0]) // 2),
                max(0, initial_patch_size - patch.shape[0] - (initial_patch_size - patch.shape[0]) // 2),
                max(0, (initial_patch_size - patch.shape[1]) // 2),
                max(0, initial_patch_size - patch.shape[1] - (initial_patch_size - patch.shape[1]) // 2),
                cv2.BORDER_REFLECT_101
            )
        
        # 调整到目标大小
        if patch.shape[0] != img_size or patch.shape[1] != img_size:
            patch = cv2.resize(patch, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        return patch
        
    except Exception as e:
        print(f"创建卫星图像块时出错: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)


def create_comparison_visualization(drone_img, sat_img, region, sample_idx, pred_coords=None, 
                                  true_coords=None, geo_error=None, save_path=None):
    """
    创建改进的对比可视化图像
    
    Args:
        drone_img (np.ndarray): 无人机图像
        sat_img (np.ndarray): 卫星图像
        region (str): 区域编号
        sample_idx (int): 样本索引
        pred_coords (tuple): 预测坐标 (lat, lon)
        true_coords (tuple): 真实坐标 (lat, lon)
        geo_error (float): 地理误差（米）
        save_path (str): 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 显示无人机图像
    axes[0].imshow(drone_img)
    axes[0].set_title(f'无人机图像 (区域 {region})', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 显示卫星图像
    axes[1].imshow(sat_img)
    
    # 如果有预测和真实坐标，在卫星图像上标注
    if pred_coords and true_coords:
        h, w = sat_img.shape[:2]
        pred_y, pred_x = int(pred_coords[0] * h), int(pred_coords[1] * w)
        true_y, true_x = int(true_coords[0] * h), int(true_coords[1] * w)
        
        # 绘制预测点和真实点
        axes[1].scatter(pred_x, pred_y, c='red', s=100, marker='x', linewidth=3, label='预测位置')
        axes[1].scatter(true_x, true_y, c='green', s=100, marker='o', linewidth=2, label='真实位置')
        
        # 绘制连接线
        axes[1].plot([pred_x, true_x], [pred_y, true_y], 'b--', linewidth=2, alpha=0.7)
        
        axes[1].legend(fontsize=10)
        
        if geo_error is not None:
            axes[1].set_title(f'卫星图像 (地理误差: {geo_error:.2f}m)', fontsize=14, fontweight='bold')
        else:
            axes[1].set_title('卫星图像', fontsize=14, fontweight='bold')
    else:
        axes[1].set_title('卫星图像', fontsize=14, fontweight='bold')
    
    axes[1].axis('off')
    
    # 添加整体标题
    if geo_error is not None:
        plt.suptitle(f'区域 {region} - 样本 {sample_idx} (误差: {geo_error:.2f}m)', 
                    fontsize=16, fontweight='bold')
    else:
        plt.suptitle(f'区域 {region} - 样本 {sample_idx}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"可视化图像已保存到: {save_path}")
    
    plt.close()
