#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
注意力可视化模块

提供注意力图的可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cv2


def visualize_attention(drone_img, sat_img, attention_maps, save_path=None, figsize=(15, 10)):
    """
    可视化注意力图
    
    Args:
        drone_img (torch.Tensor): 无人机图像 [3, H, W]
        sat_img (torch.Tensor): 卫星图像 [3, H, W]
        attention_maps (dict): 注意力图字典
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    # 转换图像为numpy格式
    drone_np = tensor_to_numpy(drone_img)
    sat_np = tensor_to_numpy(sat_img)
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle('GeoVisNet注意力可视化', fontsize=16)
    
    # 显示原始图像
    axes[0, 0].imshow(drone_np)
    axes[0, 0].set_title('无人机图像')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(sat_np)
    axes[1, 0].set_title('卫星图像')
    axes[1, 0].axis('off')
    
    # 显示注意力图
    attention_titles = [
        ('query_eca', '无人机ECA注意力'),
        ('query_spatial', '无人机空间注意力'),
        ('reference_eca', '卫星ECA注意力'),
        ('reference_spatial', '卫星空间注意力')
    ]
    
    for i, (key, title) in enumerate(attention_titles):
        if key in attention_maps:
            row = i // 2
            col = (i % 2) + 1
            
            # 获取注意力图
            attention = attention_maps[key][0]  # 取第一个样本
            
            # 如果是多通道，取平均
            if len(attention.shape) == 3:
                attention = attention.mean(dim=0)
            
            # 转换为numpy
            attention_np = attention.cpu().numpy()
            
            # 显示注意力热图
            im = axes[row, col].imshow(attention_np, cmap='hot', alpha=0.7)
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # 显示叠加图像
    if 'query_spatial' in attention_maps:
        overlay_drone = create_attention_overlay(drone_np, attention_maps['query_spatial'][0])
        axes[0, 3].imshow(overlay_drone)
        axes[0, 3].set_title('无人机+注意力叠加')
        axes[0, 3].axis('off')
    
    if 'reference_spatial' in attention_maps:
        overlay_sat = create_attention_overlay(sat_np, attention_maps['reference_spatial'][0])
        axes[1, 3].imshow(overlay_sat)
        axes[1, 3].set_title('卫星+注意力叠加')
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {save_path}")
    
    plt.show()


def plot_attention_heatmap(attention_map, title="注意力热图", save_path=None, figsize=(8, 6)):
    """
    绘制单个注意力热图
    
    Args:
        attention_map (torch.Tensor): 注意力图 [H, W]
        title (str): 标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    # 转换为numpy
    if isinstance(attention_map, torch.Tensor):
        attention_np = attention_map.cpu().numpy()
    else:
        attention_np = attention_map
    
    # 如果是多通道，取平均
    if len(attention_np.shape) == 3:
        attention_np = attention_np.mean(axis=0)
    
    plt.figure(figsize=figsize)
    plt.imshow(attention_np, cmap='hot', interpolation='bilinear')
    plt.colorbar(label='注意力强度')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力热图已保存到: {save_path}")
    
    plt.show()


def create_attention_overlay(image, attention_map, alpha=0.4):
    """
    创建图像和注意力图的叠加
    
    Args:
        image (np.ndarray): 原始图像 [H, W, 3]
        attention_map (torch.Tensor): 注意力图
        alpha (float): 注意力图透明度
        
    Returns:
        np.ndarray: 叠加后的图像
    """
    # 转换注意力图
    if isinstance(attention_map, torch.Tensor):
        attention_np = attention_map.cpu().numpy()
    else:
        attention_np = attention_map
    
    # 如果是多通道，取平均
    if len(attention_np.shape) == 3:
        attention_np = attention_np.mean(axis=0)
    
    # 归一化注意力图到0-1
    attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
    
    # 调整注意力图大小以匹配图像
    if attention_np.shape != image.shape[:2]:
        attention_np = cv2.resize(attention_np, (image.shape[1], image.shape[0]))
    
    # 创建热图
    colormap = cm.get_cmap('hot')
    heatmap = colormap(attention_np)[:, :, :3]  # 去掉alpha通道
    
    # 叠加图像
    overlay = image * (1 - alpha) + heatmap * alpha
    
    return np.clip(overlay, 0, 1)


def tensor_to_numpy(tensor):
    """
    将张量转换为numpy图像
    
    Args:
        tensor (torch.Tensor): 图像张量 [3, H, W]
        
    Returns:
        np.ndarray: 图像数组 [H, W, 3]
    """
    if isinstance(tensor, torch.Tensor):
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # 转换为numpy并调整维度
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # 反归一化
        img = img * std + mean
        
        # 限制到0-1范围
        img = np.clip(img, 0, 1)
    else:
        img = tensor
    
    return img


def compare_attention_maps(attention_maps_list, titles, save_path=None, figsize=(15, 5)):
    """
    比较多个注意力图
    
    Args:
        attention_maps_list (list): 注意力图列表
        titles (list): 标题列表
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    num_maps = len(attention_maps_list)
    fig, axes = plt.subplots(1, num_maps, figsize=figsize)
    
    if num_maps == 1:
        axes = [axes]
    
    for i, (attention_map, title) in enumerate(zip(attention_maps_list, titles)):
        # 转换为numpy
        if isinstance(attention_map, torch.Tensor):
            attention_np = attention_map.cpu().numpy()
        else:
            attention_np = attention_map
        
        # 如果是多通道，取平均
        if len(attention_np.shape) == 3:
            attention_np = attention_np.mean(axis=0)
        
        # 显示注意力图
        im = axes[i].imshow(attention_np, cmap='hot', interpolation='bilinear')
        axes[i].set_title(title)
        axes[i].axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力比较图已保存到: {save_path}")
    
    plt.show()


def visualize_attention_evolution(attention_maps_sequence, save_path=None, figsize=(15, 10)):
    """
    可视化注意力图的演化过程（例如训练过程中的变化）
    
    Args:
        attention_maps_sequence (list): 注意力图序列
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    num_steps = len(attention_maps_sequence)
    fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=figsize)
    fig.suptitle('注意力图演化过程', fontsize=16)
    
    if num_steps == 1:
        axes = axes.reshape(1, -1)
    
    for i, attention_map in enumerate(attention_maps_sequence):
        row = i // ((num_steps + 1) // 2)
        col = i % ((num_steps + 1) // 2)
        
        # 转换为numpy
        if isinstance(attention_map, torch.Tensor):
            attention_np = attention_map.cpu().numpy()
        else:
            attention_np = attention_map
        
        # 如果是多通道，取平均
        if len(attention_np.shape) == 3:
            attention_np = attention_np.mean(axis=0)
        
        # 显示注意力图
        im = axes[row, col].imshow(attention_np, cmap='hot', interpolation='bilinear')
        axes[row, col].set_title(f'Step {i+1}')
        axes[row, col].axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(num_steps, axes.size):
        row = i // ((num_steps + 1) // 2)
        col = i % ((num_steps + 1) // 2)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力演化图已保存到: {save_path}")
    
    plt.show()
