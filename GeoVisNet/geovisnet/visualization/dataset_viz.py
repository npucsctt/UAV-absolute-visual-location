#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集可视化模块

提供数据集样本和统计信息的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import cv2


def visualize_dataset_sample(drone_img, sat_img, drone_coords, sat_coords, 
                           region=None, save_path=None, figsize=(15, 6)):
    """
    可视化数据集样本
    
    Args:
        drone_img (np.ndarray): 无人机图像 [H, W, 3]
        sat_img (np.ndarray): 卫星图像 [H, W, 3]
        drone_coords (tuple): 无人机坐标 (lat, lon)
        sat_coords (dict): 卫星图像坐标范围
        region (str): 区域编号
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 标题
    title = f"数据集样本可视化"
    if region:
        title += f" - 区域 {region}"
    fig.suptitle(title, fontsize=16)
    
    # 1. 无人机图像
    axes[0].imshow(drone_img)
    axes[0].set_title(f'无人机图像\n坐标: ({drone_coords[0]:.6f}, {drone_coords[1]:.6f})')
    axes[0].axis('off')
    
    # 2. 卫星图像
    axes[1].imshow(sat_img)
    axes[1].set_title('对应卫星图像块')
    axes[1].axis('off')
    
    # 3. 坐标信息图
    axes[2].text(0.1, 0.9, '坐标信息:', fontsize=14, fontweight='bold', transform=axes[2].transAxes)
    
    info_text = f"""
无人机位置:
  纬度: {drone_coords[0]:.6f}°
  经度: {drone_coords[1]:.6f}°

卫星图像范围:
  左上角: ({sat_coords.get('LT_lat_map', 'N/A'):.6f}°, {sat_coords.get('LT_lon_map', 'N/A'):.6f}°)
  右下角: ({sat_coords.get('RB_lat_map', 'N/A'):.6f}°, {sat_coords.get('RB_lon_map', 'N/A'):.6f}°)

图像信息:
  无人机图像大小: {drone_img.shape[:2]}
  卫星图像大小: {sat_img.shape[:2]}
    """
    
    axes[2].text(0.1, 0.8, info_text, fontsize=10, transform=axes[2].transAxes, 
                verticalalignment='top', fontfamily='monospace')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据集样本可视化已保存到: {save_path}")
    
    plt.show()


def plot_coordinate_distribution(coordinates, regions=None, title="坐标分布图", 
                                save_path=None, figsize=(12, 8)):
    """
    绘制坐标分布图
    
    Args:
        coordinates (list): 坐标列表 [(lat, lon), ...]
        regions (list): 对应的区域列表
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    coordinates = np.array(coordinates)
    lats = coordinates[:, 0]
    lons = coordinates[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 地理分布散点图
    if regions is not None:
        unique_regions = sorted(set(regions))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
        
        for i, region in enumerate(unique_regions):
            mask = np.array(regions) == region
            axes[0, 0].scatter(lons[mask], lats[mask], c=[colors[i]], 
                             label=f'区域 {region}', alpha=0.7, s=20)
        axes[0, 0].legend()
    else:
        axes[0, 0].scatter(lons, lats, alpha=0.7, s=20)
    
    axes[0, 0].set_xlabel('经度')
    axes[0, 0].set_ylabel('纬度')
    axes[0, 0].set_title('地理坐标分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 纬度分布直方图
    axes[0, 1].hist(lats, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('纬度')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('纬度分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 经度分布直方图
    axes[1, 0].hist(lons, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('经度')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('经度分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 密度热图
    axes[1, 1].hist2d(lons, lats, bins=20, cmap='Blues')
    axes[1, 1].set_xlabel('经度')
    axes[1, 1].set_ylabel('纬度')
    axes[1, 1].set_title('坐标密度热图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"坐标分布图已保存到: {save_path}")
    
    plt.show()


def visualize_data_augmentation(original_img, augmented_imgs, aug_names, 
                               title="数据增强效果", save_path=None, figsize=(15, 10)):
    """
    可视化数据增强效果
    
    Args:
        original_img (np.ndarray): 原始图像
        augmented_imgs (list): 增强后的图像列表
        aug_names (list): 增强方法名称列表
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    num_augs = len(augmented_imgs)
    cols = min(4, num_augs + 1)
    rows = (num_augs + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # 显示原始图像
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 显示增强后的图像
    for i, (aug_img, aug_name) in enumerate(zip(augmented_imgs, aug_names)):
        row = (i + 1) // cols
        col = (i + 1) % cols
        
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(aug_name)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    total_plots = num_augs + 1
    for i in range(total_plots, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据增强可视化已保存到: {save_path}")
    
    plt.show()


def plot_dataset_statistics(dataset_stats, title="数据集统计信息", 
                           save_path=None, figsize=(15, 10)):
    """
    绘制数据集统计信息
    
    Args:
        dataset_stats (dict): 数据集统计信息
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 各区域样本数量
    if 'samples_per_region' in dataset_stats:
        regions = list(dataset_stats['samples_per_region'].keys())
        counts = list(dataset_stats['samples_per_region'].values())
        
        bars = axes[0, 0].bar(regions, counts, alpha=0.7)
        axes[0, 0].set_xlabel('区域')
        axes[0, 0].set_ylabel('样本数量')
        axes[0, 0].set_title('各区域样本分布')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.01,
                           str(count), ha='center', va='bottom')
    
    # 2. 训练/验证/测试集分布
    if 'split_distribution' in dataset_stats:
        splits = list(dataset_stats['split_distribution'].keys())
        split_counts = list(dataset_stats['split_distribution'].values())
        
        axes[0, 1].pie(split_counts, labels=splits, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('数据集划分')
    
    # 3. 坐标范围
    if 'coordinate_ranges' in dataset_stats:
        coord_ranges = dataset_stats['coordinate_ranges']
        
        # 纬度范围
        lat_range = coord_ranges.get('latitude', [0, 0])
        lon_range = coord_ranges.get('longitude', [0, 0])
        
        axes[0, 2].barh(['纬度', '经度'], [lat_range[1] - lat_range[0], lon_range[1] - lon_range[0]])
        axes[0, 2].set_xlabel('坐标范围 (度)')
        axes[0, 2].set_title('坐标覆盖范围')
        axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    # 4. 图像尺寸分布
    if 'image_sizes' in dataset_stats:
        sizes = dataset_stats['image_sizes']
        size_labels = [f"{w}x{h}" for w, h in sizes.keys()]
        size_counts = list(sizes.values())
        
        axes[1, 0].bar(range(len(size_labels)), size_counts, alpha=0.7)
        axes[1, 0].set_xlabel('图像尺寸')
        axes[1, 0].set_ylabel('数量')
        axes[1, 0].set_title('图像尺寸分布')
        axes[1, 0].set_xticks(range(len(size_labels)))
        axes[1, 0].set_xticklabels(size_labels, rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. 数据质量指标
    if 'quality_metrics' in dataset_stats:
        quality = dataset_stats['quality_metrics']
        metrics = list(quality.keys())
        values = list(quality.values())
        
        axes[1, 1].bar(metrics, values, alpha=0.7, color='lightgreen')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].set_title('数据质量指标')
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. 存储信息
    if 'storage_info' in dataset_stats:
        storage = dataset_stats['storage_info']
        
        info_text = f"""
总文件数: {storage.get('total_files', 'N/A')}
总大小: {storage.get('total_size_gb', 'N/A'):.2f} GB
平均文件大小: {storage.get('avg_file_size_mb', 'N/A'):.2f} MB

无人机图像: {storage.get('drone_images', 'N/A')} 张
卫星图像: {storage.get('satellite_images', 'N/A')} 张
CSV文件: {storage.get('csv_files', 'N/A')} 个
        """
        
        axes[1, 2].text(0.1, 0.9, info_text, fontsize=10, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('存储信息')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据集统计图已保存到: {save_path}")
    
    plt.show()


def visualize_satellite_coverage(sat_coords_df, drone_coords=None, 
                                title="卫星图像覆盖范围", save_path=None, figsize=(12, 8)):
    """
    可视化卫星图像覆盖范围
    
    Args:
        sat_coords_df (pd.DataFrame): 卫星坐标数据框
        drone_coords (list): 无人机坐标列表 [(lat, lon), ...]
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制卫星图像覆盖范围
    colors = plt.cm.tab10(np.linspace(0, 1, len(sat_coords_df)))
    
    for i, (_, row) in enumerate(sat_coords_df.iterrows()):
        lt_lat = row['LT_lat_map']
        lt_lon = row['LT_lon_map']
        rb_lat = row['RB_lat_map']
        rb_lon = row['RB_lon_map']
        
        # 创建矩形
        width = rb_lon - lt_lon
        height = lt_lat - rb_lat
        
        rect = patches.Rectangle((lt_lon, rb_lat), width, height,
                               linewidth=2, edgecolor=colors[i], 
                               facecolor=colors[i], alpha=0.3,
                               label=row['mapname'])
        ax.add_patch(rect)
        
        # 添加区域标签
        center_lat = (lt_lat + rb_lat) / 2
        center_lon = (lt_lon + rb_lon) / 2
        region_num = row['mapname'].replace('satellite', '').replace('.tif', '')
        ax.text(center_lon, center_lat, region_num, ha='center', va='center',
               fontsize=12, fontweight='bold', color='black')
    
    # 绘制无人机位置点
    if drone_coords is not None:
        drone_coords = np.array(drone_coords)
        ax.scatter(drone_coords[:, 1], drone_coords[:, 0], 
                  c='red', s=10, alpha=0.6, label='无人机位置')
    
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"卫星覆盖范围图已保存到: {save_path}")
    
    plt.show()
