#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果可视化模块

提供训练和测试结果的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


def plot_error_distribution(errors, title="地理误差分布", save_path=None, figsize=(12, 8)):
    """
    绘制误差分布图
    
    Args:
        errors (list): 误差列表（米）
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 直方图
    axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(errors):.2f}m')
    axes[0, 0].axvline(np.median(errors), color='green', linestyle='--', 
                       label=f'中位数: {np.median(errors):.2f}m')
    axes[0, 0].set_xlabel('地理误差 (米)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('误差直方图')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 箱线图
    axes[0, 1].boxplot(errors, vert=True)
    axes[0, 1].set_ylabel('地理误差 (米)')
    axes[0, 1].set_title('误差箱线图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 累积分布函数
    sorted_errors = np.sort(errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 0].plot(sorted_errors, cumulative_prob * 100, linewidth=2)
    axes[1, 0].set_xlabel('地理误差 (米)')
    axes[1, 0].set_ylabel('累积概率 (%)')
    axes[1, 0].set_title('累积分布函数')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加关键阈值线
    thresholds = [1, 5, 10, 25, 50, 100]
    for threshold in thresholds:
        prob = np.mean(np.array(errors) <= threshold) * 100
        if prob > 5:  # 只显示概率大于5%的阈值
            axes[1, 0].axvline(threshold, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].text(threshold, prob + 2, f'{threshold}m\n{prob:.1f}%', 
                           ha='center', va='bottom', fontsize=8)
    
    # 4. 对数尺度直方图
    axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[1, 1].set_xlabel('地理误差 (米)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('误差直方图 (对数尺度)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"误差分布图已保存到: {save_path}")
    
    plt.show()


def plot_region_comparison(errors_by_region, title="各区域性能比较", save_path=None, figsize=(14, 8)):
    """
    绘制各区域性能比较图
    
    Args:
        errors_by_region (dict): 按区域分组的误差字典
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    regions = list(errors_by_region.keys())
    region_errors = list(errors_by_region.values())
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 箱线图比较
    axes[0, 0].boxplot(region_errors, labels=regions)
    axes[0, 0].set_xlabel('区域')
    axes[0, 0].set_ylabel('地理误差 (米)')
    axes[0, 0].set_title('各区域误差分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 平均误差柱状图
    mean_errors = [np.mean(errors) for errors in region_errors]
    std_errors = [np.std(errors) for errors in region_errors]
    
    bars = axes[0, 1].bar(regions, mean_errors, yerr=std_errors, 
                         alpha=0.7, capsize=5, color='lightblue')
    axes[0, 1].set_xlabel('区域')
    axes[0, 1].set_ylabel('平均地理误差 (米)')
    axes[0, 1].set_title('各区域平均误差')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, mean_error in zip(bars, mean_errors):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{mean_error:.1f}m', ha='center', va='bottom')
    
    # 3. 准确率比较（不同阈值）
    thresholds = [1, 5, 10, 25, 50]
    accuracy_data = []
    
    for threshold in thresholds:
        accuracies = []
        for errors in region_errors:
            acc = np.mean(np.array(errors) <= threshold) * 100
            accuracies.append(acc)
        accuracy_data.append(accuracies)
    
    x = np.arange(len(regions))
    width = 0.15
    
    for i, (threshold, accuracies) in enumerate(zip(thresholds, accuracy_data)):
        axes[1, 0].bar(x + i * width, accuracies, width, 
                      label=f'{threshold}m', alpha=0.8)
    
    axes[1, 0].set_xlabel('区域')
    axes[1, 0].set_ylabel('准确率 (%)')
    axes[1, 0].set_title('各区域准确率比较')
    axes[1, 0].set_xticks(x + width * 2)
    axes[1, 0].set_xticklabels(regions)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 样本数量
    sample_counts = [len(errors) for errors in region_errors]
    axes[1, 1].bar(regions, sample_counts, alpha=0.7, color='lightgreen')
    axes[1, 1].set_xlabel('区域')
    axes[1, 1].set_ylabel('样本数量')
    axes[1, 1].set_title('各区域样本数量')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, count in enumerate(sample_counts):
        axes[1, 1].text(i, count + max(sample_counts) * 0.01, str(count), 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"区域比较图已保存到: {save_path}")
    
    plt.show()


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                        save_path=None, figsize=(15, 10)):
    """
    绘制训练曲线
    
    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        train_metrics (dict): 训练指标字典
        val_metrics (dict): 验证指标字典
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    # 确定子图数量
    num_plots = 2  # 损失曲线 + 学习率
    if train_metrics and val_metrics:
        num_plots += len(train_metrics)
    
    rows = (num_plots + 1) // 2
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('训练过程监控', fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # 1. 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 损失对数尺度
    axes[0, 1].semilogy(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0, 1].semilogy(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失 (对数尺度)')
    axes[0, 1].set_title('训练和验证损失 (对数尺度)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 其他指标
    if train_metrics and val_metrics:
        plot_idx = 2
        for metric_name in train_metrics.keys():
            if plot_idx >= num_plots:
                break
                
            row = plot_idx // cols
            col = plot_idx % cols
            
            train_values = train_metrics[metric_name]
            val_values = val_metrics.get(metric_name, [])
            
            axes[row, col].plot(epochs[:len(train_values)], train_values, 
                               'b-', label=f'训练{metric_name}', linewidth=2)
            if val_values:
                axes[row, col].plot(epochs[:len(val_values)], val_values, 
                                   'r-', label=f'验证{metric_name}', linewidth=2)
            
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric_name)
            axes[row, col].set_title(f'{metric_name}变化')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(num_plots, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()


def plot_prediction_scatter(predictions, targets, errors, title="预测结果散点图", 
                           save_path=None, figsize=(12, 10)):
    """
    绘制预测结果散点图
    
    Args:
        predictions (list): 预测坐标列表 [[lat, lon], ...]
        targets (list): 真实坐标列表 [[lat, lon], ...]
        errors (list): 误差列表
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.array(errors)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 纬度预测 vs 真实
    axes[0, 0].scatter(targets[:, 0], predictions[:, 0], c=errors, 
                      cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                   [targets[:, 0].min(), targets[:, 0].max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('真实纬度')
    axes[0, 0].set_ylabel('预测纬度')
    axes[0, 0].set_title('纬度预测对比')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 经度预测 vs 真实
    scatter = axes[0, 1].scatter(targets[:, 1], predictions[:, 1], c=errors, 
                                cmap='viridis', alpha=0.6, s=20)
    axes[0, 1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                   [targets[:, 1].min(), targets[:, 1].max()], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('真实经度')
    axes[0, 1].set_ylabel('预测经度')
    axes[0, 1].set_title('经度预测对比')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes[0, 1])
    cbar.set_label('地理误差 (米)')
    
    # 3. 误差分布在地理空间
    axes[1, 0].scatter(targets[:, 1], targets[:, 0], c=errors, 
                      cmap='hot', alpha=0.7, s=30)
    axes[1, 0].set_xlabel('经度')
    axes[1, 0].set_ylabel('纬度')
    axes[1, 0].set_title('地理空间误差分布')
    
    # 4. 预测误差向量
    # 计算预测偏移
    lat_diff = predictions[:, 0] - targets[:, 0]
    lon_diff = predictions[:, 1] - targets[:, 1]
    
    # 只显示部分样本以避免图像过于拥挤
    sample_indices = np.random.choice(len(targets), min(100, len(targets)), replace=False)
    
    axes[1, 1].scatter(targets[sample_indices, 1], targets[sample_indices, 0], 
                      c='blue', alpha=0.6, s=30, label='真实位置')
    axes[1, 1].scatter(predictions[sample_indices, 1], predictions[sample_indices, 0], 
                      c='red', alpha=0.6, s=30, label='预测位置')
    
    # 绘制误差向量
    for i in sample_indices[:20]:  # 只显示前20个向量
        axes[1, 1].arrow(targets[i, 1], targets[i, 0], 
                        lon_diff[i], lat_diff[i],
                        head_width=0.001, head_length=0.001, 
                        fc='gray', ec='gray', alpha=0.5)
    
    axes[1, 1].set_xlabel('经度')
    axes[1, 1].set_ylabel('纬度')
    axes[1, 1].set_title('预测误差向量 (样本)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测散点图已保存到: {save_path}")
    
    plt.show()


def plot_model_comparison(model_results, metric_name='mean_error', 
                         title="模型性能比较", save_path=None, figsize=(12, 8)):
    """
    绘制多个模型的性能比较图
    
    Args:
        model_results (dict): 模型结果字典 {model_name: {metric: value}}
        metric_name (str): 要比较的指标名称
        title (str): 图表标题
        save_path (str): 保存路径
        figsize (tuple): 图像大小
    """
    model_names = list(model_results.keys())
    metric_values = [model_results[name].get(metric_name, 0) for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. 柱状图
    bars = ax1.bar(model_names, metric_values, alpha=0.7, color='skyblue')
    ax1.set_xlabel('模型')
    ax1.set_ylabel(metric_name)
    ax1.set_title(f'{metric_name}比较')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values) * 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # 旋转x轴标签以避免重叠
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. 雷达图（如果有多个指标）
    metrics_to_show = ['mean_error', 'median_error', 'acc@25m', 'acc@50m']
    available_metrics = []
    
    for metric in metrics_to_show:
        if all(metric in model_results[name] for name in model_names):
            available_metrics.append(metric)
    
    if len(available_metrics) >= 3:
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2 = plt.subplot(122, projection='polar')
        
        for model_name in model_names:
            values = [model_results[model_name].get(metric, 0) for metric in available_metrics]
            # 归一化值（对于误差指标，使用倒数）
            normalized_values = []
            for i, (metric, value) in enumerate(zip(available_metrics, values)):
                if 'error' in metric:
                    # 误差指标：越小越好，使用倒数并归一化
                    max_error = max(model_results[name].get(metric, 0) for name in model_names)
                    normalized_values.append(1 - value / max_error if max_error > 0 else 1)
                else:
                    # 准确率指标：越大越好，直接归一化
                    max_acc = max(model_results[name].get(metric, 0) for name in model_names)
                    normalized_values.append(value / max_acc if max_acc > 0 else 0)
            
            normalized_values += [normalized_values[0]]  # 闭合图形
            ax2.plot(angles, normalized_values, 'o-', linewidth=2, label=model_name)
            ax2.fill(angles, normalized_values, alpha=0.25)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(available_metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('多指标性能雷达图')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    else:
        # 如果指标不足，显示参数数量比较
        param_counts = []
        for name in model_names:
            param_count = model_results[name].get('total_params', 0)
            param_counts.append(param_count / 1e6)  # 转换为百万参数
        
        ax2.bar(model_names, param_counts, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('模型')
        ax2.set_ylabel('参数数量 (百万)')
        ax2.set_title('模型参数数量比较')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型比较图已保存到: {save_path}")
    
    plt.show()
