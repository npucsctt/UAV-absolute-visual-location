#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估指标工具函数

包含训练和评估过程中使用的各种指标计算函数
"""

import torch
import numpy as np


class AverageMeter(object):
    """
    计算并存储平均值和当前值
    
    用于跟踪训练过程中的损失和指标
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计信息"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计信息
        
        Args:
            val: 当前值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(pred, target):
    """
    计算预测结果的评估指标
    
    Args:
        pred (torch.Tensor): 预测坐标 [B, 2]
        target (torch.Tensor): 真实坐标 [B, 2]
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    with torch.no_grad():
        # 计算欧几里得距离作为误差
        error = torch.sqrt(((pred - target) ** 2).sum(dim=1))
        
        # 基本统计指标
        mean_error = error.mean().item()
        median_error = error.median().item()
        std_error = error.std().item()
        max_error = error.max().item()
        min_error = error.min().item()
        
        # 计算不同阈值下的准确率（假设误差单位为归一化坐标）
        # 这里的阈值是归一化坐标的阈值，需要根据实际情况调整
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]  # 归一化坐标阈值
        accuracies = {}
        for threshold in thresholds:
            acc = (error <= threshold).float().mean().item() * 100
            accuracies[f'acc@{threshold:.2f}'] = acc
    
    metrics = {
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'max_error': max_error,
        'min_error': min_error,
        **accuracies
    }
    
    return metrics


def compute_geo_metrics(pred_coords, true_coords, sat_infos, regions, img_size=224):
    """
    计算地理坐标的评估指标
    
    Args:
        pred_coords (torch.Tensor): 预测的归一化坐标 [B, 2]
        true_coords (torch.Tensor): 真实的归一化坐标 [B, 2]
        sat_infos (list): 卫星图像信息列表
        regions (list): 区域列表
        img_size (int): 图像大小
        
    Returns:
        dict: 包含地理误差指标的字典
    """
    from .distance import calculate_geo_error_with_region_calibration
    
    geo_errors = []
    pixel_errors = []
    
    for i in range(len(pred_coords)):
        pred_lat, pred_lon = pred_coords[i][0].item(), pred_coords[i][1].item()
        true_lat, true_lon = true_coords[i][0].item(), true_coords[i][1].item()
        
        # 计算地理误差
        geo_error, pixel_error, _ = calculate_geo_error_with_region_calibration(
            pred_lat, pred_lon, true_lat, true_lon,
            sat_infos[i], regions[i], img_size
        )
        
        geo_errors.append(geo_error)
        pixel_errors.append(pixel_error)
    
    geo_errors = np.array(geo_errors)
    pixel_errors = np.array(pixel_errors)
    
    # 计算统计指标
    metrics = {
        'geo_mean_error': np.mean(geo_errors),
        'geo_median_error': np.median(geo_errors),
        'geo_std_error': np.std(geo_errors),
        'geo_max_error': np.max(geo_errors),
        'geo_min_error': np.min(geo_errors),
        'pixel_mean_error': np.mean(pixel_errors),
        'pixel_median_error': np.median(pixel_errors),
    }
    
    # 计算不同距离阈值下的准确率
    distance_thresholds = [1, 5, 10, 25, 50, 100]  # 米
    for threshold in distance_thresholds:
        acc = np.mean(geo_errors <= threshold) * 100
        metrics[f'geo_acc@{threshold}m'] = acc
    
    return metrics


def compute_loss_metrics(losses):
    """
    计算损失的统计指标
    
    Args:
        losses (list): 损失值列表
        
    Returns:
        dict: 损失统计指标
    """
    losses = np.array(losses)
    
    return {
        'loss_mean': np.mean(losses),
        'loss_std': np.std(losses),
        'loss_min': np.min(losses),
        'loss_max': np.max(losses),
        'loss_median': np.median(losses)
    }


class MetricsTracker:
    """
    指标跟踪器，用于跟踪训练过程中的多个指标
    """
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """
        更新指标
        
        Args:
            **kwargs: 指标名称和值的键值对
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = AverageMeter()
                self.history[name] = []
            
            if isinstance(value, (list, tuple)):
                # 如果是列表或元组，假设第一个是值，第二个是样本数
                val, n = value[0], value[1] if len(value) > 1 else 1
                self.metrics[name].update(val, n)
            else:
                self.metrics[name].update(value)
    
    def get_current_metrics(self):
        """
        获取当前的平均指标
        
        Returns:
            dict: 当前指标字典
        """
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def save_epoch_metrics(self):
        """保存当前epoch的指标到历史记录"""
        for name, meter in self.metrics.items():
            self.history[name].append(meter.avg)
    
    def reset(self):
        """重置所有指标"""
        for meter in self.metrics.values():
            meter.reset()
    
    def get_history(self):
        """
        获取历史指标
        
        Returns:
            dict: 历史指标字典
        """
        return self.history.copy()


def calculate_improvement(current_metric, best_metric, higher_is_better=False):
    """
    计算指标改进程度
    
    Args:
        current_metric (float): 当前指标值
        best_metric (float): 最佳指标值
        higher_is_better (bool): 是否越高越好
        
    Returns:
        tuple: (是否改进, 改进百分比)
    """
    if best_metric is None:
        return True, 0.0
    
    if higher_is_better:
        improved = current_metric > best_metric
        improvement = ((current_metric - best_metric) / abs(best_metric)) * 100
    else:
        improved = current_metric < best_metric
        improvement = ((best_metric - current_metric) / abs(best_metric)) * 100
    
    return improved, improvement


def format_metrics(metrics, precision=4):
    """
    格式化指标输出
    
    Args:
        metrics (dict): 指标字典
        precision (int): 小数点精度
        
    Returns:
        str: 格式化的指标字符串
    """
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, float):
            formatted_parts.append(f"{name}: {value:.{precision}f}")
        else:
            formatted_parts.append(f"{name}: {value}")
    
    return ", ".join(formatted_parts)


def early_stopping_check(metric_history, patience=10, min_delta=0.001, higher_is_better=False):
    """
    早停检查
    
    Args:
        metric_history (list): 指标历史记录
        patience (int): 耐心值（等待轮数）
        min_delta (float): 最小改进阈值
        higher_is_better (bool): 是否越高越好
        
    Returns:
        bool: 是否应该早停
    """
    if len(metric_history) < patience + 1:
        return False
    
    recent_metrics = metric_history[-patience-1:]
    best_metric = recent_metrics[0]
    
    for metric in recent_metrics[1:]:
        if higher_is_better:
            if metric > best_metric + min_delta:
                return False
            best_metric = max(best_metric, metric)
        else:
            if metric < best_metric - min_delta:
                return False
            best_metric = min(best_metric, metric)
    
    return True
