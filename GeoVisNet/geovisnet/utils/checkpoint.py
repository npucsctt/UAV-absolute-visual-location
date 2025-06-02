#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查点管理工具

提供模型检查点的保存和加载功能
"""

import os
import shutil
import torch
import logging
from typing import Dict, Any, Optional


def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str, 
                   filename: str = 'checkpoint.pth', best_filename: str = 'best_model.pth'):
    """
    保存模型检查点
    
    Args:
        state (dict): 包含模型状态、优化器状态等的字典
        is_best (bool): 是否是最佳模型
        save_dir (str): 保存目录
        filename (str): 检查点文件名
        best_filename (str): 最佳模型文件名
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存检查点
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    
    # 如果是最佳模型，复制为最佳模型文件
    if is_best:
        best_path = os.path.join(save_dir, best_filename)
        shutil.copyfile(checkpoint_path, best_path)
        logging.info(f"保存最佳模型到: {best_path}")


def load_checkpoint(checkpoint_path: str, map_location: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path (str): 检查点文件路径
        map_location (str): 设备映射位置
        
    Returns:
        dict or None: 检查点字典，如果加载失败返回None
    """
    if not os.path.isfile(checkpoint_path):
        logging.error(f"检查点文件不存在: {checkpoint_path}")
        return None
    
    try:
        logging.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        logging.info(f"成功加载检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint
    except Exception as e:
        logging.error(f"加载检查点失败: {e}")
        return None


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, 
                      strict: bool = True, map_location: Optional[str] = None) -> bool:
    """
    仅加载模型权重
    
    Args:
        model (torch.nn.Module): 模型
        checkpoint_path (str): 检查点文件路径
        strict (bool): 是否严格匹配键名
        map_location (str): 设备映射位置
        
    Returns:
        bool: 是否加载成功
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location)
    if checkpoint is None:
        return False
    
    try:
        # 尝试不同的键名
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 假设整个检查点就是状态字典
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
        logging.info("成功加载模型权重")
        return True
    except Exception as e:
        logging.error(f"加载模型权重失败: {e}")
        return False


def resume_training(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    从检查点恢复训练
    
    Args:
        model (torch.nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        checkpoint_path (str): 检查点文件路径
        map_location (str): 设备映射位置
        
    Returns:
        dict: 包含恢复信息的字典
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location)
    if checkpoint is None:
        return {'success': False}
    
    try:
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 获取其他信息
        start_epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', None)
        
        logging.info(f"成功恢复训练，从epoch {start_epoch}开始")
        
        return {
            'success': True,
            'start_epoch': start_epoch,
            'best_metric': best_metric,
            'checkpoint': checkpoint
        }
    except Exception as e:
        logging.error(f"恢复训练失败: {e}")
        return {'success': False}


class CheckpointManager:
    """
    检查点管理器
    
    提供更高级的检查点管理功能
    """
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器
        
        Args:
            save_dir (str): 保存目录
            max_checkpoints (int): 最大保存的检查点数量
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        # 跟踪保存的检查点
        self.checkpoint_files = []
    
    def save(self, state: Dict[str, Any], epoch: int, is_best: bool = False, 
             metric_name: str = 'loss', metric_value: float = None):
        """
        保存检查点
        
        Args:
            state (dict): 状态字典
            epoch (int): 当前epoch
            is_best (bool): 是否是最佳模型
            metric_name (str): 指标名称
            metric_value (float): 指标值
        """
        # 创建文件名
        filename = f'checkpoint_epoch_{epoch:04d}.pth'
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存检查点
        torch.save(state, filepath)
        self.checkpoint_files.append(filepath)
        
        # 保存最佳模型
        if is_best:
            best_filepath = os.path.join(self.save_dir, 'best_model.pth')
            shutil.copyfile(filepath, best_filepath)
            logging.info(f"保存最佳模型 (epoch {epoch}, {metric_name}: {metric_value:.6f})")
        
        # 清理旧的检查点
        self._cleanup_old_checkpoints()
        
        logging.info(f"保存检查点: {filename}")
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # 按文件名排序（包含epoch信息）
            self.checkpoint_files.sort()
            
            # 删除最旧的检查点
            while len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logging.debug(f"删除旧检查点: {os.path.basename(old_checkpoint)}")
    
    def load_best(self, model: torch.nn.Module, map_location: Optional[str] = None) -> bool:
        """
        加载最佳模型
        
        Args:
            model (torch.nn.Module): 模型
            map_location (str): 设备映射位置
            
        Returns:
            bool: 是否加载成功
        """
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        return load_model_weights(model, best_path, map_location=map_location)
    
    def load_latest(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                   map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        加载最新的检查点
        
        Args:
            model (torch.nn.Module): 模型
            optimizer (torch.optim.Optimizer): 优化器
            map_location (str): 设备映射位置
            
        Returns:
            dict: 恢复信息
        """
        # 查找最新的检查点
        checkpoint_files = [f for f in os.listdir(self.save_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if not checkpoint_files:
            logging.warning("没有找到检查点文件")
            return {'success': False}
        
        # 按epoch排序，获取最新的
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_checkpoint = os.path.join(self.save_dir, checkpoint_files[-1])
        
        if optimizer is not None:
            return resume_training(model, optimizer, latest_checkpoint, map_location)
        else:
            success = load_model_weights(model, latest_checkpoint, map_location=map_location)
            return {'success': success}
    
    def list_checkpoints(self) -> list:
        """
        列出所有检查点
        
        Returns:
            list: 检查点文件列表
        """
        checkpoint_files = [f for f in os.listdir(self.save_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        return checkpoint_files


def create_checkpoint_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          epoch: int, metrics: Dict[str, float], 
                          additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    创建检查点状态字典
    
    Args:
        model (torch.nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        epoch (int): 当前epoch
        metrics (dict): 指标字典
        additional_info (dict): 额外信息
        
    Returns:
        dict: 检查点状态字典
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if additional_info:
        state.update(additional_info)
    
    return state
