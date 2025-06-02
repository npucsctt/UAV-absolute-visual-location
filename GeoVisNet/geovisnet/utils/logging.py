#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块

提供日志配置和管理功能
"""

import os
import logging
import time
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir='logs', log_level=logging.INFO, model_name='geovisnet', 
                  max_bytes=10*1024*1024, backup_count=5):
    """
    设置日志配置
    
    Args:
        log_dir (str): 日志目录
        log_level (int): 日志级别
        model_name (str): 模型名称，用于日志文件名
        max_bytes (int): 单个日志文件最大字节数
        backup_count (int): 备份文件数量
        
    Returns:
        str: 日志文件路径
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(log_dir, f'{model_name}_{timestamp}.log')
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器（仅显示INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置处理器的格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 返回日志文件路径
    return log_file


def get_logger(name):
    """
    获取指定名称的日志记录器
    
    Args:
        name (str): 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)


def setup_training_logging(log_dir, experiment_name, log_level=logging.INFO):
    """
    设置训练专用的日志配置
    
    Args:
        log_dir (str): 日志目录
        experiment_name (str): 实验名称
        log_level (int): 日志级别
        
    Returns:
        tuple: (主日志文件路径, 训练日志记录器, 验证日志记录器)
    """
    # 创建实验目录
    exp_log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # 设置主日志
    main_log_file = setup_logging(exp_log_dir, log_level, 'main')
    
    # 创建训练日志记录器
    train_logger = logging.getLogger('train')
    train_handler = logging.FileHandler(os.path.join(exp_log_dir, 'train.log'))
    train_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    train_logger.addHandler(train_handler)
    train_logger.setLevel(log_level)
    
    # 创建验证日志记录器
    val_logger = logging.getLogger('validation')
    val_handler = logging.FileHandler(os.path.join(exp_log_dir, 'validation.log'))
    val_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    val_logger.addHandler(val_handler)
    val_logger.setLevel(log_level)
    
    return main_log_file, train_logger, val_logger


class TrainingLogger:
    """
    训练过程日志记录器
    
    提供结构化的训练日志记录功能
    """
    
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # 设置日志
        self.main_log_file, self.train_logger, self.val_logger = setup_training_logging(
            log_dir, experiment_name
        )
        
        # 获取主日志记录器
        self.main_logger = logging.getLogger()
        
        # 记录实验开始
        self.main_logger.info(f"开始实验: {experiment_name}")
        self.main_logger.info(f"日志目录: {os.path.join(log_dir, experiment_name)}")
    
    def log_config(self, config):
        """
        记录配置信息
        
        Args:
            config (dict): 配置字典
        """
        self.main_logger.info("实验配置:")
        for key, value in config.items():
            self.main_logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model):
        """
        记录模型信息
        
        Args:
            model: PyTorch模型
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.main_logger.info("模型信息:")
        self.main_logger.info(f"  总参数数量: {total_params:,}")
        self.main_logger.info(f"  可训练参数: {trainable_params:,}")
        self.main_logger.info(f"  模型类型: {type(model).__name__}")
    
    def log_epoch_start(self, epoch, total_epochs):
        """
        记录epoch开始
        
        Args:
            epoch (int): 当前epoch
            total_epochs (int): 总epoch数
        """
        self.main_logger.info(f"开始训练 Epoch {epoch+1}/{total_epochs}")
    
    def log_training_metrics(self, epoch, metrics, lr=None):
        """
        记录训练指标
        
        Args:
            epoch (int): 当前epoch
            metrics (dict): 训练指标
            lr (float): 学习率
        """
        log_msg = f"Epoch {epoch+1} 训练结果: "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_msg += f"{key}: {value:.6f}, "
            else:
                log_msg += f"{key}: {value}, "
        
        if lr is not None:
            log_msg += f"学习率: {lr:.8f}"
        
        self.train_logger.info(log_msg)
        self.main_logger.info(log_msg)
    
    def log_validation_metrics(self, epoch, metrics):
        """
        记录验证指标
        
        Args:
            epoch (int): 当前epoch
            metrics (dict): 验证指标
        """
        log_msg = f"Epoch {epoch+1} 验证结果: "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_msg += f"{key}: {value:.6f}, "
            else:
                log_msg += f"{key}: {value}, "
        
        self.val_logger.info(log_msg)
        self.main_logger.info(log_msg)
    
    def log_best_model(self, epoch, metric_name, metric_value):
        """
        记录最佳模型
        
        Args:
            epoch (int): epoch
            metric_name (str): 指标名称
            metric_value (float): 指标值
        """
        msg = f"发现最佳模型! Epoch {epoch+1}, {metric_name}: {metric_value:.6f}"
        self.main_logger.info(msg)
    
    def log_early_stopping(self, epoch, patience):
        """
        记录早停
        
        Args:
            epoch (int): 停止的epoch
            patience (int): 耐心值
        """
        msg = f"早停触发! Epoch {epoch+1}, 耐心值: {patience}"
        self.main_logger.warning(msg)
    
    def log_experiment_end(self, total_time, best_metrics=None):
        """
        记录实验结束
        
        Args:
            total_time (float): 总训练时间（秒）
            best_metrics (dict): 最佳指标
        """
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.main_logger.info(f"实验完成! 总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if best_metrics:
            self.main_logger.info("最佳结果:")
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    self.main_logger.info(f"  {key}: {value:.6f}")
                else:
                    self.main_logger.info(f"  {key}: {value}")
    
    def log_error(self, error_msg, exception=None):
        """
        记录错误
        
        Args:
            error_msg (str): 错误消息
            exception (Exception): 异常对象
        """
        self.main_logger.error(error_msg)
        if exception:
            self.main_logger.exception(exception)
