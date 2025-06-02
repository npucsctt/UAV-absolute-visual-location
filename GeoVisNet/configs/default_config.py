#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet默认配置

包含模型、训练、数据等的默认配置参数
"""


class DefaultConfig:
    """默认配置类"""
    
    # 数据配置
    DATA = {
        'img_size': 224,
        'sat_patch_scale': 3.0,
        'train_regions': ['01', '02', '03', '04', '05', '06'],
        'val_regions': ['07'],
        'test_regions': ['08', '10', '11'],
        'cache_satellite': True,
        'num_workers': 8,
    }
    
    # 模型配置
    MODEL = {
        'backbone': 'efficientnet_b0',
        'emb_size': 256,
        'dropout': 0.5,
        'freeze_backbone': False,
        'satellite_backbone': None,  # 如果为None，使用与backbone相同的网络
    }
    
    # 训练配置
    TRAINING = {
        'batch_size': 16,
        'epochs': 100,
        'lr': 0.0005,
        'weight_decay': 0.001,
        'warmup_epochs': 10,
        'min_lr_ratio': 0.01,
        'gradient_clip': 1.0,
        'use_amp': True,
        'freeze_backbone': False,
    }
    
    # 数据增强配置
    AUGMENTATION = {
        'enhanced_augmentation': True,
        'use_mixup': True,
        'mixup_alpha': 0.8,
        'use_cross_mixup': True,
        'cross_mixup_alpha': 0.2,
        'cross_mixup_prob': 0.3,
    }
    
    # 损失函数配置
    LOSS = {
        'loss_type': 'mse',  # 'mse' or 'smooth_l1'
        'smooth_l1_beta': 0.1,
    }
    
    # 日志和保存配置
    LOGGING = {
        'log_interval': 10,
        'eval_interval': 1,
        'save_interval': 10,
        'max_checkpoints': 5,
    }
    
    # 早停配置
    EARLY_STOPPING = {
        'patience': 20,
        'min_delta': 0.001,
        'monitor': 'val_loss',
        'mode': 'min',  # 'min' for loss, 'max' for accuracy
    }
    
    # 设备配置
    DEVICE = {
        'gpu': 0,
        'seed': 42,
    }
    
    @classmethod
    def get_config(cls):
        """获取完整配置字典"""
        config = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name.isupper():
                config[attr_name.lower()] = getattr(cls, attr_name)
        return config
    
    @classmethod
    def update_config(cls, updates):
        """更新配置"""
        config = cls.get_config()
        for key, value in updates.items():
            if key in config:
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
        return config


# 预定义的配置变体
class EfficientNetB0Config(DefaultConfig):
    """EfficientNet-B0配置"""
    MODEL = DefaultConfig.MODEL.copy()
    MODEL.update({
        'backbone': 'efficientnet_b0',
        'emb_size': 256,
        'dropout': 0.5,
    })
    
    TRAINING = DefaultConfig.TRAINING.copy()
    TRAINING.update({
        'batch_size': 16,
        'lr': 0.0005,
    })


class EfficientNetB2Config(DefaultConfig):
    """EfficientNet-B2配置"""
    MODEL = DefaultConfig.MODEL.copy()
    MODEL.update({
        'backbone': 'efficientnet_b2',
        'emb_size': 512,
        'dropout': 0.6,
    })
    
    TRAINING = DefaultConfig.TRAINING.copy()
    TRAINING.update({
        'batch_size': 12,  # 更大的模型需要更小的批次
        'lr': 0.0003,      # 更大的模型使用更小的学习率
    })


class FastTrainingConfig(DefaultConfig):
    """快速训练配置（用于调试）"""
    TRAINING = DefaultConfig.TRAINING.copy()
    TRAINING.update({
        'epochs': 20,
        'warmup_epochs': 2,
        'batch_size': 8,
    })
    
    LOGGING = DefaultConfig.LOGGING.copy()
    LOGGING.update({
        'log_interval': 5,
        'eval_interval': 1,
        'save_interval': 5,
    })


class HighQualityConfig(DefaultConfig):
    """高质量训练配置"""
    TRAINING = DefaultConfig.TRAINING.copy()
    TRAINING.update({
        'epochs': 200,
        'warmup_epochs': 20,
        'lr': 0.0003,
        'weight_decay': 0.0005,
    })
    
    AUGMENTATION = DefaultConfig.AUGMENTATION.copy()
    AUGMENTATION.update({
        'enhanced_augmentation': True,
        'use_mixup': True,
        'mixup_alpha': 1.0,
        'use_cross_mixup': True,
        'cross_mixup_alpha': 0.3,
        'cross_mixup_prob': 0.5,
    })
    
    EARLY_STOPPING = DefaultConfig.EARLY_STOPPING.copy()
    EARLY_STOPPING.update({
        'patience': 30,
        'min_delta': 0.0005,
    })


class NoAugmentationConfig(DefaultConfig):
    """无数据增强配置（用于消融实验）"""
    AUGMENTATION = {
        'enhanced_augmentation': False,
        'use_mixup': False,
        'mixup_alpha': 0.0,
        'use_cross_mixup': False,
        'cross_mixup_alpha': 0.0,
        'cross_mixup_prob': 0.0,
    }


# 配置注册表
CONFIG_REGISTRY = {
    'default': DefaultConfig,
    'efficientnet_b0': EfficientNetB0Config,
    'efficientnet_b2': EfficientNetB2Config,
    'fast': FastTrainingConfig,
    'high_quality': HighQualityConfig,
    'no_augmentation': NoAugmentationConfig,
}


def get_config(config_name='default'):
    """
    根据名称获取配置
    
    Args:
        config_name (str): 配置名称
        
    Returns:
        dict: 配置字典
    """
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"未知配置: {config_name}. 可用配置: {list(CONFIG_REGISTRY.keys())}")
    
    return CONFIG_REGISTRY[config_name].get_config()


def list_configs():
    """列出所有可用配置"""
    return list(CONFIG_REGISTRY.keys())


def merge_configs(base_config, override_config):
    """
    合并配置
    
    Args:
        base_config (dict): 基础配置
        override_config (dict): 覆盖配置
        
    Returns:
        dict: 合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    
    return merged
