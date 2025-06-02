#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet训练配置

包含不同训练场景的配置
"""

from .default_config import DefaultConfig, merge_configs


def get_training_config(config_type='default', **kwargs):
    """
    获取训练配置
    
    Args:
        config_type (str): 配置类型
        **kwargs: 额外的配置覆盖
        
    Returns:
        dict: 训练配置字典
    """
    
    # 基础配置
    base_configs = {
        'default': {
            'model': {
                'backbone': 'efficientnet_b0',
                'emb_size': 256,
                'dropout': 0.5,
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'lr': 0.0005,
                'weight_decay': 0.001,
                'warmup_epochs': 10,
                'use_amp': True,
            },
            'data': {
                'img_size': 224,
                'sat_patch_scale': 3.0,
                'enhanced_augmentation': True,
                'use_mixup': True,
                'mixup_alpha': 0.8,
            }
        },
        
        'efficientnet_b2': {
            'model': {
                'backbone': 'efficientnet_b2',
                'emb_size': 512,
                'dropout': 0.6,
            },
            'training': {
                'batch_size': 12,
                'lr': 0.0003,
                'epochs': 120,
            }
        },
        
        'fast_debug': {
            'training': {
                'epochs': 10,
                'batch_size': 8,
                'warmup_epochs': 2,
            },
            'logging': {
                'log_interval': 5,
                'eval_interval': 1,
                'save_interval': 5,
            }
        },
        
        'high_quality': {
            'training': {
                'epochs': 200,
                'lr': 0.0003,
                'weight_decay': 0.0005,
                'warmup_epochs': 20,
            },
            'data': {
                'enhanced_augmentation': True,
                'use_mixup': True,
                'mixup_alpha': 1.0,
                'use_cross_mixup': True,
                'cross_mixup_alpha': 0.3,
            },
            'early_stopping': {
                'patience': 30,
                'min_delta': 0.0005,
            }
        },
        
        'ablation_no_attention': {
            'model': {
                'use_attention': False,
            }
        },
        
        'ablation_no_mixup': {
            'data': {
                'use_mixup': False,
                'use_cross_mixup': False,
            }
        },
        
        'ablation_frozen_backbone': {
            'training': {
                'freeze_backbone': True,
                'lr': 0.001,  # 更高的学习率，因为只训练头部
            }
        },
        
        'small_dataset': {
            'training': {
                'batch_size': 8,
                'epochs': 50,
                'lr': 0.001,
            },
            'data': {
                'enhanced_augmentation': True,
                'use_mixup': True,
                'mixup_alpha': 1.2,
            }
        },
        
        'large_dataset': {
            'training': {
                'batch_size': 32,
                'epochs': 150,
                'lr': 0.0003,
                'warmup_epochs': 15,
            }
        },
        
        'fine_tuning': {
            'training': {
                'lr': 0.0001,
                'weight_decay': 0.0001,
                'epochs': 50,
                'warmup_epochs': 5,
                'freeze_backbone': False,
            }
        }
    }
    
    # 获取基础配置
    if config_type not in base_configs:
        raise ValueError(f"未知配置类型: {config_type}")
    
    # 从默认配置开始
    config = DefaultConfig.get_config()
    
    # 应用特定配置
    specific_config = base_configs[config_type]
    config = merge_configs(config, specific_config)
    
    # 应用额外的覆盖
    if kwargs:
        config = merge_configs(config, kwargs)
    
    return config


def get_ablation_configs():
    """
    获取消融实验配置列表
    
    Returns:
        dict: 消融实验配置字典
    """
    return {
        'baseline': get_training_config('default'),
        'no_attention': get_training_config('ablation_no_attention'),
        'no_mixup': get_training_config('ablation_no_mixup'),
        'frozen_backbone': get_training_config('ablation_frozen_backbone'),
        'efficientnet_b2': get_training_config('efficientnet_b2'),
    }


def get_hyperparameter_search_configs():
    """
    获取超参数搜索配置
    
    Returns:
        list: 超参数配置列表
    """
    configs = []
    
    # 学习率搜索
    for lr in [0.0001, 0.0003, 0.0005, 0.001]:
        config = get_training_config('default', training={'lr': lr})
        config['experiment_name'] = f'lr_{lr}'
        configs.append(config)
    
    # 批次大小搜索
    for batch_size in [8, 16, 24, 32]:
        config = get_training_config('default', training={'batch_size': batch_size})
        config['experiment_name'] = f'batch_{batch_size}'
        configs.append(config)
    
    # Dropout搜索
    for dropout in [0.3, 0.5, 0.7]:
        config = get_training_config('default', model={'dropout': dropout})
        config['experiment_name'] = f'dropout_{dropout}'
        configs.append(config)
    
    # MixUp alpha搜索
    for alpha in [0.2, 0.5, 0.8, 1.0, 1.5]:
        config = get_training_config('default', data={'mixup_alpha': alpha})
        config['experiment_name'] = f'mixup_alpha_{alpha}'
        configs.append(config)
    
    return configs


def validate_config(config):
    """
    验证配置的有效性
    
    Args:
        config (dict): 配置字典
        
    Returns:
        bool: 配置是否有效
        list: 错误信息列表
    """
    errors = []
    
    # 检查必需的配置项
    required_sections = ['model', 'training', 'data']
    for section in required_sections:
        if section not in config:
            errors.append(f"缺少必需的配置节: {section}")
    
    # 检查模型配置
    if 'model' in config:
        model_config = config['model']
        if 'backbone' in model_config:
            valid_backbones = ['efficientnet_b0', 'efficientnet_b2']
            if model_config['backbone'] not in valid_backbones:
                errors.append(f"无效的backbone: {model_config['backbone']}")
        
        if 'emb_size' in model_config:
            if not isinstance(model_config['emb_size'], int) or model_config['emb_size'] <= 0:
                errors.append("emb_size必须是正整数")
    
    # 检查训练配置
    if 'training' in config:
        training_config = config['training']
        if 'lr' in training_config:
            if not isinstance(training_config['lr'], (int, float)) or training_config['lr'] <= 0:
                errors.append("学习率必须是正数")
        
        if 'batch_size' in training_config:
            if not isinstance(training_config['batch_size'], int) or training_config['batch_size'] <= 0:
                errors.append("batch_size必须是正整数")
    
    # 检查数据配置
    if 'data' in config:
        data_config = config['data']
        if 'img_size' in data_config:
            if not isinstance(data_config['img_size'], int) or data_config['img_size'] <= 0:
                errors.append("img_size必须是正整数")
    
    return len(errors) == 0, errors


def print_config(config, title="配置信息"):
    """
    打印配置信息
    
    Args:
        config (dict): 配置字典
        title (str): 标题
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(config)
