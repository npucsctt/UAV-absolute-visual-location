#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据变换模块

提供训练和验证时使用的数据变换函数
"""

from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter


def get_train_transforms(img_size=224, enhanced_augmentation=False):
    """
    获取训练时的数据变换
    
    Args:
        img_size (int): 图像大小
        enhanced_augmentation (bool): 是否使用增强的数据增强
        
    Returns:
        torchvision.transforms.Compose: 训练变换组合
    """
    transforms_list = []
    
    # 基础变换
    if enhanced_augmentation:
        # 增强的数据增强
        transforms_list.extend([
            Resize((img_size, img_size)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ])
    else:
        # 基础变换
        transforms_list.append(Resize((img_size, img_size)))
    
    # 标准化变换
    transforms_list.extend([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return Compose(transforms_list)


def get_val_transforms(img_size=224):
    """
    获取验证时的数据变换
    
    Args:
        img_size (int): 图像大小
        
    Returns:
        torchvision.transforms.Compose: 验证变换组合
    """
    return Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_test_transforms(img_size=224):
    """
    获取测试时的数据变换
    
    Args:
        img_size (int): 图像大小
        
    Returns:
        torchvision.transforms.Compose: 测试变换组合
    """
    return get_val_transforms(img_size)


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反归一化图像张量，用于可视化
    
    Args:
        tensor (torch.Tensor): 归一化的图像张量 [C, H, W]
        mean (list): 归一化均值
        std (list): 归一化标准差
        
    Returns:
        torch.Tensor: 反归一化的图像张量
    """
    import torch
    
    # 确保输入是张量
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    
    # 克隆张量以避免修改原始数据
    denorm_tensor = tensor.clone()
    
    # 反归一化
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    
    # 限制值在[0, 1]范围内
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    return denorm_tensor


def tensor_to_numpy_image(tensor):
    """
    将张量转换为numpy图像数组，用于可视化
    
    Args:
        tensor (torch.Tensor): 图像张量 [C, H, W]
        
    Returns:
        np.ndarray: 图像数组 [H, W, C]，值范围[0, 255]
    """
    import numpy as np
    
    # 反归一化
    denorm_tensor = denormalize_image(tensor)
    
    # 转换为numpy数组
    numpy_image = denorm_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 转换为0-255范围
    numpy_image = (numpy_image * 255).astype(np.uint8)
    
    return numpy_image
