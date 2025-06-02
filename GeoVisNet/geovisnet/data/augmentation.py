#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据增强模块

包含MixUp和跨视图MixUp等数据增强方法
"""

import torch
import numpy as np
import random


class MixUp:
    """
    MixUp数据增强方法
    
    论文: mixup: Beyond Empirical Risk Minimization
    
    Args:
        alpha (float): Beta分布的参数，控制混合强度
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        """
        对批次数据应用MixUp
        
        Args:
            batch (dict): 包含'drone_img', 'sat_img', 'norm_lat', 'norm_lon'等键的字典
            
        Returns:
            dict: 应用MixUp后的批次数据
        """
        if self.alpha <= 0:
            return batch
            
        # 获取批次大小
        batch_size = batch['drone_img'].size(0)
        
        # 如果批次大小小于2，无法进行混合
        if batch_size < 2:
            return batch
            
        # 从Beta分布采样混合权重
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 混合无人机图像
        mixed_drone_img = lam * batch['drone_img'] + (1 - lam) * batch['drone_img'][indices]
        
        # 混合卫星图像
        mixed_sat_img = lam * batch['sat_img'] + (1 - lam) * batch['sat_img'][indices]
        
        # 混合标签（归一化坐标）
        mixed_norm_lat = lam * batch['norm_lat'] + (1 - lam) * batch['norm_lat'][indices]
        mixed_norm_lon = lam * batch['norm_lon'] + (1 - lam) * batch['norm_lon'][indices]
        
        # 混合原始坐标（如果需要）
        mixed_drone_lat = lam * batch['drone_lat'] + (1 - lam) * batch['drone_lat'][indices]
        mixed_drone_lon = lam * batch['drone_lon'] + (1 - lam) * batch['drone_lon'][indices]
        
        # 更新批次数据
        batch['drone_img'] = mixed_drone_img
        batch['sat_img'] = mixed_sat_img
        batch['norm_lat'] = mixed_norm_lat
        batch['norm_lon'] = mixed_norm_lon
        batch['drone_lat'] = mixed_drone_lat
        batch['drone_lon'] = mixed_drone_lon
        batch['mixup_lambda'] = lam
        batch['mixup_indices'] = indices
        
        return batch


class UAVSatMixUp:
    """
    无人机和卫星图像之间的MixUp，用于跨视图学习
    
    Args:
        alpha (float): Beta分布的参数，控制混合强度
        p (float): 应用MixUp的概率
    """
    
    def __init__(self, alpha=0.2, p=0.5):
        self.alpha = alpha
        self.p = p
        
    def __call__(self, batch):
        """
        对批次数据应用无人机-卫星MixUp
        
        Args:
            batch (dict): 包含'drone_img', 'sat_img'等键的字典
            
        Returns:
            dict: 应用MixUp后的批次数据
        """
        # 以概率p应用MixUp
        if random.random() > self.p:
            return batch
            
        # 获取批次大小
        batch_size = batch['drone_img'].size(0)
        
        # 从Beta分布采样混合权重
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 0.5
            
        # 混合无人机和卫星图像
        mixed_drone_img = lam * batch['drone_img'] + (1 - lam) * batch['sat_img']
        mixed_sat_img = lam * batch['sat_img'] + (1 - lam) * batch['drone_img']
        
        # 更新批次数据
        batch['drone_img'] = mixed_drone_img
        batch['sat_img'] = mixed_sat_img
        batch['cross_mixup_lambda'] = lam
        
        return batch


def mixup_data(x, y, alpha=1.0):
    """
    简单的MixUp函数，用于单独的数据和标签
    
    Args:
        x (torch.Tensor): 输入数据
        y (torch.Tensor): 标签
        alpha (float): Beta分布参数
        
    Returns:
        tuple: (混合后的数据, 混合后的标签, lambda值, 打乱的索引)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, lam, index


def cross_mixup_data(drone_imgs, sat_imgs, alpha=0.2):
    """
    跨视图MixUp函数
    
    Args:
        drone_imgs (torch.Tensor): 无人机图像
        sat_imgs (torch.Tensor): 卫星图像
        alpha (float): Beta分布参数
        
    Returns:
        tuple: (混合后的无人机图像, 混合后的卫星图像, lambda值)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 0.5

    mixed_drone_imgs = lam * drone_imgs + (1 - lam) * sat_imgs
    mixed_sat_imgs = lam * sat_imgs + (1 - lam) * drone_imgs
    
    return mixed_drone_imgs, mixed_sat_imgs, lam


class CutMix:
    """
    CutMix数据增强方法
    
    论文: CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    
    Args:
        alpha (float): Beta分布的参数
        prob (float): 应用CutMix的概率
    """
    
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch):
        """
        对批次数据应用CutMix
        
        Args:
            batch (dict): 包含图像和标签的字典
            
        Returns:
            dict: 应用CutMix后的批次数据
        """
        if random.random() > self.prob:
            return batch
            
        batch_size = batch['drone_img'].size(0)
        if batch_size < 2:
            return batch
            
        # 从Beta分布采样混合权重
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机打乱索引
        indices = torch.randperm(batch_size)
        
        # 获取图像尺寸
        _, _, H, W = batch['drone_img'].shape
        
        # 计算裁剪区域
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机选择裁剪位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 应用CutMix
        batch['drone_img'][:, :, bby1:bby2, bbx1:bbx2] = batch['drone_img'][indices, :, bby1:bby2, bbx1:bbx2]
        batch['sat_img'][:, :, bby1:bby2, bbx1:bbx2] = batch['sat_img'][indices, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda值
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # 混合标签
        batch['norm_lat'] = lam * batch['norm_lat'] + (1 - lam) * batch['norm_lat'][indices]
        batch['norm_lon'] = lam * batch['norm_lon'] + (1 - lam) * batch['norm_lon'][indices]
        batch['drone_lat'] = lam * batch['drone_lat'] + (1 - lam) * batch['drone_lat'][indices]
        batch['drone_lon'] = lam * batch['drone_lon'] + (1 - lam) * batch['drone_lon'][indices]
        
        batch['cutmix_lambda'] = lam
        batch['cutmix_indices'] = indices
        
        return batch


class RandomErasing:
    """
    随机擦除数据增强
    
    Args:
        prob (float): 应用概率
        scale (tuple): 擦除区域的尺寸范围
        ratio (tuple): 擦除区域的宽高比范围
        value (float): 填充值
    """
    
    def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        """
        对图像应用随机擦除
        
        Args:
            img (torch.Tensor): 输入图像 [C, H, W]
            
        Returns:
            torch.Tensor: 处理后的图像
        """
        if random.random() > self.prob:
            return img
            
        _, H, W = img.shape
        area = H * W
        
        for _ in range(100):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < W and h < H:
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)
                img[:, x1:x1+h, y1:y1+w] = self.value
                break
                
        return img
