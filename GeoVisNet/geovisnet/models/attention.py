#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
注意力机制模块

包含GeoVisNet中使用的各种注意力机制：
- EfficientChannelAttention (ECA): 高效通道注意力
- SpatialAttention: 空间注意力
- ChannelAttention: 通道注意力
- CBAM: 卷积块注意力模块
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientChannelAttention(nn.Module):
    """
    高效通道注意力模块 (ECA)
    
    论文: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    
    Args:
        in_channels (int): 输入通道数
        k_size (int): 1D卷积核大小，默认为3
        gamma (float): 自适应核大小计算参数
        b (float): 自适应核大小计算参数
    """
    
    def __init__(self, in_channels, k_size=3, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        self.in_channels = in_channels
        
        # 自适应计算卷积核大小
        if k_size is None:
            k_size = int(abs((math.log(in_channels, 2) + b) / gamma))
            k_size = max(k_size if k_size % 2 else k_size + 1, 3)
        
        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 注意力加权后的特征图 [B, C, H, W]
        """
        # 检查输入通道数是否与初始化时的通道数一致
        if x.size(1) != self.in_channels:
            # 动态创建新的卷积层以适应不同的通道数
            k_size = self.k_size
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False).to(x.device)
            self.in_channels = x.size(1)

        # 全局平均池化
        y = self.avg_pool(x)  # [B, C, 1, 1]

        # 转换为序列并应用1D卷积
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.squeeze(1).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        # 应用sigmoid激活
        y = self.sigmoid(y)

        # 通道注意力加权
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    通过计算空间维度的注意力权重来增强重要的空间位置
    
    Args:
        kernel_size (int): 卷积核大小，必须为3或7
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "空间注意力卷积核大小必须为3或7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 空间注意力加权后的特征图 [B, C, H, W]
        """
        # 沿着通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接特征
        y = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # 应用卷积和sigmoid激活
        y = self.conv(y)  # [B, 1, H, W]
        y = self.sigmoid(y)

        # 空间注意力加权（广播机制自动处理维度）
        return x * y


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    
    使用平均池化和最大池化来生成通道注意力权重
    
    Args:
        in_channels (int): 输入通道数
        reduction_ratio (int): 通道缩减比例
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 通道注意力权重 [B, C, 1, 1]
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    卷积块注意力模块 (CBAM)
    
    结合通道注意力和空间注意力的复合注意力机制
    
    Args:
        in_channels (int): 输入通道数
        reduction_ratio (int): 通道注意力的缩减比例
        spatial_kernel_size (int): 空间注意力的卷积核大小
    """
    
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: CBAM处理后的特征图 [B, C, H, W]
        """
        # 检查输入通道数是否与初始化时的通道数一致
        if x.size(1) != self.in_channels:
            # 动态创建新的通道注意力模块
            self.channel_attention = ChannelAttention(x.size(1), reduction_ratio=16)
            self.in_channels = x.size(1)

        # 应用通道注意力
        x = x * self.channel_attention(x)

        # 应用空间注意力
        x = self.spatial_attention(x)

        return x


class DualAttentionBlock(nn.Module):
    """
    双重注意力块
    
    结合ECA和空间注意力的双重注意力机制，这是GeoVisNet的核心组件
    
    Args:
        in_channels (int): 输入通道数
        eca_k_size (int): ECA模块的卷积核大小
        spatial_kernel_size (int): 空间注意力的卷积核大小
    """
    
    def __init__(self, in_channels, eca_k_size=3, spatial_kernel_size=7):
        super(DualAttentionBlock, self).__init__()
        self.eca = EfficientChannelAttention(in_channels, k_size=eca_k_size)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel_size)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 双重注意力处理后的特征图 [B, C, H, W]
        """
        # 先应用通道注意力
        x = self.eca(x)
        # 再应用空间注意力
        x = self.spatial(x)
        return x
