#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet模型组件

包含GeoVisNet中使用的各种组件：
- AdvancedFeatureExtractor: 高级特征提取器
- EnhancedGeoHead: 增强版地理坐标预测头
- ConvBatchNormReLU: 卷积-批归一化-激活层
- DropBlock2D: 结构化dropout
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .attention import EfficientChannelAttention, SpatialAttention


class DropBlock2D(nn.Module):
    """
    DropBlock: 一种结构化的dropout方法，用于卷积神经网络
    
    论文: DropBlock: A regularization method for convolutional networks
    
    Args:
        drop_prob (float): 丢弃概率
        block_size (int): 丢弃块的大小
    """
    
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: DropBlock处理后的特征图 [B, C, H, W]
        """
        # 形状: (batch_size, channels, height, width)
        if not self.training or self.drop_prob == 0:
            return x
        
        # 获取gamma值
        gamma = self._compute_gamma(x)
        
        # 采样掩码
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        mask = mask.to(x.device)
        
        # 计算块掩码
        block_mask = self._compute_block_mask(mask)
        
        # 应用块掩码
        out = x * block_mask[:, None, :, :]
        
        # 缩放输出
        out = out * block_mask.numel() / block_mask.sum()
        
        return out
    
    def _compute_block_mask(self, mask):
        """计算块掩码"""
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
            
        block_mask = 1 - block_mask.squeeze(1)
        
        return block_mask
    
    def _compute_gamma(self, x):
        """计算gamma值"""
        return self.drop_prob / (self.block_size ** 2)


class ConvBatchNormReLU(nn.Module):
    """
    卷积-批归一化-激活层组合
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
        dilation (int): 膨胀率
        leaky (bool): 是否使用LeakyReLU
        instance (bool): 是否使用实例归一化
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, leaky=False, instance=False):
        super(ConvBatchNormReLU, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False
        )
        
        if instance:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.BatchNorm2d(out_channels)
            
        if leaky:
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """前向传播"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class AdvancedFeatureExtractor(nn.Module):
    """
    高级特征提取器
    
    支持EfficientNet和其他模型，使用双重注意力机制，增加正则化
    
    Args:
        model_name (str): 模型名称，支持 'efficientnet_b0' 和 'efficientnet_b2'
        pretrained (bool): 是否使用预训练权重
        offline_mode (bool): 是否离线模式
        dropout (float): Dropout概率
    """
    
    def __init__(self, model_name='efficientnet_b0', pretrained=True, offline_mode=False, dropout=0.5):
        super(AdvancedFeatureExtractor, self).__init__()

        # 检查是否在离线模式
        if 'OFFLINE_MODE' in os.environ and os.environ['OFFLINE_MODE'] == '1':
            offline_mode = True

        # 初始化属性
        self.base_model = None
        self.feature_dims = None
        self.output_dim = None
        self.dropout_rate = dropout

        # 只支持 EfficientNet B0 和 B2 模型
        if model_name not in ['efficientnet_b0', 'efficientnet_b2']:
            print(f"警告: 只支持 efficientnet_b0 和 efficientnet_b2 模型，将使用 efficientnet_b0 替代 {model_name}")
            model_name = 'efficientnet_b0'

        # 加载模型
        self._load_model(model_name, pretrained, offline_mode)
        
        # 获取特征维度
        self.feature_dims = self.base_model.feature_info.channels()
        self.output_dim = self.feature_dims[-1]

        # 初始化其他组件
        self._init_components()

    def _load_model(self, model_name, pretrained, offline_mode):
        """加载基础模型"""
        model_dir = os.environ.get('TORCH_HOME', '/data1/ctt/model/detgeo/pretrained_weights')
        features_model_path = os.path.join(model_dir, f'{model_name}_features.pth')
        full_model_path = os.path.join(model_dir, f'{model_name}.pth')

        # 尝试加载特征提取器模型
        if os.path.exists(features_model_path) and pretrained:
            try:
                print(f"加载本地预训练{model_name}特征提取器模型: {features_model_path}")
                self.base_model = timm.create_model(model_name, pretrained=False, features_only=True)
                state_dict = torch.load(features_model_path)
                self.base_model.load_state_dict(state_dict)
                print("成功加载本地特征提取器模型")
            except Exception as e:
                print(f"加载特征提取器模型失败: {e}")
                self._load_full_model(model_name, full_model_path)
        elif os.path.exists(full_model_path) and pretrained:
            self._load_full_model(model_name, full_model_path)
        elif offline_mode:
            print(f"离线模式: 使用随机初始化的{model_name}模型")
            self.base_model = timm.create_model(model_name, pretrained=False, features_only=True)
        else:
            try:
                print(f"尝试在线加载{model_name}模型，超时设置为120秒")
                import urllib.request
                urllib.request.socket.setdefaulttimeout(120)
                self.base_model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
            except Exception as e:
                print(f"在线加载模型失败: {e}")
                print(f"使用随机初始化的{model_name}模型")
                self.base_model = timm.create_model(model_name, pretrained=False, features_only=True)

    def _load_full_model(self, model_name, full_model_path):
        """从完整模型加载参数"""
        try:
            print(f"加载本地预训练{model_name}完整模型: {full_model_path}")
            temp_model = timm.create_model(model_name, pretrained=False)
            temp_model.load_state_dict(torch.load(full_model_path))
            self.base_model = timm.create_model(model_name, pretrained=False, features_only=True)
            feature_dict = self.base_model.state_dict()
            full_dict = temp_model.state_dict()
            shared_keys = set(feature_dict.keys()) & set(full_dict.keys())
            copied_state_dict = {k: full_dict[k] for k in shared_keys}
            self.base_model.load_state_dict(copied_state_dict, strict=False)
            print("成功从完整模型提取并加载参数")
        except Exception as e:
            print(f"从完整模型加载参数失败: {e}")
            self.base_model = timm.create_model(model_name, pretrained=False, features_only=True)

    def _init_components(self):
        """初始化注意力模块和融合层，增加正则化"""
        # 添加ECA注意力模块
        self.eca_modules = nn.ModuleList([
            EfficientChannelAttention(dim) for dim in self.feature_dims
        ])

        # 添加空间注意力模块
        self.spatial_modules = nn.ModuleList([
            SpatialAttention(kernel_size=7) for _ in self.feature_dims
        ])

        # 添加DropBlock正则化
        self.dropblock = DropBlock2D(drop_prob=0.2, block_size=7)

        # 添加Dropout
        self.dropout = nn.Dropout2d(self.dropout_rate)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像 [B, C, H, W]
            
        Returns:
            tuple: (主要特征, 所有特征列表)
        """
        # 提取多尺度特征
        features = self.base_model(x)

        # 确保特征数量一致
        if len(features) != len(self.feature_dims):
            print(f"警告: 特征数量({len(features)})与预期({len(self.feature_dims)})不一致")
            while len(features) < len(self.feature_dims):
                features.append(features[-1])
            if len(features) > len(self.feature_dims):
                features = features[:len(self.feature_dims)]

        # 应用ECA和空间注意力
        enhanced_features = []
        for i, feature in enumerate(features):
            if len(feature.shape) != 4:
                print(f"警告: 特征 {i} 形状不是4D: {feature.shape}")
                if len(feature.shape) == 2:
                    B, C = feature.shape
                    feature = feature.view(B, C, 1, 1)
                elif len(feature.shape) == 3:
                    B, L, C = feature.shape
                    H = W = int(L ** 0.5)
                    feature = feature.transpose(1, 2).reshape(B, C, H, W)

            # 应用注意力和正则化
            channel_attention = self.eca_modules[i](feature)
            spatial_attention = self.spatial_modules[i](channel_attention)

            # 训练时应用更强的正则化
            if self.training:
                spatial_attention = self.dropblock(spatial_attention)
                spatial_attention = self.dropout(spatial_attention)

            enhanced_features.append(spatial_attention)

        # 只使用最后一层特征，简化模型
        output_feature = enhanced_features[-1]

        return output_feature, enhanced_features


class EnhancedGeoHead(nn.Module):
    """
    增强版地理坐标预测头

    使用MLP和空间金字塔池化，增加正则化

    Args:
        embed_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        dropout (float): Dropout概率
    """

    def __init__(self, embed_dim, hidden_dim=256, dropout=0.5):
        super(EnhancedGeoHead, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropblock = DropBlock2D(drop_prob=0.2, block_size=7)

        # 简化MLP结构，减少参数
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid()
        )

        # 简化SPP，只使用两个尺度
        self.spp = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size) for output_size in [1, 2]
        ])

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]

        Returns:
            torch.Tensor: 预测的地理坐标 [B, 2] (归一化的经纬度)
        """
        # 应用DropBlock
        if self.training:
            x = self.dropblock(x)

        spp_features = []
        for pool in self.spp:
            pooled = pool(x)
            flat = self.flatten(pooled)
            spp_features.append(flat)

        if len(spp_features) > 1:
            x = torch.cat(spp_features, dim=1)
            # 动态创建线性层以适应拼接后的特征维度
            x = nn.Linear(x.size(1), self.mlp[0].in_features).to(x.device)(x)
        else:
            x = spp_features[0]

        x = self.mlp(x)
        return x
