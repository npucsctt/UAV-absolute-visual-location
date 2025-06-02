#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet主模型

基于双重注意力机制的无人机-卫星图像地理定位网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import AdvancedFeatureExtractor, EnhancedGeoHead, ConvBatchNormReLU, DropBlock2D
from .attention import EfficientChannelAttention, SpatialAttention


class GeoVisNet(nn.Module):
    """
    GeoVisNet: 基于双重注意力机制的无人机-卫星图像地理定位网络
    
    该模型使用EfficientNet作为骨干网络，结合ECA和空间注意力机制，
    实现精确的地理坐标预测。
    
    Args:
        emb_size (int): 嵌入维度，默认256
        leaky (bool): 是否使用LeakyReLU激活函数
        backbone (str): 骨干网络类型，支持 'efficientnet_b0' 和 'efficientnet_b2'
        dropout (float): Dropout概率
        satellite_backbone (str): 卫星图像特征提取器的骨干网络，默认与backbone相同
    """
    
    def __init__(self, emb_size=256, leaky=True, backbone='efficientnet_b0', 
                 dropout=0.5, satellite_backbone=None):
        super(GeoVisNet, self).__init__()

        # 确保使用支持的backbone
        if backbone not in ['efficientnet_b0', 'efficientnet_b2']:
            print(f"警告: 只支持 efficientnet_b0 和 efficientnet_b2 模型，将使用 efficientnet_b0 替代 {backbone}")
            backbone = 'efficientnet_b0'

        # 无人机图像特征提取器
        self.query_extractor = AdvancedFeatureExtractor(
            model_name=backbone, 
            pretrained=True, 
            dropout=dropout
        )
        self.query_visudim = self.query_extractor.output_dim

        # 卫星图像特征提取器
        sat_backbone = satellite_backbone if satellite_backbone else backbone
        if sat_backbone not in ['efficientnet_b0', 'efficientnet_b2']:
            print(f"警告: 只支持 efficientnet_b0 和 efficientnet_b2 模型，将使用 efficientnet_b0 替代 {sat_backbone}")
            sat_backbone = 'efficientnet_b0'

        self.reference_extractor = AdvancedFeatureExtractor(
            model_name=sat_backbone, 
            pretrained=True, 
            dropout=dropout
        )
        self.reference_visudim = self.reference_extractor.output_dim

        # 特征映射层
        use_instnorm = True  # 使用实例归一化增强正则化
        self.query_mapping = ConvBatchNormReLU(
            self.query_visudim, emb_size, 1, 1, 0, 1, 
            leaky=leaky, instance=use_instnorm
        )
        self.reference_mapping = ConvBatchNormReLU(
            self.reference_visudim, emb_size, 1, 1, 0, 1, 
            leaky=leaky, instance=use_instnorm
        )

        # 地理坐标预测头
        self.geo_head = EnhancedGeoHead(emb_size, hidden_dim=256, dropout=dropout)
        
        # 双重注意力模块
        self.eca = EfficientChannelAttention(emb_size)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # 正则化模块
        self.dropblock = DropBlock2D(drop_prob=0.2, block_size=7)
        self.dropout = nn.Dropout2d(dropout)

        # 注意力融合模块
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(emb_size * 2, emb_size, kernel_size=1),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_size, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def extract_query_features(self, query_imgs):
        """
        提取无人机图像特征
        
        Args:
            query_imgs (torch.Tensor): 无人机图像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: 提取的特征 [B, emb_size, H', W']
        """
        query_fvisu, _ = self.query_extractor(query_imgs)
        query_fvisu = self.query_mapping(query_fvisu)
        return query_fvisu

    def extract_reference_features(self, reference_imgs):
        """
        提取卫星图像特征
        
        Args:
            reference_imgs (torch.Tensor): 卫星图像 [B, 3, H, W]
            
        Returns:
            torch.Tensor: 提取的特征 [B, emb_size, H', W']
        """
        reference_fvisu, _ = self.reference_extractor(reference_imgs)
        reference_fvisu = self.reference_mapping(reference_fvisu)
        return reference_fvisu

    def apply_dual_attention(self, features):
        """
        应用双重注意力机制
        
        Args:
            features (torch.Tensor): 输入特征 [B, C, H, W]
            
        Returns:
            torch.Tensor: 注意力增强后的特征 [B, C, H, W]
        """
        # 先应用ECA通道注意力
        features = self.eca(features)
        # 再应用空间注意力
        features = self.spatial_attn(features)
        return features

    def forward(self, query_imgs, reference_imgs):
        """
        前向传播
        
        Args:
            query_imgs (torch.Tensor): 无人机图像 [B, 3, H, W]
            reference_imgs (torch.Tensor): 卫星图像 [B, 3, H, W]
            
        Returns:
            tuple: (query_global_features, similarity_features, geo_predictions)
                - query_global_features: 无人机图像全局特征 [B, emb_size]
                - similarity_features: 特征相似性（为兼容性保留，返回None）
                - geo_predictions: 地理坐标预测 [B, 2]
        """
        # 提取特征
        query_fvisu = self.extract_query_features(query_imgs)
        reference_fvisu = self.extract_reference_features(reference_imgs)

        # 应用双重注意力机制
        query_fvisu = self.apply_dual_attention(query_fvisu)
        reference_fvisu = self.apply_dual_attention(reference_fvisu)

        # 训练时应用正则化
        if self.training:
            query_fvisu = self.dropblock(query_fvisu)
            reference_fvisu = self.dropblock(reference_fvisu)
            query_fvisu = self.dropout(query_fvisu)
            reference_fvisu = self.dropout(reference_fvisu)

        # 计算全局特征（用于兼容性）
        query_global = F.adaptive_avg_pool2d(query_fvisu, 1).squeeze(-1).squeeze(-1)

        # 注意力融合机制
        # 拼接特征
        concat_features = torch.cat([query_fvisu, reference_fvisu], dim=1)
        # 生成注意力权重
        attention_weights = self.attention_fusion(concat_features)
        # 应用注意力权重
        fused_feature = (attention_weights[:, 0:1, :, :] * query_fvisu + 
                        attention_weights[:, 1:2, :, :] * reference_fvisu)

        # 预测地理坐标
        geo_preds = self.geo_head(fused_feature)

        # 返回结果（保持与原始接口兼容）
        return query_global, None, geo_preds

    def get_attention_maps(self, query_imgs, reference_imgs):
        """
        获取注意力图（用于可视化）
        
        Args:
            query_imgs (torch.Tensor): 无人机图像 [B, 3, H, W]
            reference_imgs (torch.Tensor): 卫星图像 [B, 3, H, W]
            
        Returns:
            dict: 包含各种注意力图的字典
        """
        self.eval()
        with torch.no_grad():
            # 提取特征
            query_fvisu = self.extract_query_features(query_imgs)
            reference_fvisu = self.extract_reference_features(reference_imgs)
            
            # 获取ECA注意力权重（近似）
            query_eca_weights = self.eca(query_fvisu) / (query_fvisu + 1e-8)
            reference_eca_weights = self.eca(reference_fvisu) / (reference_fvisu + 1e-8)
            
            # 获取空间注意力权重（近似）
            query_spatial_weights = self.spatial_attn(query_fvisu) / (query_fvisu + 1e-8)
            reference_spatial_weights = self.spatial_attn(reference_fvisu) / (reference_fvisu + 1e-8)
            
            return {
                'query_eca': query_eca_weights,
                'query_spatial': query_spatial_weights,
                'reference_eca': reference_eca_weights,
                'reference_spatial': reference_spatial_weights
            }

    def freeze_backbone(self):
        """冻结骨干网络的权重"""
        print('冻结骨干网络的权重')
        
        # 冻结无人机图像特征提取器
        if hasattr(self.query_extractor, 'base_model'):
            for param in self.query_extractor.base_model.parameters():
                param.requires_grad = False
            print('冻结无人机图像特征提取器')

        # 冻结卫星图像特征提取器
        if hasattr(self.reference_extractor, 'base_model'):
            for param in self.reference_extractor.base_model.parameters():
                param.requires_grad = False
            print('冻结卫星图像特征提取器')

    def unfreeze_backbone(self):
        """解冻骨干网络的权重"""
        print('解冻骨干网络的权重')
        
        # 解冻无人机图像特征提取器
        if hasattr(self.query_extractor, 'base_model'):
            for param in self.query_extractor.base_model.parameters():
                param.requires_grad = True
            print('解冻无人机图像特征提取器')

        # 解冻卫星图像特征提取器
        if hasattr(self.reference_extractor, 'base_model'):
            for param in self.reference_extractor.base_model.parameters():
                param.requires_grad = True
            print('解冻卫星图像特征提取器')

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone': 'efficientnet',
            'attention_mechanism': 'dual_attention (ECA + Spatial)'
        }
