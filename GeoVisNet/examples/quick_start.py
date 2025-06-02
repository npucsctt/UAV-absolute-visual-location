#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet快速开始示例

演示如何快速使用GeoVisNet进行训练和推理
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geovisnet.models import GeoVisNet
from geovisnet.data import UAVVisLocDataset, get_train_transforms, get_val_transforms
from geovisnet.utils import setup_logging, AverageMeter


def quick_training_example():
    """快速训练示例"""
    print("GeoVisNet快速训练示例")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径（请根据实际情况修改）
    data_root = "/path/to/your/UAV_VisLoc_dataset"
    
    # 检查数据路径是否存在
    if not os.path.exists(data_root):
        print(f"数据路径不存在: {data_root}")
        print("请修改data_root变量为正确的数据集路径")
        return
    
    # 创建模型
    model = GeoVisNet(
        emb_size=256,
        backbone='efficientnet_b0',
        dropout=0.5
    )
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据变换
    train_transform = get_train_transforms(224, enhanced_augmentation=True)
    val_transform = get_val_transforms(224)
    
    # 创建数据集（使用小规模数据进行快速测试）
    train_dataset = UAVVisLocDataset(
        data_root=data_root,
        regions=['01'],  # 只使用一个区域进行快速测试
        img_size=224,
        transform=train_transform,
        augment=True,
        mode='train',
        cache_satellite=True
    )
    
    val_dataset = UAVVisLocDataset(
        data_root=data_root,
        regions=['07'],
        img_size=224,
        transform=val_transform,
        augment=False,
        mode='val',
        cache_satellite=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # 小批次用于快速测试
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.MSELoss()
    
    # 训练几个epoch
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        model.train()
        train_loss = AverageMeter()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # 只训练10个批次用于演示
                break
                
            drone_imgs = batch['drone_img'].to(device)
            sat_imgs = batch['sat_img'].to(device)
            norm_lat = batch['norm_lat'].to(device)
            norm_lon = batch['norm_lon'].to(device)
            
            targets = torch.stack([norm_lat, norm_lon], dim=1)
            
            optimizer.zero_grad()
            _, _, geo_preds = model(drone_imgs, sat_imgs)
            loss = criterion(geo_preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), drone_imgs.size(0))
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}/10, Loss: {train_loss.avg:.6f}")
        
        print(f"  训练损失: {train_loss.avg:.6f}")
        
        # 验证
        model.eval()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 5:  # 只验证5个批次用于演示
                    break
                    
                drone_imgs = batch['drone_img'].to(device)
                sat_imgs = batch['sat_img'].to(device)
                norm_lat = batch['norm_lat'].to(device)
                norm_lon = batch['norm_lon'].to(device)
                
                targets = torch.stack([norm_lat, norm_lon], dim=1)
                
                _, _, geo_preds = model(drone_imgs, sat_imgs)
                loss = criterion(geo_preds, targets)
                
                val_loss.update(loss.item(), drone_imgs.size(0))
        
        print(f"  验证损失: {val_loss.avg:.6f}")
    
    print("\n快速训练完成！")
    return model


def inference_example(model=None):
    """推理示例"""
    print("\nGeoVisNet推理示例")
    print("=" * 50)
    
    if model is None:
        # 创建模型
        model = GeoVisNet(
            emb_size=256,
            backbone='efficientnet_b0',
            dropout=0.5
        )
        
        # 这里应该加载预训练权重
        # model.load_state_dict(torch.load('path/to/pretrained/model.pth'))
        print("注意: 使用随机初始化的模型进行演示")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建随机输入用于演示
    batch_size = 4
    drone_imgs = torch.randn(batch_size, 3, 224, 224).to(device)
    sat_imgs = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"输入形状:")
    print(f"  无人机图像: {drone_imgs.shape}")
    print(f"  卫星图像: {sat_imgs.shape}")
    
    # 推理
    with torch.no_grad():
        query_features, similarity_features, geo_preds = model(drone_imgs, sat_imgs)
    
    print(f"\n输出形状:")
    print(f"  查询特征: {query_features.shape}")
    print(f"  地理坐标预测: {geo_preds.shape}")
    
    print(f"\n预测的归一化坐标:")
    for i in range(batch_size):
        lat, lon = geo_preds[i].cpu().numpy()
        print(f"  样本 {i+1}: 纬度={lat:.4f}, 经度={lon:.4f}")
    
    return geo_preds


def attention_visualization_example(model=None):
    """注意力可视化示例"""
    print("\nGeoVisNet注意力可视化示例")
    print("=" * 50)
    
    if model is None:
        model = GeoVisNet(
            emb_size=256,
            backbone='efficientnet_b0',
            dropout=0.5
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建随机输入
    drone_imgs = torch.randn(1, 3, 224, 224).to(device)
    sat_imgs = torch.randn(1, 3, 224, 224).to(device)
    
    # 获取注意力图
    attention_maps = model.get_attention_maps(drone_imgs, sat_imgs)
    
    print("注意力图形状:")
    for key, value in attention_maps.items():
        print(f"  {key}: {value.shape}")
    
    print("\n注意力可视化功能已准备就绪")
    print("可以使用matplotlib等工具将注意力图可视化")
    
    return attention_maps


def main():
    """主函数"""
    print("GeoVisNet快速开始指南")
    print("=" * 60)
    
    # 设置日志
    setup_logging('logs', model_name='quick_start')
    
    try:
        # 1. 快速训练示例
        model = quick_training_example()
        
        # 2. 推理示例
        inference_example(model)
        
        # 3. 注意力可视化示例
        attention_visualization_example(model)
        
        print("\n" + "=" * 60)
        print("快速开始指南完成！")
        print("\n接下来您可以:")
        print("1. 修改数据路径并使用完整数据集进行训练")
        print("2. 调整模型参数和训练配置")
        print("3. 使用scripts/train.py进行完整训练")
        print("4. 使用scripts/test.py进行模型评估")
        
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        print("请检查数据路径和环境配置")


if __name__ == '__main__':
    main()
