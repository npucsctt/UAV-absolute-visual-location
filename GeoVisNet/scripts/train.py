#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet训练脚本

使用命令行参数进行模型训练
"""

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geovisnet.models import GeoVisNet
from geovisnet.data import UAVVisLocDataset, get_train_transforms, get_val_transforms
from geovisnet.utils import (
    AverageMeter, setup_logging, CheckpointManager, 
    create_checkpoint_state, TrainingLogger
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GeoVisNet训练脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--train_regions', type=str, default='01,02,03,04,05,06', help='训练区域')
    parser.add_argument('--val_regions', type=str, default='07', help='验证区域')
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--sat_patch_scale', type=float, default=3.0, help='卫星图像缩放比例')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                       choices=['efficientnet_b0', 'efficientnet_b2'], help='骨干网络')
    parser.add_argument('--emb_size', type=int, default=256, help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--freeze_backbone', action='store_true', help='冻结骨干网络')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='预热轮数')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01, help='最小学习率比例')
    
    # 数据增强参数
    parser.add_argument('--enhanced_augmentation', action='store_true', help='增强数据增强')
    parser.add_argument('--use_mixup', action='store_true', help='使用MixUp')
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help='MixUp强度')
    parser.add_argument('--use_cross_mixup', action='store_true', help='使用跨视图MixUp')
    parser.add_argument('--cross_mixup_alpha', type=float, default=0.2, help='跨视图MixUp强度')
    parser.add_argument('--cross_mixup_prob', type=float, default=0.3, help='跨视图MixUp概率')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'smooth_l1'], help='损失函数类型')
    parser.add_argument('--smooth_l1_beta', type=float, default=0.1, help='Smooth L1损失的beta参数')
    
    # 训练策略参数
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪阈值')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, required=True, help='模型保存目录')
    parser.add_argument('--log_dir', type=str, required=True, help='日志目录')
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--eval_interval', type=int, default=1, help='评估间隔')
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载器工作线程数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_scheduler(base_lr, current_epoch, warmup_epochs, max_epochs, min_lr_ratio=0.01):
    """余弦学习率调度器"""
    if current_epoch < warmup_epochs:
        return base_lr * current_epoch / warmup_epochs
    else:
        min_lr = base_lr * min_lr_ratio
        return min_lr + 0.5 * (base_lr - min_lr) * (
            1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (max_epochs - warmup_epochs))
        )


def create_loss_function(loss_type, smooth_l1_beta=0.1):
    """创建损失函数"""
    if loss_type == 'smooth_l1':
        def smooth_l1_loss(pred, target):
            diff = torch.abs(pred - target)
            cond = diff < smooth_l1_beta
            loss = torch.where(cond, 0.5 * diff ** 2 / smooth_l1_beta, diff - 0.5 * smooth_l1_beta)
            return loss.mean()
        return smooth_l1_loss
    else:  # mse
        return F.mse_loss


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, args, scaler=None):
    """训练一个epoch"""
    model.train()
    loss_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(data_loader):
        # 数据移动到设备
        drone_imgs = batch['drone_img'].to(device, non_blocking=True)
        sat_imgs = batch['sat_img'].to(device, non_blocking=True)
        norm_lat = batch['norm_lat'].to(device, non_blocking=True)
        norm_lon = batch['norm_lon'].to(device, non_blocking=True)
        
        # 组合目标坐标
        targets = torch.stack([norm_lat, norm_lon], dim=1)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if args.use_amp and scaler is not None:
            with autocast():
                _, _, geo_preds = model(drone_imgs, sat_imgs)
                loss = criterion(geo_preds, targets)
            
            # 反向传播
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, _, geo_preds = model(drone_imgs, sat_imgs)
            loss = criterion(geo_preds, targets)
            
            # 反向传播
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        
        # 更新损失记录
        loss_meter.update(loss.item(), drone_imgs.size(0))
        
        # 记录日志
        if (batch_idx + 1) % args.log_interval == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss_meter.avg:.6f}')
    
    return loss_meter.avg


def validate(model, data_loader, criterion, device):
    """验证函数"""
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in data_loader:
            drone_imgs = batch['drone_img'].to(device, non_blocking=True)
            sat_imgs = batch['sat_img'].to(device, non_blocking=True)
            norm_lat = batch['norm_lat'].to(device, non_blocking=True)
            norm_lon = batch['norm_lon'].to(device, non_blocking=True)
            
            targets = torch.stack([norm_lat, norm_lon], dim=1)
            
            _, _, geo_preds = model(drone_imgs, sat_imgs)
            loss = criterion(geo_preds, targets)
            
            loss_meter.update(loss.item(), drone_imgs.size(0))
    
    return loss_meter.avg


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建实验名称
    if args.experiment_name is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        args.experiment_name = f'geovisnet_{args.backbone}_{timestamp}'
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 设置日志
    logger = TrainingLogger(args.log_dir, args.experiment_name)
    logger.log_config(vars(args))
    
    # 创建数据变换
    train_transform = get_train_transforms(args.img_size, args.enhanced_augmentation)
    val_transform = get_val_transforms(args.img_size)
    
    # 解析区域
    train_regions = args.train_regions.split(',')
    val_regions = args.val_regions.split(',')
    
    # 创建数据集
    train_dataset = UAVVisLocDataset(
        data_root=args.data_root,
        regions=train_regions,
        img_size=args.img_size,
        transform=train_transform,
        augment=True,
        mode='train',
        cache_satellite=True,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_cross_mixup=args.use_cross_mixup,
        cross_mixup_alpha=args.cross_mixup_alpha,
        cross_mixup_prob=args.cross_mixup_prob,
        sat_patch_scale=args.sat_patch_scale
    )
    
    val_dataset = UAVVisLocDataset(
        data_root=args.data_root,
        regions=val_regions,
        img_size=args.img_size,
        transform=val_transform,
        augment=False,
        mode='val',
        cache_satellite=True,
        sat_patch_scale=args.sat_patch_scale
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = GeoVisNet(
        emb_size=args.emb_size,
        backbone=args.backbone,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 冻结骨干网络（如果需要）
    if args.freeze_backbone:
        model.freeze_backbone()
    
    # 记录模型信息
    logger.log_model_info(model)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = create_loss_function(args.loss_type, args.smooth_l1_beta)
    
    # 创建混合精度训练的缩放器
    scaler = GradScaler() if args.use_amp else None
    
    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(args.save_dir)
    
    # 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 恢复训练（如果指定）
    if args.resume:
        resume_info = checkpoint_manager.load_latest(model, optimizer)
        if resume_info['success']:
            start_epoch = resume_info.get('start_epoch', 0)
            best_val_loss = resume_info.get('best_metric', float('inf'))
    
    # 训练循环
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        logger.log_epoch_start(epoch, args.epochs)
        
        # 更新学习率
        lr = cosine_scheduler(args.lr, epoch, args.warmup_epochs, args.epochs, args.min_lr_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args, scaler)
        
        # 记录训练指标
        logger.log_training_metrics(epoch, {'loss': train_loss}, lr)
        
        # 验证
        if (epoch + 1) % args.eval_interval == 0:
            val_loss = validate(model, val_loader, criterion, device)
            logger.log_validation_metrics(epoch, {'loss': val_loss})
            
            # 检查是否是最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                logger.log_best_model(epoch, 'val_loss', val_loss)
            
            # 保存检查点
            state = create_checkpoint_state(
                model, optimizer, epoch + 1,
                {'train_loss': train_loss, 'val_loss': val_loss},
                {'best_val_loss': best_val_loss}
            )
            checkpoint_manager.save(state, epoch + 1, is_best, 'val_loss', val_loss)
        
        # 定期保存
        elif (epoch + 1) % args.save_interval == 0:
            state = create_checkpoint_state(
                model, optimizer, epoch + 1,
                {'train_loss': train_loss},
                {'best_val_loss': best_val_loss}
            )
            checkpoint_manager.save(state, epoch + 1, False)
    
    # 训练完成
    total_time = time.time() - start_time
    logger.log_experiment_end(total_time, {'best_val_loss': best_val_loss})
    
    print(f'训练完成！最佳验证损失: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()
