# GeoVisNet 使用指南

本文档详细介绍如何使用GeoVisNet进行训练、测试和推理。

## 快速开始

### 1. 基本训练

```bash
python scripts/train.py \
    --data_root /path/to/UAV_VisLoc_dataset \
    --save_dir ./saved_models \
    --log_dir ./logs \
    --backbone efficientnet_b0 \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.0005
```

### 2. 基本测试

```bash
python scripts/test.py \
    --data_root /path/to/UAV_VisLoc_dataset \
    --model_path ./saved_models/best_model.pth \
    --output_dir ./test_results \
    --backbone efficientnet_b0
```

## 详细使用说明

### 训练模型

#### 基本训练命令

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --save_dir ./models \
    --log_dir ./logs \
    --experiment_name my_experiment
```

#### 高级训练选项

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --train_regions 01,02,03,04,05,06 \
    --val_regions 07 \
    --backbone efficientnet_b2 \
    --emb_size 512 \
    --batch_size 12 \
    --epochs 200 \
    --lr 0.0003 \
    --weight_decay 0.001 \
    --warmup_epochs 20 \
    --enhanced_augmentation \
    --use_mixup \
    --mixup_alpha 0.8 \
    --use_cross_mixup \
    --cross_mixup_alpha 0.2 \
    --use_amp \
    --gradient_clip 1.0 \
    --save_dir ./models \
    --log_dir ./logs \
    --experiment_name efficientnet_b2_enhanced
```

#### 恢复训练

```bash
python scripts/train.py \
    --resume ./models/checkpoint.pth \
    --data_root /path/to/data \
    --save_dir ./models \
    --log_dir ./logs
```

### 测试模型

#### 基本测试

```bash
python scripts/test.py \
    --data_root /path/to/data \
    --model_path ./models/best_model.pth \
    --output_dir ./results
```

#### 详细测试

```bash
python scripts/test.py \
    --data_root /path/to/data \
    --test_regions 08,10,11 \
    --model_path ./models/best_model.pth \
    --backbone efficientnet_b0 \
    --batch_size 32 \
    --output_dir ./results \
    --save_predictions \
    --save_detailed_results
```

### 使用配置文件

#### 创建配置文件

```python
from geovisnet.configs import get_training_config

# 获取默认配置
config = get_training_config('default')

# 获取高质量训练配置
config = get_training_config('high_quality')

# 自定义配置
config = get_training_config('default', 
    model={'backbone': 'efficientnet_b2'},
    training={'lr': 0.0001, 'epochs': 150}
)
```

#### 使用预定义配置

```bash
# 快速调试配置
python scripts/train.py --config fast_debug

# 高质量训练配置
python scripts/train.py --config high_quality

# EfficientNet-B2配置
python scripts/train.py --config efficientnet_b2
```

## 编程接口使用

### 1. 模型创建和使用

```python
from geovisnet.models import GeoVisNet
import torch

# 创建模型
model = GeoVisNet(
    emb_size=256,
    backbone='efficientnet_b0',
    dropout=0.5
)

# 加载预训练权重
model.load_state_dict(torch.load('path/to/model.pth'))

# 推理
model.eval()
with torch.no_grad():
    query_features, _, geo_preds = model(drone_imgs, sat_imgs)
```

### 2. 数据集使用

```python
from geovisnet.data import UAVVisLocDataset, get_train_transforms
from torch.utils.data import DataLoader

# 创建数据集
transform = get_train_transforms(224, enhanced_augmentation=True)
dataset = UAVVisLocDataset(
    data_root='/path/to/data',
    regions=['01', '02'],
    img_size=224,
    transform=transform,
    mode='train'
)

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8
)
```

### 3. 自定义训练循环

```python
import torch
import torch.nn.functional as F
from geovisnet.models import GeoVisNet
from geovisnet.utils import AverageMeter

# 创建模型和优化器
model = GeoVisNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 训练循环
model.train()
loss_meter = AverageMeter()

for batch in dataloader:
    drone_imgs = batch['drone_img'].cuda()
    sat_imgs = batch['sat_img'].cuda()
    targets = torch.stack([batch['norm_lat'], batch['norm_lon']], dim=1).cuda()
    
    optimizer.zero_grad()
    _, _, geo_preds = model(drone_imgs, sat_imgs)
    loss = F.mse_loss(geo_preds, targets)
    loss.backward()
    optimizer.step()
    
    loss_meter.update(loss.item(), drone_imgs.size(0))

print(f'平均损失: {loss_meter.avg:.6f}')
```

## 高级功能

### 1. 注意力可视化

```python
from geovisnet.visualization import visualize_attention

# 获取注意力图
attention_maps = model.get_attention_maps(drone_imgs, sat_imgs)

# 可视化注意力
visualize_attention(
    drone_imgs[0], 
    sat_imgs[0], 
    attention_maps,
    save_path='./attention_vis.png'
)
```

### 2. 模型分析

```python
# 获取模型信息
model_info = model.get_model_info()
print(f"总参数: {model_info['total_params']:,}")
print(f"可训练参数: {model_info['trainable_params']:,}")

# 冻结/解冻骨干网络
model.freeze_backbone()    # 冻结
model.unfreeze_backbone()  # 解冻
```

### 3. 自定义损失函数

```python
import torch.nn as nn

class CustomGeoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        return self.alpha * mse_loss + self.beta * l1_loss

# 使用自定义损失
criterion = CustomGeoLoss(alpha=1.0, beta=0.5)
```

## 实验和消融研究

### 1. 消融实验

```bash
# 无注意力机制
python scripts/train.py --config ablation_no_attention

# 无MixUp数据增强
python scripts/train.py --config ablation_no_mixup

# 冻结骨干网络
python scripts/train.py --config ablation_frozen_backbone
```

### 2. 超参数搜索

```python
from geovisnet.configs import get_hyperparameter_search_configs

# 获取超参数搜索配置
configs = get_hyperparameter_search_configs()

for config in configs:
    experiment_name = config['experiment_name']
    # 运行实验...
```

### 3. 模型比较

```bash
# EfficientNet-B0
python scripts/train.py --backbone efficientnet_b0 --experiment_name b0_baseline

# EfficientNet-B2
python scripts/train.py --backbone efficientnet_b2 --experiment_name b2_baseline
```

## 性能优化

### 1. 训练加速

```bash
# 使用混合精度训练
python scripts/train.py --use_amp

# 增加批次大小
python scripts/train.py --batch_size 32

# 使用更多工作线程
python scripts/train.py --num_workers 16
```

### 2. 内存优化

```bash
# 减小批次大小
python scripts/train.py --batch_size 8

# 不缓存卫星图像
python scripts/train.py --no_cache_satellite

# 使用梯度累积
python scripts/train.py --gradient_accumulation_steps 4
```

## 结果分析

### 1. 查看训练日志

```bash
# 查看实验日志
tail -f logs/my_experiment/main.log

# 查看训练指标
grep "训练结果" logs/my_experiment/train.log
```

### 2. 分析测试结果

```python
import json
import pandas as pd

# 加载测试结果
with open('test_results/test_results.json', 'r') as f:
    results = json.load(f)

# 查看总体结果
overall = results['overall']
print(f"平均误差: {overall['mean']:.2f}m")
print(f"中位误差: {overall['median']:.2f}m")

# 加载详细结果
df = pd.read_csv('test_results/detailed_results.csv')
print(df.describe())
```

## 常见问题

### 1. 训练不收敛
- 检查学习率是否合适
- 尝试使用预热学习率
- 检查数据预处理是否正确

### 2. 内存不足
- 减小批次大小
- 使用混合精度训练
- 关闭卫星图像缓存

### 3. 训练速度慢
- 使用更多GPU
- 增加数据加载器工作线程
- 使用更快的存储设备

---

更多详细信息请参考[API文档](api.md)。
