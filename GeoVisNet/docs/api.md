# GeoVisNet API 文档

本文档提供了GeoVisNet的详细API参考。

## 核心模块

### geovisnet.models

#### GeoVisNet

主要的地理定位网络模型。

```python
class GeoVisNet(nn.Module):
    def __init__(self, emb_size=256, leaky=True, backbone='efficientnet_b0', 
                 dropout=0.5, satellite_backbone=None):
        """
        初始化GeoVisNet模型
        
        Args:
            emb_size (int): 嵌入维度，默认256
            leaky (bool): 是否使用LeakyReLU激活函数
            backbone (str): 骨干网络类型，支持 'efficientnet_b0' 和 'efficientnet_b2'
            dropout (float): Dropout概率
            satellite_backbone (str): 卫星图像特征提取器的骨干网络
        """
```

**主要方法：**

- `forward(query_imgs, reference_imgs)`: 前向传播
- `get_attention_maps(query_imgs, reference_imgs)`: 获取注意力图
- `freeze_backbone()`: 冻结骨干网络
- `unfreeze_backbone()`: 解冻骨干网络
- `get_model_info()`: 获取模型信息

#### 注意力机制模块

```python
# ECA注意力
class EfficientChannelAttention(nn.Module):
    def __init__(self, in_channels, k_size=3, gamma=2, b=1)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7)

# 双重注意力块
class DualAttentionBlock(nn.Module):
    def __init__(self, in_channels, eca_k_size=3, spatial_kernel_size=7)
```

### geovisnet.data

#### UAVVisLocDataset

UAV-VisLoc数据集加载器。

```python
class UAVVisLocDataset(Dataset):
    def __init__(self, data_root, regions=None, img_size=224, transform=None, 
                 augment=False, mode='train', cache_satellite=False, 
                 use_mixup=False, mixup_alpha=1.0, use_cross_mixup=False, 
                 cross_mixup_alpha=0.2, cross_mixup_prob=0.3, sat_patch_scale=2.0):
        """
        初始化数据集
        
        Args:
            data_root (str): 数据集根目录
            regions (list): 要使用的区域列表
            img_size (int): 图像大小
            transform: 图像变换
            augment (bool): 是否进行数据增强
            mode (str): 'train'、'val'或'test'模式
            cache_satellite (bool): 是否缓存卫星图像
            use_mixup (bool): 是否使用MixUp数据增强
            mixup_alpha (float): MixUp强度参数
            use_cross_mixup (bool): 是否使用跨视图MixUp
            cross_mixup_alpha (float): 跨视图MixUp强度参数
            cross_mixup_prob (float): 跨视图MixUp应用概率
            sat_patch_scale (float): 卫星图像裁剪尺寸的缩放比例
        """
```

#### 数据变换函数

```python
def get_train_transforms(img_size=224, enhanced_augmentation=False):
    """获取训练时的数据变换"""

def get_val_transforms(img_size=224):
    """获取验证时的数据变换"""

def get_test_transforms(img_size=224):
    """获取测试时的数据变换"""
```

### geovisnet.utils

#### 距离计算

```python
def euclidean_geo_distance(lat1, lon1, lat2, lon2):
    """计算两点之间的欧几里得距离（平面近似）"""

def calculate_geo_error(pred_norm_lat, pred_norm_lon, true_norm_lat, true_norm_lon,
                       sat_info, img_size=224, sat_patch_scale=3.0):
    """计算地理误差，考虑卫星图像的缩放因子"""

def calculate_geo_error_with_region_calibration(pred_norm_lat, pred_norm_lon, 
                                               true_norm_lat, true_norm_lon,
                                               sat_info, region, img_size=224):
    """计算地理误差，使用区域特定的校准缩放因子"""
```

#### 评估指标

```python
class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self)
    def reset(self)
    def update(self, val, n=1)

def compute_metrics(pred, target):
    """计算预测结果的评估指标"""

def compute_geo_metrics(pred_coords, true_coords, sat_infos, regions, img_size=224):
    """计算地理坐标的评估指标"""
```

#### 检查点管理

```python
def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth', 
                   best_filename='best_model.pth'):
    """保存模型检查点"""

def load_checkpoint(checkpoint_path, map_location=None):
    """加载模型检查点"""

def load_model_weights(model, checkpoint_path, strict=True, map_location=None):
    """仅加载模型权重"""

class CheckpointManager:
    """检查点管理器"""
    def __init__(self, save_dir, max_checkpoints=5)
    def save(self, state, epoch, is_best=False, metric_name='loss', metric_value=None)
    def load_best(self, model, map_location=None)
    def load_latest(self, model, optimizer=None, map_location=None)
```

#### 日志工具

```python
def setup_logging(log_dir='logs', log_level=logging.INFO, model_name='geovisnet'):
    """设置日志配置"""

class TrainingLogger:
    """训练过程日志记录器"""
    def __init__(self, log_dir, experiment_name)
    def log_config(self, config)
    def log_model_info(self, model)
    def log_training_metrics(self, epoch, metrics, lr=None)
    def log_validation_metrics(self, epoch, metrics)
```

### geovisnet.visualization

#### 注意力可视化

```python
def visualize_attention(drone_img, sat_img, attention_maps, save_path=None, figsize=(15, 10)):
    """可视化注意力图"""

def plot_attention_heatmap(attention_map, title="注意力热图", save_path=None, figsize=(8, 6)):
    """绘制单个注意力热图"""

def create_attention_overlay(image, attention_map, alpha=0.4):
    """创建图像和注意力图的叠加"""
```

#### 结果可视化

```python
def plot_error_distribution(errors, title="地理误差分布", save_path=None, figsize=(12, 8)):
    """绘制误差分布图"""

def plot_region_comparison(errors_by_region, title="各区域性能比较", save_path=None, figsize=(14, 8)):
    """绘制各区域性能比较图"""

def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                        save_path=None, figsize=(15, 10)):
    """绘制训练曲线"""
```

#### 数据集可视化

```python
def visualize_dataset_sample(drone_img, sat_img, drone_coords, sat_coords, 
                           region=None, save_path=None, figsize=(15, 6)):
    """可视化数据集样本"""

def plot_coordinate_distribution(coordinates, regions=None, title="坐标分布图", 
                                save_path=None, figsize=(12, 8)):
    """绘制坐标分布图"""
```

### geovisnet.configs

#### 配置管理

```python
class DefaultConfig:
    """默认配置类"""
    @classmethod
    def get_config(cls)
    @classmethod
    def update_config(cls, updates)

def get_training_config(config_type='default', **kwargs):
    """获取训练配置"""

def get_config(config_name='default'):
    """根据名称获取配置"""
```

## 使用示例

### 基本模型使用

```python
import torch
from geovisnet.models import GeoVisNet

# 创建模型
model = GeoVisNet(
    emb_size=256,
    backbone='efficientnet_b0',
    dropout=0.5
)

# 前向传播
drone_imgs = torch.randn(4, 3, 224, 224)
sat_imgs = torch.randn(4, 3, 224, 224)

query_features, _, geo_preds = model(drone_imgs, sat_imgs)
print(f"地理坐标预测: {geo_preds.shape}")  # [4, 2]
```

### 数据集使用

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
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 获取一个批次
batch = next(iter(dataloader))
print(f"无人机图像: {batch['drone_img'].shape}")
print(f"卫星图像: {batch['sat_img'].shape}")
```

### 训练循环

```python
from geovisnet.utils import AverageMeter
import torch.nn.functional as F

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

### 注意力可视化

```python
from geovisnet.visualization import visualize_attention

# 获取注意力图
model.eval()
with torch.no_grad():
    attention_maps = model.get_attention_maps(drone_imgs[:1], sat_imgs[:1])

# 可视化
visualize_attention(
    drone_imgs[0], 
    sat_imgs[0], 
    attention_maps,
    save_path='attention_visualization.png'
)
```

### 配置使用

```python
from geovisnet.configs import get_training_config

# 获取默认配置
config = get_training_config('default')

# 获取自定义配置
config = get_training_config('high_quality', 
    model={'backbone': 'efficientnet_b2'},
    training={'lr': 0.0001}
)
```

## 错误处理

### 常见异常

- `FileNotFoundError`: 数据文件不存在
- `RuntimeError`: CUDA内存不足
- `ValueError`: 参数值无效
- `ImportError`: 缺少依赖包

### 调试技巧

1. 使用`model.get_model_info()`检查模型参数
2. 使用小批次大小进行调试
3. 检查数据路径和文件权限
4. 使用`torch.cuda.empty_cache()`清理GPU内存

---

更多详细信息请参考源代码和示例。
