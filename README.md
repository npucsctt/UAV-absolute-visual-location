# GeoVisNet: 基于双重注意力机制的无人机-卫星图像地理定位网络（毕业设计）

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介

GeoVisNet是一个基于深度学习的无人机-卫星图像地理定位系统，采用双重注意力机制（ECA + 空间注意力）来实现精确的地理坐标预测。该系统能够通过分析无人机图像和对应的卫星图像，准确预测无人机的地理位置。

### 🌟 主要特性

- **双重注意力机制**：结合ECA（Efficient Channel Attention）和空间注意力机制
- **多尺度特征融合**：使用EfficientNet作为骨干网络提取多尺度特征
- **正则化策略**：集成DropBlock、Dropout等正则化技术防止过拟合
- **灵活的训练配置**：支持多种损失函数、数据增强和训练策略
- **完整的评估体系**：提供详细的地理误差分析和可视化工具

### 🏗️ 模型架构

```
无人机图像 ──┐
            ├─→ 特征提取器 ──┐
卫星图像 ───┘                ├─→ 双重注意力模块 ──→ 特征融合 ──→ 地理坐标预测
                            │
                            └─→ ECA注意力 + 空间注意力
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 10.2+ (推荐)

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/GeoVisNet.git
cd GeoVisNet
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装项目：
```bash
pip install -e .
```

### 数据准备

1. 下载UAV-VisLoc数据集
2. 按照以下结构组织数据：
```
data/
├── UAV_VisLoc_dataset/
│   ├── drone_images/
│   ├── satellite_images/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
└── satellite_coordinates_range.csv
```

### 训练模型

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --save_dir ./saved_models \
    --log_dir ./logs \
    --backbone efficientnet_b0 \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.0005
```

### 测试模型

```bash
python scripts/test.py \
    --data_root /path/to/data \
    --model_path ./saved_models/best_model.pth \
    --output_dir ./test_results \
    --backbone efficientnet_b0
```

## 📁 项目结构

```
GeoVisNet/
├── geovisnet/              # 主代码包
│   ├── models/             # 模型定义
│   ├── data/              # 数据处理
│   ├── utils/             # 工具函数
│   └── visualization/     # 可视化工具
├── scripts/               # 运行脚本
├── configs/               # 配置文件
├── experiments/           # 实验相关
├── docs/                  # 文档
└── examples/              # 示例代码
```

## 🔧 高级用法

### 自定义训练

```python
from geovisnet.models import GeoVisNet
from geovisnet.data import UAVVisLocDataset
from geovisnet.utils import train_model

# 创建模型
model = GeoVisNet(backbone='efficientnet_b0')

# 创建数据集
dataset = UAVVisLocDataset(data_root='./data')

# 训练模型
train_model(model, dataset, config='./configs/default_config.py')
```

### 注意力可视化

```python
from geovisnet.visualization import visualize_attention

# 可视化注意力图
visualize_attention(model, image_pair, save_path='./attention_maps/')
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢UAV-VisLoc数据集的提供者以及开源社区的贡献。
