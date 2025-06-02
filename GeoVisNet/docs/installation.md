# GeoVisNet 安装指南

本文档提供了GeoVisNet的详细安装说明。

## 系统要求

### 硬件要求
- **GPU**: 推荐使用NVIDIA GPU（至少4GB显存）
- **内存**: 至少16GB RAM
- **存储**: 至少50GB可用空间（用于数据集和模型）

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+), Windows 10+, macOS 10.15+
- **Python**: 3.8 或更高版本
- **CUDA**: 10.2 或更高版本（如果使用GPU）

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/GeoVisNet.git
cd GeoVisNet
```

### 2. 创建虚拟环境

推荐使用conda或virtualenv创建独立的Python环境：

#### 使用Conda
```bash
conda create -n geovisnet python=3.8
conda activate geovisnet
```

#### 使用virtualenv
```bash
python -m venv geovisnet_env
source geovisnet_env/bin/activate  # Linux/macOS
# 或
geovisnet_env\Scripts\activate  # Windows
```

### 3. 安装依赖

#### 方法1: 使用pip安装（推荐）
```bash
pip install -r requirements.txt
```

#### 方法2: 手动安装核心依赖
```bash
# PyTorch (请根据您的CUDA版本选择合适的版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他核心依赖
pip install timm opencv-python pillow pandas numpy matplotlib seaborn tqdm scipy scikit-learn pyyaml
```

### 4. 安装GeoVisNet包

```bash
pip install -e .
```

这将以开发模式安装GeoVisNet，允许您修改代码并立即生效。

## 验证安装

### 1. 检查Python包导入

```python
import torch
import geovisnet
from geovisnet.models import GeoVisNet
from geovisnet.data import UAVVisLocDataset

print("GeoVisNet安装成功！")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
```

### 2. 运行快速测试

```bash
python examples/quick_start.py
```

## 数据集准备

### 1. 下载UAV-VisLoc数据集

请从官方来源下载UAV-VisLoc数据集，并按照以下结构组织：

```
data/
├── UAV_VisLoc_dataset/
│   ├── 01/
│   │   ├── 01.csv
│   │   ├── drone/
│   │   └── satellite01.tif
│   ├── 02/
│   │   ├── 02.csv
│   │   ├── drone/
│   │   └── satellite02.tif
│   ├── ...
│   └── satellite_coordinates_range.csv
```

### 2. 验证数据集

```python
from geovisnet.data import UAVVisLocDataset

# 创建数据集实例
dataset = UAVVisLocDataset(
    data_root='/path/to/your/UAV_VisLoc_dataset',
    regions=['01'],
    img_size=224,
    mode='train'
)

print(f"数据集大小: {len(dataset)}")
print("数据集加载成功！")
```

## 常见问题

### 1. CUDA相关问题

**问题**: `RuntimeError: CUDA out of memory`
**解决方案**: 
- 减小批次大小 (`--batch_size`)
- 使用混合精度训练 (`--use_amp`)
- 使用较小的模型 (`--backbone efficientnet_b0`)

**问题**: `CUDA not available`
**解决方案**:
- 检查NVIDIA驱动是否正确安装
- 确认PyTorch安装了CUDA支持版本
- 重新安装PyTorch CUDA版本

### 2. 依赖包问题

**问题**: `ImportError: No module named 'timm'`
**解决方案**:
```bash
pip install timm
```

**问题**: `ImportError: No module named 'cv2'`
**解决方案**:
```bash
pip install opencv-python
```

### 3. 数据加载问题

**问题**: `FileNotFoundError: [Errno 2] No such file or directory`
**解决方案**:
- 检查数据集路径是否正确
- 确认数据集文件结构是否符合要求
- 检查文件权限

### 4. 内存问题

**问题**: 训练时内存不足
**解决方案**:
- 减小批次大小
- 设置 `cache_satellite=False`
- 减少数据加载器的工作线程数 (`--num_workers`)

## 性能优化建议

### 1. 训练加速
- 使用混合精度训练 (`--use_amp`)
- 启用数据并行 (多GPU训练)
- 使用更快的数据加载器 (`pin_memory=True`)

### 2. 内存优化
- 使用梯度累积减少内存使用
- 启用梯度检查点
- 合理设置批次大小

### 3. 存储优化
- 使用SSD存储数据集
- 预处理并缓存数据
- 使用数据压缩

## 开发环境设置

如果您计划修改GeoVisNet代码，建议安装额外的开发工具：

```bash
pip install pytest black flake8 jupyter ipywidgets
```

### 代码格式化
```bash
black geovisnet/
```

### 运行测试
```bash
pytest tests/
```

## Docker安装（可选）

我们也提供了Docker镜像用于快速部署：

```bash
# 构建Docker镜像
docker build -t geovisnet .

# 运行容器
docker run --gpus all -it -v /path/to/data:/data geovisnet
```

## 获取帮助

如果您在安装过程中遇到问题，请：

1. 检查本文档的常见问题部分
2. 查看GitHub Issues页面
3. 提交新的Issue并提供详细的错误信息

## 更新GeoVisNet

要更新到最新版本：

```bash
git pull origin main
pip install -e . --upgrade
```

---

安装完成后，您可以继续阅读[使用指南](usage.md)来了解如何使用GeoVisNet。
