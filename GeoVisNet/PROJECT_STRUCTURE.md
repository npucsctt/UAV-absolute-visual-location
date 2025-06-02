# GeoVisNet 项目结构

本文档详细描述了重组后的GeoVisNet项目结构。

## 📁 完整项目结构

```
GeoVisNet/
├── README.md                           # 项目主文档
├── LICENSE                             # MIT许可证
├── requirements.txt                    # Python依赖列表
├── setup.py                           # 项目安装配置
├── .gitignore                         # Git忽略文件
├── test_installation.py              # 安装测试脚本
├── PROJECT_STRUCTURE.md              # 项目结构说明（本文件）
│
├── geovisnet/                         # 主代码包
│   ├── __init__.py                    # 包初始化文件
│   │
│   ├── models/                        # 模型定义模块
│   │   ├── __init__.py               # 模型包初始化
│   │   ├── geovisnet.py              # 主要的GeoVisNet模型
│   │   ├── attention.py              # 注意力机制模块
│   │   └── components.py             # 模型组件（特征提取器、预测头等）
│   │
│   ├── data/                         # 数据处理模块
│   │   ├── __init__.py               # 数据包初始化
│   │   ├── dataset.py                # UAV-VisLoc数据集加载器
│   │   ├── transforms.py             # 数据变换函数
│   │   └── augmentation.py           # 数据增强方法（MixUp等）
│   │
│   ├── utils/                        # 工具函数模块
│   │   ├── __init__.py               # 工具包初始化
│   │   ├── distance.py               # 地理距离计算函数
│   │   ├── metrics.py                # 评估指标计算
│   │   ├── logging.py                # 日志配置和管理
│   │   └── checkpoint.py             # 模型检查点管理
│   │
│   └── visualization/                # 可视化模块
│       ├── __init__.py               # 可视化包初始化
│       ├── attention_maps.py         # 注意力图可视化
│       ├── results.py                # 结果分析可视化
│       └── dataset_viz.py            # 数据集可视化
│
├── scripts/                          # 运行脚本
│   ├── train.py                      # 训练脚本
│   ├── test.py                       # 测试脚本
│   ├── evaluate.py                   # 详细评估脚本
│   └── visualize.py                  # 可视化脚本
│
├── configs/                          # 配置文件
│   ├── __init__.py                   # 配置包初始化
│   ├── default_config.py             # 默认配置定义
│   └── training_configs.py           # 训练配置管理
│
├── experiments/                      # 实验相关
│   ├── ablation/                     # 消融实验结果
│   └── configs/                      # 实验配置
│       └── ablation_study.py         # 消融实验配置生成器
│
├── docs/                             # 文档
│   ├── installation.md              # 安装指南
│   ├── usage.md                      # 使用指南
│   └── api.md                        # API文档
│
└── examples/                         # 示例代码
    └── quick_start.py                # 快速开始示例
```

## 📋 模块功能说明

### 核心模块 (geovisnet/)

#### 1. models/ - 模型定义
- **geovisnet.py**: 主要的GeoVisNet模型类，实现双重注意力机制的地理定位网络
- **attention.py**: 注意力机制实现，包括ECA、空间注意力、CBAM等
- **components.py**: 模型组件，包括特征提取器、预测头、正则化模块等

#### 2. data/ - 数据处理
- **dataset.py**: UAV-VisLoc数据集加载器，支持多区域、缓存、数据增强
- **transforms.py**: 图像预处理和数据变换函数
- **augmentation.py**: 高级数据增强方法，包括MixUp、CutMix等

#### 3. utils/ - 工具函数
- **distance.py**: 地理距离计算，支持欧几里得距离和Haversine距离
- **metrics.py**: 评估指标计算，包括准确率、误差统计等
- **logging.py**: 日志系统，支持结构化训练日志
- **checkpoint.py**: 模型检查点管理，支持自动保存和恢复

#### 4. visualization/ - 可视化
- **attention_maps.py**: 注意力图可视化，支持热图、叠加图等
- **results.py**: 结果分析可视化，包括误差分布、区域比较等
- **dataset_viz.py**: 数据集可视化，包括样本展示、坐标分布等

### 脚本模块 (scripts/)

- **train.py**: 完整的训练脚本，支持多种配置和恢复训练
- **test.py**: 测试脚本，支持批量测试和结果保存
- **evaluate.py**: 详细评估脚本，提供深入的性能分析
- **visualize.py**: 可视化脚本，生成各种分析图表

### 配置模块 (configs/)

- **default_config.py**: 默认配置类，定义所有可配置参数
- **training_configs.py**: 训练配置管理，支持预定义配置和自定义配置

### 实验模块 (experiments/)

- **configs/ablation_study.py**: 消融实验配置生成器，自动生成实验脚本

### 文档模块 (docs/)

- **installation.md**: 详细的安装指南，包括环境配置和常见问题
- **usage.md**: 使用指南，包括训练、测试、可视化等
- **api.md**: API文档，详细的函数和类说明

### 示例模块 (examples/)

- **quick_start.py**: 快速开始示例，演示基本用法

## 🔧 主要改进

### 1. 代码组织
- ✅ 模块化设计，功能清晰分离
- ✅ 统一的命名规范和代码风格
- ✅ 完整的包初始化和导入管理
- ✅ 详细的文档字符串和注释

### 2. 功能增强
- ✅ 支持多种骨干网络（EfficientNet B0/B2）
- ✅ 完整的注意力机制实现
- ✅ 高级数据增强方法
- ✅ 灵活的配置系统
- ✅ 全面的可视化工具

### 3. 易用性提升
- ✅ 简化的安装流程
- ✅ 命令行接口
- ✅ 配置文件支持
- ✅ 详细的使用文档
- ✅ 示例代码和快速开始指南

### 4. 开发友好
- ✅ 完整的测试脚本
- ✅ 自动化实验配置
- ✅ 检查点管理
- ✅ 日志系统
- ✅ 错误处理

## 🚀 使用流程

### 1. 安装
```bash
git clone https://github.com/your-username/GeoVisNet.git
cd GeoVisNet
pip install -r requirements.txt
pip install -e .
python test_installation.py  # 验证安装
```

### 2. 快速开始
```bash
python examples/quick_start.py
```

### 3. 训练模型
```bash
python scripts/train.py \
    --data_root /path/to/data \
    --save_dir ./models \
    --log_dir ./logs \
    --backbone efficientnet_b0 \
    --epochs 100
```

### 4. 测试模型
```bash
python scripts/test.py \
    --data_root /path/to/data \
    --model_path ./models/best_model.pth \
    --output_dir ./results
```

### 5. 可视化结果
```bash
python scripts/visualize.py \
    --data_root /path/to/data \
    --model_path ./models/best_model.pth \
    --output_dir ./visualizations \
    --vis_type all
```

## 📊 实验支持

### 消融实验
```bash
cd experiments/configs
python ablation_study.py  # 生成实验脚本
bash ../../run_ablation_experiments.sh  # 运行所有实验
```

### 自定义实验
```python
from geovisnet.configs import get_training_config

config = get_training_config('high_quality',
    model={'backbone': 'efficientnet_b2'},
    training={'lr': 0.0001}
)
```

## 🔍 代码质量

- ✅ 类型提示支持
- ✅ 文档字符串完整
- ✅ 错误处理机制
- ✅ 单元测试框架
- ✅ 代码格式化标准

## 📈 扩展性

项目设计支持以下扩展：

1. **新的骨干网络**: 在`models/components.py`中添加
2. **新的注意力机制**: 在`models/attention.py`中实现
3. **新的数据增强**: 在`data/augmentation.py`中添加
4. **新的评估指标**: 在`utils/metrics.py`中实现
5. **新的可视化**: 在`visualization/`模块中添加

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

---

这个重组后的项目结构为GeoVisNet提供了清晰的代码组织、完整的功能实现和良好的可扩展性，适合学术研究和工业应用。
