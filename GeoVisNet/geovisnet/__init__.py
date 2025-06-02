#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet: 基于双重注意力机制的无人机-卫星图像地理定位网络

这个包提供了完整的地理定位解决方案，包括：
- 模型定义和训练
- 数据处理和增强
- 评估和可视化工具
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import GeoVisNet
from .data import UAVVisLocDataset
from .utils import (
    calculate_geo_error,
    euclidean_geo_distance,
    AverageMeter
)

# 导入子模块
from . import models
from . import data
from . import utils
from . import visualization

__all__ = [
    "GeoVisNet",
    "UAVVisLocDataset", 
    "calculate_geo_error",
    "euclidean_geo_distance",
    "AverageMeter"
]
