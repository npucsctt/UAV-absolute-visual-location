#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet工具函数模块

包含各种工具函数：
- 距离计算函数
- 评估指标
- 日志工具
- 检查点管理
"""

from .distance import (
    euclidean_geo_distance,
    calculate_geo_error,
    calculate_geo_error_with_region_calibration,
    convert_normalized_to_geo,
    get_region_scale_factors
)
from .metrics import AverageMeter, compute_metrics
from .logging import setup_logging
from .checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "euclidean_geo_distance",
    "calculate_geo_error", 
    "calculate_geo_error_with_region_calibration",
    "convert_normalized_to_geo",
    "get_region_scale_factors",
    "AverageMeter",
    "compute_metrics",
    "setup_logging",
    "save_checkpoint",
    "load_checkpoint"
]
