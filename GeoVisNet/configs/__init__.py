#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet配置模块

包含默认配置和训练配置
"""

from .default_config import DefaultConfig
from .training_configs import get_training_config

__all__ = [
    "DefaultConfig",
    "get_training_config"
]
