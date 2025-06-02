#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UAV-VisLoc数据集加载器

处理无人机图像和卫星图像，用于地理定位任务
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import random
from tqdm import tqdm
from .augmentation import MixUp, UAVSatMixUp


class UAVVisLocDataset(Dataset):
    """
    UAV-VisLoc数据集加载器
    
    处理无人机图像和卫星图像，用于地理定位任务
    
    Args:
        data_root (str): 数据集根目录
        regions (list): 要使用的区域列表，例如['01', '02']，如果为None则使用所有区域
        img_size (int): 图像大小，将调整所有图像到这个大小
        transform: 图像变换
        augment (bool): 是否进行数据增强
        mode (str): 'train'、'val'或'test'模式
        cache_satellite (bool): 是否缓存卫星图像（消耗内存但加快速度）
        use_mixup (bool): 是否使用MixUp数据增强
        mixup_alpha (float): MixUp强度参数
        use_cross_mixup (bool): 是否使用跨视图MixUp
        cross_mixup_alpha (float): 跨视图MixUp强度参数
        cross_mixup_prob (float): 跨视图MixUp应用概率
        sat_patch_scale (float): 卫星图像裁剪尺寸的缩放比例，默认为2.0
    """
    
    def __init__(self, data_root, regions=None, img_size=224, transform=None, augment=False,
                 mode='train', cache_satellite=False, use_mixup=False, mixup_alpha=1.0,
                 use_cross_mixup=False, cross_mixup_alpha=0.2, cross_mixup_prob=0.3,
                 sat_patch_scale=2.0):
        
        self.data_root = data_root
        self.img_size = img_size
        self.transform = transform
        self.augment = augment
        self.mode = mode
        self.cache_satellite = cache_satellite
        self.sat_patch_scale = sat_patch_scale

        # MixUp数据增强设置
        self.use_mixup = use_mixup and mode == 'train'  # 仅在训练模式下使用MixUp
        self.mixup_alpha = mixup_alpha
        self.mixup = MixUp(alpha=mixup_alpha) if use_mixup else None

        # 跨视图MixUp设置
        self.use_cross_mixup = use_cross_mixup and mode == 'train'  # 仅在训练模式下使用跨视图MixUp
        self.cross_mixup_alpha = cross_mixup_alpha
        self.cross_mixup_prob = cross_mixup_prob
        self.cross_mixup = UAVSatMixUp(alpha=cross_mixup_alpha, p=cross_mixup_prob) if use_cross_mixup else None

        # 如果没有指定区域，则使用默认区域
        if regions is None:
            if mode == 'train':
                self.regions = ['01', '02', '03', '04', '05', '06']
            elif mode == 'val':
                self.regions = ['07']
            else:  # test
                self.regions = ['08', '10', '11']  # 注意：数据集中没有09区域
        else:
            self.regions = regions

        # 加载卫星图像坐标范围
        sat_coords_path = os.path.join(data_root, 'satellite_coordinates_range.csv')
        if not os.path.exists(sat_coords_path):
            # 尝试另一个可能的文件名
            sat_coords_path = os.path.join(data_root, 'satellite_ coordinates_range.csv')
        self.sat_coords_df = pd.read_csv(sat_coords_path)

        # 准备数据列表
        self.data_list = []
        self._prepare_data_list()

        # 卫星图像缓存
        self.satellite_cache = {}
        if self.cache_satellite:
            self._cache_satellite_images()

    def _prepare_data_list(self):
        """准备数据列表，包含无人机图像路径和对应的卫星图像路径及坐标信息"""
        for region in self.regions:
            # 加载区域CSV文件
            csv_path = os.path.join(self.data_root, region, f'{region}.csv')
            if not os.path.exists(csv_path):
                print(f"警告：找不到区域{region}的CSV文件")
                continue

            drone_df = pd.read_csv(csv_path)

            # 获取该区域的卫星图像信息
            sat_name = f'satellite{region}.tif'
            sat_info = self.sat_coords_df[self.sat_coords_df['mapname'] == sat_name]

            if len(sat_info) == 0:
                print(f"警告：找不到卫星图像{sat_name}的坐标信息")
                continue

            sat_path = os.path.join(self.data_root, region, sat_name)
            if not os.path.exists(sat_path):
                print(f"警告：找不到卫星图像{sat_path}")
                continue

            # 获取卫星图像的坐标范围
            lt_lat = sat_info['LT_lat_map'].values[0]
            lt_lon = sat_info['LT_lon_map'].values[0]
            rb_lat = sat_info['RB_lat_map'].values[0]
            rb_lon = sat_info['RB_lon_map'].values[0]

            # 遍历该区域的所有无人机图像
            for _, row in drone_df.iterrows():
                drone_filename = row['filename']
                drone_path = os.path.join(self.data_root, region, 'drone', drone_filename)

                if not os.path.exists(drone_path):
                    continue

                # 获取无人机图像的坐标
                drone_lat = row['lat']
                drone_lon = row['lon']

                # 检查无人机坐标是否在卫星图像范围内
                if not (lt_lat >= drone_lat >= rb_lat and lt_lon <= drone_lon <= rb_lon):
                    continue

                # 计算无人机在卫星图像中的相对位置（归一化到0-1）
                norm_lat = (lt_lat - drone_lat) / (lt_lat - rb_lat)
                norm_lon = (drone_lon - lt_lon) / (rb_lon - lt_lon)

                # 添加到数据列表
                self.data_list.append({
                    'drone_path': drone_path,
                    'sat_path': sat_path,
                    'drone_lat': drone_lat,
                    'drone_lon': drone_lon,
                    'norm_lat': norm_lat,
                    'norm_lon': norm_lon,
                    'region': region
                })

        print(f"加载了{len(self.data_list)}个样本")

    def _cache_satellite_images(self):
        """缓存所有卫星图像到内存中"""
        print("缓存卫星图像...")
        unique_sat_paths = set(item['sat_path'] for item in self.data_list)

        for sat_path in tqdm(unique_sat_paths):
            # 尝试使用OpenCV读取
            satellite_img = cv2.imread(sat_path)
            if satellite_img is None:
                # 如果无法使用OpenCV读取（可能是TIF格式），尝试使用PIL
                try:
                    img = Image.open(sat_path)
                    satellite_img = np.array(img)
                    if len(satellite_img.shape) == 2:  # 灰度图像
                        satellite_img = np.stack([satellite_img] * 3, axis=2)
                    elif satellite_img.shape[2] > 3:  # 带有Alpha通道
                        satellite_img = satellite_img[:, :, :3]
                except Exception as e:
                    print(f"无法读取卫星图像 {sat_path}: {e}")
                    continue

            # 确保图像是RGB格式
            if len(satellite_img.shape) == 2:  # 灰度图像
                satellite_img = np.stack([satellite_img] * 3, axis=2)
            elif satellite_img.shape[2] > 3:  # 带有Alpha通道
                satellite_img = satellite_img[:, :, :3]

            if satellite_img.shape[2] == 3 and satellite_img.dtype == np.uint8:
                satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)

            self.satellite_cache[sat_path] = satellite_img

    def _get_satellite_patch(self, sat_path, norm_lat, norm_lon, patch_size=224):
        """
        从卫星图像中获取对应无人机位置的图像块
        
        Args:
            sat_path (str): 卫星图像路径
            norm_lat (float): 归一化的纬度位置 (0-1)
            norm_lon (float): 归一化的经度位置 (0-1)
            patch_size (int): 最终输出的图像块大小
            
        Returns:
            np.ndarray: 卫星图像块
        """
        # 计算初始裁剪尺寸（更大的尺寸）
        initial_patch_size = int(patch_size * self.sat_patch_scale)

        # 如果已缓存，直接从缓存获取
        if self.cache_satellite and sat_path in self.satellite_cache:
            satellite_img = self.satellite_cache[sat_path]
        else:
            # 读取卫星图像
            satellite_img = cv2.imread(sat_path)
            if satellite_img is None:
                try:
                    img = Image.open(sat_path)
                    satellite_img = np.array(img)
                    if len(satellite_img.shape) == 2:
                        satellite_img = np.stack([satellite_img] * 3, axis=2)
                    elif satellite_img.shape[2] > 3:
                        satellite_img = satellite_img[:, :, :3]
                except Exception as e:
                    print(f"无法读取卫星图像 {sat_path}: {e}")
                    return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

            # 确保图像是RGB格式
            if len(satellite_img.shape) == 2:
                satellite_img = np.stack([satellite_img] * 3, axis=2)
            elif satellite_img.shape[2] > 3:
                satellite_img = satellite_img[:, :, :3]

            if satellite_img.shape[2] == 3 and satellite_img.dtype == np.uint8:
                satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)

        h, w = satellite_img.shape[:2]

        # 计算中心点坐标
        center_y = int(norm_lat * h)
        center_x = int(norm_lon * w)

        # 计算图像块的边界（使用更大的尺寸）
        half_size = initial_patch_size // 2
        y1 = max(0, center_y - half_size)
        x1 = max(0, center_x - half_size)
        y2 = min(h, center_y + half_size)
        x2 = min(w, center_x + half_size)

        # 提取图像块
        patch = satellite_img[y1:y2, x1:x2].copy()

        # 如果提取的图像块大小不足，进行填充
        if patch.shape[0] < initial_patch_size or patch.shape[1] < initial_patch_size:
            full_patch = np.zeros((initial_patch_size, initial_patch_size, 3), dtype=np.uint8)
            y_offset = max(0, half_size - center_y)
            x_offset = max(0, half_size - center_x)
            full_patch[y_offset:y_offset+patch.shape[0], x_offset:x_offset+patch.shape[1]] = patch
            patch = full_patch

        # 如果提取的图像块大小超过需要的大小，进行裁剪
        if patch.shape[0] > initial_patch_size or patch.shape[1] > initial_patch_size:
            start_y = (patch.shape[0] - initial_patch_size) // 2
            start_x = (patch.shape[1] - initial_patch_size) // 2
            patch = patch[start_y:start_y+initial_patch_size, start_x:start_x+initial_patch_size]

        # 将大尺寸图像块下采样到目标尺寸
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        return patch

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        item = self.data_list[idx]

        # 读取无人机图像
        drone_img = cv2.imread(item['drone_path'])
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        # 调整无人机图像大小
        drone_img = cv2.resize(drone_img, (self.img_size, self.img_size))

        # 获取卫星图像块
        sat_img = self._get_satellite_patch(
            item['sat_path'],
            item['norm_lat'],
            item['norm_lon'],
            self.img_size
        )

        # 数据增强
        if self.augment:
            drone_img, sat_img = self._apply_augmentation(drone_img, sat_img)

        # 应用变换
        if self.transform is not None:
            drone_img = self.transform(drone_img)
            sat_img = self.transform(sat_img)

        # 准备返回数据
        sample = {
            'drone_img': drone_img,
            'sat_img': sat_img,
            'drone_lat': item['drone_lat'],
            'drone_lon': item['drone_lon'],
            'norm_lat': item['norm_lat'],
            'norm_lon': item['norm_lon'],
            'region': item['region'],
            'idx': idx
        }

        return sample

    def _apply_augmentation(self, drone_img, sat_img):
        """应用数据增强"""
        # 随机水平翻转
        if random.random() < 0.5:
            drone_img = cv2.flip(drone_img, 1)
            sat_img = cv2.flip(sat_img, 1)

        # 随机垂直翻转
        if random.random() < 0.5:
            drone_img = cv2.flip(drone_img, 0)
            sat_img = cv2.flip(sat_img, 0)

        # 随机旋转
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            M = cv2.getRotationMatrix2D((self.img_size//2, self.img_size//2), angle, 1)
            drone_img = cv2.warpAffine(drone_img, M, (self.img_size, self.img_size))
            sat_img = cv2.warpAffine(sat_img, M, (self.img_size, self.img_size))

        # 随机亮度和对比度变化
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # 对比度
            beta = random.uniform(-10, 10)    # 亮度
            drone_img = cv2.convertScaleAbs(drone_img, alpha=alpha, beta=beta)

        return drone_img, sat_img

    def collate_fn(self, batch):
        """自定义批次收集函数，用于在DataLoader中应用MixUp"""
        # 将批次中的样本合并为一个批次
        collated_batch = {}
        for key in batch[0].keys():
            if key in ['drone_img', 'sat_img']:
                collated_batch[key] = torch.stack([sample[key] for sample in batch])
            elif key in ['drone_lat', 'drone_lon', 'norm_lat', 'norm_lon']:
                collated_batch[key] = torch.tensor([sample[key] for sample in batch], dtype=torch.float32)
            else:
                collated_batch[key] = [sample[key] for sample in batch]

        # 应用MixUp数据增强（如果启用）
        if self.use_mixup and self.mode == 'train':
            collated_batch = self.mixup(collated_batch)

        # 应用跨视图MixUp数据增强（如果启用）
        if self.use_cross_mixup and self.mode == 'train':
            collated_batch = self.cross_mixup(collated_batch)

        return collated_batch
