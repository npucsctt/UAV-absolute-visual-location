#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GeoVisNet安装测试脚本

验证GeoVisNet是否正确安装并可以正常工作
"""

import sys
import torch
import numpy as np


def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试核心模块导入
        import geovisnet
        from geovisnet.models import GeoVisNet
        from geovisnet.data import UAVVisLocDataset
        from geovisnet.utils import AverageMeter, euclidean_geo_distance
        print("✓ 核心模块导入成功")
        
        # 测试配置模块
        try:
            from configs.default_config import DefaultConfig
            print("✓ 配置模块导入成功")
        except ImportError:
            print("⚠ 配置模块导入失败，但核心功能正常")
        
        # 测试可视化模块
        from geovisnet.visualization import visualize_attention
        print("✓ 可视化模块导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from geovisnet.models import GeoVisNet
        
        # 创建模型
        model = GeoVisNet(
            emb_size=256,
            backbone='efficientnet_b0',
            dropout=0.5
        )
        
        # 检查模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 模型创建成功")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        return True, model
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False, None


def test_model_forward():
    """测试模型前向传播"""
    print("\n测试模型前向传播...")
    
    try:
        from geovisnet.models import GeoVisNet
        
        # 创建模型
        model = GeoVisNet(emb_size=256, backbone='efficientnet_b0')
        model.eval()
        
        # 创建随机输入
        batch_size = 2
        drone_imgs = torch.randn(batch_size, 3, 224, 224)
        sat_imgs = torch.randn(batch_size, 3, 224, 224)
        
        # 前向传播
        with torch.no_grad():
            query_features, similarity_features, geo_preds = model(drone_imgs, sat_imgs)
        
        # 检查输出形状
        expected_query_shape = (batch_size, 256)
        expected_geo_shape = (batch_size, 2)
        
        if query_features.shape == expected_query_shape and geo_preds.shape == expected_geo_shape:
            print("✓ 模型前向传播成功")
            print(f"  查询特征形状: {query_features.shape}")
            print(f"  地理坐标预测形状: {geo_preds.shape}")
            return True
        else:
            print(f"✗ 输出形状不匹配")
            print(f"  期望查询特征形状: {expected_query_shape}, 实际: {query_features.shape}")
            print(f"  期望地理坐标形状: {expected_geo_shape}, 实际: {geo_preds.shape}")
            return False
            
    except Exception as e:
        print(f"✗ 模型前向传播失败: {e}")
        return False


def test_attention_maps():
    """测试注意力图功能"""
    print("\n测试注意力图功能...")
    
    try:
        from geovisnet.models import GeoVisNet
        
        # 创建模型
        model = GeoVisNet(emb_size=256, backbone='efficientnet_b0')
        model.eval()
        
        # 创建随机输入
        drone_imgs = torch.randn(1, 3, 224, 224)
        sat_imgs = torch.randn(1, 3, 224, 224)
        
        # 获取注意力图
        with torch.no_grad():
            attention_maps = model.get_attention_maps(drone_imgs, sat_imgs)
        
        # 检查注意力图
        expected_keys = ['query_eca', 'query_spatial', 'reference_eca', 'reference_spatial']
        if all(key in attention_maps for key in expected_keys):
            print("✓ 注意力图功能正常")
            for key, value in attention_maps.items():
                print(f"  {key}: {value.shape}")
            return True
        else:
            print(f"✗ 注意力图键不完整")
            print(f"  期望键: {expected_keys}")
            print(f"  实际键: {list(attention_maps.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ 注意力图功能测试失败: {e}")
        return False


def test_utils():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    try:
        from geovisnet.utils import euclidean_geo_distance, AverageMeter
        
        # 测试距离计算
        lat1, lon1 = 39.9042, 116.4074  # 北京
        lat2, lon2 = 31.2304, 121.4737  # 上海
        distance = euclidean_geo_distance(lat1, lon1, lat2, lon2)
        
        # 北京到上海的距离大约是1000公里
        if 900000 < distance < 1200000:  # 900-1200公里范围
            print(f"✓ 距离计算功能正常: {distance/1000:.1f} km")
        else:
            print(f"✗ 距离计算结果异常: {distance/1000:.1f} km")
            return False
        
        # 测试平均值计算器
        meter = AverageMeter()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            meter.update(val)
        
        if abs(meter.avg - 3.0) < 1e-6:
            print("✓ 平均值计算器功能正常")
        else:
            print(f"✗ 平均值计算器结果异常: {meter.avg}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        return False


def test_data_transforms():
    """测试数据变换"""
    print("\n测试数据变换...")
    
    try:
        from geovisnet.data import get_train_transforms, get_val_transforms
        
        # 测试训练变换
        train_transform = get_train_transforms(224, enhanced_augmentation=True)
        val_transform = get_val_transforms(224)
        
        # 创建随机图像
        from PIL import Image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # 应用变换
        train_tensor = train_transform(dummy_image)
        val_tensor = val_transform(dummy_image)
        
        # 检查输出形状
        expected_shape = (3, 224, 224)
        if train_tensor.shape == expected_shape and val_tensor.shape == expected_shape:
            print("✓ 数据变换功能正常")
            print(f"  训练变换输出形状: {train_tensor.shape}")
            print(f"  验证变换输出形状: {val_tensor.shape}")
            return True
        else:
            print(f"✗ 数据变换输出形状异常")
            return False
            
    except Exception as e:
        print(f"✗ 数据变换测试失败: {e}")
        return False


def test_configs():
    """测试配置系统"""
    print("\n测试配置系统...")
    
    try:
        from configs.default_config import DefaultConfig
        from configs.training_configs import get_training_config
        
        # 测试默认配置
        default_config = DefaultConfig.get_config()
        
        # 检查必要的配置项
        required_sections = ['model', 'training', 'data']
        if all(section in default_config for section in required_sections):
            print("✓ 默认配置加载成功")
        else:
            print("✗ 默认配置缺少必要项")
            return False
        
        # 测试训练配置
        train_config = get_training_config('default')
        if 'model' in train_config and 'training' in train_config:
            print("✓ 训练配置加载成功")
        else:
            print("✗ 训练配置加载失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
        return False


def check_environment():
    """检查环境信息"""
    print("环境信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"  NumPy版本: {np.__version__}")


def main():
    """主测试函数"""
    print("GeoVisNet安装测试")
    print("=" * 50)
    
    # 检查环境
    check_environment()
    print()
    
    # 运行测试
    tests = [
        ("模块导入", test_imports),
        ("模型创建", lambda: test_model_creation()[0]),
        ("模型前向传播", test_model_forward),
        ("注意力图功能", test_attention_maps),
        ("工具函数", test_utils),
        ("数据变换", test_data_transforms),
        ("配置系统", test_configs),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name}测试出现异常: {e}")
    
    # 总结
    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 GeoVisNet安装成功！所有功能正常。")
        print("\n接下来您可以:")
        print("1. 运行 python examples/quick_start.py 进行快速体验")
        print("2. 准备数据集并开始训练")
        print("3. 查看文档了解更多使用方法")
    else:
        print("⚠️  部分功能测试失败，请检查安装。")
        print("如果问题持续存在，请查看安装文档或提交Issue。")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
