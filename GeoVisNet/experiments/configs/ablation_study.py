#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
消融实验配置

定义各种消融实验的配置
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geovisnet.configs import get_training_config


def get_ablation_configs():
    """
    获取消融实验配置列表
    
    Returns:
        dict: 消融实验配置字典
    """
    
    # 基础配置
    base_config = get_training_config('default')
    
    configs = {}
    
    # 1. 基线模型（完整模型）
    configs['baseline'] = get_training_config('default', 
        experiment_name='ablation_baseline',
        training={'epochs': 100}
    )
    
    # 2. 无ECA注意力
    configs['no_eca'] = get_training_config('default',
        experiment_name='ablation_no_eca',
        model={'use_eca_attention': False},
        training={'epochs': 100}
    )
    
    # 3. 无空间注意力
    configs['no_spatial'] = get_training_config('default',
        experiment_name='ablation_no_spatial',
        model={'use_spatial_attention': False},
        training={'epochs': 100}
    )
    
    # 4. 无双重注意力
    configs['no_attention'] = get_training_config('default',
        experiment_name='ablation_no_attention',
        model={
            'use_eca_attention': False,
            'use_spatial_attention': False
        },
        training={'epochs': 100}
    )
    
    # 5. 无MixUp数据增强
    configs['no_mixup'] = get_training_config('default',
        experiment_name='ablation_no_mixup',
        data={
            'use_mixup': False,
            'use_cross_mixup': False
        },
        training={'epochs': 100}
    )
    
    # 6. 无跨视图MixUp
    configs['no_cross_mixup'] = get_training_config('default',
        experiment_name='ablation_no_cross_mixup',
        data={'use_cross_mixup': False},
        training={'epochs': 100}
    )
    
    # 7. 无数据增强
    configs['no_augmentation'] = get_training_config('default',
        experiment_name='ablation_no_augmentation',
        data={
            'enhanced_augmentation': False,
            'use_mixup': False,
            'use_cross_mixup': False
        },
        training={'epochs': 100}
    )
    
    # 8. 冻结骨干网络
    configs['frozen_backbone'] = get_training_config('default',
        experiment_name='ablation_frozen_backbone',
        training={
            'freeze_backbone': True,
            'lr': 0.001,  # 更高的学习率
            'epochs': 50   # 更少的epoch
        }
    )
    
    # 9. 不同骨干网络
    configs['efficientnet_b2'] = get_training_config('efficientnet_b2',
        experiment_name='ablation_efficientnet_b2',
        training={'epochs': 100}
    )
    
    # 10. 不同嵌入维度
    configs['emb_128'] = get_training_config('default',
        experiment_name='ablation_emb_128',
        model={'emb_size': 128},
        training={'epochs': 100}
    )
    
    configs['emb_512'] = get_training_config('default',
        experiment_name='ablation_emb_512',
        model={'emb_size': 512},
        training={'epochs': 100}
    )
    
    # 11. 不同Dropout率
    configs['dropout_03'] = get_training_config('default',
        experiment_name='ablation_dropout_03',
        model={'dropout': 0.3},
        training={'epochs': 100}
    )
    
    configs['dropout_07'] = get_training_config('default',
        experiment_name='ablation_dropout_07',
        model={'dropout': 0.7},
        training={'epochs': 100}
    )
    
    # 12. 不同学习率
    configs['lr_0001'] = get_training_config('default',
        experiment_name='ablation_lr_0001',
        training={'lr': 0.0001, 'epochs': 100}
    )
    
    configs['lr_001'] = get_training_config('default',
        experiment_name='ablation_lr_001',
        training={'lr': 0.001, 'epochs': 100}
    )
    
    # 13. 不同批次大小
    configs['batch_8'] = get_training_config('default',
        experiment_name='ablation_batch_8',
        training={'batch_size': 8, 'epochs': 100}
    )
    
    configs['batch_32'] = get_training_config('default',
        experiment_name='ablation_batch_32',
        training={'batch_size': 32, 'epochs': 100}
    )
    
    # 14. 不同损失函数
    configs['smooth_l1_loss'] = get_training_config('default',
        experiment_name='ablation_smooth_l1',
        loss={'loss_type': 'smooth_l1'},
        training={'epochs': 100}
    )
    
    # 15. 不同卫星图像缩放比例
    configs['sat_scale_2'] = get_training_config('default',
        experiment_name='ablation_sat_scale_2',
        data={'sat_patch_scale': 2.0},
        training={'epochs': 100}
    )
    
    configs['sat_scale_4'] = get_training_config('default',
        experiment_name='ablation_sat_scale_4',
        data={'sat_patch_scale': 4.0},
        training={'epochs': 100}
    )
    
    return configs


def get_hyperparameter_search_configs():
    """
    获取超参数搜索配置
    
    Returns:
        list: 超参数配置列表
    """
    configs = []
    
    # 学习率搜索
    learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003]
    for lr in learning_rates:
        config = get_training_config('default',
            experiment_name=f'hyperparam_lr_{lr}',
            training={'lr': lr, 'epochs': 50}
        )
        configs.append(config)
    
    # 批次大小搜索
    batch_sizes = [8, 12, 16, 24, 32]
    for batch_size in batch_sizes:
        config = get_training_config('default',
            experiment_name=f'hyperparam_batch_{batch_size}',
            training={'batch_size': batch_size, 'epochs': 50}
        )
        configs.append(config)
    
    # Dropout搜索
    dropout_rates = [0.2, 0.3, 0.5, 0.7, 0.8]
    for dropout in dropout_rates:
        config = get_training_config('default',
            experiment_name=f'hyperparam_dropout_{dropout}',
            model={'dropout': dropout},
            training={'epochs': 50}
        )
        configs.append(config)
    
    # MixUp alpha搜索
    mixup_alphas = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    for alpha in mixup_alphas:
        config = get_training_config('default',
            experiment_name=f'hyperparam_mixup_{alpha}',
            data={'mixup_alpha': alpha},
            training={'epochs': 50}
        )
        configs.append(config)
    
    # 权重衰减搜索
    weight_decays = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    for wd in weight_decays:
        config = get_training_config('default',
            experiment_name=f'hyperparam_wd_{wd}',
            training={'weight_decay': wd, 'epochs': 50}
        )
        configs.append(config)
    
    return configs


def get_architecture_comparison_configs():
    """
    获取架构比较配置
    
    Returns:
        dict: 架构比较配置字典
    """
    configs = {}
    
    # EfficientNet变体
    configs['efficientnet_b0'] = get_training_config('efficientnet_b0',
        experiment_name='arch_efficientnet_b0'
    )
    
    configs['efficientnet_b2'] = get_training_config('efficientnet_b2',
        experiment_name='arch_efficientnet_b2'
    )
    
    # 不同嵌入维度
    for emb_size in [128, 256, 512, 1024]:
        configs[f'emb_{emb_size}'] = get_training_config('default',
            experiment_name=f'arch_emb_{emb_size}',
            model={'emb_size': emb_size}
        )
    
    return configs


def create_experiment_script(config_name, config, output_dir='./experiment_scripts'):
    """
    创建实验运行脚本
    
    Args:
        config_name (str): 配置名称
        config (dict): 配置字典
        output_dir (str): 输出目录
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    script_content = f"""#!/bin/bash

# 实验: {config_name}
# 自动生成的训练脚本

export CUDA_VISIBLE_DEVICES=0

python scripts/train.py \\
    --data_root /path/to/UAV_VisLoc_dataset \\
    --save_dir ./saved_models/{config_name} \\
    --log_dir ./logs/{config_name} \\
    --experiment_name {config_name} \\
    --backbone {config.get('model', {}).get('backbone', 'efficientnet_b0')} \\
    --emb_size {config.get('model', {}).get('emb_size', 256)} \\
    --dropout {config.get('model', {}).get('dropout', 0.5)} \\
    --batch_size {config.get('training', {}).get('batch_size', 16)} \\
    --epochs {config.get('training', {}).get('epochs', 100)} \\
    --lr {config.get('training', {}).get('lr', 0.0005)} \\
    --weight_decay {config.get('training', {}).get('weight_decay', 0.001)} \\
    --warmup_epochs {config.get('training', {}).get('warmup_epochs', 10)} \\
"""
    
    # 添加数据增强参数
    data_config = config.get('data', {})
    if data_config.get('enhanced_augmentation', True):
        script_content += "    --enhanced_augmentation \\\n"
    if data_config.get('use_mixup', True):
        script_content += f"    --use_mixup \\\n"
        script_content += f"    --mixup_alpha {data_config.get('mixup_alpha', 0.8)} \\\n"
    if data_config.get('use_cross_mixup', True):
        script_content += f"    --use_cross_mixup \\\n"
        script_content += f"    --cross_mixup_alpha {data_config.get('cross_mixup_alpha', 0.2)} \\\n"
    
    # 添加训练参数
    training_config = config.get('training', {})
    if training_config.get('use_amp', True):
        script_content += "    --use_amp \\\n"
    if training_config.get('freeze_backbone', False):
        script_content += "    --freeze_backbone \\\n"
    
    script_content += f"    --gradient_clip {training_config.get('gradient_clip', 1.0)}\n"
    
    # 保存脚本
    script_path = os.path.join(output_dir, f'{config_name}.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    print(f"实验脚本已保存到: {script_path}")


def main():
    """主函数 - 生成所有实验配置和脚本"""
    
    # 获取消融实验配置
    ablation_configs = get_ablation_configs()
    
    # 创建实验脚本目录
    script_dir = './experiment_scripts/ablation'
    os.makedirs(script_dir, exist_ok=True)
    
    # 为每个消融实验创建脚本
    for config_name, config in ablation_configs.items():
        create_experiment_script(config_name, config, script_dir)
    
    print(f"已生成 {len(ablation_configs)} 个消融实验脚本")
    
    # 获取超参数搜索配置
    hyperparam_configs = get_hyperparameter_search_configs()
    
    # 创建超参数搜索脚本目录
    hyperparam_script_dir = './experiment_scripts/hyperparameter'
    os.makedirs(hyperparam_script_dir, exist_ok=True)
    
    # 为每个超参数搜索实验创建脚本
    for i, config in enumerate(hyperparam_configs):
        config_name = config.get('experiment_name', f'hyperparam_{i}')
        create_experiment_script(config_name, config, hyperparam_script_dir)
    
    print(f"已生成 {len(hyperparam_configs)} 个超参数搜索脚本")
    
    # 创建批量运行脚本
    batch_script_content = """#!/bin/bash

# 批量运行消融实验

echo "开始运行消融实验..."

# 运行所有消融实验
for script in experiment_scripts/ablation/*.sh; do
    echo "运行实验: $script"
    bash "$script"
    echo "实验完成: $script"
    echo "---"
done

echo "所有消融实验完成！"
"""
    
    with open('./run_ablation_experiments.sh', 'w') as f:
        f.write(batch_script_content)
    
    os.chmod('./run_ablation_experiments.sh', 0o755)
    
    print("批量运行脚本已创建: ./run_ablation_experiments.sh")


if __name__ == '__main__':
    main()
