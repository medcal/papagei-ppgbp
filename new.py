#!/usr/bin/env python3
"""
一键创建项目配置文件
包括 .gitignore 和 configs 目录下的所有配置文件
"""

import os


def create_gitignore():
    """创建 .gitignore 文件"""
    gitignore_content = """# 数据文件和大文件
*.xlsx
*.csv
data/
datasets/
**/Data File/
**/PPG-BP Database/

# 模型文件和训练结果
*.pt
*.pth
*.pkl
*.p
trained_models/
checkpoints/
features/
logs/

# 可视化结果
visualizations/
figures/
plots/
*.png
*.pdf
*.jpg
*.jpeg

# Python缓存文件
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# IDE和编辑器文件
.vscode/
.idea/
*.swp
*.swo
*~

# 系统文件
.DS_Store
Thumbs.db

# 环境变量文件
.env
.env.local

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# WandB本地文件
wandb/

# 临时文件
*.tmp
*.temp
*.log
"""

    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("✅ .gitignore 文件创建成功!")


def create_configs():
    """创建配置文件目录和文件"""
    # 创建 configs 目录
    os.makedirs('configs', exist_ok=True)
    print("✅ configs 目录创建成功!")

    # 创建模型配置文件
    model_config = """# PPG血压估计模型配置文件

# 原始模型配置
original_models:
  papagei_s:
    architecture: "ResNet1DMoE"
    base_filters: 32
    kernel_size: 3
    stride: 2
    groups: 1
    n_block: 18
    n_classes: 512
    n_experts: 3
    target: "diasbp"

  papagei_s_svri:
    architecture: "ResNet1D"
    base_filters: 32
    kernel_size: 3
    stride: 2
    groups: 1
    n_block: 18
    n_classes: 512
    target: "hr"

  papagei_p:
    architecture: "ResNet1D"
    base_filters: 32
    kernel_size: 3
    stride: 2
    groups: 1
    n_block: 18
    n_classes: 512
    target: "sysbp"

# 简化模型配置（推荐用于小数据集）
simplified_models:
  simplified_hr:
    architecture: "SimplifiedResNet1D"
    n_classes: 64
    dropout_rate: 0.5
    target: "hr"

  simplified_sysbp:
    architecture: "SimplifiedResNet1D"
    n_classes: 64
    dropout_rate: 0.4
    target: "sysbp"

  simplified_diasbp:
    architecture: "SimplifiedResNet1D"
    n_classes: 64
    dropout_rate: 0.4
    target: "diasbp"
"""

    with open('configs/model_config.yaml', 'w', encoding='utf-8') as f:
        f.write(model_config)
    print("✅ configs/model_config.yaml 创建成功!")

    # 创建训练配置文件
    train_config = """# 训练配置文件

# 原始训练配置
original_training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 30
  weight_decay: 1e-5
  patience: 10
  scheduler: "ReduceLROnPlateau"

# 改进训练配置（针对小数据集）
improved_training:
  batch_size: 8
  learning_rate: 0.0001
  num_epochs: 100
  weight_decay: 1e-3
  patience: 20
  scheduler: "CosineAnnealingWarmRestarts"
  gradient_clip_norm: 1.0
  use_data_augmentation: true

# WandB配置
wandb:
  project: "ppg-blood-pressure-estimation"
  entity: "your-wandb-username"  # 替换为您的WandB用户名
  tags: ["ppg", "blood-pressure", "deep-learning"]
"""

    with open('configs/train_config.yaml', 'w', encoding='utf-8') as f:
        f.write(train_config)
    print("✅ configs/train_config.yaml 创建成功!")

    # 创建数据配置文件
    data_config = """# 数据配置文件

# 数据路径配置
data_paths:
  base_dir: "E:/thsiu-ppg/5459299/PPG-BP Database"  # 修改为您的数据路径
  dataset_file: "Data File/PPG-BP dataset.xlsx"
  signal_dir: "Data File/0_subject/"
  ppg_dir: "Data File/ppg/"

# 数据预处理配置
preprocessing:
  sampling_rate: 1000
  target_rate: 125
  signal_length: 1250
  normalization: "z-score"

# 数据分割配置
data_split:
  train_ratio: 0.56
  val_ratio: 0.19
  test_ratio: 0.25
  random_seed: 42

# 目标变量
targets:
  hr: "Heart Rate (bpm)"
  sysbp: "Systolic BP (mmHg)"
  diasbp: "Diastolic BP (mmHg)"
"""

    with open('configs/data_config.yaml', 'w', encoding='utf-8') as f:
        f.write(data_config)
    print("✅ configs/data_config.yaml 创建成功!")


def create_requirements():
    """创建 requirements.txt 文件"""
    requirements = """torch>=1.9.0
torchvision>=0.10.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
joblib>=1.1.0
scipy>=1.7.0
wandb>=0.12.0
openpyxl>=3.0.0
PyYAML>=6.0
"""

    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    print("✅ requirements.txt 创建成功!")


def main():
    """主函数"""
    print("🚀 开始创建项目配置文件...")
    print("=" * 40)

    # 检查是否在正确的目录
    if not os.path.exists('main.py'):
        print("⚠️  警告: 未在项目根目录运行，请确保在包含 main.py 的目录中运行此脚本")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return

    # 创建文件
    create_gitignore()
    create_configs()
    create_requirements()

    print("\n" + "=" * 40)
    print("🎉 所有配置文件创建完成!")
    print("\n📁 创建的文件:")
    print("  • .gitignore")
    print("  • requirements.txt")
    print("  • configs/model_config.yaml")
    print("  • configs/train_config.yaml")
    print("  • configs/data_config.yaml")

    print("\n📝 下一步:")
    print("1. 检查 configs/data_config.yaml 中的数据路径")
    print("2. 修改 configs/train_config.yaml 中的 WandB 用户名")
    print("3. 根据需要调整其他配置参数")
    print("4. 运行: git add . && git commit -m 'Add configuration files'")


if __name__ == "__main__":
    main()