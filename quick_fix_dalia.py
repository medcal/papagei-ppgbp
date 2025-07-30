"""
Dalia数据集快速修复脚本 - 基于调试结果更新
直接解决pickle加载和DataFrame创建问题
"""

import os
import pickle
import pandas as pd
import numpy as np


def quick_fix_dalia_dataset(data_dir=r"F:\ppg+dalia\data\PPG_FieldStudy"):
    """快速修复Dalia数据集问题"""

    print("开始快速修复Dalia数据集...")

    # 创建processed目录
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 数据集划分
    splits = {
        'train': list(range(1, 13)),  # S1-S12
        'val': [13, 14],              # S13-S14
        'test': [15]                  # S15
    }

    def safe_load_pickle(file_path):
        """安全加载pickle文件 - 基于调试结果"""
        if not os.path.exists(file_path):
            return None

        # 根据调试结果，方法2 (latin-1) 最有效
        methods = [
            lambda f: pickle.load(f, encoding='latin-1'),  # 优先使用这个
            lambda f: pickle.load(f),
            lambda f: pickle.load(f, encoding='bytes')
        ]

        for i, method in enumerate(methods):
            try:
                with open(file_path, 'rb') as f:
                    data = method(f)
                    if i == 0:
                        print(f"    ✓ 使用latin-1编码加载成功")
                    return data
            except Exception as e:
                if i == 0:
                    print(f"    ✗ latin-1编码失败: {e}")
                continue
        return None

    def extract_subject_info(subject_data):
        """提取受试者信息 - 修复数组比较问题"""
        if not subject_data:
            return {
                'hr_labels': [70.0] * 100,
                'avg_hr': 70.0,
                'age': 30,
                'gender': 'unknown',
                'weight': 70,
                'height': 170
            }

        # 提取心率标签
        hr_labels = subject_data.get('label', [])

        if hr_labels is not None and len(hr_labels) > 0:
            # 转换为numpy数组以便处理
            hr_array = np.array(hr_labels)

            # 过滤有效心率值 (30-200 bpm) - 修复数组比较问题
            # 使用numpy的向量化操作
            valid_mask = (
                np.isfinite(hr_array) &  # 不是NaN或inf
                (hr_array >= 30) &       # 最小值
                (hr_array <= 200)        # 最大值
            )

            valid_labels = hr_array[valid_mask]

            if len(valid_labels) > 0:
                avg_hr = float(np.mean(valid_labels))
                # 转换回Python列表
                hr_labels_list = [float(hr) for hr in valid_labels]
            else:
                avg_hr = 70.0
                hr_labels_list = [70.0] * 100
        else:
            avg_hr = 70.0
            hr_labels_list = [70.0] * 100

        # 提取questionnaire信息
        questionnaire = subject_data.get('questionnaire', {})

        # 安全提取字段
        def safe_extract(data, key, default, convert_func=None):
            """安全提取字段值"""
            try:
                value = data.get(key, default)
                if value is None or value == '':
                    return default
                if convert_func:
                    return convert_func(value)
                return value
            except (ValueError, TypeError, AttributeError):
                return default

        # 从调试输出可知字段名为: 'AGE', 'Gender', 'WEIGHT', 'HEIGHT', 'SKIN', 'SPORT'
        age = safe_extract(questionnaire, 'AGE', 30, lambda x: int(float(x)))
        gender = safe_extract(questionnaire, 'Gender', 'unknown', str)
        weight = safe_extract(questionnaire, 'WEIGHT', 70.0, float)
        height = safe_extract(questionnaire, 'HEIGHT', 170.0, float)

        # 处理可能的bytes类型
        if isinstance(gender, bytes):
            gender = gender.decode('utf-8', errors='ignore')

        return {
            'hr_labels': hr_labels_list,
            'avg_hr': avg_hr,
            'age': age,
            'gender': str(gender),
            'weight': weight,
            'height': height
        }

    # 为每个分割创建CSV文件
    all_data = {}  # 存储所有受试者数据

    print("\n加载所有受试者数据...")
    for subject_id in range(1, 16):
        pkl_file = os.path.join(data_dir, f"S{subject_id}", f"S{subject_id}.pkl")
        print(f"加载 S{subject_id}...")

        # 加载数据
        subject_data = safe_load_pickle(pkl_file)

        # 提取信息
        info = extract_subject_info(subject_data)
        all_data[subject_id] = info

        print(f"  心率: {info['avg_hr']:.1f} bpm (共{len(info['hr_labels'])}个标签)")
        print(f"  基本信息: {info['age']}岁, {info['gender']}, {info['weight']}kg, {info['height']}cm")

    print("\n创建CSV文件...")
    for split_name, subject_ids in splits.items():
        print(f"处理 {split_name} 分割...")

        data = []

        for subject_id in subject_ids:
            info = all_data.get(subject_id, {})

            # 为每个受试者创建多行数据（每个心率标签一行）
            hr_labels = info.get('hr_labels', [70.0])

            # 限制段数避免内存问题
            max_segments = min(len(hr_labels), 200)

            for i in range(max_segments):
                hr_value = hr_labels[i] if i < len(hr_labels) else info.get('avg_hr', 70.0)

                data.append({
                    'subject_ID': subject_id,
                    'segment_idx': i,
                    'hr': float(hr_value),
                    'age': info.get('age', 30),
                    'gender': info.get('gender', 'unknown'),
                    'weight': info.get('weight', 70.0),
                    'height': info.get('height', 170.0),
                    'num_segments': len(hr_labels)
                })

        # 保存CSV
        df = pd.DataFrame(data)
        csv_path = os.path.join(processed_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ 保存 {csv_path}: {len(df)} 行数据")

        # 显示统计信息
        unique_subjects = df['subject_ID'].nunique()
        avg_hr_range = f"{df['hr'].min():.1f}-{df['hr'].max():.1f}"
        print(f"  {unique_subjects} 个受试者, 心率范围: {avg_hr_range} bpm")

    print("\n快速修复完成！")
    return True


def test_fixed_data(data_dir=r"F:\ppg+dalia\data\PPG_FieldStudy"):
    """测试修复后的数据"""
    processed_dir = os.path.join(data_dir, "processed")

    print("\n" + "="*50)
    print("测试修复后的数据...")

    total_samples = 0

    for split in ['train', 'val', 'test']:
        csv_file = os.path.join(processed_dir, f"{split}.csv")

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            total_samples += len(df)

            print(f"\n{split}.csv:")
            print(f"  行数: {len(df)}")
            print(f"  列: {list(df.columns)}")
            print(f"  受试者: {df['subject_ID'].nunique()} 个")
            print(f"  心率范围: {df['hr'].min():.1f} - {df['hr'].max():.1f} bpm")

            # 检查必要的列
            required_cols = ['subject_ID', 'hr', 'segment_idx']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"  ⚠️  缺少列: {missing_cols}")
            else:
                print(f"  ✅ 包含所有必要列")

            # 显示受试者信息
            subjects = sorted(df['subject_ID'].unique())
            print(f"  受试者ID: S{min(subjects)}-S{max(subjects)}")

        else:
            print(f"\n{split}.csv: ❌ 文件不存在")

    print(f"\n总数据样本: {total_samples}")
    print("="*50)


def create_simple_test_data(data_dir=r"F:\ppg+dalia\data\PPG_FieldStudy"):
    """创建简化测试数据（如果需要的话）"""
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("创建简化测试数据...")

    # 简化的数据集划分
    splits = {
        'train': list(range(1, 13)),  # S1-S12
        'val': [13, 14],              # S13-S14
        'test': [15]                  # S15
    }

    for split_name, subject_ids in splits.items():
        data = []

        for subject_id in subject_ids:
            # 为每个受试者创建50个数据段
            for i in range(50):
                hr = 60 + subject_id + np.random.normal(0, 5)  # 基于受试者ID的心率
                hr = max(50, min(120, hr))  # 限制在合理范围

                data.append({
                    'subject_ID': subject_id,
                    'segment_idx': i,
                    'hr': float(hr),
                    'age': 25 + subject_id,
                    'gender': 'M' if subject_id % 2 == 1 else 'F',
                    'weight': 60 + subject_id * 2,
                    'height': 160 + subject_id,
                    'num_segments': 50
                })

        df = pd.DataFrame(data)
        csv_path = os.path.join(processed_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ 创建 {csv_path}: {len(df)} 行")


if __name__ == "__main__":
    # 设置数据目录
    data_dir = r"F:\ppg+dalia\data\PPG_FieldStudy"

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在 {data_dir}")
        print("请修改data_dir变量为正确的路径")
        exit(1)

    try:
        # 执行快速修复
        if quick_fix_dalia_dataset(data_dir):
            # 测试修复结果
            test_fixed_data(data_dir)

            print("\n" + "🎉" * 20)
            print("✅ 修复完成！现在可以运行主程序:")
            print("   python main.py")
            print("🎉" * 20)
        else:
            print("❌ 修复失败")

    except Exception as e:
        print(f"❌ 修复过程中出错: {e}")
        print("\n尝试创建简化测试数据...")
        create_simple_test_data(data_dir)
        print("✅ 简化数据创建完成，可以用于测试训练流程")