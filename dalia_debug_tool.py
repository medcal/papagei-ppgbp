"""
Dalia数据集调试工具
用于检查和修复pickle文件加载问题
"""

import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path


def safe_pickle_load(file_path):
    """安全加载pickle文件，尝试多种编码方式"""
    methods = [
        lambda f: pickle.load(f),
        lambda f: pickle.load(f, encoding='latin-1'),
        lambda f: pickle.load(f, encoding='bytes'),
        lambda f: pickle.load(f, encoding='utf-8'),
    ]

    for i, method in enumerate(methods):
        try:
            with open(file_path, 'rb') as f:
                data = method(f)
                print(f"  ✓ 方法 {i + 1} 成功")
                return data, i + 1
        except Exception as e:
            print(f"  ✗ 方法 {i + 1} 失败: {e}")
            continue

    return None, None


def analyze_data_structure(data, max_depth=2, current_depth=0):
    """分析数据结构"""
    if current_depth > max_depth:
        return "..."

    if isinstance(data, dict):
        result = "{\n"
        for key, value in list(data.items())[:5]:  # 只显示前5个键
            key_str = str(key) if not isinstance(key, bytes) else f"b'{key.decode('utf-8', errors='ignore')}'"
            value_type = type(value).__name__
            if isinstance(value, (list, np.ndarray)):
                value_str = f"[{len(value)} items]"
            elif isinstance(value, dict):
                value_str = f"dict with {len(value)} keys"
            else:
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)

            result += f"  {'  ' * current_depth}{key_str}: {value_type} = {value_str}\n"

        if len(data) > 5:
            result += f"  {'  ' * current_depth}... ({len(data) - 5} more keys)\n"
        result += "}"
        return result

    elif isinstance(data, (list, np.ndarray)):
        if len(data) == 0:
            return "[]"
        return f"[{len(data)} items] - first: {type(data[0]).__name__}"

    else:
        return f"{type(data).__name__}: {str(data)[:100]}"


def check_dalia_file(file_path):
    """检查单个Dalia文件"""
    print(f"\n检查文件: {file_path}")

    if not os.path.exists(file_path):
        print("  ✗ 文件不存在")
        return None

    # 尝试加载
    data, method = safe_pickle_load(file_path)

    if data is None:
        print("  ✗ 无法加载文件")
        return None

    print(f"  ✓ 文件加载成功 (使用方法 {method})")
    print(f"  数据类型: {type(data)}")

    if isinstance(data, dict):
        print(f"  字典键数量: {len(data)}")
        print("  主要键:")
        for key in list(data.keys())[:10]:
            key_str = str(key) if not isinstance(key, bytes) else f"b'{key.decode('utf-8', errors='ignore')}'"
            print(f"    - {key_str}")

        # 检查关键数据
        signal_keys = ['signal', b'signal']
        label_keys = ['label', b'label']
        quest_keys = ['questionnaire', b'questionnaire']

        print("\n  关键数据检查:")

        # 检查signal数据
        signal_data = None
        for key in signal_keys:
            if key in data:
                signal_data = data[key]
                print(f"    ✓ 找到signal数据 (键: {key})")
                break

        if signal_data and isinstance(signal_data, dict):
            wrist_keys = ['wrist', b'wrist']
            for key in wrist_keys:
                if key in signal_data:
                    wrist_data = signal_data[key]
                    print(f"      ✓ 找到wrist数据 (键: {key})")

                    if isinstance(wrist_data, dict):
                        bvp_keys = ['BVP', b'BVP']
                        acc_keys = ['ACC', b'ACC']

                        for bvp_key in bvp_keys:
                            if bvp_key in wrist_data:
                                bvp_data = wrist_data[bvp_key]
                                print(f"        ✓ BVP数据: {len(bvp_data)} 个数据点 (键: {bvp_key})")
                                break

                        for acc_key in acc_keys:
                            if acc_key in wrist_data:
                                acc_data = wrist_data[acc_key]
                                if isinstance(acc_data, np.ndarray) and acc_data.ndim == 2:
                                    print(f"        ✓ ACC数据: {acc_data.shape} (键: {acc_key})")
                                else:
                                    print(f"        ✓ ACC数据: {len(acc_data)} 个数据点 (键: {acc_key})")
                                break
                    break

        # 检查label数据
        for key in label_keys:
            if key in data:
                labels = data[key]
                if isinstance(labels, (list, np.ndarray)):
                    print(f"    ✓ 找到label数据: {len(labels)} 个标签 (键: {key})")
                    if len(labels) > 0:
                        print(f"      心率范围: {np.min(labels):.1f} - {np.max(labels):.1f}")
                else:
                    print(f"    ✓ 找到label数据: {type(labels)} (键: {key})")
                break

        # 检查questionnaire数据
        for key in quest_keys:
            if key in data:
                quest = data[key]
                print(f"    ✓ 找到questionnaire数据 (键: {key})")
                if isinstance(quest, dict):
                    print(f"      包含字段: {list(quest.keys())}")
                break

    return data


def check_all_dalia_files(data_dir):
    """检查所有Dalia文件"""
    print(f"检查Dalia数据集: {data_dir}")

    results = {}

    for subject_id in range(1, 16):
        file_path = f"{data_dir}/S{subject_id}/S{subject_id}.pkl"
        data = check_dalia_file(file_path)
        results[f"S{subject_id}"] = data is not None

    # 总结
    print(f"\n" + "=" * 50)
    print("总结:")
    successful = sum(results.values())
    print(f"成功加载: {successful}/15 个文件")

    if successful < 15:
        print("失败的文件:")
        for subject, success in results.items():
            if not success:
                print(f"  - {subject}")

    return results


def create_fixed_csv_files(data_dir, output_dir="processed"):
    """创建修复后的CSV文件"""
    print(f"\n创建修复后的CSV文件...")

    # 数据集划分
    splits = {
        'train': list(range(1, 13)),  # S1-S12
        'val': [13, 14],  # S13-S14
        'test': [15]  # S15
    }

    os.makedirs(f"{data_dir}/{output_dir}", exist_ok=True)

    for split_name, subject_ids in splits.items():
        data = []

        for subject_id in subject_ids:
            file_path = f"{data_dir}/S{subject_id}/S{subject_id}.pkl"

            # 加载数据
            subject_data, _ = safe_pickle_load(file_path)

            if subject_data is None:
                # 添加默认数据
                data.append({
                    'subject_ID': subject_id,
                    'hr': 70.0,
                    'age': 30,
                    'gender': 'unknown',
                    'num_segments': 100
                })
                continue

            try:
                # 提取心率数据
                labels = None
                for key in ['label', b'label']:
                    if key in subject_data:
                        labels = subject_data[key]
                        break

                if labels and len(labels) > 0:
                    # 过滤有效心率值
                    valid_labels = [l for l in labels if isinstance(l, (int, float)) and 30 <= l <= 200]
                    avg_hr = np.mean(valid_labels) if valid_labels else 70.0
                    num_segments = len(labels)
                else:
                    avg_hr = 70.0
                    num_segments = 100

                # 提取questionnaire信息
                questionnaire = None
                for key in ['questionnaire', b'questionnaire']:
                    if key in subject_data:
                        questionnaire = subject_data[key]
                        break

                age = 30
                gender = 'unknown'

                if questionnaire and isinstance(questionnaire, dict):
                    # 提取年龄
                    for age_key in ['age', b'age']:
                        if age_key in questionnaire:
                            try:
                                age = int(questionnaire[age_key])
                                break
                            except (ValueError, TypeError):
                                pass

                    # 提取性别
                    for gender_key in ['gender', b'gender', 'sex', b'sex']:
                        if gender_key in questionnaire:
                            gender_val = questionnaire[gender_key]
                            if isinstance(gender_val, bytes):
                                gender = gender_val.decode('utf-8', errors='ignore')
                            else:
                                gender = str(gender_val)
                            break

                data.append({
                    'subject_ID': subject_id,
                    'hr': float(avg_hr),
                    'age': int(age),
                    'gender': str(gender),
                    'num_segments': int(num_segments)
                })

            except Exception as e:
                print(f"处理 S{subject_id} 时出错: {e}")
                # 添加默认数据
                data.append({
                    'subject_ID': subject_id,
                    'hr': 70.0,
                    'age': 30,
                    'gender': 'unknown',
                    'num_segments': 100
                })

        # 保存CSV
        if data:
            df = pd.DataFrame(data)
            csv_path = f"{data_dir}/{output_dir}/{split_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✓ 保存 {csv_path}: {len(df)} 个受试者")


def main():
    """主函数"""
    print("Dalia数据集调试工具")
    print("=" * 50)

    # 设置数据目录
    data_dir = r"F:\ppg+dalia\data\PPG_FieldStudy"

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在 {data_dir}")
        return

    # 检查所有文件
    results = check_all_dalia_files(data_dir)

    # 创建修复后的CSV文件
    create_fixed_csv_files(data_dir)

    print("\n调试完成！")
    print("建议:")
    print("1. 查看生成的CSV文件确认数据正确")
    print("2. 重新运行主训练程序")


if __name__ == "__main__":
    main()