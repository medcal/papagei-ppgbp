import pandas as pd
import numpy as np
import os
from tqdm import tqdm
try:
    from torch_ecg._preprocessors import Normalize
except ImportError:
    print("Warning: torch_ecg not available, using simple normalization")
    class Normalize:
        def __init__(self, method='z-score'):
            self.method = method

        def apply(self, signal, fs=None):
            if self.method == 'z-score':
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            return signal, None

from utils import resample_batch_signal, preprocess_one_ppg_signal, save_segments_to_directory

def preprocess_data():
    """数据预处理主函数"""
    download_dir = r"E:\thsiu-ppg\5459299\PPG-BP Database"
    main_dir = f"{download_dir}/Data File/0_subject/"
    ppg_dir = f"{download_dir}/Data File/ppg/"

    # 创建ppg目录如果不存在
    if not os.path.exists(ppg_dir):
        os.makedirs(ppg_dir)

    # 检查必需的文件是否存在
    dataset_file = f"{download_dir}/Data File/PPG-BP dataset.xlsx"
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at {dataset_file}")
        print("Please ensure the PPG-BP dataset.xlsx file exists in the Data File directory")
        return

    if not os.path.exists(main_dir):
        print(f"Error: Subject directory not found at {main_dir}")
        print("Please ensure the 0_subject directory exists with the signal files")
        return

    # 读取数据集
    try:
        df = pd.read_excel(dataset_file, header=1)
        print(f"Dataset loaded successfully with {len(df)} subjects")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    subjects = df.subject_ID.values

    # 预处理参数
    fs = 1000
    fs_target = 125
    norm = Normalize(method='z-score')

    # 处理信号
    try:
        filenames = [f.split("_")[0] for f in os.listdir(main_dir) if f.endswith('.txt')]
        filenames = list(set(filenames))  # 去重
        print(f"Found {len(filenames)} unique subjects in signal directory")
    except Exception as e:
        print(f"Error reading signal directory: {e}")
        return

    processed_count = 0
    for f in tqdm(filenames, desc="Processing signal files"):
        try:
            segments = []
            valid_segments = 0

            for s in range(1, 4):  # 处理3个段
                signal_file = f"{main_dir}{f}_{str(s)}.txt"

                if not os.path.exists(signal_file):
                    print(f"Warning: Signal file not found: {signal_file}")
                    continue

                try:
                    # 读取信号数据
                    signal = pd.read_csv(signal_file, sep='\t', header=None)
                    signal = signal.values.squeeze()

                    # 处理数据格式问题
                    if signal.ndim > 1:
                        signal = signal.flatten()

                    # 移除最后一个元素（如果存在）
                    if len(signal) > 0:
                        signal = signal[:-1] if len(signal) > 1 else signal

                    if len(signal) == 0:
                        print(f"Warning: Empty signal in {signal_file}")
                        continue

                    # 归一化
                    signal, _ = norm.apply(signal, fs=fs)

                    # PPG信号预处理
                    signal, _, _, _ = preprocess_one_ppg_signal(waveform=signal, frequency=fs)

                    # 重采样
                    signal = resample_batch_signal(signal, fs_original=fs, fs_target=fs_target, axis=0)

                    # 填充或截断到固定长度1250
                    target_length = 1250
                    if len(signal) > target_length:
                        signal = signal[:target_length]
                    elif len(signal) < target_length:
                        padding_needed = target_length - len(signal)
                        pad_left = padding_needed // 2
                        pad_right = padding_needed - pad_left
                        signal = np.pad(signal, pad_width=(pad_left, pad_right), mode='constant', constant_values=0)

                    segments.append(signal)
                    valid_segments += 1

                except Exception as e:
                    print(f"Warning: Error processing {signal_file}: {e}")
                    continue

            # 如果有有效的段，保存数据
            if valid_segments > 0:
                if valid_segments < 3:
                    # 如果段数不足3个，用现有段重复填充
                    while len(segments) < 3:
                        segments.append(segments[-1].copy())

                segments = np.vstack(segments)
                child_dir = f.zfill(4)
                save_segments_to_directory(save_dir=ppg_dir, dir_name=child_dir, segments=segments)
                processed_count += 1
            else:
                print(f"Warning: No valid segments found for subject {f}")

        except Exception as e:
            print(f"Error processing subject {f}: {e}")
            continue

    print(f"Successfully processed {processed_count} subjects")

    # 重命名列并处理缺失值
    df = df.rename(columns={
        "Sex(M/F)": "sex",
        "Age(year)": "age",
        "Systolic Blood Pressure(mmHg)": "sysbp",
        "Diastolic Blood Pressure(mmHg)": "diasbp",
        "Heart Rate(b/m)": "hr",
        "BMI(kg/m^2)": "bmi"
    })
    df = df.fillna(0)

    # 定义受试者分割
    train_ids = [2, 6, 8, 10, 12, 15, 16, 17, 18, 19, 22, 23, 26, 31, 32, 34, 35, 38, 40, 45, 48, 50, 53, 55, 56, 58, 60, 61, 63, 65, 66, 83, 85, 87, 89, 92, 93, 97, 98, 99, 100, 104, 105, 106, 107, 112, 113, 114, 116, 120, 122, 126, 128, 131, 134, 135, 137, 138, 139, 140, 141, 146, 148, 149, 152, 153, 154, 158, 160, 162, 164, 165, 167, 169, 170, 175, 176, 179, 183, 184, 186, 188, 189, 190, 191, 193, 196, 197, 199, 205, 206, 207, 209, 210, 212, 216, 217, 218, 223, 226, 227, 230, 231, 233, 234, 240, 242, 243, 244, 246, 247, 248, 256, 257, 404, 407, 409, 412, 414, 415, 416, 417, 419]
    test_ids = [14, 21, 25, 51, 52, 62, 67, 86, 90, 96, 103, 108, 110, 119, 123, 124, 130, 142, 144, 157, 172, 173, 174, 180, 182, 185, 192, 195, 200, 201, 211, 214, 219, 221, 228, 239, 250, 403, 405, 406, 410]
    val_ids = [3, 11, 24, 27, 29, 30, 41, 43, 47, 64, 88, 91, 95, 115, 125, 127, 136, 145, 155, 156, 161, 163, 166, 178, 198, 203, 208, 213, 215, 222, 229, 232, 235, 237, 241, 245, 252, 254, 259, 411, 418]

    # 分割和保存数据
    df_train = df[df.subject_ID.isin(train_ids)]
    df_val = df[df.subject_ID.isin(val_ids)]
    df_test = df[df.subject_ID.isin(test_ids)]

    print(f"Train set: {len(df_train)} subjects")
    print(f"Validation set: {len(df_val)} subjects")
    print(f"Test set: {len(df_test)} subjects")

    # 保存分割后的数据
    try:
        df_train.to_csv(f"{download_dir}/Data File/train.csv", index=False)
        df_val.to_csv(f"{download_dir}/Data File/val.csv", index=False)
        df_test.to_csv(f"{download_dir}/Data File/test.csv", index=False)
        print("Data splits saved successfully")
    except Exception as e:
        print(f"Error saving data splits: {e}")

if __name__ == "__main__":
    preprocess_data()