import os
import numpy as np
import pandas as pd
import joblib
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_directory(path):
    """创建目录如果不存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_segments_to_directory(save_dir, dir_name, segments):
    """保存段数据到目录中"""
    dir_path = os.path.join(save_dir, dir_name)
    create_directory(dir_path)
    
    # 保存为pickle文件
    segment_path = os.path.join(dir_path, "segments.p")
    joblib.dump(segments, segment_path)

def preprocess_one_ppg_signal(waveform, frequency):
    """
    预处理单个PPG信号
    这是一个简化版本，实际实现需要根据具体需求调整
    """
    # 简单的信号预处理：去除异常值和平滑
    signal = np.array(waveform)
    
    # 去除异常值（使用3倍标准差规则）
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    signal = np.clip(signal, mean_val - 3*std_val, mean_val + 3*std_val)
    
    # 简单平滑（移动平均）
    window_size = max(1, int(frequency * 0.1))  # 0.1秒窗口
    if len(signal) > window_size:
        kernel = np.ones(window_size) / window_size
        signal = np.convolve(signal, kernel, mode='same')
    
    return signal, None, None, None

def resample_batch_signal(signal, fs_original, fs_target, axis=0):
    """重采样信号"""
    if fs_original == fs_target:
        return signal
    
    # 计算新的采样点数
    num_samples_original = signal.shape[axis]
    num_samples_target = int(num_samples_original * fs_target / fs_original)
    
    # 使用scipy的resample函数
    resampled_signal = resample(signal, num_samples_target, axis=axis)
    
    return resampled_signal

def get_data_for_ml(df, dict_embeddings, case_name, label):
    """为机器学习准备数据"""
    X_list = []
    y_list = []
    case_list = []
    
    for idx, row in df.iterrows():
        case_id = row[case_name]
        if case_id in dict_embeddings:
            X_list.append(dict_embeddings[case_id])
            y_list.append(row[label])
            case_list.append(case_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    cases = np.array(case_list)
    
    return X, y, cases

def save_embeddings(path, child_dirs, save_dir, model, batch_size, device, 
                   output_idx=0, resample=False, normalize=False, fs=125, fs_target=125):
    """保存模型特征嵌入"""
    create_directory(save_dir)
    model.eval()
    
    with torch.no_grad():
        for child_dir in tqdm(child_dirs, desc="Extracting features"):
            child_path = os.path.join(path, child_dir)
            if not os.path.exists(child_path):
                continue
            
            # 读取segments文件
            segments_path = os.path.join(child_path, "segments.p")
            if os.path.exists(segments_path):
                try:
                    segments = joblib.load(segments_path)
                except:
                    try:
                        with open(segments_path, 'rb') as f:
                            segments = pickle.load(f)
                    except:
                        continue
            else:
                continue
            
            segments = np.array(segments)
            if segments.ndim == 1:
                segments = segments.reshape(1, -1)
            
            # 准备数据
            features_list = []
            for segment in segments:
                # 确保数据长度为1250
                if len(segment) > 1250:
                    segment = segment[:1250]
                elif len(segment) < 1250:
                    segment = np.pad(segment, (0, 1250 - len(segment)), mode='constant')
                
                # 转换为tensor并添加通道维度
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).unsqueeze(0)  # (1, 1, 1250)
                segment_tensor = segment_tensor.to(device)
                
                # 提取特征
                output = model(segment_tensor)
                if isinstance(output, tuple):
                    features = output[output_idx]
                else:
                    features = output
                
                features_list.append(features.cpu().numpy().flatten())
            
            # 平均所有段的特征
            if features_list:
                avg_features = np.mean(features_list, axis=0)
                
                # 保存特征
                feature_path = os.path.join(save_dir, f"{child_dir}.p")
                joblib.dump(avg_features, feature_path)

def segment_avg_to_dict(split_dir, content):
    """将段特征转换为字典格式"""
    dict_feat = {}
    
    if not os.path.exists(split_dir):
        return dict_feat
    
    for file in os.listdir(split_dir):
        if file.endswith('.p'):
            case_id = file.replace('.p', '')
            file_path = os.path.join(split_dir, file)
            try:
                features = joblib.load(file_path)
                dict_feat[case_id] = features
            except:
                continue
    
    return dict_feat

def regression_model(estimator, param_grid, X_train, y_train, X_test, y_test):
    """运行回归模型并返回结果"""
    # 网格搜索最佳参数
    grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    # 使用最佳模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'best_params': grid_search.best_params_,
        'predictions': y_pred,
        'actual': y_test
    }
    
    return results