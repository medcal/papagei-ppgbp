import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from models.resnet import ResNet1DMoE
from utils import save_embeddings, segment_avg_to_dict
from visualization import plot_training_curves, plot_prediction_results, plot_ppg_signal_examples
import joblib
import pickle

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")
    WANDB_AVAILABLE = False


class PPGDataset(Dataset):
    def __init__(self, ppg_dir, df, case_name, label_col):
        self.ppg_dir = ppg_dir
        self.df = df
        self.case_name = case_name
        self.label_col = label_col
        self.case_ids = df[case_name].values
        self.labels = df[label_col].values

        # 检查实际的文件格式
        self.file_format = self._detect_file_format()
        print(f"检测到的文件格式: {self.file_format}")

    def _detect_file_format(self):
        """检测实际的文件保存格式"""
        if len(self.case_ids) == 0:
            return "unknown"

        case_id = self.case_ids[0]
        case_dir = os.path.join(self.ppg_dir, case_id)

        if not os.path.exists(case_dir):
            print(f"警告: 目录不存在 {case_dir}")
            return "unknown"

        files = os.listdir(case_dir)
        print(f"目录 {case_id} 中的文件: {files}")

        # 检查可能的文件格式
        if "segments.p" in files:
            return "segments.p"
        elif "segments.npy" in files:
            return "segments.npy"
        elif any(f in ["0.p", "1.p", "2.p"] for f in files):
            return "X.p"
        elif any(f.startswith("segment_") and f.endswith(".p") for f in files):
            return "segment_X.p"
        elif any(f.startswith("segment_") and f.endswith(".npy") for f in files):
            return "segment_X.npy"
        elif any(f in ["0.npy", "1.npy", "2.npy"] for f in files):
            return "X.npy"
        else:
            print(f"未知的文件格式，文件列表: {files}")
            return "unknown"

    def _load_data_file(self, file_path):
        """根据文件扩展名加载数据"""
        try:
            if file_path.endswith('.p'):
                # 尝试用 joblib 加载
                try:
                    data = joblib.load(file_path)
                    return data
                except:
                    # 如果 joblib 失败，尝试用 pickle
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    return data
            elif file_path.endswith('.npy'):
                return np.load(file_path)
            else:
                print(f"未知文件格式: {file_path}")
                return None
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None

    def __len__(self):
        if self.file_format in ["segments.npy", "segments.p"]:
            return len(self.case_ids)  # 每个受试者1个文件包含3段
        else:
            return len(self.case_ids) * 3  # 3 segments per subject

    def __getitem__(self, idx):
        if self.file_format in ["segments.npy", "segments.p"]:
            # 每个受试者一个文件包含所有段
            case_id = self.case_ids[idx]
            label = self.labels[idx]

            file_ext = ".p" if self.file_format == "segments.p" else ".npy"
            segment_path = os.path.join(self.ppg_dir, case_id, f"segments{file_ext}")

            segments = self._load_data_file(segment_path)
            if segments is not None:
                # 确保数据是正确的形状
                segments = np.array(segments)
                if segments.ndim == 2 and segments.shape[0] == 3:
                    # 随机选择一个段
                    segment_idx = np.random.randint(0, 3)
                    segment = segments[segment_idx]
                else:
                    print(f"警告: 段数据形状不正确 {segments.shape}, 使用第一个段")
                    segment = segments.flatten()[:1250] if segments.size >= 1250 else np.zeros(1250)
            else:
                print(f"文件不存在或加载失败: {segment_path}")
                segment = np.zeros(1250)
        else:
            # 每个段单独保存
            subject_idx = idx // 3
            segment_idx = idx % 3

            case_id = self.case_ids[subject_idx]
            label = self.labels[subject_idx]

            # 根据检测到的格式构建文件路径
            if self.file_format == "X.p":
                segment_path = os.path.join(self.ppg_dir, case_id, f"{segment_idx}.p")
            elif self.file_format == "segment_X.p":
                segment_path = os.path.join(self.ppg_dir, case_id, f"segment_{segment_idx}.p")
            elif self.file_format == "segment_X.npy":
                segment_path = os.path.join(self.ppg_dir, case_id, f"segment_{segment_idx}.npy")
            elif self.file_format == "X.npy":
                segment_path = os.path.join(self.ppg_dir, case_id, f"{segment_idx}.npy")
            else:
                # 尝试多种可能的格式
                possible_paths = [
                    os.path.join(self.ppg_dir, case_id, f"{segment_idx}.p"),
                    os.path.join(self.ppg_dir, case_id, f"segment_{segment_idx}.p"),
                    os.path.join(self.ppg_dir, case_id, f"segment_{segment_idx}.npy"),
                    os.path.join(self.ppg_dir, case_id, f"{segment_idx}.npy"),
                ]
                segment_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        segment_path = path
                        break

                if segment_path is None:
                    print(f"找不到段文件，尝试的路径: {possible_paths}")
                    segment = np.zeros(1250)
                    # Convert to tensor and add channel dimension
                    segment = torch.FloatTensor(segment).unsqueeze(0)
                    label = torch.FloatTensor([label])
                    return segment, label

            segment = self._load_data_file(segment_path)
            if segment is None:
                print(f"文件不存在或加载失败: {segment_path}")
                segment = np.zeros(1250)

        # 确保数据是正确的格式和长度
        segment = np.array(segment).flatten()
        if len(segment) > 1250:
            segment = segment[:1250]  # 截断
        elif len(segment) < 1250:
            # 填充到1250
            segment = np.pad(segment, (0, 1250 - len(segment)), mode='constant')

        # Convert to tensor and add channel dimension
        segment = torch.FloatTensor(segment).unsqueeze(0)  # (1, 1250)
        label = torch.FloatTensor([label])

        return segment, label


def train_papagei_s():
    """训练PaPaGei-S模型用于舒张压预测"""
    download_dir = r"E:\thsiu-ppg\5459299\PPG-BP Database"
    ppg_dir = f"{download_dir}/Data File/ppg/"
    model_save_dir = f"{download_dir}/trained_models/"
    vis_save_dir = f"{download_dir}/visualizations/"

    # 创建目录
    for dir_path in [model_save_dir, vis_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 30
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化wandb
    wandb_enabled = False
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="ppg-blood-pressure-estimation",
                name="papagei-s-diastolic-bp",
                config={
                    "model": "PaPaGei-S",
                    "target": "diastolic_bp",
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "device": device,
                    "architecture": "ResNet1DMoE",
                    "n_experts": 3
                }
            )
            wandb_enabled = True
            print("✓ WandB初始化成功，将实时记录训练过程")
        except Exception as e:
            print(f"⚠️  WandB初始化失败: {e}")
            print("   训练将继续进行，但不会记录到WandB")
            print("   提示: 运行 'python setup_wandb.py' 来配置WandB，或选择主菜单选项9")
            wandb_enabled = False
    else:
        print("⚠️  WandB未安装，训练将在本地进行")

    # 加载数据分割
    try:
        df_train = pd.read_csv(f"{download_dir}/Data File/train.csv")
        df_val = pd.read_csv(f"{download_dir}/Data File/val.csv")
        df_test = pd.read_csv(f"{download_dir}/Data File/test.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please run preprocess_data first.")
        return

    case_name = "subject_ID"
    df_train.loc[:, case_name] = df_train[case_name].apply(lambda x: str(x).zfill(4))
    df_val.loc[:, case_name] = df_val[case_name].apply(lambda x: str(x).zfill(4))
    df_test.loc[:, case_name] = df_test[case_name].apply(lambda x: str(x).zfill(4))

    # 创建数据集和数据加载器
    train_dataset = PPGDataset(ppg_dir, df_train, case_name, "diasbp")
    val_dataset = PPGDataset(ppg_dir, df_val, case_name, "diasbp")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型配置
    model_config = {
        'base_filters': 32,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 18,
        'n_classes': 512,
        'n_experts': 3
    }

    # 初始化模型
    model = ResNet1DMoE(
        in_channels=1,
        base_filters=model_config['base_filters'],
        kernel_size=model_config['kernel_size'],
        stride=model_config['stride'],
        groups=model_config['groups'],
        n_block=model_config['n_block'],
        n_classes=model_config['n_classes'],
        n_experts=model_config['n_experts']
    )

    # 添加回归头用于训练
    model.regression_head = nn.Linear(model_config['n_classes'], 1)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []

    print("开始训练PaPaGei-S模型...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training')

        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # 前向传播
            output = model(data)

            # 处理不同的模型输出格式
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output

            # 应用回归头
            output = model.regression_head(features)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        val_ppg_samples = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation')
            for batch_idx, (data, target) in enumerate(val_pbar):
                data, target = data.to(device), target.to(device)

                # 前向传播
                output = model(data)

                # 处理不同的模型输出格式
                if isinstance(output, tuple):
                    features = output[0]
                else:
                    features = output

                # 应用回归头
                pred = model.regression_head(features)
                loss = criterion(pred, target)

                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

                # 收集预测结果（仅在最后几个epoch）
                if epoch >= num_epochs - 3:
                    val_predictions.extend(pred.cpu().numpy().flatten())
                    val_targets.extend(target.cpu().numpy().flatten())
                    if len(val_ppg_samples) < 20:  # 只保存前20个样本用于可视化
                        val_ppg_samples.extend(data.cpu().numpy()[:min(5, data.shape[0])])

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)

        print(
            f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}')

        # Wandb记录
        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
                "train_rmse": np.sqrt(avg_train_loss),
                "val_rmse": np.sqrt(avg_val_loss)
            })

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 保存前移除回归头用于特征提取
            model_state = {k: v for k, v in model.state_dict().items() if not k.startswith('regression_head')}
            torch.save({
                'state_dict': model_state,
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_config': model_config
            }, f"{model_save_dir}/papagei_s.pt")
            print(f'保存最佳模型于第 {epoch + 1} 轮')

        # 早停
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("学习率过小，停止训练")
            break

    print("训练完成!")

    # 绘制训练曲线
    print("生成训练曲线图...")
    plot_training_curves(train_losses, val_losses, "PaPaGei-S", vis_save_dir)

    # 如果有验证预测结果，绘制预测对比图
    if val_predictions and val_targets:
        print("生成预测结果对比图...")
        plot_prediction_results(
            np.array(val_targets),
            np.array(val_predictions),
            "PaPaGei-S",
            "Diastolic BP (mmHg)",
            vis_save_dir
        )

        # 绘制PPG信号样例
        if val_ppg_samples:
            print("生成PPG信号样例图...")
            plot_ppg_signal_examples(
                val_ppg_samples[:6],
                val_predictions[:6],
                val_targets[:6],
                "PaPaGei-S",
                "Diastolic BP",
                vis_save_dir
            )

    # Wandb记录最终指标
    if wandb_enabled:
        if val_predictions and val_targets:
            final_mae = np.mean(np.abs(np.array(val_targets) - np.array(val_predictions)))
            final_rmse = np.sqrt(np.mean((np.array(val_targets) - np.array(val_predictions)) ** 2))

            wandb.log({
                "final_val_mae": final_mae,
                "final_val_rmse": final_rmse,
                "best_val_loss": best_val_loss
            })

            # 上传训练曲线图
            try:
                wandb.log({"training_curves": wandb.Image(f'{vis_save_dir}/PaPaGei-S_training_curves.png')})
                wandb.log({"prediction_results": wandb.Image(
                    f'{vis_save_dir}/PaPaGei-S_Diastolic BP (mmHg)_predictions.png')})
                if val_ppg_samples:
                    wandb.log({"ppg_examples": wandb.Image(f'{vis_save_dir}/PaPaGei-S_Diastolic BP_ppg_examples.png')})
            except Exception as e:
                print(f"⚠️  图片上传到WandB失败: {e}")

        wandb.finish()

    # 加载最佳模型进行特征提取
    model_path = f"{model_save_dir}/papagei_s.pt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # 创建用于特征提取的新模型（不含回归头）
        feature_model = ResNet1DMoE(
            in_channels=1,
            base_filters=model_config['base_filters'],
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts']
        )
        feature_model.load_state_dict(state_dict)
        feature_model.to(device)
        feature_model.eval()

        # 提取并保存特征
        print("开始提取特征...")
        extract_features_and_save(
            model=feature_model,
            ppg_dir=ppg_dir,
            batch_size=256,
            device=device,
            output_idx=0,
            resample=False,
            normalize=False,
            fs=125,
            fs_target=125,
            content="patient",
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            case_name=case_name,
            download_dir=download_dir
        )
        print("特征提取完成!")
    else:
        print("Warning: No trained model found for feature extraction")


def extract_features_and_save(model, ppg_dir, batch_size, device, output_idx, resample, normalize, fs, fs_target,
                              content, df_train, df_val, df_test, case_name, download_dir):
    """提取并保存特征"""
    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}

    for split in ['train', 'val', 'test']:
        df = dict_df[split]
        save_dir = f"{download_dir}/features"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_name = "papagei_s"
        if not os.path.exists(f"{save_dir}/{model_name}"):
            os.makedirs(f"{save_dir}/{model_name}")
        split_dir = f"{save_dir}/{model_name}/{split}/"

        child_dirs = np.unique(df[case_name].values)
        save_embeddings(
            path=ppg_dir,
            child_dirs=child_dirs,
            save_dir=split_dir,
            model=model,
            batch_size=batch_size,
            device=device,
            output_idx=output_idx,
            resample=resample,
            normalize=normalize,
            fs=fs,
            fs_target=fs_target
        )

        dict_feat = segment_avg_to_dict(split_dir, content)
        joblib.dump(dict_feat, f"{save_dir}/{model_name}/dict_{split}_{content}.p")


if __name__ == "__main__":
    train_papagei_s()