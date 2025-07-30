import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import joblib
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats

def plot_training_curves(train_losses, val_losses, model_name, save_dir):
    """绘制训练损失曲线"""
    plt.figure(figsize=(12, 8))

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 对数损失曲线
    ax2.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax2.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (Log Scale)')
    ax2.set_title(f'{model_name} - Loss (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 损失差异
    loss_diff = np.array(val_losses) - np.array(train_losses)
    ax3.plot(epochs, loss_diff, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation - Training Loss')
    ax3.set_title(f'{model_name} - Overfitting Monitor')
    ax3.grid(True, alpha=0.3)

    # 4. 平滑损失曲线
    window_size = max(1, len(train_losses) // 10)
    if len(train_losses) > window_size:
        train_smooth = pd.Series(train_losses).rolling(window=window_size).mean()
        val_smooth = pd.Series(val_losses).rolling(window=window_size).mean()
        ax4.plot(epochs, train_smooth, 'b-', label='Training (Smoothed)', linewidth=2)
        ax4.plot(epochs, val_smooth, 'r-', label='Validation (Smoothed)', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Smoothed Loss')
        ax4.set_title(f'{model_name} - Smoothed Loss Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{model_name}_training_curves.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_prediction_results(y_true, y_pred, model_name, target_name, save_dir):
    """绘制预测结果对比图"""
    plt.figure(figsize=(16, 12))

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 计算评估指标
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)

    # 1. 散点图 - 预测值 vs 真实值
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)

    # 添加理想线 y=x
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # 添加回归线
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(y_true, p(y_true), 'b-', alpha=0.8, linewidth=2, label=f'Regression Line')

    ax1.set_xlabel(f'True {target_name}')
    ax1.set_ylabel(f'Predicted {target_name}')
    ax1.set_title(f'{model_name} - Prediction vs Truth\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 残差图
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel(f'True {target_name}')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title(f'{model_name} - Residual Plot')
    ax2.grid(True, alpha=0.3)

    # 3. 误差分布直方图
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(residuals), color='g', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(residuals):.4f}')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{model_name} - Residual Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 时间序列预测对比（显示前100个样本）
    n_samples = min(100, len(y_true))
    indices = np.arange(n_samples)
    ax4.plot(indices, y_true[:n_samples], 'b-', linewidth=2, label='True Values', marker='o', markersize=4)
    ax4.plot(indices, y_pred[:n_samples], 'r-', linewidth=2, label='Predictions', marker='s', markersize=4)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel(target_name)
    ax4.set_title(f'{model_name} - Sample Predictions (First {n_samples} samples)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{model_name}_{target_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{model_name}_{target_name}_predictions.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_ppg_signal_examples(ppg_signals, predictions, targets, model_name, target_name, save_dir, n_examples=6):
    """绘制PPG信号波形和对应的预测结果"""
    plt.figure(figsize=(20, 12))

    n_examples = min(n_examples, len(ppg_signals))

    for i in range(n_examples):
        plt.subplot(3, 2, i + 1)

        # 绘制PPG信号
        time_axis = np.arange(len(ppg_signals[i])) / 125  # 假设采样率为125Hz
        plt.plot(time_axis, ppg_signals[i], 'b-', linewidth=1.5)

        plt.xlabel('Time (s)')
        plt.ylabel('PPG Amplitude')
        plt.title(f'Sample {i+1}: True {target_name}: {targets[i]:.2f}, '
                 f'Predicted: {predictions[i]:.2f}\nError: {abs(predictions[i] - targets[i]):.2f}')
        plt.grid(True, alpha=0.3)

        # 添加颜色编码误差
        error = abs(predictions[i] - targets[i])
        if error < 5:  # 好的预测
            plt.gca().set_facecolor('#e8f5e8')
        elif error < 10:  # 中等预测
            plt.gca().set_facecolor('#fff3e0')
        else:  # 较差的预测
            plt.gca().set_facecolor('#ffebee')

    plt.suptitle(f'{model_name} - PPG Signal Examples with {target_name} Predictions', fontsize=16)
    plt.tight_layout()

    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{model_name}_{target_name}_ppg_examples.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{model_name}_{target_name}_ppg_examples.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_model_comparison(results_dict, save_dir):
    """比较不同模型的性能"""
    plt.figure(figsize=(15, 10))

    # 提取数据
    models = []
    targets = []
    maes = []
    rmses = []

    for model_name, result in results_dict.items():
        if result is not None:
            models.append(model_name)
            if 'diasbp' in model_name or 'papagei_s' in model_name:
                targets.append('Diastolic BP (mmHg)')
            elif 'hr' in model_name or 'svri' in model_name:
                targets.append('Heart Rate (bpm)')
            elif 'sysbp' in model_name or 'papagei_p' in model_name:
                targets.append('Systolic BP (mmHg)')
            else:
                targets.append('Unknown')

            maes.append(result['mae'])
            rmses.append(result['rmse'])

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # MAE比较
    bars1 = ax1.bar(range(len(models)), maes, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Model Performance Comparison - MAE')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([f'{m}\n({t})' for m, t in zip(models, targets)], rotation=45, ha='right')

    # 添加数值标签
    for bar, mae in zip(bars1, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mae:.2f}', ha='center', va='bottom')

    # RMSE比较
    bars2 = ax2.bar(range(len(models)), rmses, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.set_title('Model Performance Comparison - RMSE')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([f'{m}\n({t})' for m, t in zip(models, targets)], rotation=45, ha='right')

    # 添加数值标签
    for bar, rmse in zip(bars2, rmses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{rmse:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/model_comparison.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def create_evaluation_report(results_dict, save_dir):
    """创建详细的评估报告"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建HTML报告
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPG Blood Pressure Estimation - Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { text-align: center; color: #333; }
            .results-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .results-table th, .results-table td { border: 1px solid #ddd; padding: 12px; text-align: center; }
            .results-table th { background-color: #f2f2f2; }
            .metric { margin: 10px 0; }
            .good { color: green; font-weight: bold; }
            .medium { color: orange; font-weight: bold; }
            .poor { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PPG Blood Pressure Estimation</h1>
            <h2>Model Evaluation Report</h2>
        </div>
        
        <table class="results-table">
            <tr>
                <th>Model</th>
                <th>Target</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>Performance</th>
            </tr>
    """

    for model_name, result in results_dict.items():
        if result is not None:
            if 'diasbp' in model_name or 'papagei_s' in model_name:
                target = 'Diastolic BP (mmHg)'
            elif 'hr' in model_name or 'svri' in model_name:
                target = 'Heart Rate (bpm)'
            elif 'sysbp' in model_name or 'papagei_p' in model_name:
                target = 'Systolic BP (mmHg)'
            else:
                target = 'Unknown'

            mae = result['mae']
            rmse = result['rmse']

            # 计算R²如果可用
            if 'actual' in result and 'predictions' in result:
                r2 = r2_score(result['actual'], result['predictions'])
            else:
                r2 = 'N/A'

            # 性能评级
            if 'hr' in model_name:  # 心率
                if mae < 5:
                    performance = '<span class="good">Excellent</span>'
                elif mae < 10:
                    performance = '<span class="medium">Good</span>'
                else:
                    performance = '<span class="poor">Needs Improvement</span>'
            else:  # 血压
                if mae < 8:
                    performance = '<span class="good">Excellent</span>'
                elif mae < 15:
                    performance = '<span class="medium">Good</span>'
                else:
                    performance = '<span class="poor">Needs Improvement</span>'

            html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{target}</td>
                <td>{mae:.4f}</td>
                <td>{rmse:.4f}</td>
                <td>{r2 if isinstance(r2, str) else f'{r2:.4f}'}</td>
                <td>{performance}</td>
            </tr>
            """

    html_content += """
        </table>
        
        <div class="metric">
            <h3>Performance Criteria:</h3>
            <ul>
                <li><strong>Heart Rate:</strong> MAE < 5 bpm (Excellent), 5-10 bpm (Good), > 10 bpm (Needs Improvement)</li>
                <li><strong>Blood Pressure:</strong> MAE < 8 mmHg (Excellent), 8-15 mmHg (Good), > 15 mmHg (Needs Improvement)</li>
            </ul>
        </div>
        
        <div class="metric">
            <h3>Notes:</h3>
            <ul>
                <li>MAE: Mean Absolute Error - Average absolute difference between predicted and actual values</li>
                <li>RMSE: Root Mean Square Error - Square root of average squared differences</li>
                <li>R²: Coefficient of determination - Proportion of variance explained by the model</li>
            </ul>
        </div>
    </body>
    </html>
    """

    # 保存HTML报告
    with open(f'{save_dir}/evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"详细评估报告已保存至: {save_dir}/evaluation_report.html")