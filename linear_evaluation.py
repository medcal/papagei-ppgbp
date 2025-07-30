import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from utils import get_data_for_ml, regression_model
from visualization import plot_model_comparison, create_evaluation_report
import joblib
import os


def linear_evaluation():
    """执行线性评估"""
    download_dir = r"E:\thsiu-ppg\5459299\PPG-BP Database"
    save_dir = f"{download_dir}/features/"
    case_name = "subject_ID"

    # 检查特征目录是否存在
    if not os.path.exists(save_dir):
        print(f"Error: Features directory not found at {save_dir}")
        print("Please run the training modules first to extract features.")
        return

    # 加载数据分割
    try:
        df_train = pd.read_csv(f"{download_dir}/Data File/train.csv")
        df_val = pd.read_csv(f"{download_dir}/Data File/val.csv")
        df_test = pd.read_csv(f"{download_dir}/Data File/test.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please run preprocess_data first.")
        return

    df_train.loc[:, case_name] = df_train[case_name].apply(lambda x: str(x).zfill(4))
    df_val.loc[:, case_name] = df_val[case_name].apply(lambda x: str(x).zfill(4))
    df_test.loc[:, case_name] = df_test[case_name].apply(lambda x: str(x).zfill(4))

    def regression(save_dir, model_name, content, df_train, df_val, df_test, case_name, label):
        """执行回归评估"""
        print(f"\n评估 {model_name} 模型用于 {label} 预测...")

        # 检查特征文件是否存在
        train_feature_file = f"{save_dir}/{model_name}/dict_train_{content}.p"
        val_feature_file = f"{save_dir}/{model_name}/dict_val_{content}.p"
        test_feature_file = f"{save_dir}/{model_name}/dict_test_{content}.p"

        if not all(os.path.exists(f) for f in [train_feature_file, val_feature_file, test_feature_file]):
            print(f"Warning: Feature files not found for {model_name}. Skipping evaluation.")
            return None

        try:
            dict_train = joblib.load(train_feature_file)
            dict_val = joblib.load(val_feature_file)
            dict_test = joblib.load(test_feature_file)
        except Exception as e:
            print(f"Error loading feature files for {model_name}: {e}")
            return None

        # 准备数据
        try:
            X_train, y_train, _ = get_data_for_ml(df=df_train, dict_embeddings=dict_train, case_name=case_name,
                                                  label=label)
            X_val, y_val, _ = get_data_for_ml(df=df_val, dict_embeddings=dict_val, case_name=case_name, label=label)
            X_test, y_test, _ = get_data_for_ml(df=df_test, dict_embeddings=dict_test, case_name=case_name, label=label)
        except Exception as e:
            print(f"Error preparing data for {model_name}: {e}")
            return None

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Warning: No valid data found for {model_name}")
            return None

        # 合并验证集和测试集用于最终评估
        X_test = np.concatenate((X_test, X_val))
        y_test = np.concatenate((y_test, y_val))

        print(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
        print(f"特征维度: {X_train.shape[1]}")

        # 定义估计器和参数网格
        estimator = Ridge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'solver': ['auto', 'cholesky', 'sparse_cg']
        }

        # 执行回归
        try:
            results = regression_model(
                estimator=estimator,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )

            print(f"最佳参数: {results['best_params']}")
            print(f"MAE: {results['mae']:.4f}")
            print(f"RMSE: {results['rmse']:.4f}")

            return results
        except Exception as e:
            print(f"Error during regression for {model_name}: {e}")
            return None

    # 执行所有模型的回归评估
    results = {}

    # PaPaGei-S 用于舒张压预测
    results['papagei_s'] = regression(
        save_dir=save_dir,
        model_name='papagei_s',
        content="patient",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        case_name=case_name,
        label="diasbp"
    )

    # PaPaGei-S sVRI 用于心率预测
    results['papagei_s_svri'] = regression(
        save_dir=save_dir,
        model_name='papagei_s_svri',
        content="patient",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        case_name=case_name,
        label="hr"
    )

    # PaPaGei-P 用于收缩压预测
    results['papagei_p'] = regression(
        save_dir=save_dir,
        model_name='papagei_p',
        content="patient",
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        case_name=case_name,
        label="sysbp"
    )

    # 打印最终结果总结
    print("\n" + "=" * 60)
    print("最终评估结果总结:")
    print("=" * 60)

    if results['papagei_s']:
        print(f"PaPaGei-S (舒张压)     MAE: {results['papagei_s']['mae']:.4f} mmHg")
    else:
        print("PaPaGei-S (舒张压)     评估失败")

    if results['papagei_s_svri']:
        print(f"PaPaGei-S sVRI (心率)  MAE: {results['papagei_s_svri']['mae']:.4f} bpm")
    else:
        print("PaPaGei-S sVRI (心率)  评估失败")

    if results['papagei_p']:
        print(f"PaPaGei-P (收缩压)     MAE: {results['papagei_p']['mae']:.4f} mmHg")
    else:
        print("PaPaGei-P (收缩压)     评估失败")

    print("=" * 60)

    # 生成可视化对比图
    vis_save_dir = f"{download_dir}/visualizations/"
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)

    print("\n生成模型性能对比图...")
    # 重新组织结果用于对比图
    comparison_results = {}
    if results['papagei_s']:
        comparison_results['PaPaGei-S'] = results['papagei_s']
    if results['papagei_s_svri']:
        comparison_results['PaPaGei-S sVRI'] = results['papagei_s_svri']
    if results['papagei_p']:
        comparison_results['PaPaGei-P'] = results['papagei_p']

    if comparison_results:
        plot_model_comparison(comparison_results, vis_save_dir)
        print(f"模型对比图已保存至: {vis_save_dir}/model_comparison.png")

        # 创建详细评估报告
        print("创建详细评估报告...")
        create_evaluation_report(comparison_results, vis_save_dir)

    # 保存结果
    try:
        results_save_path = f"{download_dir}/evaluation_results.p"
        joblib.dump(results, results_save_path)
        print(f"评估结果已保存至: {results_save_path}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")

    print(f"\n所有可视化文件已保存至: {vis_save_dir}")
    print("  - 训练曲线图: PaPaGei-S_training_curves.png")
    print("  - 预测结果对比图: PaPaGei-S_Diastolic BP (mmHg)_predictions.png")
    print("  - PPG信号样例图: PaPaGei-S_Diastolic BP_ppg_examples.png")
    print("  - 模型性能对比图: model_comparison.png")
    print("  - 详细评估报告: evaluation_report.html")

    return results


if __name__ == "__main__":
    linear_evaluation()