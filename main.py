from install_dependencies import install_dependencies
from preprocess_data import preprocess_data
from train_papagei_s import train_papagei_s
from train_papagei_s_svri import train_papagei_s_svri
from train_papagei_p import train_papagei_p
from linear_evaluation import linear_evaluation
import os


def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║      PPG Blood Pressure Estimation Pipeline         ║
    ║      基于深度学习的PPG血压估计系统                    ║
    ║                                                      ║
    ║      Models: PaPaGei-S, PaPaGei-S sVRI, PaPaGei-P   ║
    ║      Features: Real-time monitoring with WandB      ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def show_menu():
    """显示菜单选项"""
    print("\n" + "=" * 60)
    print("请选择要执行的操作：")
    print("=" * 60)
    print("1. 完整流程 - 从头开始运行所有步骤")
    print("2. 仅安装依赖包")
    print("3. 仅数据预处理")
    print("4. 仅训练 PaPaGei-S 模型 (舒张压)")
    print("5. 仅训练 PaPaGei-S sVRI 模型 (心率)")
    print("6. 仅训练 PaPaGei-P 模型 (收缩压)")
    print("7. 仅线性评估")
    print("8. 查看现有结果")
    print("9. 设置 WandB (实验跟踪)")
    print("0. 退出")
    print("=" * 60)


def check_prerequisites():
    """检查先决条件"""
    download_dir = r"E:\thsiu-ppg\5459299\PPG-BP Database"

    issues = []

    # 检查数据目录
    if not os.path.exists(download_dir):
        issues.append(f"数据目录不存在: {download_dir}")

    # 检查数据集文件
    dataset_file = f"{download_dir}/Data File/PPG-BP dataset.xlsx"
    if not os.path.exists(dataset_file):
        issues.append(f"数据集文件不存在: {dataset_file}")

    # 检查信号目录
    signal_dir = f"{download_dir}/Data File/0_subject/"
    if not os.path.exists(signal_dir):
        issues.append(f"信号目录不存在: {signal_dir}")

    if issues:
        print("\n⚠️  发现以下问题：")
        for issue in issues:
            print(f"   - {issue}")
        print("\n请确保数据文件存在后再运行程序。")
        return False

    return True


def show_results():
    """显示现有结果"""
    download_dir = r"E:\thsiu-ppg\5459299\PPG-BP Database"
    results_file = f"{download_dir}/evaluation_results.p"
    vis_dir = f"{download_dir}/visualizations/"

    print("\n" + "=" * 60)
    print("现有结果检查")
    print("=" * 60)

    if os.path.exists(results_file):
        import joblib
        try:
            results = joblib.load(results_file)
            print("✓ 找到评估结果文件")
            print("\n最新评估结果：")
            print("-" * 40)

            if results.get('papagei_s'):
                mae = results['papagei_s']['mae']
                print(f"PaPaGei-S (舒张压)     MAE: {mae:.4f} mmHg")

            if results.get('papagei_s_svri'):
                mae = results['papagei_s_svri']['mae']
                print(f"PaPaGei-S sVRI (心率)  MAE: {mae:.4f} bpm")

            if results.get('papagei_p'):
                mae = results['papagei_p']['mae']
                print(f"PaPaGei-P (收缩压)     MAE: {mae:.4f} mmHg")

        except Exception as e:
            print(f"✗ 无法读取结果文件: {e}")
    else:
        print("✗ 未找到评估结果文件")

    # 检查可视化文件
    if os.path.exists(vis_dir):
        vis_files = os.listdir(vis_dir)
        if vis_files:
            print(f"\n✓ 找到 {len(vis_files)} 个可视化文件:")
            for file in sorted(vis_files):
                print(f"   - {file}")
            print(f"\n可视化文件位置: {vis_dir}")
        else:
            print("\n✗ 可视化目录为空")
    else:
        print("\n✗ 未找到可视化目录")


def setup_wandb_wrapper():
    """WandB设置包装器"""
    try:
        from setup_wandb import setup_wandb
        setup_wandb()
    except ImportError:
        print("Error: setup_wandb.py 文件不存在")
    except Exception as e:
        print(f"Error: {e}")


def main_train():
    """主训练函数"""
    print_banner()

    while True:
        show_menu()
        choice = input("\n请输入选项 (0-9): ").strip()

        if choice == '0':
            print("退出程序")
            break

        elif choice == '1':
            # 完整流程
            print("\n🚀 开始执行完整流程...")
            if not check_prerequisites():
                continue

            try:
                print("\n1/6 安装依赖包...")
                install_dependencies()
                print("✓ 依赖包安装完成")

                print("\n2/6 数据预处理...")
                preprocess_data()
                print("✓ 数据预处理完成")

                print("\n3/6 训练 PaPaGei-S 模型...")
                train_papagei_s()
                print("✓ PaPaGei-S 训练完成")

                print("\n4/6 训练 PaPaGei-S sVRI 模型...")
                train_papagei_s_svri()
                print("✓ PaPaGei-S sVRI 训练完成")

                print("\n5/6 训练 PaPaGei-P 模型...")
                train_papagei_p()
                print("✓ PaPaGei-P 训练完成")

                print("\n6/6 线性评估...")
                linear_evaluation()
                print("✓ 线性评估完成")

                print("\n🎉 完整流程执行成功！")

            except Exception as e:
                print(f"\n❌ 执行过程中出现错误: {e}")

        elif choice == '2':
            print("\n📦 安装依赖包...")
            try:
                install_dependencies()
                print("✓ 依赖包安装完成")
            except Exception as e:
                print(f"❌ 安装失败: {e}")

        elif choice == '3':
            print("\n🔄 数据预处理...")
            if not check_prerequisites():
                continue
            try:
                preprocess_data()
                print("✓ 数据预处理完成")
            except Exception as e:
                print(f"❌ 预处理失败: {e}")

        elif choice == '4':
            print("\n🧠 训练 PaPaGei-S 模型...")
            try:
                train_papagei_s()
                print("✓ PaPaGei-S 训练完成")
            except Exception as e:
                print(f"❌ 训练失败: {e}")

        elif choice == '5':
            print("\n🧠 训练 PaPaGei-S sVRI 模型...")
            try:
                train_papagei_s_svri()
                print("✓ PaPaGei-S sVRI 训练完成")
            except Exception as e:
                print(f"❌ 训练失败: {e}")

        elif choice == '6':
            print("\n🧠 训练 PaPaGei-P 模型...")
            try:
                train_papagei_p()
                print("✓ PaPaGei-P 训练完成")
            except Exception as e:
                print(f"❌ 训练失败: {e}")

        elif choice == '7':
            print("\n📊 线性评估...")
            try:
                linear_evaluation()
                print("✓ 线性评估完成")
            except Exception as e:
                print(f"❌ 评估失败: {e}")

        elif choice == '8':
            show_results()

        elif choice == '9':
            print("\n⚙️  设置 WandB...")
            setup_wandb_wrapper()

        else:
            print("❌ 无效选项，请输入 0-9 之间的数字")

        input("\n按回车键继续...")


if __name__ == "__main__":
    main_train()