import subprocess
import sys
import os


def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        return False


def install_dependencies():
    """安装所有必需的依赖包"""
    packages = [
        "dotmap",
        "vitaldb",
        "pyPPG==1.0.41",
        "openpyxl",
        "torch_ecg",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
        "joblib",
        "scipy",
        "wandb",
        "seaborn"
    ]

    print("Installing required packages...")
    failed_packages = []

    for pkg in packages:
        print(f"Installing {pkg}...")
        if not install_package(pkg):
            failed_packages.append(pkg)

    # 尝试升级vitaldb
    print("Upgrading vitaldb...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "vitaldb"])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade vitaldb: {e}")

    if failed_packages:
        print(f"\nWarning: Failed to install the following packages: {failed_packages}")
        print("Please install them manually if needed.")
    else:
        print("\nAll packages installed successfully!")


if __name__ == "__main__":
    install_dependencies()