
"""
依赖测试脚本
"""
try:
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from tqdm import tqdm
    import joblib
    import matplotlib.pyplot as plt
    
    print("✓ 所有关键依赖导入成功！")
    print(f"  - pandas: {pd.__version__}")
    print(f"  - numpy: {np.__version__}")
    print(f"  - torch: {torch.__version__}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    
except ImportError as e:
    print(f"✗ 导入失败: {e}")
