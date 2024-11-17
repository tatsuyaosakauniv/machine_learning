import numpy as np

# データ生成 --------------------------------
np.random.seed(seed=1) # 乱数生成
N = 200
K = 3
T = np.zeros((N, 3), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3]  # X0 の範囲, 表示用
X_range1 = [-3, 3]  # X1 の範囲, 表示用
Mu = np.array([[-.5, -.5], [.5, 1.0], ])