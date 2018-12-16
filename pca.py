import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 標準化
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 相関行列の取得
R = np.corrcoef(X.T)

# 固有値分解
eigvals, eigvecs = np.linalg.eigh(R)

# 変換行列の取得
W = np.c_[eigvecs[:,-1], eigvecs[:,-2]]

# 特徴変換
X_pca = X.dot(W)

# 可視化
color = ["r","b","g"]
marker = ["s","x","o"]
for label, color, marker in zip(np.unique(y), color, marker):
    plt.scatter(X_pca[y==label, 0], X_pca[y==label, 1],c=color, marker=marker, label=label) 
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.savefig('pca.png')

print(X_pca) # 消さないでください。実行結果の確認に使います。
print(X_pca.shape)
