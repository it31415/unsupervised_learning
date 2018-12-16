import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 相関行列（13x13）を作成
R = np.corrcoef(X.T)

# 固有値分解
eigvals, eigvecs = np.linalg.eigh(R)
#print(eigvecs)
#print(eigvecs.shape)

# 可視化
plt.bar(range(13), eigvals)
plt.title("distribution of eigvals")
plt.xlabel("index")
plt.ylabel("eigvals")
plt.savefig('eigen_value.png')

print(eigvals) # 消さないでください。実行結果の確認に使います。
