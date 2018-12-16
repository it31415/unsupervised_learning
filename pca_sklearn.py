import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# PCAをインポート
#---------------------------
from sklearn.decomposition import PCA
#---------------------------

df_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 主成分分析のインスタンスを生成。主成分数は2としてください。
pca = PCA(n_components=2)

# データから変換モデルを学習し、変換する。
X_pca = pca.fit_transform(X)

# 可視化
color = ["r","b","g"]
marker = ["s","x","o"]
for label, color, marker in zip(np.unique(y), color, marker):
    plt.scatter(X_pca[y==label, 0], X_pca[y==label, 1],c=color, marker=marker, label=label) 
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.savefig('pca_sklearn.png')

print(X_pca) # 消さないでください。実行結果の確認に使います。
