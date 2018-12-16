import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs

# Xには1つのプロットの(x,y)が、Yにはそのプロットの所属するクラスター番号が入る
X,Y = make_blobs(n_samples=150,       # サンプル点の総数
               n_features=2,          # 特徴量（次元数）の指定  default:2 
               centers=6,            # ここを変えてください # クラスタの個数 
               cluster_std=0.5,       # クラスタ内の標準偏差 
               shuffle=True,          # サンプルをシャッフル
               random_state=0)        # 乱数生成器の状態を指定

plt.scatter(X[:,0], X[:,1], c="black", marker="*", s=50)
plt.grid()
plt.savefig("blobs.png")
