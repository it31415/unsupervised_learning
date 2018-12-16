import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# サンプルデータの生成
X,Y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)     

distortions = []
for i  in range(1,11):                # クラスター数1~10を一気に計算 
    km = KMeans(n_clusters=i,
                init="k-means++",     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)                         # クラスタリングのを実行
    distortions.append(km.inertia_)   # km.fitするとkm.inertia_が得られる
    
# グラフのプロット
plt.plot(range(1,11),distortions,marker="o")
plt.xticks(np.arange(1, 11, 1))
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.savefig('elbow.png')
