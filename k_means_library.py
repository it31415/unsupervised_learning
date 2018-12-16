import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# サンプルデータの生成
X,Y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)        

# KMeansクラスからkmインスタンスを作成
km = KMeans(n_clusters=3,            # クラスターの個数 # 変更してみてください
            init="random",           # セントロイドの初期値をランダムに設定  default: "k-means++"
            n_init=10,               # 異なるセントロイドの初期値を用いたk-meansの実行回数
            max_iter=300,            # k-meansアルゴリズムを繰り返す最大回数
            tol=1e-04,               # 収束と判定するための相対的な許容誤差
            random_state=0)          # 乱数発生初期化

# fit_predictメソッドによりクラスタリングを行う
Y_km = km.fit_predict(X)

# SSE値を出力
print("Distortion: %.2f"% km.inertia_)

# クラスター番号(Y_km)に応じてデータをプロット
for n in range(np.max(Y_km)+1):
    plt.scatter(X[Y_km==n,0], X[Y_km==n,1], s=50, c = cm.hsv(float(n) / 10), marker="*", label="cluster"+str(n+1))
    
# セントロイドをプロット、km.cluster_centers_には各クラスターのセントロイドの座標が入っています
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker="*", c="black", label="centroids")

plt.legend(bbox_to_anchor=(1.05, 0.7),loc="upper left")
plt.grid()
plt.savefig('k_means_library.png')
