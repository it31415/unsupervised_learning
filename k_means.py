import matplotlib.pyplot as plt 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# データセットの作成
X,Y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)        
#print(X)
#print(X.shape)
#print(Y)
#print(Y.shape)

# k-means法を行います。
km = KMeans(n_clusters=3, random_state=0)
Y_km = km.fit_predict(X) # Y_kmに各データ点が属するクラスタのラベルが入ります
#print(Y_km)
#print(Y_km.shape)
print(Y == Y_km)

# グラフの描画
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
# 元データをプロット
ax1.scatter(X[:, 0],X[:, 1],c="black")
ax1.grid()
# クラスタリング結果をプロット
ax2.scatter(X[Y_km==0, 0],X[Y_km==0, 1],c="r",s=40,label="cluster 1")
ax2.scatter(X[Y_km==1, 0],X[Y_km==1, 1],c="b",s=40,label="cluster 2")
ax2.scatter(X[Y_km==2, 0],X[Y_km==2, 1],c="g",s=40,label="cluster 3")
ax2.grid()
plt.savefig('k_means.png')
