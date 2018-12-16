import pandas as pd
import numpy as np

df_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 相関行列（13x13）を作成
R = np.corrcoef(X.T)

# 対角成分を0にしてください
_R = R - np.identity(13)
#print(_R)
# 最大相関係数をとるインデックスを1つだけ取得してください
index = np.where(_R == _R.max())[0]

print(R[index[0], index[1]])
print(index)
