import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 標準化前のデータを可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
ax1.set_title('before')
ax2.set_title('before')
ax1.scatter(X[:, 0], X[:, 1])
ax2.scatter(X[:, 5], X[:, 6])
plt.show()
print(X)

print("before")
print("mean: ", X.mean(axis=0), "\nstd: ", X.std(axis=0))

# Xに、Xを標準化したデータを代入してください
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 標準化後のデータを可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
ax1.set_title('after')
ax2.set_title('after')
ax1.scatter(X[:, 0], X[:, 1])
ax2.scatter(X[:, 5], X[:, 6])
plt.savefig('std.png')

print("after")
print("mean: ", X.mean(axis=0), "\nstd: ", X.std(axis=0))
