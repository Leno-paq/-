# -*- coding = utf-8 -*-
# @Time:2021/11/616:53
# @Author:李浩楠
# @File:K-means1.py
# @Software:PyCharm
#鸢尾花数据集二分类
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:] ##表示我们只取特征空间中的后两个维度
estimator = KMeans(n_clusters=5)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
#plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
#plt.scatter(x3[:, 0], x3[:, 1], c = "yellow", marker='o', label='label3')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()