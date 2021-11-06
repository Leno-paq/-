# -*- coding = utf-8 -*-
# @Time:2021/11/617:27
# @Author:李浩楠
# @File:SVM-2-1.py
# @Software:PyCharm
# 月亮数据集SVM二分类
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib as mpl
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# 为了显示中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.title("月亮数据",fontsize=20)
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
polynomial_svm_clf = Pipeline([
        # 将源数据 映射到 3阶多项式
        ("poly_features", PolynomialFeatures(degree=3)),
        # 标准化
        ("scaler", StandardScaler()),
        # SVC线性分类器
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])
polynomial_svm_clf.fit(X, y)
def plot_predictions(clf, axes):
    # 打表
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
#     print(y_pred)
#     print(y_decision)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

