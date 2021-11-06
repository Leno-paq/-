# -*- coding = utf-8 -*-
# @Time:2021/11/616:32
# @Author:李浩楠
# @File:LDA1.py
# @Software:PyCharm

# 鸢尾花数据集LDA线性二分类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification

class LDA():
    def Train(self, X, y):
        """X为训练数据集，y为训练label"""
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

        # 求中心点
        mju1 = np.mean(X1, axis=0)  # mju1是ndrray类型
        mju2 = np.mean(X2, axis=0)

        # dot(a, b, out=None) 计算矩阵乘法
        cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
        cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
        Sw = cov1 + cov2
        # 计算w
        w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1), 1)))
        # 记录训练结果
        self.mju1 = mju1  # 第1类的分类中心
        self.cov1 = cov1
        self.mju2 = mju2  # 第2类的分类中心
        self.cov2 = cov2
        self.Sw = Sw  # 类内散度矩阵
        self.w = w  # 判别权重矩阵
    def Test(self, X, y):
        """X为测试数据集，y为测试label"""
        # 分类结果
        y_new = np.dot((X), self.w)
        # 计算fisher线性判别式
        nums = len(y)
        c1 = np.dot((self.mju1 - self.mju2).reshape(1, (len(self.mju1))), np.mat(self.Sw).I)
        c2 = np.dot(c1, (self.mju1 + self.mju2).reshape((len(self.mju1), 1)))
        c = 1/2 * c2  # 2个分类的中心
        h = y_new - c
        # 判别
        y_hat = []
        for i in range(nums):
            if h[i] >= 0:
                y_hat.append(0)
            else:
                y_hat.append(1)
        # 计算分类精度
        count = 0
        for i in range(nums):
            if y_hat[i] == y[i]:
                count += 1
        precise = count / nums
        # 显示信息
        print("测试样本数量:", nums)
        print("预测正确样本的数量:", count)
        print("测试准确度:", precise)
        return precise
if '__main__' == __name__:
    # 产生分类数据
    n_samples = 500
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_classes=2,n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    # LDA线性判别分析(二分类)
    lda = LDA()
    # 60% 用作训练，40%用作测试
    Xtrain = X[:299, :]
    Ytrain = y[:299]
    Xtest = X[300:, :]
    Ytest = y[300:]
    lda.Train(Xtrain, Ytrain)
    precise = lda.Test(Xtest, Ytest)
    # 原始数据
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Test precise:" + str(precise))
    plt.show()
