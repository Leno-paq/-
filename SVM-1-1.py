# -*- coding = utf-8 -*-
# @Time:2021/11/617:05
# @Author:李浩楠
# @File:SVM-1-1.py
# @Software:PyCharm
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from matplotlib.colors import ListedColormap
import warnings
def plot_decision_boundary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_predict=model.predict(x_new)
    zz=y_predict.reshape(x0.shape)
    custom_cmap=ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0,x1,zz,linewidth=5,cmap=custom_cmap)
    w = model.coef_[0]
    b = model.intercept_[0]
    plot_x = np.linspace(axis[0],axis[1],200)
    up_y = -w[0]/w[1]*plot_x - b/w[1] + 1/w[1]
    down_y = -w[0]/w[1]*plot_x - b/w[1] - 1/w[1]
    up_index = (up_y>=axis[2]) & (up_y<=axis[3])
    down_index = (down_y>=axis[2]) & (down_y<=axis[3])
    plt.plot(plot_x[up_index],up_y[up_index],c='black')
    plt.plot(plot_x[down_index],down_y[down_index],c='black')
warnings.filterwarnings("ignore")
data = load_iris()
x = data.data
y = data.target
x = x[y<2,:2]
y = y[y<2]
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
svc = LinearSVC(C=1e9)
svc.fit(x,y)
plot_decision_boundary(svc,axis=[-3,3,-3,3])
plt.scatter(x[y==0,0],x[y==0,1],c='r')
plt.scatter(x[y==1,0],x[y==1,1],c='b')
plt.show()
