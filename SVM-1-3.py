# -*- coding = utf-8 -*-
# @Time:2021/11/617:08
# @Author:李浩楠
# @File:SVM-1-3.py
# @Software:PyCharm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import datasets
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
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
def RBFKernelSVC(gamma=1.0):
    return Pipeline([
        ('std_scaler',StandardScaler()),
        ('svc',SVC(kernel='rbf',gamma=gamma))
    ])
warnings.filterwarnings("ignore")
X,y = datasets.make_moons(noise=0.15,random_state=666)
svc = RBFKernelSVC(gamma=100)
svc.fit(X,y)
plot_decision_boundary(svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X[y==0,0],X[y==0,1],c='red')
plt.scatter(X[y==1,0],X[y==1,1],c='blue')
plt.show()
