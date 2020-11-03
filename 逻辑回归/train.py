from numpy import *
import numpy as np
def loadfile(filename):
    fp=open(filename)
    Xmat=[]#特征向量
    Ymat=[]#标签
    for lists in fp.readlines():
        list=lists.strip().split('\t')
        Xmat.append(list[0:-1])
        Ymat.append(list[-1])
    return mat(Xmat).astype(np.float),mat(Ymat).astype(np.int)#将字符串转为浮点数和整型
#sigmoid函数
def sigmoid(z):
    return 1/(1+exp(-z))
#训练
def train(Xmat,Ymat):
    m=shape(Xmat)[0]#样本有多少个
    Xmat=Xmat.T#列向量
    Ymat=Ymat.T#列向量
    w=np.ones((shape(Xmat)[0],1))#x向量系数初始化
    b=np.zeros((1,1))#偏置初始化
    learn=0.1#学习率
    #梯度下降法
    for i in range(1000):#迭代1000次
        z=w.T*Xmat+b#1*n
        A=sigmoid(z)#1*n 预测值
        dz=A-Ymat.T#1*n，dJ/dz
        dw=1/m*Xmat*dz.T#2*1 dJ/dw
        db=1/m*np.sum(dz)#1*1 dJ/db
        w=w-learn*dw#进行更新
        b=b-learn*db#更新
    return w,b