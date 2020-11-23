import numpy as np
from numpy import *
def loadfile(filename):
    fp=open(filename)
    Xmat=[]#特征向量
    Ymat=[]#标签
    for lists in fp.readlines():
        list=lists.strip().split('\t')
        Xmat.append(list[0:-1])
        Ymat.append(list[-1])
    return mat(Xmat).astype(np.float).T,mat(Ymat).astype(np.int).T#将字符串转为浮点数和整型
#归一化输入
def guiyihua(filename):
    Xmat,Ymat=loadfile(filename)
    ux_mean=np.sum(Xmat,axis=1)/Xmat.shape[1]
    Xmat=Xmat-ux_mean
    Sx=np.sum(np.multiply(Xmat,Xmat),axis=1)/Xmat.shape[1]
    Xmat=Xmat/np.sqrt(Sx)
    return Xmat,Ymat


def sigmoid(z):
    return 1/(1+exp(-z))

def tanh(z):
    return (exp(z)-exp(-z))/(exp(z)+exp(-z))

def Relu(z):
    return np.max(0,z)

def train(Xmat,Ymat):
    #Xmat n*m，Ymat m*1
    n=Xmat.shape[0]#特征数
    m=Xmat.shape[1]#样本数
    learnrate=0.1
    W1=0.01*np.random.randn(4,n)*np.sqrt(1/n)#4*n,乘以0.01可以防止激活函数溢出
    b1=zeros((4,1))#4*1
    W2 = 0.01*np.random.randn(1, 4)*np.sqrt(1/4)  # 1*4,乘以np.sqrt(1/4)缓解梯度消失和梯度爆炸
    b2 = zeros((1, 1))  # 1*1
    for i in range(10000):
        z1=W1*Xmat+b1#4*m
        A1=tanh(z1)#4*m
        d1=np.random.rand(A1.shape[0],A1.shape[1])<0.8#dropout正则化
        A1=np.multiply(A1,d1)/0.8
        z2 = W2 *A1+b2#1*m
        A2=sigmoid(z2)#1*m
        dz2=A2-Ymat.T#1*m
        dw2=1/m*dz2*A1.T#1*4 L2正则化
        db2=1/m*np.sum(dz2,axis=1)#1*1
        dz1=np.multiply(W2.T*dz2,(1-tanh(z1)))#4*m 矩阵对象进行数量积需要用multiply
        dw1=1/m*dz1*Xmat.T#4*n L2正则化
        db1=1/m*np.sum(dz1,axis=1)#4*1
        W1=W1-learnrate*dw1
        b1=b1-learnrate*db1
        W2=W2-learnrate*dw2
        b2=b2-learnrate*db2
    return W1,b1,W2,b2,A1
Xmat,Ymat=loadfile('testSet.txt')
W1,b1,W2,b2,A1=train(Xmat,Ymat)
print(W1)


