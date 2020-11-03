import train
from numpy import *
import numpy as np
Xmat,Ymat=train.loadfile('testSet.txt')
w,b=train.train(Xmat,Ymat)
print('特征系数:',w)
print('偏置:',b)

z=train.sigmoid(w.T*Xmat.T+b)
for i in range(shape(z)[1]):
    if z[0,i]>0.5:
        z[0,i]=1
    else:
        z[0,i]=0
count=0
for i in range(shape(z)[1]):
    if z[0,i]==Ymat[0,i]:
        count=count+1

precision=count/shape(z)[1]*100#精度
print('精度:%f%%'% precision)