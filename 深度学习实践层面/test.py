import train
from numpy import *
Xmat,Ymat=train.guiyihua('testSet.txt')
W1,b1,W2,b2=train.train(Xmat,Ymat)
z1=W1*Xmat+b1#4*m
A1=train.tanh(z1)#4*m
z2 = W2 *A1+b2#1*m
A2=train.sigmoid(z2)#1*m
for i in range(shape(A2)[1]):
    if A2[0,i]>0.5:
        A2[0,i]=1
    else:
        A2[0,i]=0
count=0
for i in range(shape(A2)[1]):
    if A2[0,i]==Ymat[i,0]:
        count=count+1

precision=count/shape(A2)[1]*100#精度
print('精度:%f%%'% precision)