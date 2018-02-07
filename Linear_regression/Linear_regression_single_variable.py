import numpy as np 
from sklearn import linear_model
from statistics import mean
import math

def predict(xtest,b0,b1):
	return b0+(xtest*b1)

def simplefindcoeflinreg(xtrain,ytrain):
	meanx=mean(xtrain)
	meany=mean(ytrain)
	print(meanx)
	print(meany)
	num=0
	den=0
	num=sum((xtrain-meanx)*(ytrain-meany))
	den=sum((xtrain-meanx)*(xtrain-meanx))
	b1=num/den
	b0=meany-(b1*meanx)
	y=b1*xtrain
	y=y+b0
	err=math.sqrt(sum(((y-ytrain)*(y-ytrain)))/len(xtrain))
	return b0,b1,err


xtrain=np.array([1,2,4,3,5])*1.0
ytrain=np.array([1,3,3,2,5])*1.0



b0,b1,err=simplefindcoeflinreg(xtrain,ytrain)
print("b0 = ",b0,"\nb1 = ",b1,"\nerror = ",err)

xtest=xtrain

print("\nprediction for test dataset is ",predict(xtest,b0,b1))

