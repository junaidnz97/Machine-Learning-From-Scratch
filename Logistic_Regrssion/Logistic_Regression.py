import numpy as np 
import math


def logisticreg(xtrain,ytrain,n_epochs):
	featurenum=len(xtrain[0])
	coef=[]
	
	alpha=0.3 

	for i in range(0,featurenum+1):
		coef.append(0.0)

	
	for k in range(0,n_epochs):
		for i in range(0,len(xtrain)):
			output=coef[0]
			for j in range(0,featurenum):
				output=output+xtrain[i][j]*coef[j+1]
			
			prediction=1/(1+math.exp(output*-1))
		
			coef[0]=coef[0]+alpha*(ytrain[i]-prediction)*prediction*(1-prediction)
			for j in range(0,featurenum):
				coef[j+1]=coef[j+1]+alpha*(ytrain[i]-prediction)*prediction*(1-prediction)*xtrain[i][j]
		

	coef=np.array(coef)
	

	return coef

def makepredictions(xtest,coef):
	predictions=[]
	featurenum=len(xtest[0])
	for i in range(0,len(xtrain)):
		output=coef[0]

		for j in range(1,featurenum+1):
			output=output+xtrain[i][j-1]*coef[j]
			

		prediction=1/(1+math.exp(output*-1))
		predictions.append(prediction)

	predictions=np.array(predictions)
	
	for i in range(0,len(predictions)):
		if (predictions[i]<0.5):
			predictions[i]=0
		else:
			predictions[i]=1

	predictions= predictions.astype(int)
	return predictions


feature1=[2.7810836,1.465489372,3.396561688,1.38807019,3.06407232,7.627531214,5.332441248,6.922596716,8.675418651,7.673756466]
feature2=[2.550537003,2.362125076,4.400293529,1.850220317,3.005305973,2.759262235,2.088626775,1.77106367,-0.242068655,3.508563011]


xtrain=[]

for i in range(0,len(feature1)):
	temp=[]
	temp.append(feature1[i])
	temp.append(feature2[i])
	xtrain.append(temp)

xtrain=np.array(xtrain)
ytrain=[0,0,0,0,0,1,1,1,1,1]

coef=logisticreg(xtrain,ytrain,10)
print("coefficients are ",coef)

xtest=xtrain 												
predictions=makepredictions(xtest,coef)

print("Final predictions are ",predictions)