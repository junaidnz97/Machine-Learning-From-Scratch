import numpy as np 
import math
from collections import Counter


def KNN(xtrain,xtest):
	euclidian_distance=[]
	for i in range(0,len(xtest)):
		temp=[]
		for j in range(0,len(xtrain)):
			tempsum=0
			for k in range(0,len(xtrain[0])):
				tempsum=tempsum+((xtest[i][k]-xtrain[j][k])**2)
			
			temp.append(math.sqrt(tempsum))
		euclidian_distance.append(temp)
	
	return euclidian_distance			

def calculate_count(voting_array):
	i=Counter(voting_array).most_common()[0][0]
	return i

def KNN_predict(ytrain,euclidian_distance,xtest,k):
	
	sorted_file=[]
	final_vote=[]
	for i in range(0,len(xtest)):
		
		sorted_file.append(zip(ytrain,euclidian_distance[i]))
		sorted_file[i]=sorted(sorted_file[i],key= lambda x:x[1] )
		voting_array=[]
		
		for j in range(0,k):
			voting_array.append(sorted_file[i][j][0])
		
		final_vote.append(calculate_count(voting_array))
	return final_vote
	

xtrain=[[3.393533211, 2.331273381], [3.110073483, 1.781539638], [1.343808831, 3.368360954], [3.582294042, 4.67917911], [2.280362439, 2.866990263], [7.423436942, 4.696522875], [5.745051997, 3.533989803], [9.172168622, 2.511101045], [7.792783481, 3.424088941], [7.939820817, 0.791637231]]
ytrain=[0,0,0,0,0,1,1,1,1,1]
xtest=[[8.093607318,3.365731514],[8.093607318,3.365731514]]
k=3

euclidian_distance=KNN(xtrain,xtest)
print(KNN_predict(ytrain,euclidian_distance,xtest,k))
