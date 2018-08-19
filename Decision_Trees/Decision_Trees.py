'''
		wrote function for train
'''
from math import log
import numpy as np

class Node:
	def __init__(self):
		self.root=None
		self.left=None
		self.right=None
		

class decision_tree:

	def __init__(self):
		self.head=Node()
		self.max_depth=3
		self.root_entropy=1

	def train(self,x_train,y_train):
		self.head=self.build(self.head,x_train,y_train,self.root_entropy,1) 
	
	def build(self,current_node,x_train,y_train,entropy_parent,cur_depth):
		if(cur_depth>self.max_depth or len(x_train)==0):
			return Node()	
		row_length=len(x_train)
		col_length=len(x_train[0])
		count={}
		prob={}
		entropy={}
		information_gain={}
		weighted_avg={}
		output_set=set(y_train)
		for i in range(0,col_length):
			count[i]={}
			prob[i]={}
			entropy[i]={}
			s=set(x_train[:,i])
			for j in s:
				count[i][j]=0
				entropy[i][j]=0
				for k in output_set:
					count[i][j+"and"+k]=0
					prob[i][j+"and"+k]=0
		for i in range(0,col_length):
			for j in range(0,row_length):
				count[i][x_train[j][i]]=count[i][x_train[j][i]]+1
				count[i][x_train[j][i]+"and"+y_train[j]]=count[i][x_train[j][i]+"and"+y_train[j]]+1
		for i in range(0,col_length):
			s=set(x_train[:,i])
			weighted_avg[i]=0
			temp_sum=0
			for j in s:
				for k in output_set:
					prob[i][j+"and"+k]=count[i][j+"and"+k]/count[i][j]
					if(prob[i][j+"and"+k]):
						entropy[i][j]=entropy[i][j]+prob[i][j+"and"+k]*(log(prob[i][j+"and"+k])/log(2))
				if(entropy[i][j]):
					entropy[i][j]=entropy[i][j]*-1
				
				weighted_avg[i]=weighted_avg[i]+entropy[i][j]*count[i][j]
				temp_sum=temp_sum+count[i][j]
			weighted_avg[i]=weighted_avg[i]/temp_sum
			information_gain[i]=entropy_parent-weighted_avg[i]
		max_key=max(information_gain,key=information_gain.get)
		split_set=set(x_train[:,max_key])
		x_train_left=[]
		x_train_right=[]
		y_train_left=[]
		y_train_right=[]
		for k,i in enumerate(split_set):
			split_data=[]
			for j in range(0,row_length):
				if(x_train[j,max_key]==i):
					split_data.append(j)
			if(k==0):
				x_train_left=x_train[split_data]
				y_train_left=y_train[split_data]
			else:
				x_train_right=x_train[split_data]
				y_train_right=y_train[split_data]
		print(entropy)
		current_node.left=Node()
		current_node.right=Node()
		current_node.left=self.build(current_node.left,x_train_left,y_train_left,entropy_parent,cur_depth+1)
		current_node.right=self.build(current_node.right,x_train_right,y_train_right,entropy_parent,cur_depth+1)
		return current_node


x_train=[["Steep","Bumpy","Yes"],["Steep","Smooth","Yes"],["Flat","Bumpy","No"],["Steep","Smooth","No"]]
y_train=["Slow","Slow","Fast","Fast"]

x_train=np.array(x_train)
y_train=np.array(y_train)


clf=decision_tree()
clf.train(x_train,y_train)