# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:37:38 2022

@author: pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


path = "c:\\data\\train.csv"
path_test = "c:\\data\\test.csv"
data = pd.read_csv(path, header=0, names=['Size', 'Price'])
data_test = pd.read_csv(path_test, header=0, names=['Size', 'Price'])
print("data test :",data_test)
print("data shape before:",data.shape)
#Remove any row that have value null in it 
df = pd.DataFrame(data)
data = df.dropna()

print(data_test)

#################################
def data_handle(data):
    data.insert(0,'ones',1)
    cols=data.shape[1]
    X=data.iloc[:,:cols-1]
    y=data.iloc[:,cols-1:cols]
    X = np.matrix(X)
    y = np.matrix(y)
    
    return X,y

################
X,y=data_handle(data)
theta=np.matrix(np.array([1,1]))


def calculCost(X,y, theta):
    error=np.power(X*theta.T-y,2)
    #print(error)
    return np.sum(error)/(2*len(X))

prime_cost=calculCost(X, y, theta)
print("prime_cost",prime_cost)

def gradientDescent(X,y,theta,alpha, iters):
    
    temp=np.zeros(theta.shape)
    paremetrs=int(X.shape[1])
    cost=np.zeros(iters)
    
    for i in range (iters):
        error=X*theta.T-y
        for j in range (paremetrs):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta=temp
        cost[i]=calculCost(X, y, theta) 
    return theta,cost
           
def normalEquation(X,y):
    step1=np.dot(X.T,X)
    step2 = np.linalg.pinv(step1)
    step3=np.dot(step2,X.T)
    step4=np.dot(step3,y)
    return step4

theta1=normalEquation(X, y)
theta,cost=gradientDescent(X, y, theta, 0.0003, 1000)
print("theta of gradient descent",theta)
print("cost of gradient descent",cost[999])

print("theta of normal Equation",theta1)
print("cost of normal Equation",calculCost(X, y, theta1.T))

# get best fit line
x=np.linspace(data.Size.min(),data.Size.max(),100)
f=theta[0,0]+(theta[0,1]*x)
f2=theta1[0,0]+(theta1[1,0]*x)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction with gradient ')
ax.plot(x, f2, 'g', label='Prediction with normal equation')
ax.scatter(data.Size, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('weight vs. height Size')


fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(1000), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

def calculDefrence(X,y, theta):
    error=X*theta.T-y
    #print(error)
    return error





X_test,y_test=data_handle(data_test)

prediction=theta[0,0]+(theta[0,1]*X_test[:,1])
defrence_test= calculDefrence(X_test, y_test, theta)

print("defrence test",defrence_test[:10])
print("prediction",prediction[:10])
data_test.insert(3,'prediction',prediction)
data_test.insert(4,'defrence',defrence_test)
print(data_test)
 

