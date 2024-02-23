# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 14:06:20 2022

@author: pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

path = "c:\\data\\data3.txt"
data = pd.read_csv(path, header=None, names=['Size','BedRooms', 'Price'])



#Rescaling Data (the data is too big )
data= (data -data.mean())/data.std()
data.insert(0,'ones',1)


#Show Graphs 
fig, ax = plt.subplots(figsize=(5,5))
data.plot(kind='scatter',x="Size",y="Price",figsize=(5,5))
ax.scatter(data.BedRooms, data.Price, label='Traning Data')
#fig = plt.figure(figsize=(4,4))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data.Size,data.BedRooms,data.Price) # plot the point (2,3,4) on the figure
#plt.show()
#####################"

cols=data.shape[1]

X=data.iloc[:,:cols-1]
y=data.iloc[:,cols-1:cols]


X=np.matrix(X.values)
y=np.matrix(y.values)


theta= np.matrix(np.array([0,0,0]))



def calculCost(X,y,theta):
    g=np.power(X*theta.T-y,2)
    return np.sum(g)/(2*len(X))

cost=calculCost(X, y, theta)
#print('Prime Cost :',cost)

def gradientDescent(X,y,theta,alpha,iters) :
    temp=np.zeros(theta.shape)
    parametrs=theta.shape[1]
    cost=np.zeros(iters)
    for i in range (iters) :
        error=(X*theta.T)-y
        
        for j in range (parametrs):
            term = np.multiply(error, X[:,j])
            theta[0,j]=theta[0,j]-np.sum(term)*(alpha/len(X))
        theta=temp
        cost[i]=calculCost(X, y, theta)
    return theta,cost

def normalEquation(X,y,theta):
    step1=np.dot(X.T,X)
    step2=np.linalg.pinv(step1)
    step3=np.dot(step2,X.T)
    step4=np.dot(step3,y)
    return step4

theta1 =normalEquation(X, y, theta)
theta,cost=gradientDescent(X, y, theta, 0.3, 1000)
print('theta Using Gradient Descent:\n',theta)
print('Cost Using Gradient Descent:\n',cost[999])

print('theta Using Normal Equation:\n',theta1)
print('Cost Using Normal Equation :\n',calculCost(X, y, theta1.T))

#Get best fit line
x1 = np.linspace(data.Size.min(), data.Size.max(), 100)
x2 = np.linspace(data.BedRooms.min(), data.BedRooms.max(), 100)

g = theta[0, 0] + (theta[0, 1] * x1)+ (theta[0, 2] * x2)
#g = theta.T[0, 0] + (theta.T[0, 1] * x1)+ (theta1.T[0, 2] * x2)




