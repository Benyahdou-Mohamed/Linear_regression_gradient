import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

path="C:\\data\\LinearRegression.csv"
data=pd.read_csv(path,header=1,names=['X','y'])
df = pd.DataFrame(data)
data = df.dropna()
print(data)

data.plot(kind="scatter",x='X',y='y',figsize=(5,5))

data.insert(0,"ones",1)
cols=data.shape[1]
rows=data.shape[0]
X=data.iloc[:rows-2,:cols-1]
y=data.iloc[:rows-2,cols-1:cols]

print("X:",X)
print("y:",y)

X = np.matrix(X,dtype=np.float64)
y = np.matrix(y,dtype=np.float64)

theta=np.matrix([1,1])
print("theta:",theta)

def calculCost(X,y, theta):
    error=np.power(X*theta.T-y,2)
    return np.sum(error)/(2*len(X))



def normalEquation(X,y):
    step1=np.dot(X.T,X)
    step2 = np.linalg.pinv(step1)
    step3=np.dot(step2,X.T)
    step4=np.dot(step3,y)
    return step4

theta=normalEquation(X, y)
cost=calculCost(X, y, theta.T)
print("theta of normal Equation",theta)
print("cost of normal Equation",calculCost(X, y, theta.T))
print(theta.shape)
# get best fit line
x=np.linspace(data.X.min(),data.X.max(),100)

f=theta[0,0]+(theta[1,0]*x)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction with gradient ')

ax.scatter(data.X, data.y, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('weight vs. height Size')



        