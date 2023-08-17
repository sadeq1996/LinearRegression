import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file=pd.read_csv("./data.csv")
# df= file.set_index("Date")

y_1= file.iloc[1500 :2000][["Code","Consumption"]]



#=================

a = np.ones(500).reshape(500, 1)
b = np.arange(1, 501).reshape(500, 1)

combined_array = np.hstack((a, b))


xhat = combined_array

yhat= np.array(y_1).reshape((500,2))





def b(X,Y):
    x1= np.dot(X.T,X)
    x2= np.linalg.inv(x1)
    x3= np.dot(x2,X.T)
    sol = np.dot(x3,Y)
    return sol



def linear_reg(xhat, yhat):
    b1= b(xhat, yhat)
    lr= lambda x: np.dot(b1.T, x)
    return lr


xnew= np.array([[1],[10]])

def plot_reg(xhat,yhat):
    point_x = []
    point_y= []
    for i in range(xhat.shape[0]):
        point_x.append(xhat[i][1])
        point_y.append(yhat[i][1])
    m1= np.min(xhat) # -1
    m2 = np.max(xhat) # +1
    t= np.arange(m1,m2,0.2)
    ls = []
    for time in t:
        ls.append((1,time))
    for i in range(len(t)):
        ls[i]= np.reshape(ls[i],(2,1))
    rg= []
    for l in ls:
        rg.append((linear_reg(xhat, yhat))(l)[1][0]) 
        
    plt.plot(t,rg, "r")
    plt.plot(point_x, point_y, "b*")
    return plt.show()

plot_reg(xhat,yhat)
