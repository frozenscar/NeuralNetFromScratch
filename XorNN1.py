#!/usr/bin/env python
# coding: utf-8

# In[332]:


#nn for XOR gate

import numpy as np
# X training ip, Y training truth output
X =[[1,1],[1,0],[0,1],[0,0]]
Y= [[0],[1],[1],[0]]

m= len(X)

xlen = 2
hlneurons = 3
ylen = 1

lrate = 0.2

theta1 = np.random.rand(hlneurons,xlen+1)
theta2 = np.random.rand(ylen,hlneurons+1)

print(theta1,theta2)

#add column of 1's (bias)
X = np.c_[np.ones((np.size(X,0),1)),X]
print(X)


# In[333]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


def FeedForward(X,Y):
    z2=np.matmul(X,np.transpose(theta1))
    a2 = sigmoid(z2)
    
    a2 = np.c_[np.ones((np.size(a2,0),1)),a2]
    z3 =np.matmul(a2,np.transpose(theta2)) 
    a3 = sigmoid(z3)
    return a3,a2


# In[334]:


hx,a2 = FeedForward(X,Y)

def cost(hx,Y):

    diff = hx-Y
    return sum(diff*diff)
print(cost(hx,Y))
print(np.size(theta2))


# In[335]:


def grad(theta1,theta2,m):
    theta1Grad=0
    theta2Grad=0
    for i in range(m):
        x=[X[i]]
        y=[Y[i]]
        
        hx,a2 = FeedForward(x,y)
        theta2Grad = theta2Grad + 2*(hx-y)*a2
        a2t = a2[:,1:]
        theta1Grad = theta1Grad +2*(hx-y)*np.matmul(np.transpose(a2t*(1-a2t)),x)
    return theta1Grad,theta2Grad


# In[337]:



for i in range(1000):

    dtheta1,dtheta2 = grad(theta1,theta2,m)
    print(cost(hx,Y))
    theta1= theta1- lrate*dtheta1
    theta2= theta2- lrate*dtheta2

    hx,a2 = FeedForward(X,Y)
    if(i%100 == 0):
        print(cost(hx,Y))

print(FeedForward([[1,0,1]],[[0]]))



# In[ ]:





# In[ ]:





# In[ ]:




