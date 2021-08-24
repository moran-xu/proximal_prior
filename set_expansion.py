#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 08:54:17 2021

@author: maoran
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import hamiltorch

def prox(beta, lam, A, b):
    """
    
    args: 
        beta: the continuous variable, a p-vector
        lam: the scalar param in proximal mapping 
        A: constriant matrix such that A@beta = b, a q*p matrix
        b: a q-vector
    returns:
        the projected variable, a p-vector
            
    """
    if len(A.shape) > 1:
        pbeta = beta - A.T @ torch.pinverse(A@A.T) @ (A@beta - b)
    else:
        pbeta = beta - 1 / torch.norm(A) ** 2 * A.T * (A@beta - b)
    dist = torch.norm((pbeta - beta))
    a = min(lam / dist, 1)
    return(a * pbeta + (1-a) * beta)

#####Example: let the constraint be theta1 + theta2 + theta3 = 1. Let beta~N(0,3) and lambda = 2.
torch.manual_seed(7)
p = 3
a1 = 1.
a2 = 1.
a3 = 1.
a4 = 1.
n = 100
A=torch.tensor([a1,a2,a3])
b = torch.tensor(a4)
lam = torch.tensor(2.)

from mpl_toolkits.mplot3d import Axes3D
plt3d = plt.figure().gca(projection='3d', azim = -88, elev = 21)
#plt3d.hold(True)

X,Y = np.meshgrid(np.linspace(-7,7,10),np.linspace(-7,7,10))
Z = (a4 - a1*X - a2*Y) / a3

plt3d.plot_surface(X, Y, Z, color = 'skyblue', alpha=.2)

y = torch.zeros([20,3])
for i in range(20):
    y[i] = torch.normal(torch.tensor([-.5,.3,1.2]), torch.tensor(3.))
plt3d.scatter(y[:,0], y[:,1], y[:,2], s=50 ,edgecolors= "black")

plt3d.scatter(-.5,.3,1.2, s=50, c = 'red' ,edgecolors= "black")
plt.show()
#########################+++++++++++++++++++++++++++++++++++++
def llh(beta):
    theta = prox(beta, lam, A, b)
    return(-(torch.norm(y-theta)**2/9+torch.norm(beta)**2/9.))

params = torch.tensor([-1.,-1.,-1.])
hamiltorch.set_random_seed(1)
params =hamiltorch.sample(log_prob_func=llh, params_init = params,  num_samples=3000, step_size=.1, num_steps_per_sample=100)

n=500
theta = torch.zeros([n, p])
beta_ = torch.zeros([n, p])
for i in range(n):
    theta[i] = prox(params[i+500], lam, A, b)
    beta_[i] = params[i+500]

dist = torch.zeros(n)
for i in range(n):
    pbeta = beta_[i] - 1 / torch.norm(A) ** 2 * A.T * (A@beta_[i] - b)
    dist[i] = torch.norm((pbeta - beta_[i])) 
idx = dist > 2 
idx1 = dist <=2   
    
plt3d = plt.figure().gca(projection='3d', azim = -88, elev = 21)
plt3d.scatter(theta[idx1,0], theta[idx1,1], theta[idx1,2],c = 'orange', s=50,edgecolors= "black")

plt3d.plot_surface(X, Y, Z, color = 'skyblue', alpha=.2)

plt3d.scatter(theta[idx,0], theta[idx,1], theta[idx,2] , s=40 ,edgecolors= "black")
plt.show()


x = np.linspace(-2,1.,10)
y = np.linspace(-1.,2, 10)
X,Y = np.meshgrid(x,y)
Z = (a4 - a1*X - a2*Y) / a3
plt3d.plot_surface(X, Y, Z, color = 'skyblue', alpha=.2)




#########Computing the prior prob
theta = torch.zeros([n, p])
beta_ = torch.zeros([n, p])
for i in range(len(beta_)):
    beta_[i,:] = torch.normal(torch.zeros(3), 3)
    theta[i,:] =  prox(beta_[i,:] , lam, A, b)

dist = torch.zeros(n)
for i in range(n):
    pbeta = beta_[i] - 1 / torch.norm(A) ** 2 * A.T * (A@beta_[i] - b)
    dist[i] = torch.norm((pbeta - beta_[i])) 
idx = dist > 2 
idx1 = dist <=2   