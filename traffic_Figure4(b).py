import hamiltorch
import statsmodels.api as sm
import pandas as pd
from igraph import Graph
import numpy as np
import matplotlib.pyplot as plt
import torch
from numba import jit, float32, int32
from torch.optim.lr_scheduler import LambdaLR

hamiltorch.set_random_seed(0)
device = torch.device('cpu')
edges = pd.read_csv ("edges.csv")
nodes = pd.read_csv ("nodes.csv")

V = nodes['node_id']
E = edges['edge_id'] 

t = 25

V1 = edges['fromnode_id']
V2 = edges['tonode_id']

p=V.shape[0]
A = np.zeros([t, p, p])
for dt in range(t):
    for i in range(V1.shape[0]):
        A[dt, V1[i] - 1, V2[i] - 1] +=  edges[edges.columns[5 + dt]][i]
       # A[dt, V2[i] - 1, V1[i] - 1] +=  - edges[edges.columns[5 + dt]][i]
   # np.fill_diagonal(A[dt], -np.sum(A[dt], 1))

A = torch.from_numpy(A)
A = A.type(torch.float32)
p=V.shape[0]
G = torch.zeros([p, p])
for i in range(V1.shape[0]):
    G[V1[i] - 1, V2[i] - 1] +=  1
    G[V2[i] - 1, V1[i] - 1] +=  -1
    
indx, indy = torch.where(A[0] > 0)

########Constructing incidence matrix
nv = len(V)
ne = len(E)

A = torch.zeros([t, nv, ne])
y = torch.zeros([t, ne])
for dt in range(t):
    for j in range(ne):
        A[dt, V1[j] - 1, j] = 1
        A[dt, V2[j] - 1, j] = -1
        y[dt, j] = edges[edges.columns[5 + dt]][j] 
def soft_threshold(beta, lam):
    return(beta.sign()*(beta.abs() - lam) * ((beta.abs() - lam) > 0))

####The ADMM step
def prox_flow(beta, sign_matrix, lam1, lam2):
    m,n = sign_matrix.shape
    z = torch.zeros(n)
    w = torch.zeros(m)
    u = torch.zeros(m)
    nu = 1. * torch.ones(1) 
    Ainv = torch.inverse(torch.eye(n) + nu*sign_matrix.T@sign_matrix)
    for i in range(50):
        z = soft_threshold(Ainv@(nu * sign_matrix.T @ (u-w) + beta), lam1)
        Hz = sign_matrix @ z
        u = soft_threshold(Hz + w, lam2 / nu)
        w = w + Hz - u
    return(z)    

#def prox_flow(beta, sign_matrix, lam1, lam2):
#    z = soft_threshold(beta, lam1)
#    return(z)    


def g(theta, sign_matrix, lam1=1., lam2=1.):
    m, n = sign_matrix.shape
    return(lam1 * theta.abs().sum() + lam2 * (sign_matrix* theta.repeat([m,1])).norm(dim = 1).sum())
    
def moreau(theta, beta, lam, sign_matrix, lam1=1., lam2=1):
    return (g(theta, sign_matrix, lam1, lam2) + ((theta - beta) ** 2).sum() / 2 / lam)

class Moreau(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, beta, lam, sign_matrix, lam1, lam2):
        ctx.save_for_backward(beta, z, lam)
        return moreau(z, beta, lam, sign_matrix, lam1, lam2)
    @staticmethod
    def backward(ctx, grad_output):
        beta, z, lam = ctx.saved_tensors
        return (torch.tensor(0.),(beta - z)/lam, torch.tensor(0.),torch.tensor(0.),torch.tensor(0.),torch.tensor(0.))
    
moreau_g = Moreau.apply
d = 10   
p = ne * d
def CoRe(params):
    beta = params[0:p].exp()
    pi_beta = -(params[0:p] ** 2).sum() / 2.
    theta = params[p:2*p].exp()
    Lam = params[2*p:(2*p+d*t)].abs()
    eta = .1
    lam = torch.tensor(1., requires_grad = True)
    lam1 = torch.tensor(5., requires_grad = True)
    lam2 = torch.tensor(50., requires_grad = True)
    #lam1 = params[-2].abs()
    #lam2 = params[-1].abs()
    Lam = Lam.reshape([d, t])
    #Lam = proj_lam(Lam, torch.quantile(Lam.norm(dim=1).abs(), min(max(b,0),1)))
    theta = theta.reshape([d, ne])
    L = - ((y.T - theta.T@Lam) ** 2).sum() / 2. 
    beta = beta.reshape([d, ne])
    core = 0
    for i in range(d):
        with torch.no_grad():
            z = prox_flow(beta[i], A[i], lam1*lam, lam2*lam)
        core += (moreau_g(z, beta[i], lam, A[i], lam1, lam2) - (g(theta[i], A[i], lam1, lam2) + ((theta[i] - beta[i])**2).sum() / 2 / lam)) / eta
    return(L + pi_beta + core-(Lam**2).sum())
params_full = torch.normal(torch.ones(2*p + d*t)) 
params_full.requires_grad = True
a = .1
lr = a
optimizer = torch.optim.Adam([params_full], lr=lr)
for epoch in range(1000):
     loss = -CoRe(params_full)
     optimizer.zero_grad()
     loss.backward(retain_graph=True)
     optimizer.step()
     if epoch % 100 ==1:
         #optimizer.param_groups[0]['lr'] = a / epoch
         print(loss.item())
beta = params_full[0:p].exp()
pi_beta = -(params_full[0:p] ** 2).sum() / 2.
theta = params_full[p:2*p].exp()
Lam = params_full[2*p:(2*p+d*t)].abs()
eta = .01
lam = torch.tensor(1., requires_grad = True)
lam1 = torch.tensor(5., requires_grad = True)
lam2 = torch.tensor(50., requires_grad = True)
#lam1 = params[-2].abs()
#lam2 = params[-1].abs()
Lam = Lam.reshape([d, t])
#Lam = proj_lam(Lam, torch.quantile(Lam.norm(dim=1).abs(), min(max(b,0),1)))
theta = theta.reshape([d, ne]) 
def proj_lam(lam, beta =.00001):
    lam = lam.reshape([d, t])
    lam_norm = lam.norm(dim = 1)
    ind = (1 - beta / lam_norm) * ((1 - beta / lam_norm) > 0) 
    return(ind.repeat([t, 1]).T * lam)
    
def CoRe_lam(params):  
    Lam = params[0:(d*t)].abs() 
    Lam = Lam.reshape([d, t])
    b = params[-1].abs()
    Lam = proj_lam(Lam, b)
    return(- ((y.T - theta.T@Lam) ** 2).sum() /1E8- Lam.norm(dim=1).sum())

import copy
params =  params_full[2*p:(2*p+d*t)].abs()
#params.requires_grad = True

print(params[-1].abs())
params_init=  torch.normal(torch.zeros(params.shape))
#inv_mass = (params ** 2+1e-4)
#params[p] = (torch.ones(1)*.01).logit()
step_size = .1
num_samples = 15000
L = 10
burn = 5000 # For results in paper burn = 2000

params_hmc_nuts = hamiltorch.sample(log_prob_func=CoRe_lam,
                                    params_init=params_init, num_samples=num_samples,
                                    step_size=step_size, num_steps_per_sample=L,
                                    desired_accept_rate=0.6,
                                    sampler=hamiltorch.Sampler.HMC,burn=burn#,inv_mass=inv_mass
                                   )

nz = []
for params in params_hmc_nuts:
    Lam = params[0:(d*t)].abs() 
    Lam = Lam.reshape([d, t])
    b = params[-1].abs()
    Lam = proj_lam(Lam, b)
    nz.append((Lam.norm(dim=1)>.1).sum()) 
plt.plot(nz)
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.figure()
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.hist(np.array(nz[0:5000]),bins=np.arange(2,8)+0.5,density=True,rwidth=.7,color='green')
matplotlib.rc('font', **font) 