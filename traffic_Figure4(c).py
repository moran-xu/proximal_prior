

import statsmodels.api as sm
import pandas as pd
from igraph import Graph
import numpy as np
import matplotlib.pyplot as plt
import torch
from numba import jit, float32, int32


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
        A[dt, V2[i] - 1, V1[i] - 1] +=  - edges[edges.columns[5 + dt]][i]
    np.fill_diagonal(A[dt], -np.sum(A[dt], 1))

A = torch.from_numpy(A)
A = A.type(torch.float32)
p=V.shape[0]
G = torch.zeros([p, p])
for i in range(V1.shape[0]):
    G[V1[i] - 1, V2[i] - 1] +=  1
    G[V2[i] - 1, V1[i] - 1] +=  -1
    
indx, indy = torch.where(G > 0)

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

@jit(nopython=True)
def soft_threshold(beta, lam):
    return(np.sign(beta)*(np.abs(beta) - lam) * ((np.abs(beta) - lam) > 0))
    
@jit(nopython=True)
def prox_flow(beta, sign_matrix, sign_matrix_inv, lam1, lam2, steps =50):
    m,n = np.shape(sign_matrix)
    w = np.zeros(m, dtype=np.float32)
    u = np.zeros(m, dtype=np.float32)
    #nu = 1.
    for i in range(steps):
        z = soft_threshold(sign_matrix_inv@(sign_matrix.T @ (u-w) + beta), lam1)
        Hz = sign_matrix @ z
        u = soft_threshold(Hz + w, lam2)
        w = w + Hz - u
    return(z) 
    
    
    
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
d = 3

p = ne * d
Ainv = torch.inverse(torch.eye(A[0].shape[1]) + A[0].T@A[0]).detach().numpy()

def CoRe(params):
    beta = params[0:p].exp()
    pi_beta = -(params[0:p] ** 2).sum() / 2.
    theta = params[p:2*p].exp()
    Lam = params[2*p:].abs()
    eta = 1.
    lam = torch.tensor(1., requires_grad = True)
    lam1 = torch.tensor(10., requires_grad = True)
    lam2 = torch.tensor(20., requires_grad = True)
    #lam1 = params[-2].abs()
    #lam2 = params[-1].abs()
    Lam = Lam.reshape([d, t])
    #Lam = proj_lam(Lam, torch.quantile(Lam.norm(dim=1).abs(), min(max(b,0),1)))
    theta = theta.reshape([d, ne])
    L = - ((y.T - theta.T@Lam) ** 2).sum() / 2. 
    beta = beta.reshape([d, ne])
    core = 0
    z = torch.zeros([d,ne])
    for i in range(d):
        beta_numpy = beta[i].detach().numpy()
        Ai_numpy = A[i].detach().numpy()
        z = torch.tensor(prox_flow(beta_numpy, Ai_numpy, Ainv, lam1.detach().numpy(), lam2.detach().numpy()))
        core += (moreau_g(z, beta[i], lam, A[i], lam1, lam2) - (g(theta[i], A[i], lam1, lam2) + ((theta[i] - beta[i])**2).sum() / 2 / lam)) / eta
    return(L + pi_beta + core-(Lam**2).sum() + theta.log().sum() + beta.log().sum())
      
from torch.optim.lr_scheduler import LambdaLR
params = torch.normal(torch.zeros(2*p + d*t))
params.requires_grad = True



a = .1
lr = a
optimizer = torch.optim.Adam([params], lr=lr)
for epoch in range(1000):
     loss = -CoRe(params)
     optimizer.zero_grad()
     loss.backward(retain_graph=True)
     optimizer.step()
     if epoch % 10 ==1:
         #optimizer.param_groups[0]['lr'] = lr / epoch ** .9
         print(loss.item())


import hamiltorch
#params = torch.normal(torch.zeros(2*p + d*t))
#params.requires_grad = True
sampler = hamiltorch.Sampler.HMC_NUTS

#lams = torch.tensor([10.,20.])
#lams.requires_grad = True
#params_init = torch.hstack([params, lams])
H = torch.autograd.functional.hessian(CoRe, params)

M = 1 / torch.diag(H).abs() 
num_samples = 1000
step_size = .0001
num_steps_per_sample = 10
hamiltorch.set_random_seed(1)
#M = torch.ones(len(params))
params_hmc = hamiltorch.sample(log_prob_func=CoRe, params_init=params,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, desired_accept_rate = .6, inv_mass = M)
params_hmc = torch.vstack(params_hmc)

ll = torch.zeros([1000, 3, 25])
for _ in range(1000):
    params_ = params_hmc[_]
    Lam = params_[2*p:].abs()
    Lam = Lam.reshape([d, t])
    ll[_] = Lam

ll_median = torch.quantile(ll,.5,0).detach().numpy() 
color = [ 'red','steelblue',  'orange', '#d62728',
              '#17becf', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22','#bcbd22','#bcbd22']
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
rcParams['figure.dpi'] = 300
rcParams['font.size'] = 20
plt.figure()
s = 0
for i in range(3):
    s+=1
    plt.plot(ll_median[i], c=color[i], linewidth = 5, label='l='+str(i+1))   
    plt.xlabel('t')
plt.legend()