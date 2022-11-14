# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:09:51 2022

@author: Maoran
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import hamiltorch
hamiltorch.set_random_seed(123)
n = 100
r = 10
p = 2
u = 2
M = torch.normal(torch.zeros([r, r])).cuda()
O = torch.svd_lowrank(M, q = r)[0]
Gamma = O[:,0:u]
Gamma0 = O[:, u:]
omega = torch.ones(u).cuda()
omega0 = 200*torch.ones(r-u).cuda()
Sigma = Gamma @ torch.diag(omega) @ Gamma.T + Gamma0 @ torch.diag(omega0) @ Gamma0.T
X = torch.randn([p, n]).cuda() * 5.

m = torch.distributions.MultivariateNormal(torch.zeros(r).cuda(), covariance_matrix = Sigma) 
eps = m.sample((n,)).cuda()
eta = torch.randn([u, p]).cuda()
Y = (Gamma@eta@X).T + eps

def softthreshold(beta, mu):
    p = beta.shape[0]
    
    abs_w = torch.abs(beta)    
    t = abs_w- mu
    
    s = torch.sign(beta)

    
    eta = s.t()*torch.max(t.t(), torch.zeros(p).cuda())
    return eta


softplus = torch.nn.functional.softplus
def logjac_softplus(x):
    return x - softplus(x)

def llh_full(params):
    idx = 0
    idx1 = r*r
    R = params[idx:idx1].reshape(r, r)
    R_prior = -(R**2).sum()
    idx = idx1
    idx1 += r - 2
    lam = params[idx: idx1]
    lam_prior = -lam.abs().sum() 
    idx = idx1
    idx1 += 1
    mu = torch.sigmoid(params[idx: idx1])
    mu_q = -mu.log()
    lam = softthreshold(lam, mu_q) 
    ra_log_prior = mu.log() + (1-mu).log()
    lam = torch.hstack([torch.ones(1).cuda(),lam,torch.zeros(1).cuda()]) 
    Lam = torch.diag(lam)
    lam1 = 1 / (lam+1E-6) * (lam / (lam + 1E-6))
    A = R @ Lam 
    Rinv = R.inverse()
    Lam1 = torch.diag(lam1)
    ATAm = Lam1@Rinv@Rinv.T@Lam1
    AATm = Rinv.T @ Lam1**2 @ Rinv
    sATA = (A.T@A).svd()[1]
    ATA_det = torch.prod(sATA[sATA > 1E-4])
    idx = idx1
    idx1 += r*p
    beta = params[idx:idx1].reshape(r,p)
    PA = torch.eye(r).cuda() - A @ ATAm @ A.T
    theta = PA @ beta
    idx = idx1
    idx1 += r * r
    W1 = params[idx:idx1].reshape([r, r])
    W_prior = -(W1**2).sum()
    W = W1.T @ W1
    Omg = PA @ W @ PA 
    sOmg = Omg.svd()[1]
    Omg_det = torch.prod(sOmg[sOmg > 1E-4])
    data_llh =  - ((Y - X.T @ theta.T) @ Omg @ (Y.T - theta @ X) ).trace() / 2  - (Y@AATm@Y.T).trace() / 2 + n / 2 * Omg_det.log() - n / 2 * ATA_det.log()
    llh_ = R_prior + lam_prior + ra_log_prior  + W_prior + data_llh
    return(llh_, lam, theta)
 
def llh_1full(params):
    theta = params[0:p*r].reshape(p,r)
    return(-((Y-X.T@theta)**2).sum() - (theta**2).sum(), 1)
def llh(params):
    return(llh_full(params)[0])

params = torch.normal(torch.zeros(r*r+r+1+r*p+r*r+1)).cuda()
params = params.requires_grad_() 

optimizer = torch.optim.Adam([params], lr=1E-1)

for t in range(500): 
    loss =  -llh(params) 
    
    if t%100==0:
        print(t,  loss)
          
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward(retain_graph=True)

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

optimizer = torch.optim.Adam([params], lr=1E-2)

for t in range(10000): 
    loss =  -llh(params) 
    
    if t%100==0:
        print(t,  loss)
          
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward(retain_graph=True)

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
          
params_init=params #+ torch.normal(torch.zeros(params.shape).cuda())*.01
inv_mass = (params ** 2+1e-4)
#params[p] = (torch.ones(1)*.01).logit()
step_size = .0001
num_samples = 10000
L = 10
burn = 5000 # For results in paper burn = 2000

params_hmc_nuts = hamiltorch.sample(log_prob_func=llh,
                                    params_init=params_init, num_samples=num_samples,
                                    step_size=step_size, num_steps_per_sample=L,
                                    desired_accept_rate=0.6,
                                    sampler=hamiltorch.Sampler.HMC,burn=burn
                                   )   
params_hmc_nuts = torch.vstack(params_hmc_nuts).cpu().detach()
lam_ = [llh_full(_)[1] for _ in params_hmc_nuts.cuda()]
rank = [torch.sum((torch.abs(_)>0.))  for _ in lam_]
theta_ = torch.stack([llh_full(_)[2].cuda() for _ in params_hmc_nuts.cuda()])
mse = ((theta_.mean(0)-Gamma@eta)**2).mean()
from statsmodels.tsa.stattools import acf as autocorr
def neff(arr):
    n = len(arr)
    acf = autocorr(arr, nlags=n, fft=True)
    sums = 0
    for k in range(1, len(acf)):
        sums = sums + acf[k] * (n-k) / n
    return 1./(1+2*sums)
 
ESS = neff(theta_.cpu().detach()[:,0,0][::5])

params = torch.normal(torch.zeros(r*r+r+1+r*p+r*r+1)).cuda()
params = params.requires_grad_() 
def llh(params):
    return(llh_1full(params)[0])
optimizer = torch.optim.Adam([params], lr=1E-2)

for t in range(2000): 
    loss =  -llh(params) 
    
    if t%100==0:
        print(t,  loss)
          
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward(retain_graph=True)

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step() 
theta_null = params[0:p*r].reshape(p,r)
mse_null =  ((theta_null.T-Gamma@eta)**2).mean()

print('ratio =', mse/mse_null)
