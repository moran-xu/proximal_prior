 
# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
start =  time.time()
def run_rjmcmc(rep,n,p,d,correlated): 
    start =  time.time()
    torch.manual_seed(rep)
    X = torch.randn([n,p])
    correlated = 1
    if correlated == 0:
        X = torch.randn([n,p]) 
    else:  
        corr = torch.zeros(p,p)
        sig = .5
        for i in range(p):
            for j in range(p):
                corr[i,j] = sig ** np.abs(i-j)
        L = corr.cholesky()
        X = torch.normal(torch.zeros([n,p]))@L.T
        
    w0 = torch.zeros([p])
    w0[:d] = 5
    y = X@w0 + torch.randn([n])
    n_iter = 20000
    beta_list = torch.zeros([n_iter, p])
    tau_list = torch.zeros(n_iter)
    q_list = torch.zeros(n_iter)
    idx_list = []
    ######hyperparameters
    a_tau = 1 * torch.ones(1)
    b_tau = 1 * torch.ones(1) ###(tau^2 ~ IG(a_tau, b_tau))
    lam = .1  ###### Global shrinkage
    def update_beta(idx, tau, q):
        Xi = X[:, idx]
        Sig = (Xi.T@Xi + 1./lam * torch.eye(q)).inverse()
        Sig_L = (tau*Sig).cholesky()
        mu = Sig @ Xi.T @ y
        beta[idx] = Sig_L @ torch.normal(torch.zeros(q)) + mu
        return(beta)
    def update_tau(idx, beta, q):
        a_star = a_tau + (n+q)/2
        eps = (y - X[:, idx] @ beta[idx])
        b_star = b_tau + (eps.T @ eps + beta[idx].T @ beta[idx] / lam) / 2
        ig = torch.distributions.Gamma(a_star, b_star)
        tau = 1. / ig.sample()
        return(tau)
    def update_idx(idx, q):
        Xi = X[:, idx]
        Sig = (Xi.T@Xi + 1./lam * torch.eye(q)).inverse()
        Sig_L = (tau*Sig).cholesky()
        V =  torch.eye(n) - X[:, idx] @ ( X[:,idx].T@X[:,idx] + torch.eye(q) / lam).inverse() @ X[:, idx].T
        llh = - .5 * (torch.eye(q) + lam * X[:,idx].T@X[:,idx]).det().log() - (a_tau + n / 2) * (1 + y.T @ V @ y / 2 / b_tau).log()
        u = torch.rand(1)
        if u>.5 or q==1:###birth move
            mask = torch.ones(p, dtype=bool)
            mask[idx] = False
            item_new = torch.torch.arange(p)[mask][torch.randperm(p-q)[0]]
            idx_new = torch.hstack([item_new, idx])
            g_diff = np.log((p-q)/(q+1))
            pi_diff = -np.log(p)
        else: ###death_move
            idx_new = idx[torch.randperm(q)[0:q-1]]
            g_diff = np.log(q/(p-q+1))
            pi_diff = np.log(p)
        q_new = len(idx_new)
        V_new =  torch.eye(n) - X[:, idx_new] @ ( X[:,idx_new].T@X[:,idx_new] + torch.eye(q_new) / lam).inverse() @ X[:, idx_new].T
        llh_new = - .5 * (torch.eye(q_new) + lam * X[:,idx_new].T@X[:,idx_new]).det().log() - (a_tau + n / 2) * (1 + y.T @ V_new @ y / 2 / b_tau).log()
        prob = llh_new - llh + pi_diff + g_diff
        u = torch.rand(1)
        s = 1
       # print(prob)
        if u.log() > prob:
            idx_new = idx
            s = 0
        return(idx_new, s)
    beta = torch.normal(torch.zeros(p))
    tau = torch.rand(1)
    q = 1
    idx = torch.randperm(p)[0:q]
    acc = 0
    for i in range(n_iter):
        idx, s = update_idx(idx, q)
        acc += s 
        q = len(idx) 
        beta = update_beta(idx, tau, q)
        tau = update_tau(idx, beta, q) 
        q_list[i] = q
        beta_list[i] = beta
        tau_list[i] = tau
        idx_list.append(idx)  
    from statsmodels.tsa.stattools import acf as autocorr
    def neff(arr):
        n = len(arr)
        acf = autocorr(arr, nlags=n, fft=True)
        sums = 0
        for k in range(1, len(acf)):
            sums = sums + (n-k)*acf[k]/n
        return 1./(1+2*sums) 
    ESS = np.zeros(5)
    plt.plot(q_list)
    nq =  neff(q_list[10000:]) 
    print(nq)
    for i in range(5):
        ESS[i] = neff(beta_list[10000:,i]) * nq
    theta_list = []
    for i in range(n_iter-10001):
        idxi = torch.zeros(p)
        idxi[idx_list[i+10000]]=1
        theta_list.append(beta_list[i+10000] * idxi)
    mse = np.mean([ ((theta_list[i]-w0)**2).sum() for i in range(len(theta_list))])
    stop = time.time()
    return(ESS, np.mean(mse),stop - start) 
    #np.savetxt(str(d)+'rjmcmc.txt',q_list) 
ESS = np.zeros([20, 5])
mse = np.zeros(20)
times = np.zeros(20)
for rep in range(1):
    ESS[rep], mse[rep], times[rep] = run_rjmcmc(rep,200, 500, 5,1)