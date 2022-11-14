import hamiltorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
start =  time.time()
hamiltorch.set_random_seed(0)
cuda0 = torch.device('cuda:0')
torch.cuda.set_device(cuda0)
rep = 1
mse = torch.zeros(rep) 
aab = torch.zeros(rep) 
mab = torch.zeros(rep)
n = 200
p = 300
kt = 15
n_iter = 10000
Lamb = torch.zeros([p,kt]).cuda()
numeff = kt + torch.randperm(kt).cuda()
for h in range(kt):
    temp = torch.randperm(p).cuda()[0:numeff[h]]#numeffh
    Lamb[temp,h] = torch.normal(torch.zeros(numeff[h])).cuda() * 3

mu = torch.zeros(p).cuda()
Ot1 = Lamb@Lamb.T  + .25* torch.eye(p).cuda()

m = torch.distributions.MultivariateNormal(torch.zeros(p).cuda(), Ot1) 
y = m.sample((n,)).cuda() 
y = y-y.mean(0)
vy = y.var(0)
VY = torch.outer(vy,vy)
y = y / vy.sqrt()
#===============================
Ot = Ot1*(1 / VY.sqrt())

alpha = .8*torch.ones(1).cuda() ## param for geometric prior
pz = .5*torch.ones(1).cuda() ##prob for birth/death
a = 2. *torch.ones(1).cuda()
b = 1. *torch.ones(1).cuda()
delt = 100* torch.ones(1).cuda()
###params in clude: k, Gamma, Sigma

def update_Omega(k, Gamma, Sigma):
    cov = (torch.eye(k).cuda() + Gamma.T@torch.diag(1./Sigma).cuda()@Gamma).inverse()
    L = torch.linalg.cholesky(cov)
    F = (L @ torch.normal(torch.zeros([k,n])).cuda() + cov @ Gamma.T @ torch.diag(1./Sigma).cuda() @ y.T).T 
    for i in range(p):
        cov = (torch.eye(k).cuda() * delt + 1. / Sigma[i]* F.T @ F).inverse()
        L = torch.linalg.cholesky(cov)
        Gamma[i] = cov @ F.T @ y[:,i] / Sigma[i] + L@torch.normal(torch.zeros(k)).cuda()
    for i in range(p):
        ss = ((y[:,i] - Gamma[i]@F.T)**2).sum()
        m = torch.distributions.Gamma(a+ n /2 , b+.5*ss)
        Sigma[i] = 1./m.sample()
    return(Gamma, Sigma)

def update_k(k, Gamma, Sigma):
    Omega = Gamma @ Gamma.T + torch.diag(Sigma)
    Om1 = Omega.inverse()
    c = 1E-5* torch.ones(1).cuda() 
    bd = torch.rand(1).cuda()
    acc = 0
    u = torch.rand(1).cuda()
    g_prior = torch.eye(p).cuda()  *delt
    if bd < pz or k==1:
    ###Birth 
        indic = 'birth'
        cov =   (g_prior -  Om1 @ y.T @ y @ Om1).inverse()
        L,s,_ = cov.svd()
        Gamma_k1 = L@torch.diag(s.sqrt())@torch.normal(torch.zeros(p)).cuda().reshape([p,1])
        Gamma_k1 = L@torch.normal(torch.zeros(p)).cuda().reshape([p,1])
        acc_prob = -n / 2 * (1+Gamma_k1.T@Om1@Gamma_k1).abs().log() + alpha.log()
        acc_prob += .5 * (Gamma_k1.T@Om1@y.T@y@Om1@Gamma_k1).trace() *  (1. / (1+Gamma_k1.T@Om1@Gamma_k1) - c)
        acc_prob += ((1-pz) / (k+1)).log()   
        acc_prob += - .5 * ((g_prior - c*Om1@y.T@y@Om1)/delt).logdet() #+ .5 * p * delt.log() 
        if u.log() < acc_prob:
            acc = 1
            k += 1
            Gamma = torch.hstack([Gamma,Gamma_k1.reshape([p,1])])
    else: ### Death
        indic = 'death'
        l = torch.randperm(k).cuda()[0]
        Gamma_l = Gamma[:,torch.arange(Gamma.size(1)).cuda()!=l]  
        Om_star1 = (Gamma_l@Gamma_l.T + torch.diag(Sigma) ).inverse()
        acc_1 = .5 * ((g_prior - c*Om_star1@y.T@y@Om_star1)/delt).logdet()-alpha.log() 
        acc_2 = n / 2 * (1+Gamma[:,l].T@Om_star1@Gamma[:,l]).abs().log()
        acc_3 = -.5 * ((Gamma[:,l].reshape([1,p]))@Om_star1@y.T@y@Om_star1@Gamma[:,l]) * (1. / (1 + Gamma[:,l].T@Om_star1@Gamma[:,l])-c)
        acc_prob = acc_1+acc_2+acc_3+(pz*k / (1-pz)).log()   
       # print(acc_1,acc_2,acc_3)
        if u.log() < acc_prob:
            acc = 1
            k = k - 1
            Gamma = Gamma_l 
    return(k, Gamma, acc, acc_prob, indic)


acc_count = 0
k_list = torch.zeros(n_iter)
Gamma_list = []
Sigma_list = torch.zeros([n_iter, p]).cuda() 
k = kt+5#int(np.log(p)*5) 
Gamma = torch.normal(torch.zeros([p,k])).cuda() 
Sigma = torch.normal(torch.zeros(p)).abs().cuda() + 1
for i in range(n_iter):
    Gamma, Sigma = update_Omega(k, Gamma, Sigma )
    Gamma_list.append(Gamma)
    Sigma_list[i] = Sigma
    k, Gamma, acc, pb, ind = update_k(k, Gamma, Sigma)
    k_list[i] = k
    acc_count += acc
    if i%30==0 and i > 1: 
       print(i, acc_count/i, pb, ind, k)
stop = time.time()
print(stop - start) 
###===============save=results=
np.savetxt('klist'+str(kt), k_list,delimiter = ',')
gamma1 = []
indx = np.where((Lamb@Lamb.T).cpu()>0)[0][0]
indy = np.where((Lamb@Lamb.T).cpu()>0)[1][0]
for _ in Gamma_list:
    gamma1.append((_@_.T)[indx, indy].cpu())
np.savetxt('gamma_nonzero'+str(kt), gamma1,delimiter = ',')
np.savetxt('sigma'+str(kt), Sigma_list[:,0].cpu(), delimiter = ',')
Omega = Gamma @ Gamma.T + torch.diag(Sigma)
mse = (((Omega-Ot)**2).mean())
aab = ((Omega-Ot).abs().mean())
mab = ((Omega-Ot).abs().max())
print((mse, aab, mab))