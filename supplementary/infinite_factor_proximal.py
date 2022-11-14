 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import hamiltorch
import numpy as np
import matplotlib.pyplot as plt
#===============================Generating Data===================
#hamiltorch.set_random_seed(0)
cuda0 = torch.device('cuda:0')
torch.cuda.set_device(cuda0)
rep = 20
mse = torch.zeros(rep)
aab = torch.zeros(rep)
mab = torch.zeros(rep)
n = 200
p = 100
k = 5

Lamb = torch.zeros([p,k], device = cuda0)
numeff = k + torch.randperm(k)
for h in range(k):
    temp = torch.randperm(p).cuda()[0:numeff[h]]
    Lamb[temp,h] = torch.normal(torch.zeros(numeff[h])).cuda() * 3

mu = torch.zeros(p).cuda()
Ot = Lamb@Lamb.T  + .2* torch.eye(p).cuda()

m = torch.distributions.MultivariateNormal(torch.zeros(p).cuda(), Ot) 
for ii in range(rep): 
    #hamiltorch.set_random_seed(ii)
    y = m.sample((n,)).cuda()  
    #===============================
    k1 = int(np.log(p)*5) 
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
    
    def log_prob(params):
        idx = 0
        idx1 = p*k1
        Lam =  params[idx:idx1] 
        logprior_Lam = -(Lam**2).sum()  
# =============================================================================
#         idx = idx1
#         idx1 = idx + 1
#         mu_l = softplus(params[idx: idx1])
#         Lam = softthreshold(Lam, mu_l)
#         r_l = Lam.abs().sum()
#         r_l_log_prior =  -r_l / 10.  + logjac_softplus(params[idx]) #-  (1./k+1.) * (torch.log(1+torch.abs(r) * k/alpha))  + logjac_softplus(params[idx: idx1])
# =============================================================================
        Lam = Lam.reshape([p,k1])
        idx = idx1
        idx1 += k1
        beta = softplus(params[idx:idx1])
        logprior_shrink = -(beta**2).sum() + logjac_softplus(params[idx: idx1]).sum() 
     
        idx = idx1
        idx1 = idx + 1
        mu = softplus(params[idx: idx1])
        shrink = softthreshold(beta, mu)
        r = shrink.abs().sum()
        r_log_prior =  -r * 8 + logjac_softplus(params[idx])  
        
        idx = idx1
        idx1 += p
        sig = softplus(params[idx:idx1] )
        logprior_sig = -2*torch.log(sig ).sum() - (.3/ sig).sum() + logjac_softplus(params[idx: idx1]).sum()
        Lam_lite = Lam @torch.diag(shrink) 
        Omega =  Lam_lite @ Lam_lite.T + torch.diag(sig)
        m = torch.distributions.MultivariateNormal(torch.zeros(p).cuda(), Omega)
        datalikelihood  = m.log_prob(y).sum()
        est = logprior_Lam + logprior_shrink + logprior_sig + datalikelihood  +  r_log_prior#+r_l_log_prior  
        return(est, Lam, shrink, sig, Omega)
    
    params = torch.normal(torch.zeros(p*k1+k1+p+2) ).cuda()
    params = params.requires_grad_()
    
    def llh(params):
        return(log_prob(params)[0])
    optimizer = torch.optim.Adam([params], lr=1E-2) 
    
    for t in range(6000): 
        loss =  -log_prob(params)[0]
        
        if t%100==0:
            print(t,  loss)
              
        optimizer.zero_grad()
    
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward(retain_graph=True)
    
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
     
          
    
    params_init = params
    step_size = .01
    num_samples = 15010 # For results in plot num_samples = 12000
    L = 1
    burn = 10 # For results in plot burn = 2000
    
    params_hmc_nuts = hamiltorch.sample(log_prob_func=llh,
                                        params_init=params_init, num_samples=num_samples,
                                        step_size=step_size, num_steps_per_sample=L,
                                        desired_accept_rate=0.6,
                                        sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn
                                       )
    
    param_trace_factor = torch.vstack(params_hmc_nuts)[1000:]
 
    params = param_trace_factor.mean(0)
    Lam = log_prob(params)[1]
    shrink = log_prob(params)[2]
    sig = log_prob(params)[3]
    Omega = log_prob(params)[4]  
        
    mse[ii] = (((Omega-Ot)**2).mean())
    aab[ii] = ((Omega-Ot).abs().mean())
    mab[ii] = ((Omega-Ot).abs().max())

nz = []
for pa in param_trace_factor:    
    shrink = log_prob(pa)[2]
    nz.append((shrink>.01).sum().cpu())

plt.plot(nz)
#plt.plot(bimf[2000:3000])
plt.yticks([5,6,7])
plt.ylim([4.5,8.5])
plt.xlabel("Iteration")
plt.ylabel("No. of Component")

from statsmodels.tsa.stattools import acf as autocorr
def neff(arr):
    n = len(arr)
    acf = autocorr(arr, nlags=n, fft=True)
    sums = 0
    for k in range(1, len(acf)):
        sums = sums + (n-k)*acf[k]/n

    return n/(1+2*sums)
bimf_10 = np.loadtxt('C:/Users/laosu/Documents/bimf_10.txt')
plt.plot(bimf_10[2000:3000])
plt.yticks([5,6,7])
plt.ylim([4.5,8.5])
plt.xlabel("Iteration")
plt.ylabel("No. of Component")
ESS_MGPS = neff(bimf_10)
