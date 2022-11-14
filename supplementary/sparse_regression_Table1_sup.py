 import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import hamiltorch
from copy import deepcopy
rep=0
p=500
n=200
d=5
correlated=1
start =  time.time()
hamiltorch.set_random_seed(rep)
def softthreshold(beta, mu):
    p = beta.shape[0]
    
    abs_w = torch.abs(beta)    
    t = abs_w- mu
    
    s = torch.sign(beta)

    
    eta = s.t()*torch.max(t.t(), torch.zeros(p))
    return eta 
#######Change correlated from 1 to 0 to produce iid design
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
    
X = torch.normal(torch.zeros([n,p]))  
w0 = torch.zeros([p])  
w0[:d] = torch.ones(d)*5.

y = X@w0 + torch.randn([n])*1.

softplus = torch.nn.functional.softplus
def logjac_softplus(x):
    return x - softplus(x) 
def log_prob_full(params, b): 
    lam = 10.
    idx = 0
    idx1 = idx + p
    beta = params[idx: idx1] 
    beta_log_prior =  - beta.abs().sum() * lam #(1./k+1.) * (torch.log(1+torch.abs(beta) * k / alpha)).sum() # 
    idx = idx1
    idx1 = idx + 1
    mu = softplus(params[idx: idx1])
    theta = softthreshold(beta, mu)
    #r = theta.abs().sum()
    #r_log_prior =  -r * 10#torch.log(1+r*r) #-  (1./k+1.) * (torch.log(1+torch.abs(r) * k/alpha))  + logjac_softplus(params[idx: idx1])
    quant = (-mu*lam).exp()*.99
    r_log_prior=  b*((1-quant).log())
    
    idx = idx1
    idx1 = idx + 1
    sigma2 =torch.ones(1)# softplus(params[idx: idx1])#
    sigma2_log_prior = 0.#- 3*torch.log(sigma2) - 1./sigma2  + logjac_softplus(params[idx: idx1]) 
    psi = X@theta

    lik = -((y - psi)**2).sum()/sigma2/2.0 - n*torch.log(sigma2)/2.0

    total_posterior= lik + r_log_prior + sigma2_log_prior+beta_log_prior 
    return [total_posterior, beta,mu, sigma2,theta] 


     
def log_prob(params):
    return log_prob_full(params, p*2)[0]


params = torch.randn(p+1+1)
params = params.requires_grad_()
#true_param = torch.hstack((w0+1., torch.ones(1), torch.ones(1)))
 
optimizer = torch.optim.Adam([params], lr=1E-2)

for t in range(10000):
    loss =  -log_prob_full(params,1)[0]
    
    if t%100==0:
        print(t,  loss)
          
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward(retain_graph=True)

    optimizer.step() 
    
params_init = params#torch.normal( torch.hstack((w0+1., torch.ones(1), torch.ones(1))))
step_size = .0001 
num_samples = 20000  # For results in plot num_samples = 12000
L = 15
burn = 10000   # For results in plot burn = 2000

params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob,
                                    params_init = params_init, num_samples=num_samples,
                                    step_size=step_size, num_steps_per_sample=L, desired_accept_rate=.6,
                                    sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn,verbose=False,debug=2 ,inv_mass=(params.abs()+.001)**2
                                   )
print('repeat experiment ===',str(rep), 'acc_rate ===',str(params_hmc_nuts[1]))

 
param_trace = torch.vstack(params_hmc_nuts[0]) 
theta_trace  = torch.vstack([log_prob_full(param_trace[i],p*2)[4] for i in range(0,num_samples-burn)]).numpy()
 
nz = [np.sum(np.abs(_)>.000001) for _ in theta_trace] 
plt.plot(nz[1000:],  linewidth=3) 
# =============================================================================
# 
# plt.plot(nz,  linewidth=3)  
# 
# plt.xlabel("Iteration")  
# plt.legend()
# 
# =============================================================================
from statsmodels.tsa.stattools import acf as autocorr
mse = np.mean([((theta_trace[i]-w0.numpy())**2).sum() for i in range(len(theta_trace))] )
def neff(arr):
    n = len(arr)
    acf = autocorr(arr, nlags=n, fft=True)
    sums = 0
    for k in range(1, len(acf)):
        sums = sums + acf[k] * (n-k) / n
    return 1./(1+2*sums)
ESS = np.zeros([5])
for j in range(5):
    t1 = theta_trace[:,j][::5]
    ESS[j] = neff(t1)
nz = [np.sum(np.abs(_)>1E-4) for _ in theta_trace] 
ESS[0] = neff(nz[::5])
stop = time.time() 
#np.savetxt(str(d)+'prox.txt',nz)