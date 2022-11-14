import hamiltorch
import torch 
import numpy as np
import copy
hamiltorch.set_random_seed(1)
def softthreshold(beta, mu):
    p = beta.shape[0]
    
    abs_w = torch.abs(beta)    
    t = abs_w- mu
    
    s = torch.sign(beta)

    
    eta = s.t()*torch.max(t.t(), torch.zeros(p))
    return eta
p= int(200)

n = int(100) 

#######Change correlated from 1 to 0 to produce iid design
correlated = 1
if correlated == 0:
    X = torch.randn([n,p]) 
else:  
    corr = torch.zeros(p,p)
    sig = .5
    for i in range(p):
        for j in range(p):
            corr[i,j] = .5 ** np.abs(i-j)
    L = corr.cholesky()
    X = torch.normal(torch.zeros([n,p]))@L.T
    
d = 5 
w0 = torch.zeros([p])  
w0[:d] = torch.randn([d])* 0.001+ torch.tensor([1,1,1,1,1])*5.

y = X@w0 + torch.randn([n])

softplus = torch.nn.functional.softplus
def logjac_softplus(x):
    return x - softplus(x) 
def log_prob_full(params): 
    idx = 0
    idx1 = idx + p
    beta = params[idx: idx1]
    
    beta_log_prior =   - beta.abs().sum() / .1 #(1./k+1.) * (torch.log(1+torch.abs(beta) * k / alpha)).sum() 
    idx = idx1
    idx1 = idx + 1
    mu = softplus(params[idx: idx1])
    theta = softthreshold(beta, mu)
    r = theta.abs().sum()
    r_log_prior =  -r / 10 #torch.log(1+r*r) #-  (1./k+1.) * (torch.log(1+torch.abs(r) * k/alpha))  + logjac_softplus(params[idx: idx1])
    
    
    idx = idx1
    idx1 = idx + 1
    sigma2 = torch.ones(1)#softplus(params[idx: idx1])
    sigma2_log_prior = 0#-3*torch.log(sigma2) - sigma2  + logjac_softplus(params[idx: idx1]) 
    psi = X@theta

    lik = -((y - psi)**2).sum()/sigma2/2.0 - n*torch.log(sigma2)/2.0

    total_posterior= lik + r_log_prior + sigma2_log_prior+beta_log_prior 
    return [total_posterior, beta,mu, sigma2,theta] 


     
def log_prob(params):
    return log_prob_full(params)[0]


params = torch.randn(p+1+1)
params = params.requires_grad_()
optimizer = torch.optim.Adam([params], lr=1E-1)

for t in range(3000):
    

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
import pylab as plt
theta = log_prob_full(params)[4] 
inv_mass = torch.ones( p+1+1)
rep = 10
mc_len = 3000
log_post = np.zeros([rep, mc_len])
theta_trace = np.zeros([rep, mc_len])
for i in range(rep):
    params_init = torch.normal(torch.zeros(params.shape)) 
    if i>5:
        params_init = copy.deepcopy(params.detach()) 
        params_init += torch.rand(params.shape)  
    step_size = .001
    num_samples = mc_len+1 # For results in plot num_samples = 12000
    L = 50
    burn = 1 # For results in plot burn = 2000
    
    params_hmc_nuts = hamiltorch.sample(log_prob_func=log_prob,
                                        params_init=params_init, num_samples=num_samples,
                                        step_size=step_size, num_steps_per_sample=L,
                                        desired_accept_rate=0.6,
                                        sampler=hamiltorch.Sampler.HMC ,burn=burn,
                                        inv_mass = inv_mass
                                       )
    
    
    
    param_trace = torch.vstack(params_hmc_nuts)
    
    trace_np = param_trace.detach().cpu().numpy()
    theta_tracei = torch.vstack([log_prob_full(param_trace[i])[4] for i in range(mc_len)]).numpy()
    theta_trace[i] = theta_tracei[:,0]
    #plt.plot(trace_np[:,9])
    log_posti = torch.vstack([log_prob_full(param_trace[i])[0] for i in range(mc_len)]).flatten().numpy()
    log_post[i] = log_posti
    
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
    
matplotlib.rc('font', **font)

plt.figure(figsize = [8,6])
for i in range(rep):
    plt.plot(theta_trace[i,0:1000],alpha=.5, linewidth=3)
    plt.scatter(0, theta_trace[i,0],s=10)
plt.axhline(5.,label='true model',linestyle = 'dashed', linewidth=3)
plt.xlabel("iteration")
plt.ylabel("theta[1]")
plt.legend()
plt.savefig("hmc_converge_theta_"+str(correlated))

plt.figure(figsize = [8,6])
true_param = torch.hstack((w0, torch.ones(1)*0, torch.ones(1)))
true_post = log_prob(true_param)[0]
null_param = torch.zeros(params.shape)
null_param[-2] = 1.
null_param[-1] = 1.
null_post = log_prob(null_param)[0]
high_post = log_prob(params)[0]
for i in range(rep):
    plt.plot(log_post[i,0:1000],alpha=.5, linewidth=3)
    plt.scatter(0, log_post[i,0],s=10)
plt.axhline(true_post,label='true model',linestyle = 'dashed', linewidth=3)
plt.axhline(high_post.detach().numpy(),label='highest probability model',color='red',linestyle = 'dotted', linewidth=3)
plt.axhline(null_post.detach().numpy(),label='null model',color='grey',linestyle = 'dashdot', linewidth=3)
plt.xlabel("iteration")
plt.ylabel("log-posterior")
plt.legend() 
plt.savefig("hmc_converge_llh_"+str(correlated))
