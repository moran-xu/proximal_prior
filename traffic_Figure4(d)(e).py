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
d = 3
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
params = torch.normal(torch.ones(2*p + d*t)) 
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
         #optimizer.param_groups[0]['lr'] = a / epoch
         print(loss.item())
beta = params[0:p].exp()
pi_beta = -(params[0:p] ** 2).sum() / 2.
theta = params[p:2*p].exp()
Lam = params[2*p:(2*p+d*t)].abs()
eta = .01
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
from matplotlib import cm
import igraph
import networkx as nx
nodes_sorted = nodes.sort_values(by=['node_id'])
#g.to_undirected()
x_coord, y_coord = nodes_sorted['x'],nodes_sorted['y']
layout = zip(list(x_coord.to_numpy()), list(-y_coord.to_numpy())) 
nodes_pos = (nodes_sorted[['x','y']])
pos_array = []
for i in range(len(nodes_pos)):
    pos_array.append([i,[nodes_pos.iloc[i]['x'],nodes_pos.iloc[i]['y']]])
pos= dict(pos_array)
cmap = cm.get_cmap('Reds', 10)
X = np.zeros((d, nv, nv))
for dt in range(d):
    z = prox_flow(beta[dt].abs(), A[0], 50.,500).detach().numpy()
    for i in range(ne):
        X[dt, V1[i] - 1, V2[i] - 1] += z[i]    
plt.figure()
color = [  'red','steelblue', 'orange',]
for ni in range(3):
    #cmap = cm.get_cmap('Oranges', 10)
    a =  X[ni]
    G = igraph.Graph.Adjacency(a.tolist())
    e = G.get_edgelist()
    Gr = nx.DiGraph(e) 
    for _ in Gr.edges:
        Gr[_[0]][_[1]]['weight'] = a[_]
    #print(G.nodes)
    for n in Gr.nodes:
        Gr.nodes[n]['pos'] = (nodes_sorted.iloc[n, -2], nodes_sorted.iloc[n,-1])
    #color = [X[ni][u][v]*10. for u,v in Gr.edges()]
    widths = nx.get_edge_attributes(Gr, 'weight')
      #  nx.draw(Gr,pos=pos,alpha=.3, node_size = 1, node_color = 'black', edge_color = 'gray',linewidths = .01)
    nx.draw(Gr, pos=pos, edge_color= color[ni], alpha=1.,  width =5, node_size = 1, node_color = 'black',arrowstyle='->',arrowsize=40)
  
def CoRe1(params, Lam):
    beta = params[0:p].exp()
    pi_beta = -(params[0:p] ** 2).sum() / 2.
    theta = params[p:2*p].exp() 
    eta = .1
    lam = torch.tensor(1., requires_grad = True)
    lam1 = torch.tensor(5., requires_grad = True)
    lam2 = torch.tensor(50., requires_grad = True)
    #lam1 = params[-2].abs()  
    #Lam = proj_lam(Lam, torch.quantile(Lam.norm(dim=1).abs(), min(max(b,0),1)))
    theta = theta.reshape([d, ne])
    L = - ((y.T - theta.T@Lam) ** 2).sum() / 2. 
    beta = beta.reshape([d, ne])
    core = 0
    for i in range(d):
        with torch.no_grad():
            z = prox_flow(beta[i], A[i], lam1*lam, lam2*lam)
        core += (moreau_g(z, beta[i], lam, A[i], lam1, lam2) - (g(theta[i], A[i], lam1, lam2) + ((theta[i] - beta[i])**2).sum() / 2 / lam)) / eta
    return(L + pi_beta + core)

def llh(params):
    return(CoRe1(params, Lam))

    
params_init = params[0:2*p]
step_size = .0001 
num_samples = 2000  # For results in plot num_samples = 12000
L = 1
burn = 1000   # For results in plot burn = 2000
params_hmc_nuts = hamiltorch.sample(log_prob_func=llh,
                                    params_init = params_init, num_samples=num_samples,
                                    step_size=step_size, num_steps_per_sample=L, desired_accept_rate=.6,
                                    sampler=hamiltorch.Sampler.HMC_NUTS,burn=burn,verbose=False,debug=2)
                                  
params_hmc_nuts = torch.vstack(params_hmc_nuts[0])
X1 = np.zeros((d, nv, nv))
for dt in range(d):
    z = prox_flow(beta[dt].abs(), A[0], 50.,500).detach().numpy()
    z =( z - min(z)) / (max(z) - min(z))
    for i in range(ne):
        X1[dt, V1[i] - 1, V2[i] - 1] += z[i]  
        
plt.figure()
def normalize(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:value*factor for key,value in d.items()}
color = [  'red','steelblue', 'orange',]
for ni in range(3):
    #cmap = cm.get_cmap('Oranges', 10)
    a =  X[ni]
    G = igraph.Graph.Adjacency(a.tolist())
    e = G.get_edgelist()
    Gr = nx.Graph(e) 
    for _ in Gr.edges:
        Gr[_[0]][_[1]]['weight'] = X1[ni][_]
    #print(G.nodes)
    for n in Gr.nodes:
        Gr.nodes[n]['pos'] = (nodes_sorted.iloc[n, -2], nodes_sorted.iloc[n,-1])
    #color = [X[ni][u][v]*10. for u,v in Gr.edges()]
    widths = nx.get_edge_attributes(Gr, 'weight')
    alpha = nx.get_edge_attributes(Gr,'weight') 
    #  #  nx.draw(Gr,pos=pos,alpha=.3, node_size = 1, node_color = 'black', edge_color = 'gray',linewidths = .01)
    nx.draw(Gr, pos=pos, edge_color= color[ni], alpha=.01,  width =5, node_size = 1, node_color = 'black')
    [nx.draw_networkx_edges(Gr, pos= pos,edgelist=[_], edge_color= color[ni],  alpha=Gr[_[0]][_[1]]['weight'],width=5) for _ in Gr.edges] #loop through edges and draw them
