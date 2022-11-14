 
import pandas as pd
import igraph
from igraph import Graph
import numpy as np
import matplotlib.pyplot as plt
import torch
#import pickle
import hamiltorch
import statsmodels.api as sm
#import copy
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 8
# rcParams['figure.dpi'] = 300

device = torch.device('cpu')
edges = pd.read_csv ("edges.csv")
nodes = pd.read_csv ("nodes.csv")

V = nodes['node_id']
E = edges['edge_id'] 

t = 25

V1 = edges['fromnode_id']
V2 = edges['tonode_id']

p = V.shape[0]
#A: t*p*p, the flow network
A = np.zeros([t, p, p])
for dt in range(t):
    for i in range(V1.shape[0]):
        A[dt, V1[i] - 1, V2[i] - 1] +=  edges[edges.columns[5 + dt]][i]
        A[dt, V2[i] - 1, V1[i] - 1] +=  - edges[edges.columns[5 + dt]][i]
    np.fill_diagonal(A[dt], -np.sum(A[dt], 1))

A = torch.from_numpy(A)
A = A.type(torch.float32)
#G: the graph of connected nodes
p=V.shape[0]
G = torch.zeros([p, p])
for i in range(V1.shape[0]):
    G[V1[i] - 1, V2[i] - 1] +=  1
    G[V2[i] - 1, V1[i] - 1] +=  -1
 
indx, indy = torch.where(G > 0)
n_up = len(indx)
Y = torch.zeros([t, n_up])
for i in range(t):
    Y[i] = A[i][indx, indy]
Y = torch.sign(Y)*torch.log(Y.abs()+1)
d = 3

softplus = torch.nn.functional.softplus
def logjac_softplus(x):
    return x - softplus(x)

def llh(params):
    log_pos = 0
    idx = 0
    idx1 = n_up * d
    F = params[idx : idx1].reshape(d, n_up)
    F1 = torch.zeros([d, p, p]) 
    for i in range(d):
        F1[i, indx, indy] = F[i]
    idx = idx1 
    idx1 += p*d
    tau_i = softplus(params[idx : idx1]).reshape(d, p)
    log_pos += logjac_softplus(params[idx : idx1]).sum()
    idx = idx1
    idx1 += n_up * d
    tau = softplus(params[idx : idx1]).reshape(d, n_up)
    tau_j = torch.zeros([d, p, p])
    for i in range(d):
        tau_j[i, indx, indy] = tau[i]
    log_pos += logjac_softplus(params[idx : idx1]).sum()
    idx = idx1
    idx1 += 1
    lam = softplus(params[idx])
    log_pos += logjac_softplus(params[idx : idx1]).sum()
    idx = idx1
    idx1 += d*t
    weight = softplus(params[idx : idx1]).reshape(d,t)
    log_pos += logjac_softplus(params[idx : idx1]).sum()
    sig = lam**2 * tau_i.repeat_interleave(p,dim=1).reshape([d,p,p]) * tau_j
    log_pos +=  - (F1 **2 *100 / (sig+1E-5)).sum() - (1 + tau_i **2).log().sum() - (1 + tau_j **2).log().sum() - (1 + lam**2 ).log() 
    Yhat = weight.T @ F
    log_pos += -((Yhat - Y)**2).sum()
    Rho = torch.zeros([d, d])
    for i in range(d):
        for j in range(d):
            Rho[i,j] = torch.sum(F[i] * F[j]) / ((F[i] ** 2).sum() * (F[j] ** 2).sum())
    log_pos += (Rho.logdet()) * 2
    return(log_pos)
  
params = torch.rand(n_up * d * 2 + p*d + n_up * d + 1 + d*t)

llh(params)

params = params.requires_grad_() 

optimizer = torch.optim.Adam([params], lr=1E-2)

for tt in range(2000): 
    loss =  -llh(params) 
    
    if tt%100==0:
        print(tt,  loss)
          
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward(retain_graph=True)

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

 

log_pos = 0
idx = 0
idx1 = n_up * d
F = params[idx : idx1].reshape(d, n_up)
F1 = torch.zeros([d, p, p]) 
for i in range(d):
    F1[i, indx, indy] = F[i]


X = F1.detach()  
X[X.abs()<1]=0
for i in range(d):
    a =  X[i].detach().numpy()
    np.fill_diagonal(a,0)
    #a[a<.01] = 0
    #a = a.round()
    nodes_sorted = nodes.sort_values(by=['node_id'])
    g = igraph.Graph.Adjacency((a > 0).tolist())
    g.to_undirected()
    x_coord, y_coord = nodes_sorted['x'],nodes_sorted['y']
    layout = zip(list(x_coord.to_numpy()), list(-y_coord.to_numpy())) 

np.fill_diagonal(a,0)
#a[a<.01] = 0
#a = a.round()
nodes_sorted = nodes.sort_values(by=['node_id'])
#g.to_undirected()
x_coord, y_coord = nodes_sorted['x'],nodes_sorted['y']
layout = zip(list(x_coord.to_numpy()), list(-y_coord.to_numpy())) 
import networkx as nx 
nodes_pos = (nodes_sorted[['x','y']])
pos_array = [] 
for i in range(len(nodes_pos)):
    pos_array.append([i,[nodes_pos.iloc[i]['x'],nodes_pos.iloc[i]['y']]])
pos= dict(pos_array)
from matplotlib import cm  
c = ['Reds','OrRd','Blues']
for ni in range(3): 
    cmap = cm.get_cmap(c[ni], 10)
    a =  X[ni].detach().numpy()  
    g = igraph.Graph.Adjacency(a.tolist())
    e = g.get_edgelist()
    Gr = nx.DiGraph(e) 
    for _ in Gr.edges:
        Gr[_[0]][_[1]]['weight'] = a[_]
    #print(G.nodes)
    for n in Gr.nodes:
        Gr.nodes[n]['pos'] = (nodes_sorted.iloc[n, -2], nodes_sorted.iloc[n,-1])
    color = [X[ni][u][v] for u,v in Gr.edges()]
    color = ['orange','steelblue','red'][ni]
    widths = nx.get_edge_attributes(Gr, 'weight')
      #  nx.draw(Gr,pos=pos,alpha=.3, node_size = 1, node_color = 'black', edge_color = 'gray',linewidths = .01)
    nx.draw(Gr, pos=pos, edge_color = color, edge_cmap=cmap, width =5,node_size = 1, node_color = 'black',arrowstyle='->',arrowsize=25)
    # =============================================================================
#     for edge in Gr.edges(data='weight'):
#         nx.draw_networkx_edges(Gr, pos, edgelist=[edge], edge_color = color, edge_cmap=cmap,width =5, node_size = .1,arrowstyle='->',arrowsize=40)
#    
# =============================================================================
    
