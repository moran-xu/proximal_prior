import hamiltorch
import statsmodels.api as sm
import pandas as pd
from igraph import Graph
import numpy as np
import matplotlib.pyplot as plt
import torch
from numba import jit, float32, int32
from torch.optim.lr_scheduler import LambdaLR


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
from matplotlib import cm
import igraph
import networkx as nx
nodes_sorted = nodes.sort_values(by=['node_id']) 
x_coord, y_coord = nodes_sorted['x'],nodes_sorted['y']
layout = zip(list(x_coord.to_numpy()), list(-y_coord.to_numpy())) 
nodes_pos = (nodes_sorted[['x','y']])
pos_array = []
for i in range(len(nodes_pos)):
    pos_array.append([i,[nodes_pos.iloc[i]['x'],nodes_pos.iloc[i]['y']]])
pos= dict(pos_array)  
plt.figure()
a =  G
G = igraph.Graph.Adjacency(a.tolist())
e = G.get_edgelist()
Gr = nx.DiGraph(e)  
Gr.to_undirected()  
for _ in Gr.edges:
    Gr[_[0]][_[1]]['weight'] = a[_]
#print(G.nodes)
for n in Gr.nodes:
    Gr.nodes[n]['pos'] = (nodes_sorted.iloc[n, -2], nodes_sorted.iloc[n,-1])
widths = nx.get_edge_attributes(Gr, 'weight')
nx.draw(Gr,pos=pos,alpha=.3, node_size = 1, node_color = 'black', edge_color = 'gray',linewidths = .01) 
#plt.savefig("map"+str(ni)+"orange.png", format="PNG")
