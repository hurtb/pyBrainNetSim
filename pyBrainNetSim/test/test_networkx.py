# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 07:46:49 2015

@author: brian
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec


def draw_histogram(graph, ax):
    # Create histogram of node connections
    mybars = nx.degree_histogram(graph)
    mynodes = range(len(mybars))
    ax.bar(mynodes, mybars)
    ax.set_title("Histogram of Number of Edges per Node")
#    ax.show() # show histogram 
    return ax
    
def draw_power_law_network(num_nodes=100, num_edges_per_node=4, p_tri=0.5, ax=None):
    G = nx.powerlaw_cluster_graph(num_nodes, num_edges_per_node, p_tri)
    nx.draw_circular(G, ax=ax)
    return G

def draw_small_world_network(num_nodes, num_edges, seed=None, ax=None):
#    G = nx.barabasi_albert_graph(num_nodes, num_edges, seed=seed)
    G = nx.watts_strogatz_graph(num_nodes, num_edges, 0.5, seed=seed)
    nx.draw_circular(G, ax=ax)
    return G

if __name__ == '__main__':
    
    gridlen = (4,4)
    gs = gridspec.GridSpec(*gridlen)
    ax1 = plt.subplot(gs[:, 0:-1])
    ax2 = plt.subplot(gs[-1,-1])
#    fig, axarr = plt.subplots(nrows=1, ncols=2)

    # Create Powerlaw Graph
    num_nodes, num_edges_per_node, p_tri = 30, 4, 0.5
#    G = draw_power_law_network(num_nodes, num_edges_per_node, p_tri, axarr[0])   

    G = draw_small_world_network(num_nodes, num_edges_per_node, ax=ax1) 
 
    draw_histogram(G, ax2)
        
    
    
   