# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 20:59:54 2016

@author: brian
"""
import sys
sys.path.append('..')
import pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
import networkx as nx
import numpy as np
from pyBrainNetSim.models import SimNet
from pyBrainNetSim.generators import inhibitory_feedback

scale = 1./255.
base_color = (252,246,200)
fire_color = scale * np.array([247,155,144])
inactive_color = .9 * np.array([1,1,1])

def get_fig_A(nd):

    nc = scale * np.array([base_color]*len(nd.nodes()))
    nc[nd.presyn_nodes] = fire_color
    nc[nd.inactive_nodes] = inactive_color
    
    out_e = [n for n in range(len(nd.edges())) if nd.edges()[n][0] in nd.presyn_nodes]
    ec = scale * np.array([base_color]*len(nd.edges()))
    ec[out_e] = fire_color
    
    nx.draw(nd, nx.get_node_attributes(nd,'pos'), node_color=nc, edge_color=ec,
            with_labels=True)
    
def get_fig_B(nd):
    scale = 1./255.    
    nc = scale * np.array([base_color]*len(nd.nodes()))
    nc[nd.presyn_nodes] = fire_color
    nc[nd.inactive_nodes] = inactive_color
    nc[nd.postsyn_nodes] = fire_color
    ec = scale * np.array([base_color]*len(nd.edges()))
    nx.draw(nd, nx.get_node_attributes(nd,'pos'), node_color=nc, edge_color=ec,
            with_labels=True)
            
def draw_net():
    
    pylab.show()
    plt.axis('equal')
    plt.legend(loc='center')
    per = perA = perB = .2
    for n in range(max_iterations):
        print n    
        d1 = ts.simdata[n]
        get_fig_A(d1)
        pylab.draw()
        pause(perA)
        plt.cla()
        
        get_fig_B(d1)
        pylab.draw()
        pause(perB)

#network = np.array([[0,3,0,1],[1,0,0,0],[1,1,0,1], [1,0,0,0]])
#network = nx.connected_watts_strogatz_graph(10, 3, 0.6)
#network = excitatory_feedforward(3, value=0., spont_driver_prob=0.99, threshold=.99, in_per=2)
network = inhibitory_feedback(spont_driver_prob=.95, threshold=0.3, in_per=0)
#network = hopfield_machine()

max_iterations = 18
init_fire = [0]
sim_options = {'max_iter':max_iterations, 'dt': 1, 'inactive_per':1}
ts = SimNet(network, initial_fire='prescribed', prescribed=init_fire, threshold=0.1)    
ts.simulate(**sim_options)
print ts.simdata.fire_data
print ts.simdata.active
d0 = ts.simdata[0]
d1 = ts.simdata[1]

draw_net()

