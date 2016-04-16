# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:40:51 2016

@author: brian
"""
import networkx as nx
import numpy as np
import pandas as pd


class NeuralNetData(nx.DiGraph):
    """
    Data structure holding the neural network data state. This is a networkx
    graph with added features for a real temporal neural network
    
    
    """
    def __init__(self, network, time=None, inactive_node=None, pre_fire=None, *args, **kwargs):
        super(NeuralNetData, self).__init__(network, *args)
        self.t = time
        self.inactive_nodes = self.set_default(inactive_node)
        self.presyn_nodes = None
        self.postsyn_nodes = None
        self.postsyn_signal = None
        self.spont_nodes = None
        self.driven_nodes = None
        self.prop_vector = None
        self.dead_nodes = None
        
    def set_default(self, input_val):
        return None if input_val is None else input_val
        
    def update_properties(self):
        self.__update_props(self.active_nodes, **{'state': 'active'})
        self.__update_props(self.inactive_nodes, **{'state': 'inactive', 'value': 0.})
        self.__update_props(self.postsyn_nodes, **{'state': 'post_fire', 'value': 0.})
        self.__update_props(self.presyn_nodes, **{'state': 'pre_fire', 'value': 1.})
        for nID in self.presyn_nodes:
            self.node[nID]['energy_value'] = self.node[nID]['energy_dynamics'](self.node[nID]['energy_value'],
                                                                               self.node[nID]['energy_consumption'])
        self.dead_nodes = [nID for nID in self.nodes() if self.node[nID]['energy_value'] <=0.]

    def __update_props(self, nIDs, **kwargs):
        if nIDs is None:
            return
        for nID in nIDs:
            self.node[nID].update(**kwargs)

    @property
    def synapses(self):
        weights = nx.to_numpy_matrix(self) # connection matrix with edges
        for nID, data in self.nodes_iter(data=True):
            if data['node_type'] == 'I':
                print "Inhibitory node: %s" % nID
                weights[self.nID_to_nIx[nID]] *= -1.

        for nID in self.dead_nodes:
            weights[self.nID_to_nIx[nID]] *= 0.
        return weights
    @property
    def nID_to_nIx(self):
        return {nID:i for i,nID in enumerate(self.nodes())}
    @property
    def nIx_to_nID(self):
        return {i:nID for i,nID in enumerate(self.nodes())}
    @property
    def active_nodes(self):
        return [self.nIx_to_nID[nIx] for nIx in np.where(self.inactive_vector == 0)[0]]
    @property
    def active_vector(self):        
        return self.nodeIDs_to_vector(self.active_nodes)
    @property
    def inactive_vector(self):
        return self.nodeIDs_to_vector(self.inactive_nodes)
    @property
    def presyn_vector(self):
        return self.nodeIDs_to_vector(self.presyn_nodes)
    @property
    def postsyn_vector(self):
        return self.nodeIDs_to_vector(self.postsyn_nodes)
    @property
    def spont_vector(self):
        return self.nodeIDs_to_vector(self.spont_nodes)
    @property
    def driven_vector(self):        
        return self.nodeIDs_to_vector(self.driven_nodes)
    @property
    def dead_vector(self):
        return self.nodeIDs_to_vector(self.dead_nodes)
    @property
    def is_dead(self):
        return True if np.all(self.dead_vector == 1.) else False
        
    def nodeIDs_to_vector(self, nIDs):
        vector = np.zeros(self.number_of_nodes())
        if nIDs is None or len(nIDs) == 0:
            return vector        
        vector[[self.nID_to_nIx[nID] for nID in nIDs]] = 1.
        return vector
        
    def vector_to_nodeIDs(self, vector):
        return np.array(self.nodes())[vector==1.].tolist()          

    def __repr__(self):
        return "t: %d\nTotal Nodes: %d\nPre: %s\nInactive: %s\nSpont: %s\nPost: %s" \
            %(self.t, self.number_of_nodes(), self.presyn_nodes, \
              self.inactive_nodes, self.spont_nodes, self.postsyn_nodes)
        
     
class NeuralNetSimData(list):
    """
    Collection of the simulation parameters and simulation data
    
    examples data from time 4ms
    {t: 4 ms,
        NeuralNetData: overidden networkx graph
        ...
     ...
     }  
    """
    def __init__(self, *args, **kwargs):
        super(NeuralNetSimData, self).__init__(*args, **kwargs)
        self._fire_data = {}  # nodeID: time_series list
        self._active = {}          
    
    @property
    def fire_data(self):
        return pd.DataFrame(self._fire_data)
        
    @property
    def active(self):
        return pd.DataFrame(self._active)
       
    def get_node_dynamics(self, aname):
        out = []        
        for state in self:
            out.append(getattr(state, aname))
        return pd.DataFrame(out, columns=self[-1])
        
    def neuron_ts(self, neuronID=None, prop=None):
        if neuronID is not None:
            ts = {}  # dict where neuronID is the top key
            for i in range(len(self)):
                if not ts.has_key(neuronID):
                    ts.update({neuronID:[]})
                ts[neuronID].append((i, self[i].node[neuronID][prop]))
        else:
            print "HERE"
            ts = {}
#            if prop is None:
#                ts = {} # array of props
#                
#            for i in range(len(self)):
#                if neuronID not in self[i].nodes():
#                    ts
#                    if not ts.has_key(nID):
#                        ts.update({nID:[]})
#                    ts[nID].append((i,prop))
                    
        return ts
    @property
    def edge_ts(self):
        ts = {} # dict where neuronID is the top key
        for i in range(len(self)):
            if not ts.has_key(neuronID):
                ts.update({neuronID:[]})
            ts[neuronID].append((i,self[i].node[neuronID][prop]))