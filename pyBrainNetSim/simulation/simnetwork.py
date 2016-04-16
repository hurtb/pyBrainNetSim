# -*- coding: utf-8 -*-
"""
simnetwork.py

classes used to simulate neuronal spikes through networks.
Created on Mon Dec 28 09:07:44 2015

@author: brian
"""

import networkx as nx
import numpy as np
import pandas as pd
from pyBrainNetSim.models.network import *


class SimNetBase(nx.DiGraph):
    """
    
    """
    def __init__(self, t0_network, initial_fire='rand', threshold=0.5,
                 initial_N=None, prescribed=None, *args, **kwargs):
        super(SimNetBase, self).__init__(t0_network, *args)
        self.initial_net = nx.DiGraph(t0_network,**kwargs)
        self.initial_nodes = self.initial_fire(mode=initial_fire,
                                               threshold=threshold,
                                               initial_N=initial_N,
                                               prescribed=prescribed)
        self.threshold = threshold
        self.is_initialized = False
        
    def simulate(self, **kwargs):
        """ Simulation mode of the network."""
        # initialize and setup simulation parameters
        if not self.is_initialized:        
            self.initiate_simulation(**kwargs)
        # simulation loop
        while self.t < self.T and not self.simdata[-1].is_dead:
            self.evolve_time_step()      

    def initiate_simulation(self, max_iter=5, dt=1):
        # setup parameters
        self.t, self.dt, self.N = 0, dt, max_iter # global time, dt, max iter
        self.T = self.N * self.dt
        self.simdata = NeuralNetSimData()
        self.is_initialized = True
        
    def start_timestep(self):
        if self.t ==0:
            self.simdata.append(NeuralNetData(self.initial_net.copy(), time=self.t))
            self.simdata[-1].presyn_nodes = self.initial_nodes
        else:
            self.simdata.append(NeuralNetData(self.simdata[-1].copy(), time=self.t))
            self.simdata[-1].presyn_nodes = self.simdata[-2].postsyn_nodes
            
    def initial_fire(self, mode='rand', threshold=0.9, rand_N=None, initial_N=0.5, prescribed=None):
        if mode == 'rand':
            out = np.random.permutation(self.nodes())[np.random.rand(self.number_of_nodes()) < rand_N]
        elif mode == 'rand_pct_N':
            out = np.random.permutation(self.nodes())[0:int(np.floor(rand_N * self.number_of_nodes()))]
        elif mode == 'prescribed':
            out = prescribed
        return out
        
    def evolve_time_step(self, driven_nodes=None):
        if self.t > 0.:
            if self.simdata[-1].is_dead:
                return
        self.start_timestep()
        nd = self.simdata[-1]

        self.add_driven(nd, driven_nodes=driven_nodes) # Externally (forced) action potentials
        self.add_spontaneous(nd) # add spontaneous firing
        self.find_inactive(nd) # get potential post-synaptic nodes
        self.integrate_action_potentials(nd) # integrate action potentials
        self.propigate_AP(nd) # Combine propigating and spontaneous action potentials
        nd.update_properties()  # update data structures
        self.synapse_plasticity(nd) # Plasticity
        self.migration(nd) # Migrate
        self.birth_death(nd) # Neuron Birth

#        self.simdata.update()
        
        self.t += self.dt  # step forward in time
        
    def add_driven(self, ng, driven_nodes=None):
        """Set pre-synaptic driving neurons."""
        pass
    
    def add_spontaneous(self, ng):
        """Fxn to setup spontaneuos firing."""
        pass
        
    def find_inactive(self, ng):
        """Retrieves a list of neurons firing within the last 'inactive_period' for each neuron."""
        pass
     
    def integrate_action_potentials(self, ng):
        """Integration of action potentials"""
        pass

    def propigate_AP(self, ng):
        """ Propagate AP from pre to post given the network's thresholds."""
        pass
        
    def synapse_plasticity(self, ng, *args, **kwargs):
        """Apply plasticity rules."""
        pass
    
    def migration(self, ng, *args, **kwargs):
        """Alter (x,y,[z]) position of neurons."""        
        pass
    
    def birth_death(self, ng, *args, **kwargs):
        """Add/subtract neurons."""        
        pass
    
    def generate_spontaneous(self, threshold=0.9, active_vector=None):
        """ 
        Generate spontaneous firing. Uses a basic random number generator with
        thresholding. FUTURE: add random "voltage" to the presynaptic inputs.
        """
        out = (np.random.rand(len(self.nodes())) < threshold).astype(float)
        if active_vector is None:
            return np.where(out == 1)[0]
        return np.multiply(out, active_vector)  # vector of 0|1 ordered by .nodes()
    

class SimNetBasic(SimNetBase):
    def add_driven(self, ng, driven_nodes=None):
        """Set pre-synaptic driving neurons."""
        ng.driven_neurons = []  # use this in external fxn/class to set
    
    def add_spontaneous(self, ng):
        """Fxn to setup spontaneous firing."""
        thresh = nx.get_node_attributes(ng, 'spontaneity').values()
        ng.spont_nodes = ng.vector_to_nodeIDs(
            self.generate_spontaneous(threshold=thresh, active_vector=ng.active_vector))
        
    def find_inactive(self, ng):
        """Retrieves a list of neurons firing within the last 'inactive_period' for each neuron."""
        inact = []
        for nID, in_per in nx.get_node_attributes(ng, 'inactive_period').items():
            if in_per == 0.:
                continue
            if in_per > len(self):
                in_per = len(self)
            numfire = self.simdata.get_node_dynamics('presyn_vector')[nID][-int(in_per):].sum()
            if numfire > 0.:
                inact.append(nID)
        ng.inactive_nodes = inact
     
    def integrate_action_potentials(self, ng):
        """Integration of action potentials"""
        ng.postsyn_signal = np.multiply(ng.synapses.T, ng.presyn_vector).T.sum(axis=0).A[0]

    def propigate_AP(self, ng):
        """Propigate AP from pre to post given the network's thresholds."""
        thresh = nx.get_node_attributes(ng, 'threshold').values()
        ng.prop_vector = (ng.postsyn_signal + ng.spont_vector + ng.driven_vector > thresh).astype(float)
        AP_vector = np.multiply(ng.prop_vector, ng.active_vector)
        ng.postsyn_nodes = ng.vector_to_nodeIDs(AP_vector)


class HebbianNetworkBasic(SimNetBasic):
    def synapse_plasticity(self, ng, *args, **kwargs):
        """Basic hebbian learning rule."""
        node = ng.nodes()
        eta_pos = 0.1
        eta_neg = -0.05
        x = np.matrix([nx.get_node_attributes(ng, 'value')[nID] for nID in ng.nodes()])
        dW = eta_pos * np.multiply(x.T * x, ng.synapses > 0)
        Wn = ng.synapses + dW
        vals = dict(((node[i],node[j]), Wn[i,j]) \
            for i in range(Wn.shape[1]) \
            for j in range(Wn.shape[0]) if (Wn[i,j]>0. and i != j))
        nx.set_edge_attributes(ng, 'weight', vals)