# -*- coding: utf-8 -*-
"""
simnetwork.py

classes used to simulate neuronal spikes through networks.
Created on Mon Dec 28 09:07:44 2015

@author: brian
"""


from pyBrainNetSim.models.network import *
from scipy.stats import uniform


class SimNetBase(object):
    """

    """
    def __init__(self, t0_network, initial_fire='rand', threshold=0.5,
                 initial_N=None, prescribed=None, *args, **kwargs):
        self.initial_net = NeuralNetData(t0_network,**kwargs)
        self.initial_nodes = self.initial_fire(mode=initial_fire, threshold=threshold,
                                               initial_N=initial_N,
                                               prescribed=prescribed)
        self.threshold = threshold
        self.is_initialized = False
        self.initiate_simulation()
        
    def simulate(self, **kwargs):
        """ Simulation mode of the network."""
        if not self.is_initialized:  # initialize and setup simulation parameters
            self.initiate_simulation(**kwargs)
        while self.t < self.T and not self.simdata[-1].is_dead:  # simulation loop
            self.evolve_time_step()      

    def initiate_simulation(self, max_iter=5, dt=1): # setup parameters
        self.t, self.dt, self.N = 0, dt, max_iter # global time, dt, max iter
        self.T = self.N * self.dt
        self.simdata = NeuralNetSimData()
        self.simdata.append(NeuralNetData(self.initial_net.copy()))
        self.is_initialized = True

    def initial_fire(self, mode='rand', threshold=0.9, rand_N=None, initial_N=0.5, prescribed=None):
        if mode == 'rand':
            out = np.random.permutation(self.initial_net.nodes())[np.random.rand(self.initial_net.number_of_nodes()) < rand_N]
        elif mode == 'rand_pct_N':
            out = np.random.permutation(self.initial_net.nodes())[0:int(np.floor(rand_N * self.initial_net.number_of_nodes()))]
        elif mode == 'prescribed':
            out = prescribed
        return out
        
    def evolve_time_step(self, driven_nodes=None):
        nd = self.simdata[-1]

        self.add_driven(nd, driven_nodes=driven_nodes) # Externally (forced) action potentials
        self.add_spontaneous(nd)  # add spontaneous firing
        self.find_inactive(nd)  # get potential post-synaptic nodes
        self.integrate_action_potentials(nd)  # integrate action potentials
        self.propigate_AP(nd)  # Combine propagating and spontaneous action potentials
        nd.update_properties()  # update data structures
        self.synapse_plasticity(nd)  # Plasticity
        self.migration(nd)  # Migrate
        self.birth_death(nd)  # Neuron Birth

        if self.simdata[-1].is_dead:
            return
        self.simdata.append(NeuralNetData(self.simdata[-1].copy(), time=self.t))
        self.simdata[-1].dead_nodes = self.simdata[-2].dead_nodes
        self.simdata[-1].presyn_nodes = self.simdata[-2].postsyn_nodes
        self.t += self.dt  # step forward in time

    def add_energy(self, amount):
        amount_per_neuron = np.floor(float(amount) / self.simdata[-1].number_of_nodes())
        for n_id in self.simdata[-1].nodes():
            self.simdata[-1].node[n_id]['energy_value'] += amount_per_neuron

    def add_driven(self, ng, driven_nodes=None):
        """Set pre-synaptic driving neurons.
        :param ng:
        :param driven_nodes:
        """
        pass
    
    def add_spontaneous(self, ng):
        """Fxn to setup spontaneuos firing.
        :param ng:
        """
        pass
        
    def find_inactive(self, ng):
        """Retrieves a list of neurons firing within the last 'inactive_period' for each neuron.
        :param ng:
        """
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
    

class SimNetBasic(SimNetBase):
    def add_driven(self, ng, driven_nodes=None):
        """Set pre-synaptic driving neurons."""
        ng.driven_neurons = []  # use this in external fxn/class to set
    
    def add_spontaneous(self, ng):
        """Fxn to setup spontaneous firing."""
        # ng.spont_nodes = ng.vector_to_nodeIDs(self.generate_spontaneous(ng))
        ng.spont_signal = self.generate_spontaneous(ng)

        
    def find_inactive(self, ng):
        """Retrieves a list of neurons firing within the last 'inactive_period' for each neuron."""
        inact = []
        for nID, in_per in nx.get_node_attributes(ng, 'inactive_period').items():
            if in_per == 0.:
                continue
            if in_per > len(ng):
                in_per = len(ng)
            numfire = self.simdata.node_group_properties('presyn_vector')[nID][-int(in_per):].sum()
            if numfire > 0.:
                inact.append(nID)
        ng.inactive_nodes = inact
     
    def integrate_action_potentials(self, ng):
        """Integration of action potentials"""
        ng.postsyn_signal = np.multiply(ng.synapses.T, ng.presyn_vector).T.sum(axis=0).A[0]

    def propigate_AP(self, ng):
        """Propigate AP from pre to post given the network's thresholds."""
        thresh = nx.get_node_attributes(ng, 'threshold').values()
        ng.prop_vector = (ng.postsyn_signal + ng.spont_signal + ng.driven_vector > thresh).astype(float)
        AP_vector = np.multiply(ng.prop_vector, ng.active_vector)
        AP_vector = np.multiply(AP_vector, ng.alive_vector)
        ng.postsyn_nodes = ng.vector_to_nodeIDs(AP_vector)

    def generate_spontaneous_thresh(self, ng):
        """
        Generate spontaneous firing. Uses a basic random number generator with
        thresholding. FUTURE: add random "voltage" to the presynaptic inputs.
        """
        out = (np.random.rand(len(ng.nodes())) < nx.get_node_attributes(ng, 'threshold').values()).astype(float)
        if ng.active_vector is None:
            return np.where(out == 1)[0]
        out[[ng.nID_to_nIx[n_id] for n_id in ng.dead_nodes]] = 0.
        out = np.multiply(out, ng.active_vector)
        return out  # vector of 0|1 ordered by .nodes()

    def generate_spontaneous(self, ng):
        """
        Generate spontaneous firing. Uses a basic random number generator with
        thresholding. FUTURE: add random "voltage" to the presynaptic inputs.
        """
        out = []
        for n_id in ng.nodes():
            smin = ng.node[n_id]['spontaneity']*ng.node[n_id]['threshold']
            smax = smin + 1.
            out.append(uniform(loc=smin, scale=smax).rvs())
        out = np.array(out)
        out[[ng.nID_to_nIx[n_id] for n_id in ng.inactive_nodes]] = 0.
        out[[ng.nID_to_nIx[n_id] for n_id in ng.dead_nodes]] = 0.
        out = np.multiply(out, ng.active_vector)
        return out  # vector of 0|1 ordered by .nodes()


class HebbianNetworkBasic(SimNetBasic):
    def synapse_plasticity(self, ng, *args, **kwargs):
        """Basic hebbian learning rule."""
        node = ng.nodes()
        eta_pos = 0.1
        eta_neg = -0.05
        x = np.matrix([nx.get_node_attributes(ng, 'value')[nID] for nID in ng.nodes()])
        dW = eta_pos * np.multiply(x.T * x, ng.synapses > 0)
        Wn = ng.synapses + dW
        vals = dict(((node[i],node[j]), Wn[i, j]) for i in range(Wn.shape[1]) for j in range(Wn.shape[0])
                    if (Wn[i, j]> 0. and i != j))
        nx.set_edge_attributes(ng, 'weight', vals)
