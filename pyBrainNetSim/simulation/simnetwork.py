# -*- coding: utf-8 -*-
"""
simnetwork.py

classes used to simulate neuronal spikes through networks.
Created on Mon Dec 28 09:07:44 2015

@author: brian
"""

from pyBrainNetSim.models.network import *
import copy


class SimNetBase(object):
    """

    """
    def __init__(self, t0_network, initial_fire='rand', threshold=0.5, initial_n=None, prescribed=None,
                 data_cutoff=None, data_save_frequency=None, *args, **kwargs):
        self.simdata = []
        self.initial_net = NeuralNetData(t0_network, **kwargs)
        self.initial_nodes = self.initial_fire(mode=initial_fire, threshold=threshold, initial_n=initial_n,
                                               prescribed=prescribed)
        self.threshold = threshold
        self.is_initialized = False
        self.data_cutoff, self.data_save_frequency = data_cutoff, data_save_frequency
        
    def simulate(self, max_iter=5, **kwargs):
        """ Simulation mode of the network."""
        if not self.is_initialized:  # initialize and setup simulation parameters
            self.initiate_simulation(max_iter=max_iter, dt=1, **kwargs)
        while self.t < self.T and not self.simdata[-1].is_dead:  # simulation loop
            self.evolve_time_step()      

    def initiate_simulation(self, initial_network=None, t0=0, max_iter=5, dt=1): # setup parameters
        self.t, self.dt, self.n, self.N = t0, dt, 0, max_iter # global time, dt, max iter
        self.T = self.N * self.dt + t0
        self.simdata = NeuralNetSimData(t0=t0)
        self.simdata.append(NeuralNetData(self.initial_net.copy()))
        self.is_initialized = True

    def initial_fire(self, mode='rand', threshold=0.9, rand_N=None, initial_n=0.5, prescribed=None):
        if mode == 'rand':
            out = np.random.permutation(self.initial_net.nodes())[np.random.rand(self.initial_net.number_of_nodes()) < rand_N]
        elif mode == 'rand_pct_N':
            out = np.random.permutation(self.initial_net.nodes())[0:int(np.floor(rand_N * self.initial_net.number_of_nodes()))]
        elif mode == 'prescribed':
            out = prescribed
        return out
        
    def evolve_time_step(self, driven_nodes=None):
        """
        Main method to run one simulation time step. Each method call within this method can be overidden with
        custom methods. An example of this is the 'HebbianNetworkBasic' class.
        :param driven_nodes: A way to manually drive certain nodes in the model.
        :return: None
        """
        self.simdata.append(NeuralNetData(copy.deepcopy(self.simdata[-1]), time=self.t))
        if self.simdata[-1].is_dead:
            return
        nd = self.simdata[-1]
        nd.presyn_nodes = self.simdata[-2].postsyn_nodes
        nd.initialize()
        self.add_driven(nd, driven_nodes=driven_nodes) # Externally (forced) action potentials
        self.add_spontaneous(nd)  # add spontaneous firing
        self.find_inactive(nd)  # get potential post-synaptic nodes
        self.integrate_action_potentials(nd)  # integrate action potentials
        self.propagate_action_potentials(nd)  # Combine propagating and spontaneous action potentials
        nd.update_properties()  # update data structures
        self.synapse_plasticity(nd)  # Plasticity
        self.migration(nd)  # Migrate
        self.birth_death(nd)  # Neuron Birth
        self.update()

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
        """Fxn to setup spontaneous firing.
        :param NeuralNetData ng: pass the dat
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

    def propagate_action_potentials(self, ng):
        """ Propagate action potential from pre to post given the network's thresholds."""
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

    def multiply_value(self, attr, value=1., time_id=-1, node_id=None):
        nn = self.simdata[time_id]
        value = 1. if not isinstance(value, (int, float)) else value
        if node_id in nn:
            nn[node_id][attr] *= value
        else:
            for nid in nn.node.iterkeys():
                nn.node[nid][attr] *= value

    def add_value(self, attr, value=0., time_id=-1, node_id=None):
        value = 0. if not isinstance(value, (int, float)) else value
        if node_id in self.simdata[time_id]:
            self.simdata[time_id].node[node_id][attr] += value
        else:
            for nid, nn in self.simdata[time_id].node.iteritems():
                self.simdata[time_id].node[nid][attr] += value

    def update(self):
        for nid in self.simdata[-1].dead_nodes:
            for onid in self.simdata[-1].out_edges(nid):
                self.simdata[-1].edge[nid][onid[1]]['weight'] = 0.
        if isinstance(self.data_cutoff, (int, float)):
            if len(self.simdata) > self.data_cutoff:
                del self.simdata[1:(len(self.simdata) - self.data_cutoff)]  # pop the first item in list
        self.t += self.dt  # step forward in time
        self.n += 1


class SimNet(SimNetBase):
    def add_driven(self, ng, driven_nodes=None):
        """Set pre-synaptic driving neurons."""
        ng.driven_neurons = []  # use this in external fxn/class to set
    
    def add_spontaneous(self, ng):
        """Fxn to setup spontaneous firing."""
        ng.spont_signal = self.generate_spontaneous(ng)

    def find_inactive(self, ng):
        """Retrieves a list of neurons firing within the last 'inactive_period' for each neuron."""
        inact = []
        for nid, in_per in nx.get_node_attributes(ng, 'inactive_period').items():
            if in_per == 0.:
                continue
            if in_per > len(ng):
                in_per = len(ng)
            numfire = self.simdata.neuron_group_property_ts('presyn_vector')[nid][-int(in_per):].sum()
            if numfire > 0.:
                inact.append(nid)
        ng.inactive_nodes = inact
     
    def integrate_action_potentials(self, ng):
        """Integration of action potentials"""
        ng.postsyn_signal = np.multiply(ng.synapses.T, ng.presyn_vector).T.sum(axis=0).A[0]

    def propagate_action_potentials(self, ng):
        """Propagate AP from pre to post given the network's thresholds."""
        # thresh = nx.get_node_attributes(ng, 'threshold').values()
        thresh = ng.attr_vector('threshold')
        ng.afferent_signal = ng.postsyn_signal + ng.spont_signal + ng.driven_vector
        ng.prop_vector = (ng.afferent_signal > thresh).astype(float)
        AP_vector = np.multiply(ng.prop_vector, ng.active_vector)
        AP_vector = np.multiply(AP_vector, ng.alive_vector)
        ng.postsyn_nodes = ng.vector_to_nodeIDs(AP_vector)

    def generate_spontaneous(self, ng):
        """Generate spontaneous firing. Outputs a voltage based on the spontaneity (0-1, like a percentage) and the
        threshold number. Spontaneity=1."""
        out = []
        for nid in ng.nodes():
            out.append(0. if ng.node[nid]['spontaneity'] != 1. else 1.)
        out = np.array(out)
        out[[ng.nID_to_nIx[nid] for nid in ng.inactive_nodes]] = 0.
        out[[ng.nID_to_nIx[nid] for nid in ng.dead_nodes]] = 0.
        out = np.multiply(out, ng.active_vector)
        return out  # vector of numbers ordered by .nodes()


class HebbianNetworkBasic(SimNet):
    def __init__(self, t0_network, initial_fire='rand', threshold=0.5, initial_n=None, prescribed=None,
                 pos_synapse_growth=0.1, neg_synapse_growth=-0.05, data_cutoff=None, *args, **kwargs):
        super(HebbianNetworkBasic, self).__init__(t0_network, initial_fire='rand', threshold=threshold,
                 initial_N=None, prescribed=None, data_cutoff=data_cutoff, *args, **kwargs)
        self.pos_synapse_growth = pos_synapse_growth
        self.neg_synapse_growth = neg_synapse_growth

    def synapse_plasticity(self, ng, *args, **kwargs):
        """Basic Hebbian learning rule."""
        nodes = ng.nodes()
        eta_pos = self.pos_synapse_growth
        eta_neg = self.neg_synapse_growth
        x = np.matrix([nx.get_node_attributes(ng, 'value')[nID] for nID in nodes])
        xn = (x == 0).astype(float)  # nodes not firing; matrix xn.T * x represents nodes not firing
        dW = eta_pos * np.multiply(x.T * x, ng.abs_synapses > 0) \
             + eta_neg * np.multiply(xn.T * xn, ng.abs_synapses > 0)  # positive + negative reinforcement
        Wn = ng.abs_synapses + dW
        np.fill_diagonal(Wn, 0.)  # reinforce no auto-synapsing
        vals = dict(((nodes[i], nodes[j]), Wn[i, j]) for i in range(Wn.shape[1]) for j in range(Wn.shape[0])
                    if (Wn[i, j] > 0. and i != j))
        nx.set_edge_attributes(ng, 'weight', vals)
