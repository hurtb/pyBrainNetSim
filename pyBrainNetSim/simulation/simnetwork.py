# -*- coding: utf-8 -*-
"""
simnetwork.py

classes used to simulate neuronal spikes through networks.
Created on Mon Dec 28 09:07:44 2015

@author: brian
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyBrainNetSim.models.network import *
from pyBrainNetSim.drawing.viewers import draw_networkx
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
        
    def simulate(self, max_iter=5, **kwargs):
        """ Simulation mode of the network."""
        if not self.is_initialized:  # initialize and setup simulation parameters
            self.initiate_simulation(max_iter, dt=1, **kwargs)
        while self.t < self.T and not self.simdata[-1].is_dead:  # simulation loop
            self.evolve_time_step()      

    def initiate_simulation(self, max_iter=5, dt=1): # setup parameters
        self.t, self.dt, self.n, self.N = 0, dt, 0, max_iter # global time, dt, max iter
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
        self.propagate_action_potential(nd)  # Combine propagating and spontaneous action potentials
        nd.update_properties()  # update data structures
        if self.simdata[-1].is_dead:
            return
        self.simdata.append(NeuralNetData(self.simdata[-1].copy(), time=self.t))
        nd = self.simdata[-1]
        self.synapse_plasticity(nd)  # Plasticity
        self.migration(nd)  # Migrate
        self.birth_death(nd)  # Neuron Birth
        self.simdata[-1].dead_nodes = self.simdata[-2].dead_nodes
        self.simdata[-1].presyn_nodes = self.simdata[-2].postsyn_nodes
        self.t += self.dt  # step forward in time
        self.n += 1

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

    def propagate_action_potential(self, ng):
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

    def draw_networkx(self, t=None, axs=None):
        """Draw networkx graphs for time points indicated"""
        max_cols, max_axs, color, alpha = 5, 20, '#D9F2FA', 0.5
        if t is None or not isinstance(t, [int, float, list, tuple, np.ndarray]):
            t = range(self.n + 1) if t is None or isinstance(t, [int, float]) else t  # get all points
        else:
            t = [t] if isinstance(t, [int, float]) else t  # get all points
        num_ax = len(t)
        if axs is None or num_ax != len(axs):
            rows = int(np.floor(num_ax / max_cols)) + 1
            cols = num_ax if num_ax <= max_cols else max_cols
            axs = [plt.subplot2grid((rows, cols), (i/cols, i % cols)) for i in range(num_ax)]

        for i, ax in enumerate(axs):
            # i_subg = self.simdata[-1].subgraph(self.simdata[-1].nodes('Internal'))
            # i_points = np.array([p for p in nx.get_node_attributes(i_subg, 'pos').itervalues()])
            # i_hull = ConvexHull(i_points)
            #
            # m_subg = self.simdata[-1].subgraph(self.simdata[-1].nodes('Motor'))
            # s_subg = self.simdata[-1].subgraph(self.simdata[-1].nodes('Sensory'))
            #

            axs[i] = draw_networkx(self.simdata[i], ax=ax)
            axs[i].axis('square')
            axs[i].set(xticklabels=[], yticklabels=[])
            # axs[i].add_patch(patches.Polygon([i_points[k] for k in i_hull.vertices], color=color, alpha=alpha))
            # for m_id, attr in m_subg.node.iteritems():
            #     axs[i].arrow(attr['pos'][0], attr['pos'][1], attr['force_direction'][0]/2, attr['force_direction'][1]/2,
            #                  head_width=0.05, head_length=0.1, fc='k', ec='k')

        plt.tight_layout(pad=0.05)
        return axs


class SimNet(SimNetBase):
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

    def propagate_action_potential(self, ng):
        """Propagate AP from pre to post given the network's thresholds."""
        thresh = nx.get_node_attributes(ng, 'threshold').values()
        ng.prop_vector = (ng.postsyn_signal + ng.spont_signal + ng.driven_vector > thresh).astype(float)
        AP_vector = np.multiply(ng.prop_vector, ng.active_vector)
        AP_vector = np.multiply(AP_vector, ng.alive_vector)
        ng.postsyn_nodes = ng.vector_to_nodeIDs(AP_vector)

    def generate_spontaneous(self, ng):
        """
        Generate spontaneous firing. Outputs a voltage based on the spontaneity (0-1, like a percentage) and the
        threshold number. Spontaneity=1
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
        return out  # vector of numbers ordered by .nodes()


class HebbianNetworkBasic(SimNet):
    def __init__(self, t0_network, initial_fire='rand', threshold=0.5,
                 initial_N=None, prescribed=None, pos_synapse_growth=0.1, neg_synapse_growth=-0.05, *args, **kwargs):
        super(HebbianNetworkBasic, self).__init__(t0_network, initial_fire='rand', threshold=0.5,
                 initial_N=None, prescribed=None, *args, **kwargs)
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
