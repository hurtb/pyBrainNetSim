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
    def __init__(self, network=None, time=None, inactive_node=None, pre_fire=None, *args, **kwargs):
        super(NeuralNetData, self).__init__(network, *args)
        self.t = time
        self.inactive_nodes = self.set_default(inactive_node)
        self.presyn_nodes = []
        self.postsyn_nodes = []
        self.postsyn_signal = []
        # self.spont_nodes = []
        self.spont_signal = np.zeros(self.number_of_nodes())
        self.driven_nodes = []
        self.prop_vector = []
        self.dead_nodes = []

    @staticmethod
    def set_default(input_val):
        return [] if input_val is None else input_val
        
    def update_properties(self):
        self.__update_props(self.active_nodes, **{'state': 'active'})
        self.__update_props(self.inactive_nodes, **{'state': 'inactive', 'value': 0.})
        self.__update_props(self.postsyn_nodes, **{'state': 'post_fire', 'value': 0.})
        self.__update_props(self.presyn_nodes, **{'state': 'pre_fire', 'value': 1.})
        for nID in self.presyn_nodes:
            self.node[nID]['energy_value'] = self.node[nID]['energy_dynamics'](self.node[nID]['energy_value'],
                                                                               self.node[nID]['energy_consumption'])
        self.dead_nodes = [nID for nID in self.nodes() if self.node[nID]['energy_value'] <= 0.]
        for i, nID in enumerate(self.nodes()):
            self.node[nID]['postsyn_signal'] = self.postsyn_signal[i]

    def __update_props(self, nIDs, **kwargs):
        if nIDs is None:
            return
        for nID in nIDs:
            self.node[nID].update(**kwargs)

    def nodes(self, node_class=None):
        nodes = super(NeuralNetData, self).nodes()
        if isinstance(node_class, str):
            nodes = [n_id for n_id in nodes if self.node[n_id]['node_class'] == node_class]
        return nodes

    @property
    def synapses(self):
        weights = nx.to_numpy_matrix(self) # connection matrix with edges
        for nID, data in self.nodes_iter(data=True):
            if data['node_type'] == 'I':
                weights[self.nID_to_nIx[nID]] *= -1.
        for nID in self.dead_nodes:
            weights[self.nID_to_nIx[nID]] *= 0.
        return weights

    @property
    def abs_synapses(self):
        return np.abs(self.synapses)

    @property
    def nID_to_nIx(self):
        return {nID: i for i,nID in enumerate(self.nodes())}

    @property
    def nIx_to_nID(self):
        return {i: nID for i,nID in enumerate(self.nodes())}

    @property
    def active_nodes(self):
        return [self.nIx_to_nID[nIx] for nIx in np.where(self.inactive_vector == 0)[0]]

    @property
    def active_vector(self):        
        return self.nodeIDs_to_vector(self.active_nodes)

    def is_node_active(self, node_id):
        return True if node_id in self.active_nodes else False

    @property
    def inactive_vector(self):
        return self.nodeIDs_to_vector(self.inactive_nodes)

    @property
    def presyn_vector(self):
        return self.nodeIDs_to_vector(self.presyn_nodes)

    def is_node_firing(self, node_id):
        return True if node_id in self.presyn_nodes else False

    @property
    def postsyn_vector(self):
        return self.nodeIDs_to_vector(self.postsyn_nodes)

    @property
    def spont_vector(self):
        return self.spont_signal
        # return self.nodeIDs_to_vector(self.spont_nodes)

    @property
    def driven_vector(self):        
        return self.nodeIDs_to_vector(self.driven_nodes)

    @property
    def energy_consumption_vector(self):
        out = np.zeros(self.number_of_nodes())
        for i, n_id in enumerate(self.nodes()):
            if n_id in self.postsyn_nodes:
                out[i] = self.node[n_id]['energy_consumption']
        return out

    @property
    def energy_vector(self):
        return np.array([self.node[n_id]['energy_value'] for n_id in self.nodes()])

    @property
    def total_energy(self):
        return self.energy_vector.sum()

    @property
    def total_energy_consumed(self):
        return self.energy_consumption_vector.sum()

    @property
    def total_neurons(self):
        return self.number_of_nodes()

    @property
    def alive_vector(self):
        return (self.dead_vector == 0).astype(float)

    @property
    def alive_nodes(self):
        return self.vector_to_nodeIDs(self.alive_vector)

    def is_node_alive(self, node_id):
        return True if node_id in self.alive_nodes else False

    @property
    def dead_vector(self):
        return self.nodeIDs_to_vector(self.dead_nodes)

    def is_node_dead(self, node_id):
        return True if node_id in self.dead_nodes else False

    @property
    def is_dead(self):
        return True if np.all(self.dead_vector == 1.) else False

    @property
    def excitatory_to_inhibitory_ratio(self):
        nc = np.array([attr['node_type'] for attr in self.node.itervalues() if attr['node_class'] == 'Internal'])
        return (nc == 'E').sum().astype(np.float) / len(self.nodes(node_class='Internal'))

    @property
    def max_energy_consumption_per_time_period(self):
        return np.sum(nx.get_node_attributes(self, 'energy_consumption').values())

    def eval_min_time_periods_remaining(self, energy):
        return np.round(energy / self.max_energy_consumption_per_time_period)

    @property
    def min_time_periods_remaining(self):
        """Estimate of the total energy of the SensorMover divided by the maximum energy that could be consumed in one
         time period.
        .. math::
            \frac{\sum_{i \in all neurons} energy_i}{\sum_{all neurons} possible energy consumed per time step}
         = total_energy / maxim

         """
        return self.eval_min_time_periods_remaining(self.total_energy)

    def nodeIDs_to_vector(self, nIDs):
        vector = np.zeros(self.number_of_nodes())
        if nIDs is None or len(nIDs) == 0:
            return vector        
        vector[[self.nID_to_nIx[nID] for nID in nIDs]] = 1.
        return vector
        
    def vector_to_nodeIDs(self, vector):
        return np.array(self.nodes())[vector == 1.].tolist()

    def incoming_nodes(self, to_nid):
        w = nx.get_edge_attributes(self, 'weight')

        return [f_id for f_id, t_id in w.iterkeys() if t_id == to_nid]

    def __repr__(self):
        return "t: %s\nTotal Nodes: %d\nPre: %s\nInactive: %s\nPost: %s" \
            % (self.t, self.number_of_nodes(), self.presyn_nodes,
               self.inactive_nodes, self.postsyn_nodes)
        
     
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
    def __init__(self, t0=0, *args, **kwargs):
        super(NeuralNetSimData, self).__init__(*args, **kwargs)
        self._fire_data = {}  # nodeID: time_series list
        self._active = {}
        self.t0 = t0
    
    @property
    def fire_data(self):
        return pd.DataFrame(self._fire_data)
        
    @property
    def active(self):
        return pd.DataFrame(self._active)

    def neuron_group_property_ts(self, node_attribute):
        """
        Return a dataframe with the
        :param node_attribute:
        :return pandas DataFrame:
        """
        out = []
        for state in self:
            out.append(getattr(state, node_attribute))
        if isinstance(out[0], (list, tuple, np.ndarray)):
            out = pd.DataFrame(out, columns=self[-1].nodes())
        else:
            out = pd.Series(out, index=range(len(self)))
        return out
        # return pd.DataFrame(out, columns=sorted(self[-1].nodes()))

    def neuron_ts(self, prop, neuron_id=None):
        """
        Return a dict for the time-series data for the node's property.
        :param prop: any node property
        :param neuron_id: any neuron ID. Can be list, or string. If none, returns every node.
        :return: dict of
        """
        if prop is None:  # entire list of attributes;
            return []  # TEMPORARY
        ts = {}
        for i, nn in enumerate(self):
            t = i + self.t0  # adjust to world time
            for node, attr in nn.node.iteritems():
                if node not in ts:
                    ts.update({node: {}})
                if isinstance(prop, (list, tuple, np.ndarray)):
                    for pr in prop:
                        if pr not in ts[node]:
                            ts[node].update({pr: {}})
                        if pr not in attr:
                            ts[node][pr].update({t: None})
                        else:
                            ts[node][pr].update({t: attr[pr]})
                else:
                    if prop not in attr:
                        ts[node].update({t: None})
                    else:
                        ts[node].update({t: attr[prop]})
        out = ts
        if isinstance(neuron_id, (list, tuple, np.ndarray)):
            out = {n_id: ts[n_id] for n_id in neuron_id}
        elif isinstance(neuron_id, str):
            out = ts[neuron_id]
        return out

    def edge_ts(self, pairs, prop):
        """
        Return a dict for the time-series data for the connection edge's property.
        :param pairs: (node1, node2)
        :param prop: any edge attribute
        :return: dict
        """
        ts = {} # dict where {(afferent, efferent): [time series]}
        pairs = pairs if len(pairs) > 2 and len(pairs[0]) > 1 else [pairs]
        for i, nn in enumerate(self):
            t = i + self.t0  # adjust to world time
            ed = nx.get_edge_attributes(nn, prop)
            for pair in pairs:
                if pair not in ed:
                    continue
                if pair not in ts:
                    ts.update({pair: {}})
                ts[pair].append({t: ed[pair]})
        return ts

    @property
    def total_energy(self):
        return pd.DataFrame(self.neuron_ts('energy_value')).sum(axis=1)

    @property
    def total_neurons(self):
        return pd.Series({t: nn.number_of_nodes() for t, nn in enumerate(self)})

    @property
    def total_neurons_alive(self):
        return pd.Series({t: nn.alive_vector.sum() for t, nn in enumerate(self)})

    @property
    def total_neurons_dead(self):
        return pd.Series({t: nn.dead_vector.sum() for t, nn in enumerate(self)})

    @property
    def fraction_neurons_alive(self):
        return self.total_neurons_alive / self.total_neurons

    @property
    def fraction_neurons_dead(self):
        return 1. - self.fraction_neurons_alive
