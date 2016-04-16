# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:12:04 2016

@author: brian
"""

import networkx as nx
import numpy as np
from network_properties import *


def excitatory_feedforward(N, value=0., spont_driver_prob=.8, spontaneity=.1, threshold=1., in_per=1., *args, **kwargs):
    """
    Produces a network motif
    :param N:
    :param value:
    :param spont_driver_prob:
    :param spontaneity:
    :param threshold:
    :param in_per:

    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(range(N - 1), range(1, N)))
    for n in G.nodes():
        G.node[n]['pos'] = (10. * n, 0)
        G.node[n].update(BASIC_EXCITATORY_NEURON)
    nx.set_node_attributes(G, 'value', value)
    nx.set_node_attributes(G, 'threshold', threshold)
    nx.set_node_attributes(G, 'inactive_period', in_per)
    nx.set_node_attributes(G, 'spontaneity', spontaneity)
    G.node[0]['spontaneity'] = spont_driver_prob
    #    G.node[0]['inactive_period'] = 0.
    G.name = "Excitatory FeedForward Motif"
    return G


def excitatory_circular(N, value=0., spont_driver_prob=.8, spontaneity=.9, threshold=1., in_per=1., *args, **kwargs):
    G = excitatory_feedforward(N, value, spont_driver_prob, spontaneity, threshold, in_per, *args, **kwargs)
    G.add_edge(N - 1, 0)
    return G


def inhibitory_feedback(spont_driver_prob=.8, threshold=1., in_per=1., *args, **kwargs):
    """
    Produces a network motif
    """
    G = nx.DiGraph()
    G.add_node(0, pos=(0, 0), **BASIC_EXCITATORY_NEURON)
    G.add_node(1, pos=(10, 0), **BASIC_EXCITATORY_NEURON)
    G.add_node(2, pos=(40, 0), **BASIC_EXCITATORY_NEURON)
    G.add_node(3, pos=(20, 10), **BASIC_INHIBITORY_NEURON)
    nx.set_node_attributes(G, 'threshold', threshold)
    G.node[0]['threshold'] = 1. - spont_driver_prob
    G.node[2]['threshold'] = 3 * threshold
    G.node[1]['threshold'] = 3 * threshold
    nx.set_node_attributes(G, 'inactive_period', in_per)
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 1, weight=-1)
    G.name = "Inhibitory Feedback Motif"
    return G
