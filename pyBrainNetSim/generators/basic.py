

import networkx as nx
import numpy as np
from network_properties import *
from random import random_internal_neuron_prop, random_sensory_neuron_prop, random_motor_neuron_prop, random_neuron_props
from scipy.stats import lognorm, bernoulli

EDGE_WEIGHT_DISTRIBUTION = lognorm(0.5)


def __generate_weights(weights, num_edges):
    if weights == 'rand':
        my_weights = EDGE_WEIGHT_DISTRIBUTION.rvs(size=num_edges)
    elif isinstance(weights, (list, np.ndarray)) and len(weights) >= num_edges:
        my_weights = weights
    elif isinstance(weights, (int, float)):
        my_weights = weights * np.ones(num_edges)
    else:
        my_weights = np.ones(num_edges)
    return my_weights


def __set_edge_weights(G, weights):
    i = 0
    for n in range(G.number_of_nodes()):
        for m in range(G.number_of_nodes()):
            if n != m:
                G.add_edge('I%d' % n, 'I%d' % m, weight=weights[i])
                i += 1
    return G


def __setup_positions(num_neurons):
    mylen = np.ceil(np.sqrt(num_neurons)) + 1
    x, y = np.mgrid[1:mylen:1, 1:mylen:1]
    i_pos = np.vstack([x.ravel(), y.ravel()]).T
    return i_pos


def network_noE(internal_neurons=16, node_props=None, weights=None, **kwargs):
    """
    Produces a network where firing & transmitting do not require energy.
    :param internal_neurons:
    :param node_props:
    :param weights:
    """
    G = nx.DiGraph()
    i_pos = __setup_positions(internal_neurons)  # setup positions
    myprops = node_props if node_props is not None and isinstance(node_props, 'dict') else I_INTERNAL
    my_weights = __generate_weights(weights, internal_neurons ** 2 - internal_neurons)
    for k in range(internal_neurons):
        G.add_node('I%d' % k, pos=i_pos[k], **myprops)
    G = __set_edge_weights(G, my_weights)
    G.name = "Regular Network"
    return G


def network_withE(internal_neurons=16, excitatory_props=None, inhibitory_props=None, weights=None, **kwargs):
    """
    Produces a network where neurons require energy to recieve/fire/transmit signals.
    :param internal_neurons: Number of neurons
    :param excitatory_props: properties of excitatory neurons. Default to 'BASIC_EXCITATORY_NEURON' if 'None'
    :param inhibitory_props: properties of excitatory neurons. Default to 'BASIC_INHIBITORY_NEURON' if 'None'
    :param props:
    :param weights:

    """

    G = network_noE(internal_neurons, props, weights, **kwargs)
    for nID in G.nodes():
        G.node[nID].update({'energy_value': INTERNAL_ENERGY0,
                            'energy_consumption': INTERNAL_DELTA_ENERGY,
                            'energy_dynamics': EnergyDynamics.linear})
    G.name = "Energy Consuming Network"
    return G


def sensor_mover(sensor_dir=None, motor_dir=None, internal_neurons=16, props=None, weights=None, **kwargs):
    """
    Produces a network with Sensory and Motor function. To be applied within an "environment"

    todo: Work on making the functional parameters more flexible.
    """
    s_pos = range(1, len(sensor_dir) + 1)
    m_pos = range(1, len(motor_dir) + 1)
    G = network_withE(internal_neurons, props, weights, **kwargs)

    for i, sdir in enumerate(sensor_dir):
        G.add_node('S%d' % i, pos=(0, s_pos[i]), sensor_direction=sdir, **BASIC_SENSORY_NEURON_PROP)
        G.node['S%d' % i].update({'energy_value': SENSORY_ENERGY0,
                                  'energy_consumption': SENSORY_DELTA_ENERGY,
                                  'energy_dynamics': EnergyDynamics.linear})
        G.add_edge('S%d' % i, 'I%d' % i, weight=1.)

    for j, fdir in enumerate(motor_dir):  # alter for i,dir in ...:
        G.add_node('M%d' % j, pos=(m_pos[j], 0), force_direction=fdir, **BASIC_MOTOR_NEURON_PROPS)
        G.node['M%d' % j].update({'energy_value': MOTOR_ENERGY0,
                                  'energy_consumption': MOTOR_DELTA_ENERGY,
                                  'energy_dynamics': EnergyDynamics.linear})
        G.add_edge('I%d' % max([internal_neurons - j - 1, 0]), 'M%d' % j, weight=1.)

    G.name = "Hopfield Machine"
    return G

# MARKED FOR DELETION
# def rand_internal_network(internal_neurons=16, pct_excit=0.5, edge_weights='rand', **kwargs):
#     G = nx.DiGraph()
#     i_pos = __setup_positions(internal_neurons)  # setup positions
#     my_weights = __generate_weights(edge_weights, internal_neurons ** 2 - internal_neurons)
#     node_polarity = bernoulli(pct_excit)  # random variable
#     for i in range(internal_neurons):
#         G.add_node('I%d' % i, pos=i_pos[i], **random_internal_neuron_prop(ntype=node_polarity, **kwargs).copy())
#     G = __set_edge_weights(G, my_weights)
#     return G
#
#
# def rand_sensor_mover(internal_neurons=16, edge_weights=None, sensor_dir=None, motor_dir=None, **kwargs):
#     G = rand_internal_network(internal_neurons=internal_neurons, edge_weights=edge_weights, **kwargs)
#     s_pos, m_pos = range(1, len(sensor_dir) + 1), range(1, len(motor_dir) + 1)
#     sensor_props = random_sensory_neuron_prop().copy()
#     motor_props = random_motor_neuron_prop().copy()
#     for i, sdir in enumerate(sensor_dir):
#         G.add_node('S%d' % i, pos=(0, s_pos[i]), sensor_direction=sdir, **sensor_props.copy())
#         G.add_edge('S%d' % i, 'I%d' % i, weight=1.)
#
#     for j, fdir in enumerate(motor_dir):  # alter for i,dir in ...:
#         G.add_node('M%d' % j, pos=(m_pos[j], 0), force_direction=fdir, **motor_props.copy())
#         G.add_edge('I%d' % max([internal_neurons - j - 1, 0]), 'M%d' % j, weight=1.)
#
#     G.name = "Hopfield Machine"
#     return G
#
