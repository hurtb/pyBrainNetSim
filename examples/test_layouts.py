import networkx as nx
from pyBrainNetSim.drawing.layouts import grid_layout
import pyBrainNetSim.simulation.simnetwork as sim
import pyBrainNetSim.generators.networks as rnd

internal_props = {'number_neurons': 10, 'excitatory_to_inhibitory': 0.01,
                  'spontaneity': 0.00, 'inactive_period': 0., 'threshold': 1.}
ipd = rnd.InternalNodeProperties(**internal_props)
sensor_props = {'threshold': 5.}
weight_props = {'int_to_int': 1., 'int_to_motor': 1., 'sensor_to_int': 0.5,
                'sensory_to_internal_max_connections': 1, 'int_to_motor_max_connections': 1.,
                'edge_motor_min_cutoff': .2, 'edge_sensor_min_cutoff': .2}

sm_prop_dist = rnd.SensorMoverProperties(internal=ipd,
                                         sensors=rnd.SensoryNodeProperties(**sensor_props),
                                         weights=rnd.EdgeProperties(**weight_props))
my_network = sim.HebbianNetworkBasic(sm_prop_dist.create_digraph(), pos_synapse_growth=0.0, neg_synapse_growth=-0.05)
my_network.simulate(max_iter=1)
g = grid_layout(my_network.simdata[-1])
print g
print nx.get_node_attributes(g, 'pos')