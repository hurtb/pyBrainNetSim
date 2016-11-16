import scipy.stats as stats
import pyBrainNetSim.generators.special_functions as sfunct


NODE_PROPS = {
    'Generic': {
        'node_class': 'Generic',
        'number_of_nodes': 24,
        'excitatory_to_inhibitory': .5,
        'physical_distribution': {'type': 'Grid', 'size': 10},
        'identical': True,
        'identical_within_type': True,
        'node_type_acceptable': ['E', 'I'],
        'identical_within_class': True,
        'threshold': 1.,
        'threshold_change_fxn': 'Future',
        'energy_value': 50,
        'energy_consumption': 1,
        'energy_dynamics': sfunct.linear,
        'spontaneity': 0.,
        'inactive_period': 0,
        'value': 0.,
        'postsyn_signal': 0.,
        'max_outgoing': {'Generic': None, 'Internal': None, 'Motor': None, 'Sensory': None},
        'max_incoming': {'Generic': None, 'Internal': None, 'Motor': None, 'Sensory': None},
    },
    'Internal': {
        'node_class': 'Internal',
        'number_of_nodes': 24,
        'excitatory_to_inhibitory': .5,
        'physical_distribution': {'type': 'Grid', 'size': 10.},
        'identical': True,
        'identical_within_type': True,
        'identical_within_class': True,
        'threshold': 1.,
        'threshold_change_fxn': 'Future',
        'energy_value': 50,
        'energy_consumption': 1,
        'energy_dynamics': sfunct.linear,
        'spontaneity': 0.,
        'inactive_period': 0,
        'value': 0.,
        'postsyn_signal': 0.,
        'max_outgoing': {'Generic': None, 'Internal': None, 'Motor': None, 'Sensory': None},
        'max_incoming': {'Generic': None, 'Internal': None, 'Motor': None, 'Sensory': None},
    },
    'Motor': {
        'node_class': 'Motor',
        'number_of_nodes': 4,
        'physical_distribution': {'type':'by_direction', 'offset': -1., 'extra_distance': 1.},
        'node_type': 'E',
        'identical': True,
        'identical_within_type': True,
        'direction': [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)],  # unit vectors (2-D)
        'energy_value': 50,
        'energy_consumption': 1,
        'energy_dynamics': sfunct.linear,
        'threshold': 1.,
        'threshold_change_fxn': 'Future',
        'spontaneity': .1,
        'inactive_period': 0,
        'value': 0.,
        'postsyn_signal': 0.,
        'max_outgoing': {'Generic': 0, 'Internal': 0, 'Motor': 0, 'Sensory': 0},
        'max_incoming': {'Generic': 1, 'Internal': 1, 'Motor': 0, 'Sensory': 0},
    },
    'Sensory': {
        'node_class': 'Sensory',
        'number_of_nodes': 4,
        'physical_distribution': {'type':'by_direction', 'offset': 1., 'extra_distance': 1.},
        'node_type': 'E',
        'identical': True,
        'identical_within_type': True,
        'direction': [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)],
        'energy_value': 50,
        'energy_consumption': 1,
        'energy_dynamics': sfunct.linear,
        'threshold': 1.,
        'threshold_change_fxn': 'Future',
        'spontaneity': .1,
        'stimuli_sensitivity': 1.,  # kappa
        'stimuli_max': 1.,  # maximum probability of firing
        'stimuli_min': 0.,  # minimum probability of firing
        'sensory_mid': 0.5,  # inflection point for signal
        'stimuli_fxn': sfunct.sigmoid,  # mapping function of stimuli to probability of firing
        'value': 0.,
        'signal': 0.,
        'postsyn_signal': 0.,
        'max_outgoing': {'Generic': 1, 'Internal': 1, 'Motor': 0, 'Sensory': 0},
        'max_incoming': {'Generic': 0, 'Internal': 0, 'Motor': 0, 'Sensory': 0},
    },
}

SYNAPSE_DIST = 1.
SYNAPSES = [
    {'class_node_to':'Internal',
     'class_node_from':'Internal',
     'weight': SYNAPSE_DIST,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Internal',
     'class_node_from': 'Motor',
     'weight': None,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Internal',
     'class_node_from': 'Sensory',
     'weight': SYNAPSE_DIST,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Sensory',
     'class_node_from': 'Internal',
     'weight': None,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Sensory',
     'class_node_from': 'Sensory',
     'weight': None,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Sensory',
     'class_node_from': 'Motor',
     'weight': None,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Motor',
     'class_node_from': 'Internal',
     'weight': SYNAPSE_DIST,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Motor',
     'class_node_from': 'Sensory',
     'weight': None,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Motor',
     'class_node_from': 'Motor',
     'weight': None,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    {'class_node_to': 'Undefined',
     'class_node_from': 'Undefined',
     'weight': 1.,
     'identical': True,
     'minimum_cutoff': 0.2,
     'maximum_value': None},
    ]

ENVIRONMENT = {'origin': (-10., -10.), 'maximum_point': (10., 10.), 'permeability': 1.}
ATTRACTOR = {'position': (2., 2.), 'strength': 10.}
POPULATION = {'initial_size': 10, 'reproductive_cost': 0.1, 'reproductive_threshold': 20}
TIME_ITERATIONS = 20

