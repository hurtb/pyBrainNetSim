import scipy.stats as stats
import pyBrainNetSim.generators.network_properties as props
# from pyBrainNetSim.generators.settings.base import NODE_PROPS

NODE_PROPS = {
    'Generic': {
        # 'node_class': 'Generic',
        'number_of_nodes': stats.binom(32, 0.8),
        'excitatory_to_inhibitory': stats.uniform(loc=0., scale=1.),  # Ratio of excitatory/inhibitory neurons. 0=100%
        # 'physical_distribution': 'Grid',
        # 'identical': True,
        # 'identical_within_type': True,
        # 'node_type_acceptable': ['E', 'I'],
        # 'identical_within_class': True,
        # 'threshold': 1.,
        # 'threshold_change_fxn': 'Future',
        'energy_value': stats.binom(50, .9),
        # 'energy_consumption': 1,
        # 'energy_dynamics': props.linear,
        'spontaneity': stats.uniform(loc=0., scale=1.),
        # 'inactive_period': 0,
        # 'value': 0.,
        # 'postsyn_signal': 0.,
    },
    'Internal': {
        # 'node_class': 'Internal',
        'number_of_nodes': stats.binom(32, 0.8),
        'excitatory_to_inhibitory': stats.uniform(loc=0., scale=1.),  # Ratio of excitatory/inhibitory neurons. 0=100%
        # 'physical_distribution': 'Grid',
        # 'identical': True,
        # 'identical_within_type': True,
        # 'identical_within_class': True,
        # 'threshold': 1.,
        # 'threshold_change_fxn': 'Future',
        'energy_value': stats.binom(50, .9),
        # 'energy_consumption': 1,
        # 'energy_dynamics': props.linear,
        'spontaneity': stats.uniform(loc=0., scale=1.),
        # 'inactive_period': 0,
        # 'value': 0.,
        # 'postsyn_signal': 0.,
    },
    'Motor': {
        # 'node_class': 'Motor',
        # 'number_of_nodes': 4,
        # 'physical_distribution': 'xstack',
        # 'node_type': 'E',
        # 'identical': True,
        # 'identical_within_type': True,
        # 'force_direction': [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)],  # unit vectors (2-D)
        'energy_value': stats.binom(50, .9),
        # 'energy_consumption': 1,
        # 'energy_dynamics': props.linear,
        # 'threshold': 1.,
        # 'threshold_change_fxn': 'Future',
        # 'spontaneity': .1,
        # 'inactive_period': 0,
        # 'value': 0.,
        # 'postsyn_signal': 0.,
    },
    'Sensory': {
        # 'node_class': 'Sensory',
        # 'number_of_nodes': 4,
        # 'physical_distribution': 'ystack',
        # 'node_type': 'E',
        # 'identical': True,
        # 'identical_within_type': True,
        # 'sensor_direction': [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)],
        'energy_value': stats.binom(50, .9),
        # 'energy_consumption': 1,
        # 'energy_dynamics': props.linear,
        # 'threshold': 1.,
        # 'threshold_change_fxn': 'Future',
        # 'spontaneity': .1,
        'stimuli_sensitivity': stats.poisson(1, loc=1),
        'stimuli_max': stats.norm(loc=1., scale=0.5),
        'stimuli_min': stats.norm(loc=0.2, scale=0.1),
        'sensory_mid': stats.norm(loc=0.2, scale=0.1),  # inflection point for signal
        'stimuli_fxn': props.sigmoid,
        # 'value': 0.,
        # 'signal': 0.,
        # 'postsyn_signal': 0.,
    },
}

SYNAPSE_DIST = stats.norm(loc=1., scale=0.5)
SYNAPSES = [
    {'class_node_to':'Internal',
     'class_node_from':'Internal',
     'weight': SYNAPSE_DIST,
     'identical': False,
     'minimum_cutoff': 0.2,
     'maximum_value': None
     },
    {'class_node_to': 'Internal',
     'class_node_from': 'Motor',
     'weight': SYNAPSE_DIST,
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
     'weight': SYNAPSE_DIST,
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
     'weight': None,
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

# # ----------------------------------------------
# # INTERNAL NODE POPULATION PROPERTIES
# INTERNAL_NODE_CLASS = 'Internal'
# INTERNAL_EXCIT_TO_INHIB = stats.uniform(loc=0., scale=1.)  # Ratio of excitatory to inhibitory neurons. 0=100% internal
# INTERNAL_NUMBER_OF_NODES = stats.binom(32, 0.8)
# INTERNAL_PHYSICAL_DISTRIBUTION = 'Grid'  # placeholder; to be used in future
# INTERNAL_IDENTICAL = False  # T/F whether  each internal neuron have constant intrinsic properties; overrides below
# INTERNAL_IDENTICAL_WITHIN_CLASS = True  # T/F whether each internal class of neuron have constant intrinsic properties
#
# # Internal neuron attributes
# INTERNAL_ENERGY = stats.binom(50, .9)
# INTERNAL_ENERGY_CONSUMPTION = 1.  #
# INTERNAL_ENERGY_DYNAMICS = props.linear
# INTERNAL_THRESHOLD = stats.uniform(loc=0.7, scale=1.2)
# INTERNAL_THRESHOLD_FXN = 'Future'  # placeholder; to
# INTERNAL_SPONTANEITY = stats.uniform(loc=0., scale=1.)  #
# INTERNAL_INACTIVE_PERIOD = 0.  # may change the way inactivation works in the future by hyperpolarization
# INTERNAL_VALUE = 0.  # initial value
#
# # ----------------------------------------------
# # MOTOR NODE POPULATION PROPERTIES
# MOTOR_NUMBER_OF_NODES = 4
# MOTOR_DIRECTION = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]  # unit vectors (2-D by default)
# MOTOR_NODE_CLASS = 'Motor'
# MOTOR_NODE_TYPE = 'E'
# MOTOR_IDENTICAL = True
#
# # Individual motor unit attributes
# MOTOR_ENERGY = stats.binom(50, .9)
# MOTOR_ENERGY_CONSUMPTION = 1.
# MOTOR_ENERGY_DYNAMICS = props.linear
# MOTOR_THRESHOLD = 1.
# MOTOR_THRESHOLD_FXN = 'Future'
# MOTOR_SPONTANEITY = 0.
# MOTOR_INACTIVE_PERIOD = 0.
# MOTOR_VALUE = 0.  # initial value
#
# # ----------------------------------------------
# # SENSORY NEURONS
# SENSORY_NUMBER_OF_NEURONS = 4
# SENSORY_NODE_CLASS = 'Sensory'
# SENSORY_NODE_TYPE = 'E'  #
# SENSORY_IDENTICAL = True  # Whether or not the population of sensory neurons have constant intrinsic properties
#
# # Individual sensory neuron attributes
# SENSORY_ENERGY = stats.binom(50, .9)
# SENSORY_ENERGY_CONSUMPTION = 1.
# SENSORY_ENERGY_DYNAMICS = props.linear
# SENSORY_INACTIVE_PERIOD = 0.
# SENSORY_THRESHOLD = 0.
# SENSORY_THRESHOLD_FXN = 'Future'  # placeholder; will use to alter sensory dynamics
# SENSORY_SPONTANEITY = 0.
# SENSORY_SENSITIVITY = stats.poisson(1, loc=1)
# SENSORY_MAX = stats.norm(loc=1., scale=0.5)
# SENSORY_MIN = stats.norm(loc=0.2, scale=0.1)
# STIMULI_MID = stats.norm(loc=0.2, scale=0.1)  # inflection point for signal
# SENSORY_TO_STIMULI_FXN = props.sigmoid
# SENSORY_SIGNAL0 = 0.  # initial signal
# SENSORY_VALUE0 = 0.  # initial value
