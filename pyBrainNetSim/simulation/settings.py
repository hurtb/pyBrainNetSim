"""
settings.py

used to setup simulations

"""

POPULATION_SIZE = 2
ITERATIONS = 5

internal_props = {
    'number_neurons': 10,
    'excitatory_to_inhibitory': 1,
    'physical_distribution': None,
    'node_class': 'E',
    'value': 1,
    'energy_value': 10,
    'energy_consumption': 1.,
    'energy_dynamics': None,
    'threshold': None,
    'threshold_change_fx': None,
    'spontaneity': None,
    'inactive_period': None,
    'postsyn_signal': 0.
}

motor_props = {
    'number_of_units': 4,
    'force_direction': None,
    'node_class': None,
    'node_type': None,
    'energy_value': None,
    'energy_consumption': None,
    'energy_dynamics': None,
    'threshold': 1.,
    'threshold_change_fx': None,
    'spontaneity': None,
    'inactive_period': None,
    'value': 1.,
    'postsyn_signal': None
}

sensory_props = {
    'number_of_neurons': 4,
    'sensor_direction': None,
    'node_class': None,
    'node_type': None,
    'energy_value': None,
    'energy_consumption': None,
    'energy_dynamics': None,
    'inactive_period': None,
    'threshold': None,
    'threshold_change_fx': None,
    'spontaneity': None,
    'stimuli_sensitivity': None,
    'stimuli_max': 1,
    'stimuli_min': 0,
    'stimuli_mid': .5,
    'stimuli_fxn': None,
    'value': 0.,
    'signal': 0.,
    'postsyn_signal': 0.
}

weight_props = {
    'int_to_int': None,
    'edge_internal_min_cutoff': None,
    'int_to_motor': None,
    'edge_motor_min_cutoff': None,
    'sensor_to_int': None,
    'edge_sensor_min_cutoff': None,
    'sensory_max_connections': 1.,
    'motor_max_connections': 1.,
    'sensory_to_internal_max_connections': 1.,
    'int_to_motor_max_connections': 1.,
    'explicit_connections': None,
}