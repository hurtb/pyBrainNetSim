# # Top-level settings
# DEFAULT_NEURON_CLASS = stats.randint(low=0, high=3)
# DEFAULT_NEURON_TYPE = stats.randint(low=0, high=2)
# DEFAULT_THRESHOLD = stats.poisson(1, loc=1.)
# DEFAULT_SPONTANEITY = stats.norm(loc=0., scale=1.5)
# DEFAULT_ENERGY = stats.binom(50, .9)
# DEFAULT_ENERGY_CONSUMPTION = stats.poisson(1, loc=1.)
# ACCEPTABLE_NODE_CLASSES = ['Internal', 'Motor', 'Sensory']
#
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
#
# # ----------------------------------------------
# # INITIAL NETWORK CONNECTIVITY PROPERTIES
