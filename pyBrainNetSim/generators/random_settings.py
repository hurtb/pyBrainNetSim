from scipy.stats import norm, binom, randint, poisson, uniform
from network_properties import *

DEFAULT_NEURON_CLASS = randint(low=0, high=3)
DEFAULT_NEURON_TYPE = randint(low=0, high=2)
DEFAULT_THRESHOLD = poisson(1, loc=1.)
DEFAULT_SPONTANEITY = norm(loc=0., scale=1.5)
DEFAULT_ENERGY = binom(50, .9)
DEFAULT_ENERGY_CONSUMPTION = poisson(1, loc=1.)

# ----------------------------------------------

# Internal node population properties
INTERNAL_NODE_CLASS = 'Internal'
INTERNAL_EXCIT_TO_INHIB = uniform(loc=0., scale=1.)  # Ratio of excitatory to inhibitory neurons. 0=100% internal
INTERNAL_NUMBER_OF_NEURONS = binom(32, 0.5)
INTERNAL_PHYSICAL_DISTRIBUTION = 'Grid'  # placeholder; to be used in future

# Individual neurons
INTERNAL_ENERGY = binom(50, .9)
INTERNAL_ENERGY_CONSUMPTION = 1.  #
INTERNAL_ENERGY_DYNAMICS = EnergyDynamics.linear
INTERNAL_THRESHOLD = uniform(loc=0.7, scale=1.2)
INTERNAL_THRESHOLD_FXN = 'Future'  # placeholder; to
INTERNAL_SPONTANEITY = uniform(loc=0.7, scale=1.2)
INTERNAL_INACTIVE_PERIOD = 0.  # may change the way inactivation works in the future by hyperpolarization
INTERNAL_VALUE = 0.  # initial value
# ----------------------------------------------

# Motor node population properties
MOTOR_NUMBER_OF_UNITS = 4
MOTOR_DIRECTION = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]

MOTOR_NODE_CLASS = 'Motor'
MOTOR_NODE_TYPE = 'E'
MOTOR_ENERGY = binom(50, .9)
MOTOR_ENERGY_CONSUMPTION = 1.
MOTOR_ENERGY_DYNAMICS = EnergyDynamics.linear
MOTOR_THRESHOLD = 0.
MOTOR_THRESHOLD_FXN = 'Future'
MOTOR_SPONTANEITY = 0.
MOTOR_INACTIVE_PERIOD = 0.
MOTOR_VALUE = 0.  # initial value

# ----------------------------------------------
SENSORY_NUMBER_OF_NEURONS = 4
SENSORY_NODE_CLASS = 'Sensory'
SENSORY_NODE_TYPE = 'E'
SENSORY_ENERGY = binom(50, .9)
SENSORY_ENERGY_CONSUMPTION = 1.
SENSORY_ENERGY_DYNAMICS = EnergyDynamics.linear
SENSORY_INACTIVE_PERIOD = 0.
SENSORY_THRESHOLD = 0.
SENSORY_THRESHOLD_FXN = 'Future'  # placeholder; to
SENSORY_SPONTANEITY = 0.
SENSORY_SENSITIVITY = poisson(1, loc=1)
SENSORY_MAX = norm(loc=1., scale=0.5)
SENSORY_MIN = norm(loc=0.2, scale=0.1)
STIMULI_MID = norm(loc=0.2, scale=0.1)  # inflection point for signal
SENSORY_TO_STIMULI_FXN = StimuliSensitivity.sigmoid
SENSORY_SIGNAL0 = 0.  # initial signal
SENSORY_VALUE0 = 0.  # initial value

EDGE_INTERNAL_TO_INTERNAL = norm(loc=1., scale=0.5)
EDGE_INTERNAL_MIN_CUTOFF = 0.2
EDGE_INTERNAL_TO_MOTOR = norm(loc=1., scale=0.5)
EDGE_MOTOR_MIN_CUTOFF = 1.
EDGE_SENSORY_TO_INTERNAL = norm(loc=1., scale=0.5)
EDGE_SENSORY_MIN_CUTOFF = 1.
SENSORY_MAX_CONNECTIONS = 3.
MOTOR_MAX_CONNECTIONS = 3.
