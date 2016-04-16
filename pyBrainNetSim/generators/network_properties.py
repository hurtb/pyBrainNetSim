from numpy import maximum
from scipy.special import expit

NEURON_CLASSES = ['Internal', 'Sensory', 'Motor']
NEURON_TYPES = ['E', 'I']

MOTOR_ENERGY0, MOTOR_DELTA_ENERGY = 50., 2.
INTERNAL_ENERGY0, INTERNAL_DELTA_ENERGY = 20., 1.
SENSORY_ENERGY0, SENSORY_DELTA_ENERGY = 50., 1.


class StimuliSensitivity(object):
    @staticmethod
    def sigmoid(x, fmin, fmax, sensitivity, x_mid):
        return (fmax - fmin) * expit(sensitivity * (x - x_mid)) + fmin


class EnergyDynamics(object):
    @staticmethod
    def linear(e, de):
        return maximum(0., e - de)

BASIC_NEURON = {'node_class': 'Basic',
                'node_type': 'E',  # 'E' for excitatory or 'I' for inhibitory
                'threshold': 1.,
                'threshold_change_fxn': None,
                'value': 0.,
                'inactive_period': 0.,
                'spontaneity': None,
                'energy_value': 0.,
                'energy_consumption': 0.,
                'energy_dynamics': EnergyDynamics.linear}  # function

I_INTERNAL = {'node_class': 'Internal',
              'threshold': 0.5,
              'inactive_period':1,
              'value': 0.,
              'spontaneity':0.5,  # p(spont fire)
              }

BASIC_EXCITATORY_NEURON = {'node_class': 'Internal',
              'threshold': 1.,
              'inactive_period':1,
              'value':0.,
              'spontaneity':.2,  # p(spont fire)
              'node_type': 'E',
                           }
BASIC_INHIBITORY_NEURON = {'node_class': 'Internal',
              'threshold': 1.,
              'inactive_period':1,
              'value':0.,
              'spontaneity':.2,  # p(spont fire)
              'node_type': 'I',
                    }
BASIC_SENSORY_NEURON_PROP = {'node_class': 'Sensory',
                             'threshold': 0.9,
                             'inactive_period': 2,
                             'spontaneity':0.3,
                             'value': 0.,
                             'stimuli_fxn': StimuliSensitivity.sigmoid,
                             'stimuli_sensitivity':0.5,
                             'stimuli_max':.95,
                             'stimuli_min':.1,
                             'sensory_mid':.2}

BASIC_MOTOR_NEURON_PROPS = {'node_class': 'Motor',
                            'threshold': 0.9,
                            'inactive_period': 0,
                            'spontaneity': 0.0,
                            'value': 0.
                            }
