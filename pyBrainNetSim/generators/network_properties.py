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


def linear(e, de):
    return maximum(0., e - de)


def sigmoid(x, fmin, fmax, sensitivity, x_mid):
    return (fmax - fmin) * expit(sensitivity * (x - x_mid)) + fmin
