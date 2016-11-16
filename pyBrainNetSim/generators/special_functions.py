from numpy import maximum
from scipy.special import expit
import numpy as np


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
    return (fmax - fmin) * expit(sensitivity * np.log(x/x_mid)) + fmin
