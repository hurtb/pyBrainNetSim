from numpy import maximum
from scipy.special import expit
import numpy as np


class StimuliSensitivity(object):
    @staticmethod
    def sigmoid(x, fmin, fmax, sensitivity, x_mid):
        return (fmax - fmin) * expit(sensitivity * (x - x_mid)) + fmin


class EnergyDynamics(object):
    @staticmethod
    def linear(e, fire_loss, *args, **kwargs):
        de = fire_loss if e > fire_loss else e
        return de

    @staticmethod
    def linear_with_decay(e, fire_loss, basal_rate=0.7, *args, **kwargs):
        de = fire_loss if EnergyDynamics.linear(e, fire_loss) - basal_rate >= 0. else 0.
        return de


def linear(e, fire_loss):
    de = fire_loss if e > fire_loss else e
    return de


def sigmoid(x, fmin, fmax, sensitivity, x_mid):
    return (fmax - fmin) * expit(sensitivity * np.log(x/x_mid)) + fmin
