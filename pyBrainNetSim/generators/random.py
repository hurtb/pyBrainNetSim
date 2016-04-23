import numpy as np
import networkx as nx
import types
from itertools import permutations, chain
from random_settings import *


def __chk_rand(vals, default):
    def chk_num(numb):
        if isinstance(numb, (int, float, str)):
            num = numb
        elif numb is None:
            num = default.rvs()
        else:
            num = numb.rvs()
        return num
    if isinstance(vals, (list, np.ndarray)):
        nums = [chk_num(val) for val in vals]
    else:
        nums = chk_num(vals)
    return nums


class PropertyDistribution(object):
    def set_kwargs(self, **kwargs):
        for key,val in kwargs.iteritems():
            setattr(self, key, val)

    def sample_field(self, field, name=None):
        return self._chk_rand(field)

    def sample(self, nID = None, *args, **kwargs):
        out = {key: self.sample_field(val, key) for key, val in self.__dict__.iteritems()}
        out.update({'id': nID})
        return out

    def _chk_rand(self, vals):
        def chk_num(numb):
            if isinstance(numb, (int, float, str, types.FunctionType)):
                num = numb
            else:
                num = numb.rvs()
            return num
        if isinstance(vals, (list, np.ndarray)):
            nums = [chk_num(val) for val in vals]
        else:
            nums = chk_num(vals)
        return nums

    @staticmethod
    def get_grid_positions(num_neurons):
        mylen = np.ceil(np.sqrt(num_neurons)) + 1
        x, y = np.mgrid[1:mylen:1, 1:mylen:1]
        i_pos = np.vstack([x.ravel(), y.ravel()]).T
        return i_pos


class InternalPropertyDistribution(PropertyDistribution):
    def __init__(self, *args, **kwargs):
        self.number_neurons = INTERNAL_NUMBER_OF_NEURONS
        self.excitatory_to_inhibitory = INTERNAL_EXCIT_TO_INHIB
        self.physical_distribution = INTERNAL_PHYSICAL_DISTRIBUTION

        self.node_class = INTERNAL_NODE_CLASS
        self.value = INTERNAL_VALUE
        self.energy_value = INTERNAL_ENERGY
        self.energy_consumption = MOTOR_ENERGY_CONSUMPTION
        self.energy_dynamics = INTERNAL_ENERGY_DYNAMICS
        self.threshold = INTERNAL_THRESHOLD
        self.threshold_change_fx = INTERNAL_THRESHOLD_FXN
        self.spontaneity = INTERNAL_SPONTANEITY
        self.inactive_period = INTERNAL_INACTIVE_PERIOD

        self.set_kwargs(**kwargs)

    def sample(self, *args, **kwargs):
        node = []
        number_of_neurons = self._chk_rand(self.number_neurons)
        items = self.__dict__.copy()
        _tmp = [items.pop(val) for val in ['number_neurons', 'physical_distribution', 'excitatory_to_inhibitory']]
        pos = self.get_positions(number_of_neurons, self._chk_rand(self.physical_distribution))
        for nID in range(number_of_neurons):
            prop = {key: self.sample_field(val, key) for key, val in items.iteritems()}
            prop.update({'node_type': 'E' if self.sample_field(self.excitatory_to_inhibitory) > .5 else 'I',
                         'pos': pos[nID]})
            node.append(('I%d' % nID, prop))
        return node

    def get_positions(self, num_neurons, layout):
        if layout == 'Grid':
            pos = self.get_grid_positions(num_neurons)
        return pos


class MotorPropertyDistribution(PropertyDistribution):
    def __init__(self, *args, **kwargs):
        self.number_of_units = MOTOR_NUMBER_OF_UNITS
        self.force_direction = MOTOR_DIRECTION
        self.node_class = MOTOR_NODE_CLASS
        self.node_type = MOTOR_NODE_TYPE
        self.energy_value = MOTOR_ENERGY
        self.energy_consumption = MOTOR_ENERGY_CONSUMPTION
        self.energy_dynamics = MOTOR_ENERGY_DYNAMICS
        self.threshold = MOTOR_THRESHOLD
        self.threshold_change_fx = MOTOR_THRESHOLD_FXN
        self.spontaneity = MOTOR_SPONTANEITY
        self.inactive_period = MOTOR_INACTIVE_PERIOD
        self.value = MOTOR_VALUE

        self.set_kwargs(**kwargs)

    def sample(self, *args, **kwargs):
        node = []
        number_of_units = int(self._chk_rand(self.number_of_units))
        items = self.__dict__.copy()
        items.pop('number_of_units')
        items.pop('force_direction')
        pos = self.get_positions(number_of_units)
        for uID in range(number_of_units):
            prop = {key: self.sample_field(val, key) for key, val in items.iteritems()}
            prop.update({'pos': pos[uID], 'force_direction': self.force_direction[uID], 'node_class': MOTOR_NODE_CLASS})
            node.append(('M%d' % uID, prop))
        return node

    def get_positions(self, number_units):
        pos = []
        for i in range(number_units):
            pos.append([0., i])
        return np.array(pos)


class SensoryPropertyDistribution(PropertyDistribution):
    def __init__(self, *args, **kwargs):
        self.number_of_neurons = SENSORY_NUMBER_OF_NEURONS
        self.sensor_direction = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]

        self.node_class = SENSORY_NODE_CLASS
        self. node_type = SENSORY_NODE_TYPE
        self.energy_value = SENSORY_ENERGY
        self.energy_consumption = SENSORY_ENERGY_CONSUMPTION
        self.energy_dynamics = SENSORY_ENERGY_DYNAMICS
        self.inactive_period = SENSORY_INACTIVE_PERIOD
        self.threshold = SENSORY_THRESHOLD
        self.threshold_change_fx = SENSORY_THRESHOLD_FXN
        self.spontaneity = SENSORY_SPONTANEITY
        self.stimuli_sensitivity = SENSORY_SENSITIVITY
        self.stimuli_max = SENSORY_MAX
        self.stimuli_min = SENSORY_MIN
        self.sensory_mid = STIMULI_MID  # inflection point
        self.stimuli_fxn = SENSORY_TO_STIMULI_FXN
        self.value = SENSORY_VALUE0
        self.signal = SENSORY_SIGNAL0

        self.set_kwargs(**kwargs)

    def sample(self, *args, **kwargs):
        node = []
        number_of_neurons = int(self._chk_rand(self.number_of_neurons))
        items = self.__dict__.copy()
        _tmp = [items.pop(val) for val in ['number_of_neurons', 'sensor_direction']]
        pos = self.get_positions(number_of_neurons)
        for uID in range(number_of_neurons):
            prop = {key: self.sample_field(val, key) for key, val in items.iteritems()}
            prop.update({'pos': pos[uID], 'sensor_direction': self.sensor_direction[uID]})
            node.append(('S%d' % uID, prop))
        return node

    def get_positions(self, number_of_neurons):
        pos = []
        for i in range(number_of_neurons):
            pos.append([0., i])
        return np.array(pos)


class WeightPropertyDistribution(PropertyDistribution):
    def __init__(self, *args, **kwargs):
        self.int_to_int = EDGE_INTERNAL_TO_INTERNAL
        self.edge_internal_min_cutoff = EDGE_INTERNAL_MIN_CUTOFF
        self.int_to_motor = EDGE_INTERNAL_TO_MOTOR
        self.edge_motor_min_cutoff = EDGE_MOTOR_MIN_CUTOFF
        self.sensor_to_int = EDGE_SENSORY_TO_INTERNAL
        self.edge_sensor_min_cutoff = EDGE_SENSORY_MIN_CUTOFF
        self.sensory_max_connections = SENSORY_MAX_CONNECTIONS
        self.motor_max_connections = MOTOR_MAX_CONNECTIONS
        self.set_kwargs(**kwargs)

    def sample_edges(self, node, *args, **kwargs):
        """
        Return a dict where {(nodeID1, nodeID2): weight_value, ...}
        Return a container of the form [(nodeID1, nodeID2, weight_value), ...]
        """
        if not isinstance(node, (dict, list, np.ndarray)):
            return {}
        nodes = node
        if isinstance(node, dict):
            nodes = node.keys()
        weight = {}
        for n1, n2 in permutations(nodes, 2):  # get all permutations of nodes (no auto-signaling)
            w = 0.
            sensor_numbers = {}
            motor_numbers = {}
            if n1.startswith('I') and n2.startswith('I'):
                w = self._chk_rand(self.int_to_int)
                w = w if w > self.edge_internal_min_cutoff else 0.
            elif n1.startswith('I') and n2.startswith('M'):
                motor_numbers[n2] = 1 if n2 not in motor_numbers.keys() else motor_numbers[n2]+1
                if motor_numbers[n2] <= self.motor_max_connections:
                    w = self._chk_rand(self.int_to_motor)
                    w = w if w > self.edge_motor_min_cutoff else 0.
            elif n1.startswith('S') and n2.startswith('I'):
                sensor_numbers[n1] = 1 if n1 not in sensor_numbers.keys() else sensor_numbers[n1]+1
                if sensor_numbers[n1] <= self.sensory_max_connections:
                    w = self._chk_rand(self.sensor_to_int)
                    w = w if w > self.edge_sensor_min_cutoff else 0.
            if w > 0.:
                weight.update({(n1, n2): w})

        return weight

    def sample_edge_weights(self, node, *args, **kwargs):
        """
        Return a container of the form [(nodeID1, nodeID2, weight_value), ...]
        """
        if not isinstance(node, (dict, list, np.ndarray)):
            return {}
        nodes = node
        if isinstance(node, dict):
            nodes = node.keys()
        weight = []
        for n1, n2 in permutations(nodes, 2):  # get all permutations of nodes (no auto-signaling)
            w = 0.
            sensor_numbers = {}
            motor_numbers = {}
            if n1.startswith('I') and n2.startswith('I'):
                w = self._chk_rand(self.int_to_int)
                w = w if w > self.edge_internal_min_cutoff else 0.
            elif n1.startswith('I') and n2.startswith('M'):
                motor_numbers[n2] = 1 if n2 not in motor_numbers.keys() else motor_numbers[n2]+1
                if motor_numbers[n2] <= self.motor_max_connections:
                    w = self._chk_rand(self.int_to_motor)
                    w = w if w > self.edge_motor_min_cutoff else 0.
            elif n1.startswith('S') and n2.startswith('I'):
                sensor_numbers[n1] = 1 if n1 not in sensor_numbers.keys() else sensor_numbers[n1]+1
                if sensor_numbers[n1] <= self.sensory_max_connections:
                    w = self._chk_rand(self.sensor_to_int)
                    w = w if w > self.edge_sensor_min_cutoff else 0.
            if w > 0.:
                weight.append((n1, n2, w))
        return weight


class SensorMoverPropertyDistribution(PropertyDistribution):
    def __init__(self, internal=None, sensors=None, motor=None, weights=None, *args, **kwargs):
        self.internal = internal if internal is not None else InternalPropertyDistribution()
        self.sensors = sensors if sensors is not None else SensoryPropertyDistribution()
        self.motor = motor if motor is not None else MotorPropertyDistribution()
        self.weights = weights if weights is not None else WeightPropertyDistribution()

    def sample(self, *args, **kwargs):
        """
        Generate a dict library of the properties of a sensor-mover object. Use this to feed simulations.
        :param args: to be used in the future
        :param kwargs: to be used in the future
        :return: dict
        """
        ins = self.internal.sample()
        ss = self.sensors.sample()
        ms = self.motor.sample()
        nodes = ins + ss + ms
        weights = self.weights.sample_edge_weights(list(chain.from_iterable([[n[0] for n in ins],
                                                                            [n[0] for n in ss],
                                                                            [n[0] for n in ms]])))
        return nodes, weights

    def create_digraph(self, *args, **kwargs):
        nodes, weights = self.sample(*args, **kwargs)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(weights)
        return G

# TODO: DEPRECIATE BELOW
#
# def random_neuron_props(nclass=None, ntype=None, threshold=None, spontaneous=None, energy=None):
#     """
#     Generate properties of a random neuron.
#     :param nclass: None, scipy distribution, int, float, or str. If None, will use default random generator
#     :param ntype: None, scipy distribution, int, float, or str. If None, will use default random generator
#     :param threshold: None, scipy distribution, int, or float. If None, will use default random generator
#     :param spontaneous: None, scipy distribution, int, or float. If None, will use default random generator
#     :param energy: None, scipy distribution, int, or float. If None, will use default random generator
#     :return: dict of properties of a random neuron
#     """
#
#     props = BASIC_NEURON
#
#     props.update({'node_class': NEURON_CLASSES[__chk_rand(nclass, DEFAULT_NEURON_CLASS)],
#                   'node_type': NEURON_TYPES[__chk_rand(ntype, DEFAULT_NEURON_TYPE)],
#                   'threshold': __chk_rand(threshold, DEFAULT_THRESHOLD),
#                   'spontaneity': __chk_rand(spontaneous, DEFAULT_SPONTANEITY),
#                   'energy_value': __chk_rand(energy, DEFAULT_ENERGY),
#                   'energy_consumption': __chk_rand(energy, DEFAULT_ENERGY_CONSUMPTION),
#                   })
#
#     return props
#
#
# def random_internal_neuron_prop(**kwargs):
#     props = random_neuron_props(nclass=0, **kwargs).copy()
#     return props
#
#
# def random_sensory_neuron_prop(sensitivity=None, maxv=None, minv=None, midv=None, **kwargs):
#     props = random_neuron_props(nclass=1, **kwargs).copy()
#     sens_min = np.maximum(__chk_rand(maxv, SENSORY_MIN), 0.)
#     sens_max = np.maximum(__chk_rand(minv, SENSORY_MAX), sens_min)
#     stim_mid = np.maximum(__chk_rand(midv, STIMULI_MID), 0.)
#     props.update({'stimuli_fxn': StimuliSensitivity.sigmoid,
#                   'stimuli_sensitivity': __chk_rand(sensitivity, SENSORY_SENSITIVITY),
#                   'stimuli_max': sens_max,
#                   'stimuli_min': sens_min,
#                   'sensory_mid': stim_mid,
#                   })
#     return props
#
#
# def random_motor_neuron_prop(**kwargs):
#     props = random_neuron_props(nclass=2, **kwargs).copy()
#     return props
#
#
# def sensor_mover_prop(sensor_mover_distribution):
#     """
#     Create a random sensor-mover based on the set of various 'PropertyDistributions.
#     MORE CONVENIENT METHOD TO GENERATE 'sensor_mover' individuals.
#     :param sensor_mover_distribution: class instance of SensorMoverPropertyDistribution
#     :return:
#     """
#     props = {}
#     # First create a dictionary of all of the neural properties. Set edges. Return graph
#     smp = sensor_mover_distribution
#     for n in range(__chk_rand(smp.internal.number_neurons)):
#         pass
#
#
