import copy
import types
from itertools import permutations
import networkx as nx
import numpy as np
import scipy as sp
from pyBrainNetSim.generators.settings.base import *

NODE_PROPERTY_NAMES = [
    'name',
    'node_class',
    'node_type',
    'number_of_nodes',
    'identical',
    'identical_within_type',
    'identical_within_class',
    'threshold',
    'threshold_change_fxn',
    'energy_value',
    'energy_consumption',
    'energy_dynamics',
    'spontaneity',
    'inactive_period',
    'value',
    'physical_distribution',
    'postsyn_signal',
]
NODE_PROPERTY_IGNORE = [
    'name',
    'identical',
    'identical_within_class',
    'number_of_nodes',
    'physical_distribution',
]
NODE_PROPERTY_IGNORE_SAMPLING = {
    'Generic': NODE_PROPERTY_IGNORE,
    'Internal': NODE_PROPERTY_IGNORE + ['excitatory_to_inhibitory'],
    'Motor': NODE_PROPERTY_IGNORE + ['force_direction'],
    'Sensory': NODE_PROPERTY_IGNORE + ['sensor_direction'],
}


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


class Properties(object):
    """Base class for the distribution classes that generate the neural networks."""
    def __init__(self, name=None, *args, **kwargs):
        self._initialized = False
        self._property_names = []
        self._sample_names = []
        self._sampled_props = {}
        self.name=name

    def set_kwargs(self, **kwargs):
        for key, val in kwargs.iteritems():
            if key in self._property_names:
                setattr(self, key, val)

    @staticmethod
    def _chk_rand(value):
        def check_num(numb):
            if isinstance(numb, (int, float, str, types.FunctionType, types.NoneType)):
                num = numb
            else:
                num = numb.rvs()
            return num
        if isinstance(value, (list, np.ndarray)):
            nums = [check_num(val) for val in value]
        else:
            nums = check_num(value)
        return nums

    def copy_intrinsic_props(self):
        iterable_properties = {key: val for key, val in self.__dict__.copy().iteritems() if
                               not key.startswith('_') and key in self._sample_names}
        return iterable_properties

    def sample_field(self, field, significant_figs=None):
        sample = self._chk_rand(field)
        # if isinstance(sample, float) and isinstance(significant_figs, (int, float)):
        #     sample = np.around(sample, decimals=significant_figs)
        return sample

    def reset_props(self):
        pass


class NodeProperties(Properties):
    """Base class for the node/neuron property distribution classes in generating neural networks."""
    node_class = 'Generic'
    _node_id_prefix = 'N'
    _node_type_acceptable = ['E', 'I']

    def __init__(self, name=None, *args, **kwargs):
        super(NodeProperties, self).__init__(*args, **kwargs)
        self._property_names = NODE_PROPS[self.__class__.node_class].keys()
        self._sample_names = [s for s in self._property_names
                              if s not in NODE_PROPERTY_IGNORE_SAMPLING[self.__class__.node_class]]
        self.set_kwargs(**NODE_PROPS[self.__class__.node_class])
        self._number_of_node_types = len(self._node_type_acceptable)
        self._i = 0
        self._pos = []
        self.set_kwargs(**kwargs)
        self.reset_props()

    def sample_old(self, node_id_prefix=None, *args, **kwargs):
        out = {key: self.sample_field(val, key) for key, val in self.__dict__.iteritems()}
        out.update({'id': node_id_prefix})
        return out

    def sample(self, *args, **kwargs):
        self.reset_props()
        number_of_nodes = int(self._chk_rand(self.number_of_nodes))
        self.set_positions(number_of_nodes, self._chk_rand(self.physical_distribution))
        nodes = []
        for i in range(number_of_nodes):
            prop = self.get_one_sample()
            prop.update({'pos':self._pos[self._i]})
            prop.update(self.extra_props())
            node_id = '%s%d' % (self._node_id_prefix, self._i)
            nodes.append((node_id, prop))
            self._i += 1
        return nodes

    def get_one_sample(self, node_type=None, addn_props=None, **kwargs):
        node_type = node_type if isinstance(node_type, str) else self.evaluate_node_type()
        if node_type not in self._node_type_acceptable:
            return
        if (self.identical or self.identical_within_type) and self._initialized:
            prop = copy.copy(self._sampled_props[node_type])
        else:
            prop = {key: self.sample_field(val, key) for key, val in self.copy_intrinsic_props().iteritems()}
        prop.update({'node_type': node_type, 'node_class': self.node_class})
        return prop

    def extra_props(self, *args, **kwargs):
        return {}

    def evaluate_node_type(self):
        return self.node_type

    def set_positions(self, number_of_nodes, layout):
        if layout == 'Grid':
            self._pos = self.get_grid_positions(number_of_nodes)
        elif layout == 'ystack':
            self._pos = self.get_ystack_positions(number_of_nodes)
        elif layout == 'xstack':
            self._pos = self.get_xstack_positions(number_of_nodes)

    def reset_props(self):
        self._number_of_node_types = len(self._node_type_acceptable)
        self._initialized = False
        if self.identical_within_type:
            self._sampled_props.update({n_type: self.get_one_sample(n_type) for n_type in self._node_type_acceptable})
        if self.identical:
            _tmp_sample = self.get_one_sample(self._node_type_acceptable[0])
            self._sampled_props.update({n_type: _tmp_sample for n_type in self._node_type_acceptable})
        self._initialized = True
        self._i = 0

    @staticmethod
    def get_grid_positions(num_neurons):
        mylen = np.ceil(np.sqrt(num_neurons)) + 1
        x, y = np.mgrid[1:mylen:1, 1:mylen:1]
        i_pos = np.vstack([x.ravel(), y.ravel()]).T
        return i_pos

    @staticmethod
    def get_ystack_positions(number_of_nodes):
        pos = []
        for i in range(number_of_nodes):
            pos.append([0., i + 1.])
        return np.array(pos)

    @staticmethod
    def get_xstack_positions(number_of_nodes):
        pos = []
        for i in range(number_of_nodes):
            pos.append([i + 1., 0.])
        return np.array(pos)

    def __repr__(self):
        str = ""
        for nm, attr in self.__dict__.iteritems():
            str += "%s: \t%s\n" % (nm, attr)
        return str


class InternalNodeProperties(NodeProperties):
    node_class = 'Internal'
    _node_id_prefix = 'I'
    _node_type_acceptable = ['E', 'I']

   # TO DELETE
    # def __init__(self, name=None, *args, **kwargs):
    #     super(InternalNodeProperties, self).__init__(name=name, *args, **kwargs)
    #     print "internal: %s" %self.__class__.my_node_class
        # self.node_class = INTERNAL_NODE_CLASS
        # self.number_of_nodes = INTERNAL_NUMBER_OF_NODES
        # self.excitatory_to_inhibitory = INTERNAL_EXCIT_TO_INHIB
        # self.physical_distribution = INTERNAL_PHYSICAL_DISTRIBUTION
        # self.value = INTERNAL_VALUE
        # self.energy_value = INTERNAL_ENERGY
        # self.energy_consumption = MOTOR_ENERGY_CONSUMPTION
        # self.energy_dynamics = INTERNAL_ENERGY_DYNAMICS
        # self.threshold = INTERNAL_THRESHOLD
        # self.threshold_change_fxn = INTERNAL_THRESHOLD_FXN
        # self.spontaneity = INTERNAL_SPONTANEITY
        # self.inactive_period = INTERNAL_INACTIVE_PERIOD
        # self.identical = INTERNAL_IDENTICAL
        # self.identical_within_type = INTERNAL_IDENTICAL_WITHIN_CLASS

        # self.set_kwargs(**NODE_PROPS['Internal'])  # First set default properties

        # self._ignore_sample.extend(['number_of_nodes', 'physical_distribution', 'excitatory_to_inhibitory'])
        # self._node_id_prefix = 'I'

        # self.set_kwargs(**kwargs) # Then set kwargs input properties
        # self.reset_props()

    def evaluate_node_type(self):
        return 'E' if self.sample_field(sp.stats.bernoulli.rvs(self.excitatory_to_inhibitory)) > .5 else 'I'

    def reset_props(self):
        super(InternalNodeProperties, self).reset_props()
        self.excitatory_to_inhibitory = self._chk_rand(self.excitatory_to_inhibitory)


class MotorNodeProperties(NodeProperties):
    """For generating motor units."""
    node_class = 'Motor'
    _node_id_prefix = 'M'
    _node_type_acceptable = ['E']

    # TO DELETE
    # def __init__(self, name=None, *args, **kwargs):
    #     super(MotorNodeProperties, self).__init__(name, *args, **kwargs)
    #     self.number_of_nodes = MOTOR_NUMBER_OF_NODES
    #     self.force_direction = MOTOR_DIRECTION
    #     self.node_class = MOTOR_NODE_CLASS
    #     self.node_type = MOTOR_NODE_TYPE
    #     self.energy_value = MOTOR_ENERGY
    #     self.energy_consumption = MOTOR_ENERGY_CONSUMPTION
    #     self.energy_dynamics = MOTOR_ENERGY_DYNAMICS
    #     self.threshold = MOTOR_THRESHOLD
    #     self.threshold_change_fx = MOTOR_THRESHOLD_FXN
    #     self.spontaneity = MOTOR_SPONTANEITY
    #     self.inactive_period = MOTOR_INACTIVE_PERIOD
    #     self.value = MOTOR_VALUE
    #     self.identical = MOTOR_IDENTICAL
    #     self._node_types_acceptable = ['E']
    #     # self._ignore_sample.extend(['force_direction'])
    #     self._node_id_prefix = 'M'
    #     self.physical_distribution = 'xstack'
    #
    #     self.set_kwargs(**kwargs)
    #     self.reset_props()

    def extra_props(self, *args, **kwargs):
        return {'force_direction': np.array(self.force_direction[self._i])}


class SensoryNodeProperties(NodeProperties):
    """For generating motor units."""
    node_class = 'Sensory'
    _node_id_prefix = 'S'
    _node_type_acceptable = ['E', 'I']

    # TO DELETE
    # def __init__(self, name=None, *args, **kwargs):
    #     super(SensoryNodeProperties, self).__init__(name, *args, **kwargs)
    #     self.number_of_nodes = SENSORY_NUMBER_OF_NEURONS
    #     self.sensor_direction = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]
    #     self.node_class = SENSORY_NODE_CLASS
    #     self.node_type = SENSORY_NODE_TYPE
    #     self.energy_value = SENSORY_ENERGY
    #     self.energy_consumption = SENSORY_ENERGY_CONSUMPTION
    #     self.energy_dynamics = SENSORY_ENERGY_DYNAMICS
    #     self.inactive_period = SENSORY_INACTIVE_PERIOD
    #     self.threshold = SENSORY_THRESHOLD
    #     self.threshold_change_fxn = SENSORY_THRESHOLD_FXN
    #     self.spontaneity = SENSORY_SPONTANEITY
    #     self.stimuli_sensitivity = SENSORY_SENSITIVITY
    #     self.stimuli_max = SENSORY_MAX
    #     self.stimuli_min = SENSORY_MIN
    #     self.sensory_mid = STIMULI_MID  # inflection point
    #     self.stimuli_fxn = SENSORY_TO_STIMULI_FXN
    #     self.value = SENSORY_VALUE0
    #     self.signal = SENSORY_SIGNAL0
    #     self.identical = SENSORY_IDENTICAL
    #     # self._ignore_sample.extend(['sensor_direction'])
    #     self._node_id_prefix = 'S'
    #     self.physical_distribution = 'ystack'
    #
    #     self.set_kwargs(**kwargs)
    #     self.reset_props()

    def extra_props(self, *args, **kwargs):
        return {'sensor_direction': np.array(self.sensor_direction[self._i])}


class EdgeProperty(Properties):
    def __init__(self, prop, named_node_from=None, named_node_to=None, class_node_from=None, class_node_to=None, *args, **kwargs):
        super(EdgeProperty, self).__init__(*args, **kwargs)
        self._props = prop if isinstance(prop, dict) else {}
        self._property_names = ['named_node_from', 'named_node_to', 'class_node_from', 'class_node_to',
                                'weight', 'minimum_cutoff', 'maximum_value', 'identical']
        self.named_node_from = named_node_from
        self.named_node_to = named_node_to
        self.class_node_from = class_node_from if not named_node_from else None
        self.class_node_to = class_node_to if not named_node_to else None
        self.weight = None
        self.minimum_cutoff = None
        self.maximum_value = None
        self.identical = False

        self.set_kwargs(**self._props)
        self.reset_props()

    def sample(self):
        if not self._initialized:
            self.reset_props()
        edge = self._sampled_props if self.identical else self._get_one_sample()
        return {'weight': edge}

    def _get_one_sample(self, significant_figs=2):
        if self.weight is None:
            return 0.
        val = self.sample_field(self.weight, significant_figs=significant_figs)
        if val < self.minimum_cutoff:
            val = 0.
        elif val <= self.maximum_value:
            val = self.maximum_value
        return val

    def reset_props(self):
        self._initialized = False
        if self.identical:
            self._sampled_props = self._get_one_sample()
        self._initialized = True

    def __repr__(self):
        return "\n".join(["%s: %s" %(key, getattr(self, key, val)) for key, val in self.__dict__.iteritems()])


class EdgeProperties(Properties):
    def __init__(self, prop=None, *args, **kwargs):
        super(EdgeProperties, self).__init__(*args, **kwargs)
        self._default_props = SYNAPSES
        self._prop = SYNAPSES if not isinstance(prop, list) else prop
        self.edges = {}
        self.reset_props(prop=self._prop)

    def sample(self, node_list):
        w = []  # list of tuples [(nID1, nID2, weight_value), (nID1, ...)
        for n in list(permutations(node_list, 2)):
            # TODO: get existing number of incoming and outgoing edges before adding another
            w.append((n[0][0], n[1][0], self.edges[n[0][1]['node_class']][n[1][1]['node_class']].sample()['weight']))
        return w

    def reset_props(self, prop=None):
        self.__update_edges(self._default_props)
        if isinstance(prop, list):
            self.__update_edges(prop)

    def __update_edges(self, prop):
        for p in prop:
            if p['class_node_from'] in self.edges:
                self.edges[p['class_node_from']].update({p['class_node_to']: EdgeProperty(p)})
            else:
                self.edges.update({p['class_node_from']: {p['class_node_to']: EdgeProperty(p)}})


class SensorMoverProperties():
    def __init__(self, internal=None, sensors=None, motor=None, weights=None, *args, **kwargs):
        self.internal = internal if internal is not None else InternalNodeProperties()
        self.sensors = sensors if sensors is not None else SensoryNodeProperties()
        self.motor = motor if motor is not None else MotorNodeProperties()
        self.weights = weights if weights is not None else EdgeProperties()

    def sample(self, *args, **kwargs):
        """
        Generate a dict library of the properties of a sensor-mover object. Use this to feed simulations.
        :param args: to be used in the future
        :param kwargs: to be used in the future
        :return: dict
        """
        internal_nodes = self.internal.sample()
        sensory_nodes = self.sensors.sample()
        motor_nodes = self.motor.sample()
        nodes = internal_nodes + sensory_nodes + motor_nodes
        weights = self.weights.sample(nodes)
        return nodes, weights

    def reset_props(self):
        self.internal.reset_props()
        self.sensors.reset_props()
        self.motor.reset_props()
        self.weights.reset_props()

    def create_digraph(self, *args, **kwargs):
        nodes, weights = self.sample(*args, **kwargs)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(weights)
        return G