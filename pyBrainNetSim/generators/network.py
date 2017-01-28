import copy
import types
from itertools import permutations
import networkx as nx
import numpy as np
import scipy as sp
import random
from pyBrainNetSim.generators.settings.base import *
from pyBrainNetSim.models.network import NeuralNetData
import pyBrainNetSim.utils as utils

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
        elif callable(getattr(numb, 'rvs', None)):
            num = numb.rvs()
        else:
            num = numb
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
            if callable(getattr(numb, 'rvs', None)):
                num = numb.rvs()
            else:
                num = numb
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

    def sample(self, *args, **kwargs):
        self.reset_props()
        number_of_nodes = int(self._chk_rand(self.number_of_nodes))
        # self.set_positions(number_of_nodes, self._chk_rand(self.physical_distribution['type']), *args, **kwargs)
        nodes = []
        for i in range(number_of_nodes):
            prop = self.get_one_sample()
            # prop.update({'pos':self._pos[self._i]})
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

    def set_positions(self, number_of_nodes, layout, *args, **kwargs):
        if layout == 'Grid':
            self._pos = self.get_grid_positions(number_of_nodes, *args, **kwargs)
        elif layout == 'ystack':
            self._pos = self.get_ystack_positions(number_of_nodes, *args, **kwargs)
        elif layout == 'xstack':
            self._pos = self.get_xstack_positions(number_of_nodes, *args, **kwargs)
        elif layout == 'by_direction':
            self._pos = self.get_bydirection_positions(number_of_nodes, *args, **kwargs)

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

    def get_grid_positions(self, num_neurons, *args, **kwargs):
        half_grid_size = np.round(self.physical_distribution['size'] / 2.)
        mylen = np.ceil(np.sqrt(num_neurons))
        x, y = np.meshgrid(np.linspace(-half_grid_size, half_grid_size, num=mylen),
                           np.linspace(-half_grid_size, half_grid_size, num=mylen))
        pts = np.vstack([x.ravel(), y.ravel()]).T
        return pts

    @staticmethod
    def get_ystack_positions(number_of_nodes, *args, **kwargs):
        return np.array([[0., i + 1.] for i in range(number_of_nodes)])

    @staticmethod
    def get_xstack_positions(number_of_nodes, *args, **kwargs):
        return np.array([[i + 1., 0.] for i in range(number_of_nodes)])

    def get_bydirection_positions(self, number_of_nodes, *args, **kwargs):
        pos = []
        if 'existing_points' in kwargs:
            pts = np.array([p['pos'] for i, p in kwargs['existing_points']])
        else:
            return self.get_xstack_positions(number_of_nodes)  # temporary
        c = utils.centroid(pts)
        mx = np.max(pts, axis=0)
        mn = np.min(pts, axis=0)
        mxdist = np.max(mx-mn)
        offset = self._chk_rand(self.physical_distribution['offset'])
        extra = self._chk_rand(self.physical_distribution['extra_distance'])
        print self.direction
        for d in self.direction:  # loop through each direction
            extra_dim = np.zeros(len(d) + 1)
            extra_dim[-1] = 1.
            ortho = np.cross(d, extra_dim)[:len(d)]
            p = c + (mxdist + extra) * np.array(d) + offset * ortho
            pos.append(p)
        return pos

    def __repr__(self):
        str = ""
        for nm, attr in self.__dict__.iteritems():
            str += "%s: \t%s\n" % (nm, attr)
        return str


class InternalNodeProperties(NodeProperties):
    node_class = 'Internal'
    _node_id_prefix = 'I'
    _node_type_acceptable = ['E', 'I']

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

    def extra_props(self, *args, **kwargs):
        return {'force_direction': np.array(self.direction[self._i])}


class SensoryNodeProperties(NodeProperties):
    """For generating motor units."""
    node_class = 'Sensory'
    _node_id_prefix = 'S'
    _node_type_acceptable = ['E', 'I']

    def extra_props(self, *args, **kwargs):
        return {'sensor_direction': np.array(self.direction[self._i])}


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

    def sample(self, pruning_method='proximity', *args, **kwargs):
        """
        Generate a dict library of the properties of a sensor-mover object. Use this to feed simulations.
        :param args: to be used in the future
        :param kwargs: to be used in the future
        :return: dict
        """
        internal_nodes = self.internal.sample()
        sensory_nodes = self.sensors.sample(existing_points=internal_nodes)
        motor_nodes = self.motor.sample(existing_points=internal_nodes)
        nodes = internal_nodes + sensory_nodes + motor_nodes
        weights = self.weights.sample(nodes)
        weights = self.prune_edges(nodes, weights, pruning_method=pruning_method)  # apply the max_incoming/max_outgoing
        return nodes, weights

    def reset_props(self):
        self.internal.reset_props()
        self.sensors.reset_props()
        self.motor.reset_props()
        self.weights.reset_props()

    def create_digraph(self, nodes=None, edges=None, *args, **kwargs):
        if not nodes or not edges:
            nodes, edges = self.sample(*args, **kwargs)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)
        return G

    def __list_to_dict(self, v):
        return {nid: val for nid, val in v}

    def __edge_list_to_dict(self, lst):
        o = {}
        for nf, nt, val in lst:
            if nf in o:
                o[nf].update({nt: {'weight': val}})
            else:
                o.update({nf: {nt: {'weight': val}}})
        return o

    def prune_edges(self, nodes, edges, pruning_method='furthest_proximity'):
        """`prune_edges` is a method for capping the outgoing/incoming edges of each node according to the
        `'max_incoming'` and `'max_outgoing'` properties for a given `'node_class'` These attributes are prescribed in
        the input each node distribution, or as defaulted in `settings.base.NODE_PROPS[group_name]`. The method works
        via:
        1. Loop through nodes
        2. Get a list of incoming/outgoing nodes by 'node_class'
        3. For each of those lists, prune the incoming/outgoing node edges accordingly
        :param nodes: list of tuple, where each tuple is (node_id, property dict)
        :param edges: tuple list, where each tuple is (node from, node to, edge weight)
        :param pruning_method: 'furthest_proximity' - prunes edges that are outside the maximum allowed nodes, 'random'
        :return: edges, as above.
        """
        G = NeuralNetData(self.create_digraph(nodes=nodes, edges=edges))
        nodes = self.__list_to_dict(nodes)
        edges = self.__edge_list_to_dict(edges)
        for n_id, n_props  in nodes.iteritems():
            for n_class, val in n_props['max_outgoing'].iteritems():
                nc = G.nodes(node_class=n_class)
                if not isinstance(val, (int, float)) or len(nc) == 0:
                    continue
                prune_nid = G.get_n_neighbors_by(n_id, n=len(nc)-val, node_class=n_class, method=pruning_method,
                                                 connected=True)
                for pid in prune_nid:
                    if pid in edges[n_id]:
                        edges[n_id][pid]['weight'] = 0.
            for n_class, val in n_props['max_incoming'].iteritems():
                nc = G.nodes(node_class=n_class)
                if not isinstance(val, (int, float)) or len(nc) == 0:
                    continue
                prune_nid = G.get_n_neighbors_by(n_id, n=len(nc)-val, node_class=n_class, method=pruning_method,
                                                 connected=True)
                for pid in prune_nid:
                    if pid in edges and n_id in edges[pid]:
                        edges[pid][n_id]['weight'] = 0.
        edges = [(nf, nt, attr['weight']) for nf in edges.keys() for nt, attr in edges[nf].iteritems()]
        return edges