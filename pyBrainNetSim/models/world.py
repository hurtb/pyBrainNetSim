# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:16:58 2016

@author: brian
"""
from scipy.spatial.distance import euclidean
from scipy.stats import randint
from scipy.ndimage import zoom
import networkx as nx
import numpy as np
import pyBrainNetSim.utils as utils
import copy
import random


class ScalarField(object):
    def __init__(self, field_type, c_grid, *args, **kwargs):
        self.field_type = field_type
        self.d_size = len(c_grid)
        self.c_grid = c_grid
        self.sources = {}
        self._update_field()

    def function(self, *args, **kwargs):
        return np.zeros_like(self.c_grid[0])

    def add_point_source(self, a_id, location, strength):
        self.sources.update({a_id: {'location': location, 'strength': strength}})
        self._update_field()

    def rm_point_source(self, point_source_id):
        del(self.sources[point_source_id])
        self._update_field()

    def _update_field(self):
        _field = np.zeros_like(self.c_grid[0])
        for source_id, source_properties in self.sources.iteritems():
            _field += self.function(**source_properties)
        self._field = _field
        self._grad = np.gradient(self._field)

    def field(self, source_id=None):
        _field = self.function(**self.sources[source_id]) if source_id in self.sources else self._field
        return _field

    def field_at(self, location, source_id=None):
        indx = self._loc_to_indx(location)  # TODO: PROBLEM IN 1-D
        _field = 0.
        if self.d_size == 1:
            _field = self.field(source_id)[indx]
        elif self.d_size == 2:
            _field = self.field(source_id)[indx[1]][indx[0]]
        elif self.d_size == 3:
            _field = self.field(source_id)[indx[2]][indx[1]][indx[0]]
        return _field

    def gradient(self, source_id=None):
        _grad = np.gradient(self.field(source_id)) if source_id in self.sources\
            else self._grad
        if self.d_size > 1:
            _grad.reverse()  # REVERSE ORDERED - DOCUMENTED INCORRECTLY IN THE NUMPY DOCS
        return _grad

    def gradient_at(self, location, source_id=None):
        indx = self._loc_to_indx(location)
        _grad = grad = self.gradient(source_id)
        if self.d_size == 1:
            grad = np.array([_grad[indx]])
        elif self.d_size == 2:
            grad = np.array([_grad[i][indx[0]][indx[1]] for i in range(len(indx))])
        elif self.d_size == 3:
            grad = np.array([_grad[i][indx[0]][indx[1]][indx[2]] for i in range(len(indx))])
        return grad

    def _loc_to_indx(self, p):
        """"Map the cartesian coordinates to closest index"""
        cg = self.c_grid
        idx = []
        if len(cg) == 2:
            idx.append(np.where(cg[0][0] == p[0])[0][0])
            idx.append(np.where(cg[1].T[0] == p[1])[0][0])
        elif len(cg) == 1.:
            idx.append(np.where(cg[0] == p[0])[0][0])
        else:
            print '>2 dimensions not implemented' # TODO: Generalize this functionality for N-D
        return np.array(idx, dtype=int)

    def get_polar_grid(self, x0):
        pg = []
        if self.d_size == 2:
            pg = utils.cart2pol(self.c_grid, x0=x0)
        elif self.d_size == 1:
            pg = self.c_grid - x0
        return pg


class LinearDecayScalarField(ScalarField):
    def __init__(self, field_type, c_grid, field_permeability=1., *args, **kwargs):
        super(LinearDecayScalarField, self).__init__(field_type, c_grid, *args, **kwargs)
        self.field_permeability = field_permeability

    def function(self, location=(0, 0), strength=1, *args, **kwargs):
        return np.maximum(strength - abs((self.c_grid[0] - location))* self.field_permeability, 0.)


class ExponentialDecayScalarField(ScalarField):
    def __init__(self, field_type, c_grid, field_permeability=1., *args, **kwargs):
        super(ExponentialDecayScalarField, self).__init__(field_type, c_grid, *args, **kwargs)
        self.field_permeability = field_permeability

    def function(self, location=(0, 0), strength=1, *args, **kwargs):
        pg = self.get_polar_grid(location)
        return strength * np.exp(-abs(pg[0] * self.field_permeability))


class Environment(object):
    def __init__(self, origin=(0., 0.), max_point=(20., 20.), deltas=1., field_decay=LinearDecayScalarField,
                 field_permeability=0.5, *args, **kwargs):
        self.d_size, self.origin, self.max_point, self.deltas, self.c_grid = None, None, None, None, None
        self.change_world(origin, max_point, deltas)
        self._individuals = {}
        self.attractors = {}
        self.field_permeability = field_permeability
        self.fields = {'Sensory': field_decay('Sensory', self.c_grid, self.field_permeability)}
        self.an = 0  # counter for attractors

    def change_world(self, origin, max_point, deltas=1.):
        self.d_size = len(max_point)
        if self.d_size > 3:  # too big
            return
        if np.any(np.array(origin) == np.array(max_point)):
            return
        for i in range(self.d_size):
            if origin[i] > max_point[i]:
                origin, max_point = max_point, origin
                break
        self.origin, self.max_point = origin, max_point
        if isinstance(deltas, (int, float)):
            self.deltas = deltas * np.ones(self.d_size)
        elif len(deltas) == self.d_size:
            self.deltas = deltas
        else:
            self.deltas = np.ones(self.d_size)

        self.ticks = [np.arange(self.origin[i], self.max_point[i] + self.deltas[i], self.deltas[i])
                      for i in range(self.d_size)]
        self.c_grid = np.meshgrid(*self.ticks, sparse=False)

    def add_individual(self, individual):
        if isinstance(individual, Individual):
            self._individuals.update({individual.ind_id: individual})

    def rm_individuals(self):
        self._individuals = {}
        
    def add_attractor(self, attr, field_type):
        """ e.g. food
        :param attr:
        """
        attr.a_id = 's%s' % self.an if attr.a_id is None else attr.a_id
        self.attractors.update({attr.a_id: attr})
        self.fields[field_type].add_point_source(attr.a_id, attr.position, attr.strength)
        self.an += 1

    def add_random_attractor(self, field_type='Sensory', strength=1., energy=100):
        pos = np.array([randint(low=self.origin[i], high=self.max_point[i]).rvs() for i in range(self.d_size)])
        attract = Attractor(self, pos, a_id='s%s' % self.an, strength=strength, energy_value=energy,
                            attractor_type=field_type)
        self.add_attractor(attract, field_type)

    def rm_attractor(self, a_id):
        self.fields[self.attractors[a_id].field_type].rm_point_source(a_id)
        del(self.attractors[a_id])

    def attractor_field(self, field_type='Sensory', attractor_id=None):
        return self.fields[field_type].field(source_id=attractor_id)

    def attractor_field_at(self, location, field_type='Sensory', attractor_id=None):
        """
        Returns the summed field at the location. Assumes the fields are of the same type.
        :param location:
        :param attractor_id:
        :return:
        """
        return self.fields[field_type].field_at(location, source_id=attractor_id)

    def attractor_gradient_at(self, location, field_type='Sensory', attractor_id=None):
        """
        Returns the summed gradient at the location. Assumes the fields are of the same type.
        :param location:
        :param attractor_id:
        :return:
        """
        return self.fields[field_type].gradient_at(location, source_id=attractor_id)

    def __upsample_attractor_field(self, factor=5, order=1, **kwargs):
        x = zoom(self.c_grid[0], factor, order=order)
        y = zoom(self.c_grid[1], factor, order=order)
        z = zoom(self.attractor_field(**kwargs), factor, order=order)
        return x, y, z

    def generate_position(self, position=None):
        if position is None:  # generate random position
            pos = np.array([randint(low=self.origin[i], high=self.max_point[i]).rvs() for i in range(self.d_size)])
        else:
            pos = position
        return pos

    @property
    def positions(self):
        return []
    
    @property
    def individuals(self):
        return self._individuals
        
    def _build_env(self):
        pass


class Individual(object):
    
    def __init__(self, environment=None, position=None, ind_id=None, reproduction_cost=0.5, reproductive_threshold=10,
                 reproduction_mode='asexual', data_cutoff=None, global_time_born=0, *args, **kwargs):
        self.parents, self.children, self.generation, self.kin = [], [], 0, 0
        self._cloned_from, self._not_cloned_from = None, None
        self.t, self.t_birth = 0, global_time_born
        self.reproduction_cost = reproduction_cost
        self.reproduction_threshold = reproductive_threshold
        self.reproduction_mode = reproduction_mode
        self._reproduced = False
        self._environment, self._position, self.d_size, self.data_cutoff = None, None, 2, data_cutoff
        self.ind_id = self._generate_id(ind_id)
        self.set_environment(environment)
        self.set_position(position)
        self.set_trajectory([self.position.copy()])
        self.reset_data()

    def reset_data(self):
        self.set_trajectory([self.position.copy()])

    def _generate_id(self, ind_id=None, num_chars=4, choices='0123456789'):
        if not ind_id or not isinstance(ind_id, str):
            ind_id = ''.join(random.choice(choices) for i in range(num_chars))
        return ind_id
        
    def move(self, vector):
        if len(vector) == self.d_size:
            self._position += np.array(vector, dtype=np.float)
            self._trajectory.append(self._position.copy())

    def in_world(self, pos):
        inworld = True
        pos = np.array(pos)
        if np.any(pos > self.environment.max_point) or np.any(pos < self.environment.origin):
            inworld = False
        return inworld

    def set_environment(self, environment):
        env = environment
        if isinstance(env, Environment):
            env.add_individual(self)
        elif not environment:
            env = Environment()
        self.d_size = env.d_size
        self._environment = env

    @property
    def environment(self):
        return self._environment

    def set_trajectory(self, trajectory):
        self._trajectory = trajectory

    def trajectory_vector_to(self, target):
        pass

    @property
    def trajectory(self):
        return np.array(self._trajectory) if not isinstance(self._trajectory, np.ndarray) else self._trajectory

    def set_position(self, position):
        en = self.environment
        if position is None:
            self._position = np.zeros(en.d_size)
        elif len(position) == en.d_size:
            self._position = np.array(position).astype(np.float)
        elif position == 'rand':
            self._position = [np.random.randint(0, en.max_point[i]) for i in range(en.d_size)]
        else:
            self._position = np.zeros(en.d_size)

    @property
    def age(self):
        return self.t - self.t_birth

    @property
    def position(self):
        return np.array(self._position)
        
    @property
    def velocity(self):
        return np.vstack((np.zeros(self.d_size), self.trajectory[1:]-self.trajectory[:-1]))

    def dist_to(self, target):
        return euclidean(self.position, target)

    def vector_to(self, target, normalized=True):
        target = np.array(target) if not isinstance(target, np.ndarray) else target
        vector = (self.position - target)
        if normalized:
            vector /= euclidean(self.position, target)
        return vector

    def efficiency(self, *args, **kwargs):
        """Method to evaluate an objective function. To be overridden."""
        pass

    def reproduce(self, mode='asexual', partner=None, new_environment=True, error_rate=0., *args, **kwargs):
        child = None
        if mode == 'asexual':
            child = self._reproduce_asexually(error_rate)
        elif mode == 'sexual':
            child = self._reproduce_sexually(partner)
        if new_environment:
            env = copy.copy(self.environment)
            env.rm_individuals()
        else:
            env = self.environment
        # env = self.environment if not new_environment else copy.deepcopy(self.environment)
        child.set_environment(env, *args, **kwargs)
        _tmp = [child.environment.add_attractor(attractor, attractor.field_type)
                for attractor in self.environment.attractors.values()]
        # child.set_position()
        self.generation += 1
        child.generation = self.generation
        child.kin = child._generate_id()
        child.ind_id = "G%d_I%s" % (child.generation, child.kin)
        child.reset_data()
        child.t_birth = self.t + 1
        self.children.append(child)

        return child

    def _reproduce_asexually(self, error_rate=0.):
        child = copy.copy(self)
        child.parents = [self]
        child._cloned_from, child._not_cloned_from = self, None
        return child

    def _reproduce_sexually(self, partner):  # TODO: Broken.
        """Sexual reproduction between two networks"""
        inherited_state = -1  # -1 would be most recent
        network_props = ['num_nodes']
        node_props = ['threshold', 'energy_consumption', 'spontaneity']
        # node_props = ['threshold']
        edge_props = ['weight']
        child = copy.deepcopy(self)
        partner.children.append(child)
        # partner.reproductive_energy_cost = self.reproductive_energy_cost
        child.parents, child.children = [self, partner], []
        if np.random.randint(0, 2) == 1:
            internal_net = copy.deepcopy(self.internal)
            child._cloned_from, child._not_cloned_from = self, partner
        else:
            internal_net = copy.deepcopy(partner.internal)
            child._cloned_from, child._not_cloned_from = partner, self
        # print "Kin with %d neurons, copied from net with %d neurons" %(internal_net.simdata[-1].number_of_nodes(), self.internal.simdata[-1].number_of_nodes())
        child.set_internal_network(copy.deepcopy(internal_net), t0=self.t)
        child.internal.simdata[inherited_state] = copy.copy(internal_net.simdata[inherited_state])

        choices = np.random.randint(2, size=(2, len(node_props)))  # randomly choose attributes
        for j, n in enumerate(node_props):
            p1 = nx.get_node_attributes(self.internal.simdata[inherited_state], n)
            p2 = nx.get_node_attributes(partner.internal.simdata[inherited_state], n)
            # add/remove nodal information based on the inherited number of nodes
            # chosen = self if choices[0][j] else partner
            # print "Using %s(N%d) for %s" %(chosen.ind_id, chosen.internal.simdata[inherited_state].number_of_nodes(), n)
            utils.set_node_attributes(child.internal.simdata[inherited_state], n, p1 if choices[0][j] else p2)

        for j, e in enumerate(edge_props):
            p1 = nx.get_edge_attributes(self.internal.simdata[inherited_state], e)
            p2 = nx.get_edge_attributes(partner.internal.simdata[inherited_state], e)
            utils.set_edge_attributes(child.internal.simdata[inherited_state], n, p1 if choices[1][j] else p2)
        return child


class Attractor(object):
    def __init__(self, environment, position, strength, a_id=None, energy_value=100,
                 attractor_type='Sensory'):
        self.a_id = a_id
        self.position = np.array(position)
        self.strength = strength
        self.energy_value = energy_value  # used to give the SensorMover if they converge to it.
        environment.add_attractor(self, attractor_type)
        self.environment = environment

        self.field_type = 'Sensory'

    @property
    def field_permeability(self):
        return self.environment.field_permeability