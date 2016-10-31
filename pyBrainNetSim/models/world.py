# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:16:58 2016

@author: brian
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from pyBrainNetSim.drawing.viewers import vTrajectory
from scipy.spatial.distance import euclidean
from scipy.stats import randint
from scipy.ndimage import zoom
from pyBrainNetSim.utils import cart2pol
import pyBrainNetSim.utils as utils
import copy
import random


class Environment(object):
    
    def __init__(self, origin=(0., 0.), max_point=(20., 20.), deltas=1., field_permeability=0.5, *args, **kwargs):

        self.d_size, self.origin, self.max_point, self.deltas, self.c_grid = None, None, None, None, None
        self.change_world(origin, max_point, deltas)
        self._individuals = {}
        self.attractors = {}
        self.field_permeability = 0.5
        self.fields = {'Sensory': ExponentialDecayScalarField('Sensory', self.c_grid, self.field_permeability)}
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

        slices = [slice(self.origin[i], self.max_point[i]+self.deltas[i], self.deltas[i]) for i in range(self.d_size)]
        self.c_grid = np.mgrid[slices]  # cartesian grid

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

    def add_random_attractor(self, field_type='Sensory'):
        pos = np.array([randint(low=self.origin[i], high=self.max_point[i]).rvs() for i in range(self.d_size)])
        attract = Attractor(self, pos, a_id='s%s' % self.an,
                            strength=randint(low=8, high=20),
                            energy_value=randint(low=50, high=100),
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

    def plot_attractor_field(self, field_type='Sensory', attractor_id=None, upsample_factor=1, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        dx, dy = 0.05, 0.05
        cmap = plt.get_cmap('Blues')
        if upsample_factor > 1:
            x, y, z = self.__upsample_attractor_field(factor=int(upsample_factor), order=1,
                                                      **{'field_type': field_type, 'attractor_id': attractor_id})
        else:
            x, y = self.c_grid
            z = self.attractor_field(field_type, attractor_id)
        z = z[:-1, :-1]
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
        ax.contourf(x[:-1, :-1] + dx/2.,
                    y[:-1, :-1] + dy/2., z, levels=levels,
                    cmap=cmap)
        for attr in self.attractors.itervalues():
            ax.scatter(*attr.position, s=10, c='k', edgecolors='w', alpha=0.9)
        self._format_plot(ax)
        return ax

    def plot_individual_trajectory(self, individual=None, at_time=None, show_attractor_field=True, field_type='Sensory',
                                   attractor_id=None, upsample_factor=1, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.cla()
        individual = [individual] if isinstance(individual, str) else individual
        if show_attractor_field:
            ax = self.plot_attractor_field(field_type, attractor_id, upsample_factor, ax, **kwargs)
        individuals = self.individuals if individual is None else {s_id: self.individuals[s_id] for s_id in individual}
        for ind_id, ind in individuals.iteritems():
            t = at_time if isinstance(at_time, int) else len(ind.trajectory)
            trajectory = ind.trajectory[:t]
            if len(trajectory) == 1:
                ax.scatter(*trajectory[-1], s=15, marker='s', c='r', edgecolors='w', alpha=0.9)
            elif t > 0:
                points = np.array(utils.points_to_segments(trajectory))
                lines = vTrajectory(points)
                lines.text.set_fontsize(9)
                ax.scatter(*trajectory[-1], s=15, marker='s', c='r', edgecolors='w', alpha=0.9)
                ax.add_collection(lines)
        self._format_plot(ax)
        return ax

    def _format_plot(self, ax):
        ax.set_aspect('equal', adjustable='box')
        ax.set(xlim=[self.origin[0], self.max_point[0]], ylim=[self.origin[1], self.max_point[1]], title="Environment")

    def __upsample_attractor_field(self, factor=5, order=1, **kwargs):
        x = zoom(self.c_grid[0], factor, order=order)
        y = zoom(self.c_grid[1], factor, order=order)
        z = zoom(self.attractor_field(**kwargs), factor, order=order)
        return x, y, z

    def generate_position(self, position):
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


class Individual(object):
    
    def __init__(self, environment=None, position=None, ind_id=None, reproduction_cost=0.5, reproductive_threshold=10,
                 reproduction_mode='asexual', *args, **kwargs):
        self.parents, self.children, self.generation, self.kin, self.t = [], [], 0, 0, 0.
        self.reproduction_cost = reproduction_cost
        self.reproduction_threshold = reproductive_threshold
        self.reproduction_mode = reproduction_mode
        self._environment, self._position, self.d_size = None, None, 2
        self.ind_id = self._generate_id(ind_id)
        self.set_environment(environment)
        self.set_position(position)
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
    def position(self):
        return np.array(self._position)
        
    @property
    def velocity(self):
        # return np.vstack(([0., 0.], self.trajectory[1:]-self.trajectory[:-1]))
        return self.trajectory[1:]-self.trajectory[:-1]

    def dist_to(self, target):
        return euclidean(self.position, target)

    def vector_to(self, target, normalized=True):
        target = np.array(target) if not isinstance(target, np.ndarray) else target
        vector = (self.position - target)
        if normalized:
            vector /= euclidean(self.position, target)
        return vector

    def efficiency(self, *args, **kwargs):
        """
        Method to evaluate an objective function. To be overriden.
        :param to:
        :return:
        """
        pass

    def reproduce(self, mode='asexual', partner=None, new_environment=True, error_rate=0., *args, **kwargs):
        child = None
        if mode == 'asexual':
            child = self._reproduce_asexually(error_rate)
        elif mode == 'sexual':
            child = self._reproduce_sexually(partner)
        env = self.environment if not new_environment else None
        child.set_environment(env, *args, **kwargs)
        _tmp = [child.environment.add_attractor(attractor, attractor.field_type)
                for attractor in self.environment.attractors.values()]
        # child.set_position()
        self.generation += 1
        child.generation = self.generation
        child.kin = child._generate_id()
        child.ind_id = "G%d_I%s" % (child.generation, child.kin)
        self.children.append(child)

        return child

    def _reproduce_asexually(self, error_rate=0.):
        child = copy.copy(self)
        child.parents = [self]
        return child

    def _reproduce_sexually(self, partner):
        child = copy.copy(self)
        child.parents = [self, partner]
        return child


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
        indx = self._loc_to_indx(location).astype(np.int)
        _field = 0.
        if self.d_size == 1:
            _field = self.field(source_id)[indx]
        elif self.d_size == 2:
            _field = self.field(source_id)[indx[0]][indx[1]]
        elif self.d_size == 3:
            _field = self.field(source_id)[indx[0]][indx[1]][indx[2]]
        return _field

    def gradient(self, source_id=None):
        _grad = np.gradient(self.field(source_id)) if source_id in self.sources\
            else self._grad
        _grad.reverse()  # REVERSE ORDERED - DOCUMENTED INCORRECTLY IN THE NUMPY DOCS
        return _grad

    def gradient_at(self, location, source_id=None):
        indx = self._loc_to_indx(location).astype(np.int)
        _grad = grad = self.gradient(source_id)
        if self.d_size == 1:
            grad = np.array([_grad[indx]])
        elif self.d_size == 2:
            grad = np.array([_grad[i][indx[0]][indx[1]] for i in range(len(indx))])
        elif self.d_size == 3:
            grad = np.array([_grad[i][indx[0]][indx[1]][indx[2]] for i in range(len(indx))])
        return grad

    def _loc_to_indx(self, location):
        return location


class ExponentialDecayScalarField(ScalarField):
    def __init__(self, field_type, c_grid, field_permeability=1., *args, **kwargs):
        super(ExponentialDecayScalarField, self).__init__(field_type, c_grid, *args, **kwargs)
        self.field_permeability = field_permeability

    def function(self, location=(0, 0), strength=1):
        p_grid = cart2pol(self.c_grid, location)
        return strength * np.exp(-p_grid[0] * self.field_permeability)
