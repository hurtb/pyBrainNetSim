# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:16:58 2016

@author: brian
"""
import numpy as np
import matplotlib.pyplot as plt
from pyBrainNetSim.drawing.viewers import vTrajectory
from scipy.spatial.distance import euclidean
from pyBrainNetSim.utils import  cart2pol


class Environment(object):
    
    def __init__(self, origin=(0., 0.), max_point=(20., 20.), deltas=1., *args, **kwargs):

        self.d_size, self.origin, self.max_point, self.deltas, self.c_grid = None, None, None, None, None
        self.change_world(origin, max_point, deltas)
        self._individuals = []
        self.attractors = []
        self.decay_rate = 0.5
        self.fields = {'Sensory': ExponentialDecayScalarField('Sensory', self.c_grid, self.decay_rate )}

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
        if type(individual) in ['Individual']:
            self._individuals.append(individual)
        
    def add_attractor(self, attr, field_type):
        """ e.g. food
        :param attr:
        """
        self.attractors.append(attr)
        self.fields[field_type].add_point_source(attr.position, attr.strength)

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

    def plot_individual_trajectories(self, ax=None):
        # TODO 3. Create a viewing method.
        if ax is None:
            fig, ax = plt.subplots()
        # lines = vTrajectory(x, y)
        # lines.text.set_fontsize(9)
        # ax.hold()
        # ax.scatter(*lines.get_segments()[-1][-1])
        # ax.add_collection(lines)
        # ax.set_xlim(left=self.x_min, right=self.x_max)
        # ax.set_ylim(bottom=self.y_min, top=self.y_max)
    
    @property
    def positions(self):
        return []
    
    @property
    def individuals(self):
        return self._individuals
        
    def _build_env(self):
        pass


class Attractor(object):
    def __init__(self, environment, position, strength, energy_value=100,
                 attractor_type='Sensory'):
        self.position = position
        self.strength = strength
        self.energy_value = energy_value  # used to give the SensorMover if they converge to it.
        environment.add_attractor(self, attractor_type)
        self.environment = environment

        self.field_type = 'exponential' # 'linear'|'exponential'
        self.decay_rate = 1.

    # def field_function(self, c_grid):
    #     p_grid = cart2pol(c_grid, x0=self.position)
    #     if self.field_type == 'exponential':
    #         scalar = self._exp_field(p_grid[0])
    #     elif self.field_type == 'linear':
    #         scalar = self._linear_field(p_grid[0])
    #     return scalar
    #
    # def _linear_field(self, r):
    #     return np.max([self.strength - r * self.decay_rate, 0])
    #
    # def _exp_field(self, r):
    #     return self.strength * np.exp(-r * self.decay_rate)
    #
    # def _exp_field_cartesian(self, pos, pos0):
    #     return self._exp_field(euclidean(pos, pos0))
    #
    # def _linear_field_cartesian(self, pos, pos0):
    #     return self._linear_field(euclidean(pos, pos0))

    #
    #
    # def field_at(self, position):
    #     """Return attractor field strength at position
    #     :param position:
    #     """
    #     if self.field_type == "linear":
    #         return self._linear_field(euclidean(self.position, position))
    #     elif self.field_type == "exponential":
    #         return self._exp_field(euclidean(self.position, position))
    #
    # def gradient_at(self, position):
    #     grad = np.gradient(self.field, dtype=np.float)
    #     return [grad[dim][position][dim] for dim in range(len(position))]
    #
    # def evolve_diffuse(self, permeability):
    #     """ Diffusion process $dF/dt = \alpha\nabla^2 F$ """
    #     # TODO: create dynamically changing fields
    #     pass

    
class Individual(object):
    
    def __init__(self, environment=None, position=None, *args, **kwargs):
        if isinstance(environment, Environment):
            environment.add_individual(self)
            self.d_size = environment.d_size
        self.environment = environment
        if position  is None:
            self._position = np.array([0, 0])
        elif len(position) == environment.d_size:
            self._position = np.array(position)
        elif position == 'rand':
            self._position = [np.random.randint(0, environment.max_point[0]),
                              np.random.randint(0, environment.max_point[1])]
        else:
            self._position = np.array([0,0])

        self._trajectory = [self._position.copy()]        
        
    def move(self, vector):
        if len(vector) == self.d_size:
            self._position += np.array(vector)
            self._trajectory.append(self._position.copy())
            
    def set_trajectory(self, trajectory):
        self._trajectory = trajectory

    def in_world(self, pos):
        inworld = True
        pos = np.array(pos)
        if np.any(pos > self.environment.max_point) or np.any(pos < self.environment.origin):
            inworld = False
        return inworld

    @property
    def trajectory(self):
        return np.array(self._trajectory) if not isinstance(self._trajectory, np.ndarray) else self._trajectory
        
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

    def trajectory_vector_to(self, target):
        pass

    def efficiency(self, *args, **kwargs):
        """
        Method to evaluate an objective function. To be overriden.
        :param to:
        :return:
        """
        pass


class ScalarField(object):
    def __init__(self, field_type, c_grid, *args, **kwargs):
        self.field_type = field_type
        self.d_size = len(c_grid)
        self.c_grid = c_grid
        self.sources = {}
        self._update_field()

    def function(self, *args, **kwargs):
        return np.zeros_like(self.c_grid[0])

    def add_point_source(self, location, strength):
        self.sources.update({'%s' % (len(self.sources)+1): {'location': location, 'strength': strength}})
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
        indx = self._loc_to_indx(location)
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
        indx = self._loc_to_indx(location)
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
    def __init__(self, field_type, c_grid, decay_rate=1., *args, **kwargs):
        super(ExponentialDecayScalarField, self).__init__(field_type, c_grid, *args, **kwargs)
        self.decay_rate = decay_rate

    def function(self, location=(0, 0), strength=1):
        p_grid = cart2pol(self.c_grid, location)
        return strength * np.exp(-p_grid[0] * self.decay_rate)


