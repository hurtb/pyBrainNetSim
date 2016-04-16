# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:16:58 2016

@author: brian
"""
import numpy as np
import matplotlib.pyplot as plt
from pyBrainNetSim.drawing.viewers import vTrajectory


class Environment(object):
    
    def __init__(self, origin=(0., 0.), max_point=(20., 20.), deltas=1., *args, **kwargs):

        self.d_size, self.origin, self.max_point, self.deltas, self.grid = None, None, None, None, None
        self.change_world(origin, max_point, deltas)
        self._individuals = []
        self.attractors = []

    def change_world(self, origin, max_point, deltas=1.):
        self.d_size = len(max_point)
        if self.d_size >3: # too big
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
        self.grid = np.mgrid[slices]

    def add_individual(self, individual):
        if type(individual) in ['Individual']:
            self._individuals.append(individual)
        
    def add_attractor(self, attr):
        """ e.g. food
        :param attr:
        """
        self.attractors.append(attr)
        
    def get_attractor_at_position(self, position):
        """Sum of all attractor fields at position.
        :param position:
        """
        return np.sum([attr.field_at(position) for attr in self.attractors])

    def get_attractor_field(self):
        pass

    def plot_individual_trajectories(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        lines = vTrajectory(x, y)
        lines.text.set_fontsize(9)
        ax.hold()
        ax.scatter(*lines.get_segments()[-1][-1])
        ax.add_collection(lines)
        ax.set_xlim(left=self.x_min, right=self.x_max)
        ax.set_ylim(bottom=self.y_min, top=self.y_max)
    
    @property
    def positions(self):
        return []
    
    @property
    def individuals(self):
        return self._individuals
        
    def _build_env(self):
        pass
        
from scipy.spatial.distance import euclidean
class Attractor(object):
    def __init__(self, environment, location, strength, decay_rate):
        self.loc = location
        self.strength = strength
        self.field_type = 'exponential' # 'linear'|'exponential'
        self.decay_rate = decay_rate
        environment.add_attractor(self)
        self.environment = environment
    
    def _linear_field(self, r):
        return np.max([self.strength - r * self.decay_rate, 0])
        
    def _exp_field(self, r):
        return self.strength * np.exp(-r * self.decay_rate)

    @property
    def field(self):
        env = self.environment

        r = np.sqrt(self.loc[0] - env.grid[0]) # FINISH STATEMENT
        return None


    def field_at(self, position):
        """Return attractor field strength at position
        :param position:
        """
        
        if self.field_type == "linear":
            return self._linear_field(euclidean(self.loc, position))
        elif self.field_type == "exponential":
            return self._exp_field(euclidean(self.loc, position))
        
    def evolve_diffuse(self, permeability):
        """ to do: Diffusion process $dF/dt = \alpha\nabla^2 F$ """
        pass

    
class Individual(object):
    
    def __init__(self, environment=None, position=None, *args, **kwargs):
        if isinstance(environment, Environment):
            environment.add_individual(self)
            self.d_size = environment.d_size
        self.environment = environment
        if position  is None:
            self._position = np.array([0,0])
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
        return self._position
        
    @property
    def velocity(self):
        return self.trajectory[1:]-self.trajectory[:-1]
