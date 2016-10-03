# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 19:54:59 2016

File holding the different models

@author: brian
"""

import numpy as np
from world import Individual
import pandas as pd


class SensorMover(Individual):
    def __init__(self, environment, position=None, initial_network=None, ind_id=None, *args, **kwargs):
        """
        Class used to simulate the how a 'SensorMover' individual operates in the environment it is placed in.
        :param environment:
        :param position:
        :param initial_network:
        :param args:
        :param kwargs:
        :return:
        """
        super(SensorMover, self).__init__(environment, position, ind_id, **kwargs)
        self.internal = initial_network
        self.internal.initiate_simulation()
        self.motor_activations = []
        self._sensory_gradients = []  # along trajectory
        self._internal_sensory_signals = []  # what the sensors sense.
        self.parent = None
        self.consumed = []
        self.t = 0.
         
    def sim_time_step(self):
        """
        Evolve internal network and apply external sensory inputs and motor outputs.
        """
        if self.is_dead:
            return
        if self.found_target():  # 1. Award energy & 2. Move/Create new attractor
            self.internal.add_energy(self.consumed[-1].energy_value)
            self.environment.rm_attractor(self.consumed[-1].a_id)

        self._sensory_gradients.append(self.environment.attractor_gradient_at(self.position, field_type='Sensory'))
        # get sensory field for each sensor at individual's position fire sensory neurons accordingly
        sens_firing = []
        self.internal.evolve_time_step(driven_nodes=sens_firing)  # evolve 1 internal state's time-period
        self._evaluate_sensory_signals(self.internal.simdata[-1])
        self.move(self._motor_actions(self.internal.simdata[-1]))  # apply motor output
        self.t += 1
        
    def sim_time_steps(self, max_iter=10):
        """
        Simulate in time.
        :param int max_iter: Maximum number of iterations
        :return: None
        """
        while self.t < max_iter:
            self.sim_time_step()
        
    def _evaluate_sensory_signals(self, ng):
        """
        Evaluate the sensory signals and transmit the signal.
        :param ng: neural network graph (NeuralNetData)
        """
        sn_ids = [nID for nID, nAttrDict in ng.node.items()
                  if (nAttrDict['node_class'] == 'Sensory')]
        for nID in sn_ids:   # 'Attractor' field value at each sensory position
            pos = self.__filter_pos(self.position + ng.node[nID]['sensor_direction'])
            ng.node[nID]['signal'] = \
                self.environment.attractor_field_at(pos, field_type='Sensory')
            ng.node[nID]['spontaneity'] = ng.node[nID]['stimuli_fxn'](
                 ng.node[nID]['signal'], ng.node[nID]['stimuli_min'], ng.node[nID]['stimuli_max'],
                 ng.node[nID]['stimuli_sensitivity'], ng.node[nID]['sensory_mid'])

    def _motor_actions(self, ng):
        mn_ids = [n_id for n_id in ng.presyn_nodes if n_id in ng.nodes(node_class='Motor')]
        move = np.array([0., 0.])
        if mn_ids is None:
            return move
        for nID in mn_ids:
            move += np.array(ng.node[nID]['force_direction'])
        if not self.in_world(self.position+move):
            for i in range(len(move)):
                if not self.in_world(move[i]+self._position[i]):
                    move[i] = 0.
        return move

    def __filter_pos(self, pos):
        if not self.in_world(pos):
            pos -= 1.
        return pos

    def _update_energy(self, ng):
        for nID in ng.nodes():
            ng.node[nID]['energy_value'] = \
                ng.node[nID]['energy_dynamics'](ng.node[nID]['energy_value'], ng.node[nID]['energy_consumption'])

    @property
    def sensory_gradients(self):
        """
        Return a time-series of the sensory gradients as a function of the trajectory and the environment.
        :return: numpy.ndarray
        """
        return np.array(self._sensory_gradients)

    @property
    def is_dead(self):
        return self.internal.simdata[-1].is_dead

    @property
    def is_living(self):
        return not self.is_dead

    def found_target(self):
        out = False
        for attr in self.environment.attractors.itervalues():
            if np.all(self.position == attr.position):
                out = True
                self.consumed.append(attr)
        return out

    def efficiency(self, to='Sensory', *args, **kwargs):
        """
        Method used to measure of how much motor energy has been expended in an effort to get to the target.
        efficiency = convolution(sensory gradient, velocity) / motor energy expenditure
        :param to:
        :param args:
        :param kwargs:
        :return:
        """
        sn = self.internal.simdata
        motor_energy = sn.neuron_group_property_ts('energy_vector')[sn[-1].nodes(node_class='Motor')]
        motor_energy[motor_energy == 0.] = 1.  # such that no energy exp
        denominator = motor_energy[1:].sum(axis=1) if self.is_living else motor_energy.sum(axis=1)
        eff = np.multiply(self.velocity, self.sensory_gradients).sum(axis=1)/denominator
        return eff

    def plot_efficiency(self, **kwargs):
        eff = pd.Series(self.efficiency())
        eff.plot(style='o-')


class SensorMoverFactory(object):
    """
    Used to create multiple individuals within 1 world.
    """

    def __init__(self, *args, **kwargs):
        pass

    def create_new(self, *args, **kwargs):
        pass

    def create_offspring(self, *args, **kwargs):
        pass
