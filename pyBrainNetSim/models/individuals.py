# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 19:54:59 2016

File holding the different models

@author: brian
"""

import numpy as np
from world import Individual
import pandas as pd
import copy


class SensorMover(Individual):
    def __init__(self, environment, position=None, initial_network=None, ind_id=None, reproduction_mode='asexual',
                 reproduction_cost=0.5, reproductive_threshold=10, *args, **kwargs):
        """
        Class used to simulate the how a 'SensorMover' individual operates in the environment it is placed in.
        :param environment:
        :param position:
        :param initial_network:
        :param args:
        :param kwargs:
        :return:
        """
        super(SensorMover, self).__init__(environment, position=position, ind_id=ind_id,
                                          reproduction_cost=reproduction_cost,
                                          reproductive_threshold=reproductive_threshold,
                                          reproduction_mode=reproduction_mode, **kwargs)
        self.internal = []
        self.motor_activations = []
        self._sensory_gradients = []  # along trajectory
        self._internal_sensory_signals = []  # what the sensors sense.
        self.consumed = []
        self.set_internal_network(initial_network)
         
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
        child = self.reproduce(mode=self.reproduction_mode) if self.to_reproduce else None
        return child

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
            # print "pos %s | filtered pos %s" %(self.position + ng.node[nID]['sensor_direction'], pos)
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

    @property
    def to_reproduce(self):
        return True if self.reproductive_energy > self.reproductive_energy_threshold else False

    def __filter_pos(self, pos):
        if not self.in_world(pos):
            for i, p in enumerate(pos):
                if p > self.environment.max_point[i]:
                    pos[i] = self.environment.max_point[i]
                elif p < self.environment.origin[i]:
                    pos[i] = self.environment.origin[i]
        return pos

    def _update_energy(self, ng):
        for nID in ng.nodes():
            ng.node[nID]['energy_value'] = \
                ng.node[nID]['energy_dynamics'](ng.node[nID]['energy_value'], ng.node[nID]['energy_consumption'])

    def set_internal_network(self, network, t0=0):
        self.internal = network
        if len(self.internal.simdata)>0:
            self.internal.initial_net = network.simdata[-1]
        self.internal.initiate_simulation(network, t0=t0)

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

    def reproduce(self, mode='asexual', new_environment=True, energy_cost=0.50, *args, **kwargs):
        child = super(SensorMover, self).reproduce(mode=mode, new_environment=new_environment,
                                                   *args, **kwargs)
        # Create new SensorMover with copied internal features (with error)
        child.set_internal_network(copy.deepcopy(self.internal), t0=self.t)
        p_self_e, p_child_e = self.internal.simdata[-1].total_energy, child.internal.simdata[-1].total_energy
        energy_reduction_factor = (1-self.reproduction_cost) / 2.
        child.internal.multiply_value('energy_value', factor=energy_reduction_factor)
        self.internal.multiply_value('energy_value', factor=energy_reduction_factor)
        # print "%s (e %s [%s], prior e: %s) @ t=%s to create %s (e %s, prior %s)" % \
        #       (self.ind_id, self.internal.simdata[-1].total_energy, self.reproductive_energy_threshold, p_self_e, self.t, child.ind_id,
        #        child.internal.simdata[-1].total_energy, p_child_e)
        return child

    @property
    def reproductive_energy_threshold(self):
        return self.internal.simdata[-1].max_energy_consumption_per_time_period * self.reproduction_threshold

    @property
    def reproductive_energy(self):
        return (1. - self.reproduction_cost) * self.internal.simdata[-1].total_energy


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
