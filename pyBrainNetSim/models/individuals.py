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
                 reproduction_cost=0.5, reproductive_threshold=10, data_cutoff=None, global_time_born=0, *args, **kwargs):
        """Class used to simulate the how a 'SensorMover' individual operates in the environment it is placed in."""
        self.internal = []
        self.motor_activations = []
        self._sensory_gradients = []  # along trajectory
        self._internal_sensory_signals = []  # what the sensors sense.
        self.consumed = []
        self.set_internal_network(initial_network)
        super(SensorMover, self).__init__(environment, position=position, ind_id=ind_id,
                                          reproduction_cost=reproduction_cost,
                                          reproductive_threshold=reproductive_threshold,
                                          reproduction_mode=reproduction_mode,
                                          data_cutoff=data_cutoff,
                                          global_time_born=global_time_born,
                                          *args,**kwargs)
        self._energy_final, self._energy_initial, self._energy_consumed,self._energy_lost = {}, {}, {}, {}
        self._energy_added, self._num_nodes_firing = {}, {}
        self.t = -1
        number_of_children, number_of_parents = 1., 2.
        self.energy_reduction_factor = number_of_parents / (number_of_children + number_of_parents)
        self.reproductive_energy_cost = 0.
         
    def sim_time_step(self, driven_nodes=None):
        """Evolve internal network and apply external sensory inputs and motor outputs."""
        self.initialize_step()
        if self.is_dead:
            return
        target = self.found_target()
        if target:  # 1. Award energy & 2. Move/Create new attractor
            self.internal.add_value('energy_value',
                                    value=target.energy_value/float(self.internal.simdata[-1].number_of_nodes()),
                                    time_id=-1)
            self.environment.rm_attractor(target.a_id)
            self._energy_added[self.t] = target.energy_value
        self._sensory_gradients.append(self.environment.attractor_gradient_at(self.position, field_type='Sensory'))
        self._evaluate_sensory_signals(self.internal.simdata[-1])  # get field & fire sensory neurons at position
        self.internal.evolve_time_step(driven_nodes=driven_nodes)  # evolve 1 internal state's time-period
        self.move(self._motor_actions(self.internal.simdata[-2]))  # apply motor output
        self.update(self.internal.simdata)
        return

    def sim_time_steps(self, max_iter=10):
        """
        Simulate in time.
        :param int max_iter: Maximum number of iterations
        :return: None
        """
        while self.t < max_iter:
            self.sim_time_step()

    def initialize_step(self):
        self.reproductive_energy_cost = 0.
        self.t += 1
        self._energy_added.update({self.t: 0.})

    def reset_data(self):
        super(SensorMover, self).reset_data()
        self._sensory_gradients = [self.environment.attractor_gradient_at(self.position, field_type='Sensory')]
        self._energy_final = {nid: {} for nid in self.internal.simdata[-1].nodes()}
        self._energy_initial = {nid: {} for nid in self.internal.simdata[-1].nodes()}
        self._energy_consumed = {nid: {} for nid in self.internal.simdata[-1].nodes()}
        self._energy_lost = {nid: {} for nid in self.internal.simdata[-1].nodes()}
        self._energy_added = {}
        self._num_nodes_firing = {}
        self._prior_energy = self.internal.simdata[-1].energy_vector
        
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
        mn_ids = [n_id for n_id in ng.postsyn_nodes if n_id in ng.nodes(node_class='Motor')]
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

    def update(self, nns):
        for i, nid in enumerate(nns[-1].nodes()):
            if self._energy_final.has_key(nid):
                if len(nns) < 2:
                    self._energy_initial[nid].update({self.t: 0.})
                    self._energy_consumed[nid].update({self.t: 0.})
                else:
                    self._energy_initial[nid].update({self.t: nns[-2].energy_vector[i]})
                    self._energy_consumed[nid].update({self.t: nns[-1].energy_consumption_vector[i]})
                self._energy_final[nid].update({self.t: nns[-1].energy_vector[i]})
                self._energy_lost[nid].update({self.t: self.reproductive_energy_cost})
            else:
                if len(nns) < 2:
                    self._energy_initial.update({nid: {self.t: 0.}})
                    self._energy_consumed.update({nid: {self.t: 0.}})
                else:
                    self._energy_initial.update({nid: {self.t: nns[-2].energy_vector[i]}})
                    self._energy_consumed.update({nid: {self.t: nns[-1].energy_consumption_vector[i]}})
                self._energy_final.update({nid: {self.t: nns[-1].energy_vector[i]}})
                self._energy_lost.update({nid: {self.t: self.reproductive_energy_cost}})

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

    def set_internal_network(self, network, t0=0):
        self.internal = network
        if len(self.internal.simdata)>0:
            self.internal.initial_net = network.simdata[-1]
        self.internal.initiate_simulation(network, t0=t0)

    @property
    def sensory_gradients(self):
        """Return a time-series of the sensory gradients as a function of the trajectory and the environment.
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
        target = None
        for attr in self.environment.attractors.itervalues():
            if np.all(self.position == attr.position):
                target = attr
                self.consumed.append(attr)
        return target

    def efficiency(self, to='Sensory', *args, **kwargs):
        """Method used to measure of how much motor energy has been expended in an effort to get to the target.
        efficiency = convolution(sensory gradient, velocity) / motor energy expenditure."""
        sn = self.internal.simdata
        # motor_energy = sn.neuron_group_property_ts('energy_vector')[sn[-1].nodes(node_class='Motor')]
        motor_energy = self.energy[sn[-1].nodes(node_class='Motor')]
        numerator = np.multiply(self.velocity, self.sensory_gradients)[1:].sum(axis=1)
        denominator = motor_energy.sum(axis=1) if self.is_living else motor_energy.sum(axis=1)
        denominator[denominator == 0] = 1.
        eff = numerator / denominator
        return eff

    def reproduce(self, mode='asexual', new_environment=True, energy_cost=0.50, partner=None, *args, **kwargs):
        child = super(SensorMover, self).reproduce(mode=mode, new_environment=new_environment, partner=partner,
                                                   *args, **kwargs)
        if child is None:
            return None
        clone = child._cloned_from
        clone.reproductive_energy_cost = -clone.reproduction_cost * \
                            float(clone.total_energy.iloc[-1]) / clone.internal.simdata[-1].number_of_nodes()
        if partner:
            partner = child._not_cloned_from
            partner.reproductive_energy_cost = clone.reproductive_energy_cost
            partner_e = partner.internal.simdata[-1].total_energy / partner.internal.simdata[-1].number_of_nodes()
            child.internal.add_value('energy_value', value=partner_e, time_id=-1)
            child.internal.add_value('energy_value', value=partner.reproductive_energy_cost, time_id=-1)
        child.internal.add_value('energy_value', value=clone.reproductive_energy_cost, time_id=-1)
        child.internal.multiply_value('energy_value', value=clone.energy_reduction_factor/2., time_id=-1)
        child.update(child.internal.simdata)
        for p in child.parents:
            p.internal.add_value('energy_value', value=p.reproductive_energy_cost, time_id=-1)
            p.internal.multiply_value('energy_value', value=p.energy_reduction_factor, time_id=-1)
            p.update(p.internal.simdata)
        return child

    @property
    def reproductive_energy_threshold(self):
        return self.internal.simdata[-1].max_energy_consumption_per_time_period * self.reproduction_threshold

    @property
    def reproductive_energy(self):
        return (1. - self.reproduction_cost) * self.internal.simdata[-1].total_energy

    @property
    def energy_initial(self, *args, **kwargs):
        return pd.DataFrame(self._energy_initial)

    @property
    def energy_final(self, *args, **kwargs):
        return pd.DataFrame(self._energy_final)

    @property
    def energy_consumed(self):
        return pd.DataFrame(self._energy_consumed)

    @property
    def energy_lost(self):
        return pd.DataFrame(self._energy_lost)

    @property
    def energy_added(self):
        return pd.Series(self._energy_added)

    @property
    def total_energy(self):
        return self.energy_initial.sum(axis=1)

    @property
    def num_nodes_firing(self):
        return pd.Series(self._num_nodes_firing)

    @property
    def energy_balance(self):
        df = pd.concat([self.energy_initial.sum(axis=1), self.energy_final.sum(axis=1), self.energy_consumed.sum(axis=1)
                           , -self.energy_lost.sum(axis=1), self.energy_added,
                        # self.energy_final.sum(axis=1) + self.energy_consumed.cumsum() - self.energy_lost.cumsum()
                        # + self.energy_added
                        ], axis=1)
        df.rename(
            columns={0: "Initial E", 1: "Final E", 2: "Consumed E", 3: "Lost E", 4: "Added E", 5: "Global E"},
            inplace=True)
        return df

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
