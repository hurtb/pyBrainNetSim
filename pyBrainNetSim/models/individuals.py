# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 19:54:59 2016

File holding the different models

@author: brian
"""

import numpy as np
from scipy.spatial.distance import euclidean
import network as nx

# from pyBrainNetSim.generators.basic import sensor_mover, rand_sensor_mover
from pyBrainNetSim.simulation.simnetwork import HebbianNetworkBasic
from world import Individual


class SensorMoverBase(Individual):
    def __init__(self, environment, position=None, initial_network=None, sensors=None,
                 motor_dir=None, internal_neurons=16, props = None,
                 weights=None, *args, **kwargs):

        super(SensorMoverBase, self).__init__(environment, position, **kwargs)
        if initial_network is not None:
            self.initial_net = initial_network
        # else:
            # TODO: implement default SensorMover Network
            # self.initial_net = rand_sensor_mover(internal_neurons=internal_neurons, edge_weights='Rand',
            #                                      sensor_dir=sensors, motor_dir=motor_dir, pct_excit=0.7)
        self.internal = HebbianNetworkBasic(self.initial_net, **kwargs)
        self.internal.initiate_simulation()
        self.motor_activations = []
        self._sensory_gradients = []  # along trajectory
        self._internal_sensory_signals = []  # what the sensors sense.
        self._best_path_vector = []  # along trajectory of the individual
        self.t = 0.
         
    def sim_time_step(self):
        """
        Evolve internal network and apply external sensory inputs and motor outputs.
        """
        self._sensory_gradients.append(self.environment.attractor_gradient_at(self.position, field_type='Sensory'))
        # self._best_path_vector.append(self.vector_to())
        # get sensory field for each sensor at individual's position fire sensory neurons accordingly
        sens_firing = []
        self.internal.evolve_time_step(driven_nodes=sens_firing)  # evolve 1 internal state's timeperiod
        
        # apply motor output accordingly
        self._sensory_pos(self.internal.simdata[-1])
        self.move(self._motor_actions(self.internal.simdata[-1]))
        self.t += 1
        
    def sim_time_steps(self, max_iter=10):
        while self.t < max_iter:
            self.sim_time_step()
        
    def _sensory_pos(self, ng):
        sn_ids = [nID for nID, nAttrDict in ng.node.items()
                  if (nAttrDict['node_class'] == 'Sensory')]
        # 'Attractor' field value at each sensory position
        for nID in sn_ids:
            ng.node[nID]['signal'] = \
                self.environment.attractor_field_at(self.position + ng.node[nID]['sensor_direction'],
                                                    field_type='Sensory')
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

    def _update_energy(self, ng):
        for nID in ng.nodes():
            ng.node[nID]['energy_value'] = ng.node[nID]['energy_dynamics'](ng.node[nID]['energy_value'],
                                                                           ng.node[nID]['energy_consumption'])

    @property
    def sensory_gradients(self):
        return np.array(self._sensory_gradients)

    def efficiency(self, to='food', *args, **kwargs):
        """
        Method used to measure of how much motor energy has been expended in an effort to get to the target.
        efficiency = convolution(sensory gradient, velocity) / motor energy expenditure
        :param to:
        :param args:
        :param kwargs:
        :return:
        """
        sn = self.internal.simdata
        motor_energy = sn.get_node_dynamics('energy_vector')[sn[-1].nodes(node_class='Motor')]
        motor_energy[motor_energy == 0.] = 1.  # such that no energy exp
        eff = np.multiply(self.velocity, self.sensory_gradients).sum(axis=1)/motor_energy[1:].sum(axis=1)
        return eff


class SensorMover(SensorMoverBase):
    pass


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
