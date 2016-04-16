# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 19:54:59 2016

File holding the different models

@author: brian
"""

import numpy as np
from scipy.spatial.distance import euclidean

from pyBrainNetSim.generators.basic import sensor_mover, rand_sensor_mover
from pyBrainNetSim.simulation.simnetwork import HebbianNetworkBasic
from world import Individual


class SensorMoverBase(Individual):
    def __init__(self, environment, position=None, initial_network=None, sensors=None,
                 motor_dir=None, internal_neurons=16, props = None,
                 weights=None, *args, **kwargs):

        super(SensorMoverBase, self).__init__(environment, position, **kwargs)
        if initial_network is not None:
            self.initial_net = initial_network
        else:
            self.initial_net = rand_sensor_mover(internal_neurons=internal_neurons, edge_weights='Rand',
                                                 sensor_dir=sensors, motor_dir=motor_dir, pct_excit=0.7)
        self.internal = HebbianNetworkBasic(self.initial_net, **kwargs)
        self.internal.initiate_simulation()
        self.motor_activations = []
        self.t = 0.
         
    def sim_time_step(self):
        """
        Evolve internal network and apply external sensory inputs and
        motor outputs.
        """
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
        snIDs = [nID for nID,nAttrDict in ng.node.items() 
                if (nAttrDict['node_class']=='Sensory')]
        # 'Attractor' field value at each sensory position
        for nID in snIDs:
            ng.node[nID]['signal'] = \
            self.environment.get_attractor_at_position(self.position + ng.node[nID]['sensor_direction'])
            ng.node[nID]['spontaneity'] = ng.node[nID]['stimuli_fxn']\
                (ng.node[nID]['signal'], ng.node[nID]['stimuli_min'], ng.node[nID]['stimuli_max'],
                 ng.node[nID]['stimuli_sensitivity'], ng.node[nID]['sensory_mid'])

    def _motor_actions(self, ng):
        mnIDs = [nID for nID,nAttrDict in ng.node.items()
                 if (nAttrDict['node_class']=='Motor' and nAttrDict['value']>0.)]
        move = np.array([0.,0.])
        if mnIDs is None:
            return move
        for nID in mnIDs:
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

    def dist_to(self, target):
        return euclidean(self.position, target)


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