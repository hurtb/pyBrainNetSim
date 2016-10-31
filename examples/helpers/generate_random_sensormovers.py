# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 14:48:56 2016

@author: brian
"""

import sys
sys.path.append('../')
import numpy as np
from pyBrainNetSim.generators.network import SensorMoverProperties, \
 SensoryNodeProperties, InternalNodeProperties, \
 EdgeProperties, MotorNodeProperties
from pyBrainNetSim.models.network import NeuralNetData
from pyBrainNetSim.models.world import Environment, Individual, Attractor
from pyBrainNetSim.models.individuals import SensorMover
from pyBrainNetSim.drawing.viewers import vTrajectory
from pyBrainNetSim.simulation.simnetwork import HebbianNetworkBasic

# Method 1: Simple, using default values/distributions
smpd1 = SensorMoverProperties()
G1 = smpd1.create_digraph()  # creates a networkx DiGraph

# Method 2: Explicit use of distributions
ipd = InternalNodeProperties(**{"number_neurons": 5})
spd = SensoryNodeProperties()
mpd = MotorNodeProperties()
wpd = EdgeProperties()
smpd2 = SensorMoverProperties(ipd, spd, mpd, wpd)
G2 = smpd2.create_digraph()

# Create environment
scale , x0sm, y0sm, x0att, y0att =10., .5, .5, .1, .1
w1 = Environment(max_point = scale * np.array([1.,1.]))
a0 = Attractor(environment=w1, position=(scale * x0att, scale * y0att), strength=10.)
sm1 = SensorMover(environment=w1, initial_network=HebbianNetworkBasic(G1), position=[scale * x0sm, scale * x0sm])
sm1n0 = sm1.internal.simdata[0]
sm1.sim_time_steps(max_iter=5)
sm1n1 = sm1.internal.simdata[-1]
sm1ds = sm1.internal.simdata
#print sm1n1.energy_vector
#print sm1.internal.simdata.neuron_group_property_ts('postsyn_vector')[['M1','M2']]
#print sm1.efficiency()
print sm1.trajectory
print sm1.sensory_gradients
print sm1.efficiency()

nodes = ['S0','S1','S2','S3']
#print sm1.internal.simdata.neuron_group_property_ts('spont_vector')[['M0','M1','M2','M3']]
print "Pre:\n%s" % sm1.internal.simdata.neuron_group_property_ts('presyn_vector')[nodes]
print "Post:\n%s" % sm1.internal.simdata.neuron_group_property_ts('postsyn_vector')[nodes]
