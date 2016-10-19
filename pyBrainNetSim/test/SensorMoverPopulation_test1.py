# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 14:48:56 2016

@author: brian
"""

import sys
sys.path.append('../../')
import numpy as np
from pyBrainNetSim.generators.networks import SensorMoverProperties, \
 SensoryNodeProperties, InternalNodeProperties, \
 EdgeProperties, MotorNodeProperties
from pyBrainNetSim.models.network import NeuralNetData
from pyBrainNetSim.models.world import Environment, Individual, Attractor
from pyBrainNetSim.models.individuals import SensorMover
from pyBrainNetSim.drawing.viewers import vTrajectory
from pyBrainNetSim.simulation.evolution import SensorMoverPopulation
from scipy.stats import binom
import matplotlib.pyplot as plt
import pandas as pd

# Method 2: Explicit use of distributions
ipd = InternalNodeProperties(**{"number_neurons": 4,
                                      'energy_value': 2.,
                                      'threshold': 1.,
                                      'spontaneity': 1.})
spd = SensoryNodeProperties(**{'energy_value': 2, 'spontaneity': 1., 'threshold': 1.})
mpd = MotorNodeProperties(**{'energy_value': 2})
wpd = EdgeProperties()
smpd = SensorMoverProperties(ipd, spd, mpd, wpd)
G = smpd.create_digraph()


# Create environment
scale , x0sm, y0sm, x0att, y0att =10., .5, .5, .1, .2
w1 = Environment(max_point = scale * np.array([1.,1.]))
a0 = Attractor(environment=w1, position=(scale * x0att, scale * y0att), strength=10.)

smp = SensorMoverPopulation(w1, smpd, initial_population_size=9)
smp.sim_time_steps(max_iter=7)
traj = smp.trajectory
te = smp.network_attr('total_energy')

sm1 = smp.individuals['G0_I0']
#print sm1.efficiency()
smp.draw_top_networkx()
print "IS DEAD: %s" % sm1.is_dead
print smp.individual_efficiency()
#sm1sd = sm1.internal.simdata
#sm1n1 = smp.individuals['G0_I0'].internal.simdata[-1]   
#print "SIGNAL"
#print pd.DataFrame(sm1sd.neuron_ts('signal'))
#print "ENERGY"
#print pd.DataFrame(sm1sd.neuron_ts('energy_value'))
#print "DEAD_VECT"
#print pd.DataFrame(sm1sd.neuron_group_property_ts('excitatory_to_inhibitory_ratio'))
#print "THRESHOLD"
#print pd.DataFrame(sm1sd.neuron_ts('threshold'))
#print "SPONT_VECT"
#print pd.DataFrame(sm1sd.neuron_group_property_ts('spont_vector'))[sorted(sm1n1.nodes())]
#smp.hist_population_attr_at_time('energy_value', 3, stacked=True)
#smp.plot_efficiency()