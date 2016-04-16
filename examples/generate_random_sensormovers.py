# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 14:48:56 2016

@author: brian
"""

import sys
sys.path.append('../../')
from pyBrainNetSim.generators.random import SensorMoverPropertyDistribution, \
 SensoryPropertyDistribution, InternalPropertyDistribution, \
 WeightPropertyDistribution, MotorPropertyDistribution

# Method 1: Simple, using default values/distributions
smpd1 = SensorMoverPropertyDistribution()
G1 = smpd1.create_digraph()

# Method 2: Explicit use of distributions
ipd = InternalPropertyDistribution()
spd = SensoryPropertyDistribution()
mpd = MotorPropertyDistribution()
wpd = WeightPropertyDistribution()
smpd2 = SensorMoverPropertyDistribution(ipd, spd, mpd, wpd)
G2 = smpd2.create_digraph()