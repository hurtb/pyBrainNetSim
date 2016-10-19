# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:14:31 2016

@author: brian
"""

from pyBrainNetSim.generators.networks import SensorMoverProperties
from pyBrainNetSim.models.network import NeuralNetData
smpd = SensorMoverProperties()
rdn = smpd.create_digraph()
nnd = NeuralNetData(rdn)
print nnd.synapses