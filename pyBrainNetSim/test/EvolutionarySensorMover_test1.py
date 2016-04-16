# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:12:09 2016

@author: brian
"""


import sys
sys.path.append('../../')
from pyBrainNetSim.models.individuals import SensorMover
from pyBrainNetSim.models.world import Environment, Attractor
from pyBrainNetSim.solvers.solver import SensorMoverEvolutionarySolver
from pyBrainNetSim.generators.random import SensorMoverPropertyDistribution


class EvolutionarySolverTest(SensorMoverEvolutionarySolver):
    def test1(self):
        self.simulate_generation()

if __name__ == '__main__':  
    
    e0 = Environment((10,10))
    a0 = Attractor(environment=e0, location=(0, 0), strength=1., decay_rate=0.1)
    smpd = SensorMoverPropertyDistribution()
    G = smpd.create_digraph()    
    ev = EvolutionarySolverTest(e0, num_individuals=10)
    ev.test1()
#    print ts.trajectory
#    print ts.internal.simdata.get_node_dynamics('presyn_vector')
#    print dat.neuron_ts(neuronID='S0',prop='energy_value')
#    print dat.get_node_dynamics('presyn_vector')[sorted(d0.nodes())]