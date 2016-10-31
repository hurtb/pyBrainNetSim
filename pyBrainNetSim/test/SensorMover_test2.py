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
from pyBrainNetSim.generators.network import SensorMoverProperties


class EvolutionarySolverTest(SensorMoverEvolutionarySolver):
    def test1(self):
        self.solve()


class SensorMoverTest(SensorMover):
    
    def test1(self):
        sim_options = {'max_iter':20}        
        self.sim_time_steps(**sim_options)

if __name__ == '__main__':  
    
    e0 = Environment((10,10))
    a0 = Attractor(environment=e0, location=(0, 0), strength=1., decay_rate=0.1)
    smpd = SensorMoverProperties()
    G = smpd.create_digraph()    
    ts = SensorMoverTest(e0, position=(5,5), initial_network=G, \
                            **{'initial_fire':'prescribed', 'prescribed':['S0']})    
    ev = EvolutionarySolverTest(e0)
    ev.test1()
    
    ts.test1()
    dat = ts.internal.simdata
    d0 = dat[0]    
    d1 = dat[1]
    d2 = dat[2]
#    print ts.trajectory
#    print ts.internal.simdata.neuron_group_property_ts('presyn_vector')
#    print dat.neuron_ts(neuronID='S0',prop='energy_value')
#    print dat.neuron_group_property_ts('presyn_vector')[sorted(d0.nodes())]