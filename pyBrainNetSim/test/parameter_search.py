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
from pyBrainNetSim.generators.basic import rand_sensor_mover
from pyBrainNetSim.generators.random import InternalPropertyDistribution, MotorPropertyDistribution, SensoryPropertyDistribution
from scipy.stats import norm, binom, randint, poisson, uniform

INTERNAL = {'number': randint(low=10, high=40),
            'energy': binom(50, .9),
            'energy_consumption': randint(low=1, high=4),
            'threshold': uniform(loc=0.7, scale=1.2),
            'spontaneity': uniform(loc=0.7, scale=1.2)}
internal = InternalPropertyDistribution(INTERNAL)

MOTOR = {'energy': binom(50, 0.5),
         'energy_consumption': randint(low=1, high=4),
         'threshold': uniform(loc=0.7, scale=1.2),
         'spontaneity': uniform(loc=0.0, scale=1.0),
         'direction': [(1, 0), (-1, 0), (0, 1), (0, -1)]}
motor = MotorPropertyDistribution(MOTOR)

SENSORY = {'energy': binom(50, 0.5),
           'energy_consumption': randint(low=1, high=4),
           'threshold': uniform(loc=0.7, scale=1.2),
           'spontaneity':uniform(loc=0.0, scale=1.0),
           'sensitivity': uniform(loc=0.0, scale=1.0),
           'max_stimulation': 1.,
           'min_stimulation': 0.,
           'mid_sensory': uniform(loc=.05, scale=.95),
           }
sensor = SensoryPropertyDistribution(SENSORY)

MAX_ITER = 50


class SensorMoverTest(SensorMover):

    def test1(self):
        sim_options = {'max_iter': MAX_ITER}
        self.sim_time_steps(**sim_options)

if __name__ == '__main__':

    e0 = Environment((10,10))
    a0 = Attractor(environment=e0, location=(0, 0), strength=1., decay_rate=0.1)
    sm = None
    ts = SensorMoverTest(e0, position=(5,5), sensors=[(0,1),(0,-1)],
                         motor_dir=motor.direction, internal_neurons=6,
                            weights = 2.,
                            **{'initial_fire':'prescribed', 'prescribed':['S0']})

    ev = SensorMoverEvolutionarySolver(e0)
    ev.test1()

    ts.test1()
    dat = ts.internal.simdata
    d0 = dat[0]
    d1 = dat[1]
    d2 = dat[2]
    print ts.trajectory
    print ts.internal.simdata.node_group_properties('presyn_vector')