import matplotlib as mpl
import pyBrainNetSim.models.world as world
import pyBrainNetSim.generators.network as nw
import pyBrainNetSim.drawing.viewers as vw
import pyBrainNetSim.simulation.evolution as evo
from pyBrainNetSim.generators.settings import test as settings
import cPickle as pickle
import numpy as np
import pandas as pd
import networkx as nx


class SensorMoverPopulationTest(object):
    def __init__(self, number_of_individuals=2, time_iterations=5):
        self.number_of_individuals, self.time_iterations = number_of_individuals, time_iterations

    def run_simulation(self, time_iterations=None):
        time_iterations = self.time_iterations if time_iterations is None else time_iterations

        my_environment = world.Environment(origin=(0, 0), max_point=(20, 20), field_permeability=1.)
        food = world.Attractor(environment=my_environment, position=(5, 5), strength=10.)  # add "food"
        sm_pd = nw.SensorMoverProperties(internal=nw.InternalNodeProperties(**settings.NODE_PROPS['Internal']),
                                         motor=nw.MotorNodeProperties(**settings.NODE_PROPS['Motor']),
                                         sensors=nw.SensoryNodeProperties(**settings.NODE_PROPS['Sensory']),
                                         weights=nw.EdgeProperties(prop=settings.SYNAPSES))
        self.smp = evo.SensorMoverPopulation(my_environment, sm_pd,
                                        initial_population_size=self.initial_number_of_individuals,
                                        reproductive_cost=0.1,
                                        reproductive_threshold=20,
                                        reproductive_mode='sexual', data_cutoff=2)
        self.smp.sim_time_steps(max_iter=time_iterations)

    def validate_movement(self):
        mwe = {}
        for iid, i in self.smp.individuals.iteritems():
            mec = i.energy_consumed[i.internal.simdata[-1].nodes(node_class='Motor')]
            speed = pd.Series([np.linalg.norm(v) for v in i.velocity[1:]], index=(range(0, len(i.velocity) - 1)))
            mac_tot = mec.sum(axis=1)
            sn = i.internal.simdata
            if len(mac_tot) > 0:
                s1 = speed > 0.
                m1 = mac_tot == 0.
                movement_without_energy = pd.Series(s1 & m1).any()

    def validate_energy(self):
        energy = {}
        for iid, i in self.smp.individuals.iteritems():
            energy.update({iid: i.energy})


if __name__ == '__main__':
    smpt = SensorMoverPopulationTest(number_of_individuals=2, time_iterations=5)
    smpt.run_simulation()
    smpt.validate_movement()



