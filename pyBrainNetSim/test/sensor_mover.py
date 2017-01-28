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


class SensorMoverTest(object):
    def __init__(self, number_of_individuals=2, time_iterations=5):
        self.time_iterations = time_iterations

    def run_simulation(self, time_iterations=5):
        time_iterations = self.time_iterations if time_iterations is None else time_iterations

        my_environment = world.Environment(origin=(0, 0), max_point=(20, 20), field_permeability=1.)
        food = world.Attractor(environment=my_environment, position=(5, 5), strength=10.)  # add "food"
        smpd = nw.SensorMoverProperties(internal=nw.InternalNodeProperties(**settings.NODE_PROPS['Internal']),
                                         motor=nw.MotorNodeProperties(**settings.NODE_PROPS['Motor']),
                                         sensors=nw.SensoryNodeProperties(**settings.NODE_PROPS['Sensory']),
                                         weights=nw.EdgeProperties(prop=settings.SYNAPSES))

        smpd.sample()

        self.smp = evo.SensorMoverPopulation(my_environment, smpd,
                                             initial_population_size=self.initial_number_of_individuals,
                                             reproductive_cost=0.1,
                                             reproductive_threshold=20,
                                             reproductive_mode='sexual', data_cutoff=2)
        self.smp.sim_time_steps(max_iter=time_iterations)