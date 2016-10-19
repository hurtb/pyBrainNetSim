import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pyBrainNetSim.models.world as world
import pyBrainNetSim.generators.networks as rnd
import pyBrainNetSim.drawing.viewers as vis
import pyBrainNetSim.simulation.evolution as evo
from pyBrainNetSim.drawing import animations as ani
import pickle

mpl.rcParams['figure.figsize'] = (12, 9)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
plt.rcParams.update({'axes.titlesize': 'small'})
my_dir = r'C:\Users\brian\Dropbox\projects\Neurosurgery-Ojemann\NeuralNets\simulations\20160528'

POPULATION_SIZE = 2
ITERATIONS = 5

internal_props = {'number_neurons': 50,
                  'excitatory_to_inhibitory': 0.5,
                  'spontaneity': 0.00,
                  'inactive_period': 0.,
                  'threshold': 1.
                  }
sensor_props = {'threshold': 5.}
weight_props = {'int_to_int': 1.,
                'int_to_motor': 1.,
                'sensor_to_int': 0.5,
                'int_to_motor_max_connections': 1,
                'sensory_to_internal_max_connections': 1,
                'edge_motor_min_cutoff': .2,
                'edge_sensor_min_cutoff': .2
                }

sm_prop_dist = rnd.SensorMoverProperties(internal=rnd.InternalNodeProperties(**internal_props),
                                         sensors=rnd.SensoryNodeProperties(**sensor_props),
                                         weights=rnd.EdgeProperties(**weight_props))

my_environment = world.Environment(origin=(-10, -10), max_point=(10, 10), field_permeability=1.)
food = world.Attractor(environment=my_environment, position=(3, 3), strength=10.)  # add "food"
smp = evo.SensorMoverPopulation(my_environment, sm_prop_dist, initial_population_size=POPULATION_SIZE)
smp.sim_time_steps(max_iter=ITERATIONS)

# Saving the objects:
with open('objs.pickle', 'w') as f:
    pickle.dump([my_environment, food, smp], f)