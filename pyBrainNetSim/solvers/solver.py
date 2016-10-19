from pyBrainNetSim.generators.networks import SensorMoverProperties
from pyBrainNetSim.models.individuals import SensorMover


class SensorMoverEvolutionarySolver(object):
    """
    An Evolutionary solver to find the best 'SensorMover' for the environment. This simulates I individuals at time 0.
    Each individual moves in the environment, using energy, and "trying" to find the food source. An individual must
    find the food or they will perish.

    At a defined time period P, the remaining individuals mate to produce M offspring individuals with a mixture of the
    two mating individual's attributes. The choice/likelihood to mate is determined by internal energy levels of
    individuals.

    In this way, the idea is that the individuals with the "best" configurations eventually emerge from the population.

    """
    def __init__(self, environment, sensormover_distribution=None, num_individuals=100, max_generations = 10,  *args, **kwargs ):
        self.environment = environment
        self.sm_dist = sensormover_distribution if sensormover_distribution is not None\
            else SensorMoverProperties()
        self.n = 0  # iteration number
        self.g = 0  # generation number
        self.num_individuals = num_individuals
        self.max_generations = max_generations
        # Create N individuals @ t0
        self.data = {}

    def solve(self):
        while self.n < self.max_generations:
            self.simulate_generation()
            self.evolve()

    def simulate_generation(self):
        ts, i= [], 0
        while i < self.num_individuals:
            ts.append(SensorMover(self.environment, position=(5, 5),
                                  initial_network=self.sm_dist.create_digraph(),
                                  **{'initial_fire':'prescribed', 'prescribed':['S0']}))
            i += 1
        self.data['G%d' % self.g] = ts

    def evolve(self, *args, **kwargs):
        """
        Defines how individuals evolve from one generation to the next
        :param args:
        :param kwargs:
        :return:
        """
        self.n += 1
