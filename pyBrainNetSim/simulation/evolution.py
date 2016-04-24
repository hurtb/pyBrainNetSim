
from pyBrainNetSim.generators.random import SensorMoverPropertyDistribution
from pyBrainNetSim.models.individuals import SensorMover
from pyBrainNetSim.simulation.simnetwork import HebbianNetworkBasic
import pandas as pd
import copy
import matplotlib.pyplot as plt


class SensorMoverPopulationBase(object):

    def __init__(self, environment, sensor_mover_driver_distribution, network_type=HebbianNetworkBasic,
                 initial_population_size=25, share_world=False):
        self.share_world = share_world
        self.environment = environment  # base environment
        self.smd_dist = sensor_mover_driver_distribution
        self.network_type = network_type
        self.initial_population_size = initial_population_size
        self.individuals = {}
        self.t, self.n = 0, 0
        self._create_initial_population()

    def _create_initial_population(self):
        for i in range(self.initial_population_size):
            self.add_individual()

    def add_individual(self):
        network = self.network_type(self.smd_dist.create_digraph())
        env = self.environment if self.share_world else copy.deepcopy(self.environment)
        sm = SensorMover(env, position=(1, 1), initial_network=network)  # TODO: alter position
        self.individuals.update({'G0_I%s' % self.n: sm})
        self.n += 1

    def sim_time_step(self):
        for sm in self.individuals.itervalues():
            sm.sim_time_step()
        self.t += 1

    def sim_time_steps(self, max_iter=10):
        while self.t < max_iter:
            self.sim_time_step()

    def __pandas_df_from_series(self, attr):
        return pd.DataFrame({sm_id: getattr(sm.internal.simdata, attr) for sm_id, sm in self.individuals.iteritems()})

    @property
    def trajectory(self):
        return {sm_id: sm.trajectory for sm_id, sm in self.individuals.iteritems()}

    def network_attr(self, attr):
        """ Generic attribute getter referencing attributes in the NeuralNetSimData class, or the objects within the
        individuals dictionary.
        :param attr:
        :return pandas.DataFrame:
        """
        return self.__pandas_df_from_series(attr)

    def plot_network_attr(self, attr, ax=None, **kwargs):
        self.network_attr(attr)

    def population_node_attr_at_time(self, attr, t):
        pna = {}
        for sm_id, sm in self.individuals.iteritems():
            pna[sm_id] = {}
            for n_id in sm.internal.simdata[t].nodes():
                if attr in sm.internal.simdata[t].node[n_id]:
                    pna[sm_id][n_id] = sm.internal.simdata[t].node[n_id][attr]
        return pd.DataFrame(pna)

    def hist_population_attr_at_time(self, attr, t, ax=None, **kwargs):
        df = self.population_node_attr_at_time(attr, t)
        df.plot(kind='hist', ax=ax, alpha=0.5, **kwargs)

    def plot_efficiency(self, **kwargs):
        df = pd.DataFrame({sm_id: sm.efficiency() for sm_id, sm in self.individuals.iteritems()})
        df.cumsum().plot(style='o-')


class SensorMoverPopulation(SensorMoverPopulationBase):
    """

    """
    pass