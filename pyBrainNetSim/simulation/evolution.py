
from pyBrainNetSim.generators.network import SensorMoverProperties
from pyBrainNetSim.models.individuals import SensorMover
from pyBrainNetSim.simulation.simnetwork import HebbianNetworkBasic
from pyBrainNetSim.drawing.viewers import draw_networkx
import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np


class SensorMoverPopulationBase(object):

    def __init__(self, environment, sensor_mover_driver_distribution, network_type=HebbianNetworkBasic,
                 initial_population_size=25, share_world=False, reproductive_cost=0.5, reproductive_threshold=10,
                 *args, **kwargs):
        self.reproductive_cost, self.reproductive_threshold = reproductive_cost, reproductive_threshold
        self.reproduction_mode = 'asexual'
        self.share_world = share_world
        self.environment = environment  # base environment
        self.smd_dist = sensor_mover_driver_distribution
        self.network_type = network_type
        self.initial_population_size = initial_population_size
        self._population_size = {'existing':[], 'born': [], 'found_target': [], 'died': []}
        self.individuals = {}
        self.t, self.n = 0, 0
        self._create_initial_population()

    def _create_initial_population(self):
        for i in range(self.initial_population_size):
            self.create_and_add_individual()

    def create_and_add_individual(self, position=None):
        network = self.network_type(self.smd_dist.create_digraph())
        env = self.environment if self.share_world else copy.copy(self.environment)
        env.rm_individuals()
        _id = 'G0_I%s' % self.n
        sm = SensorMover(env, position=env.generate_position(position), initial_network=network, ind_id=_id,
                         reproduction_cost=self.reproductive_cost, reproductive_threshold=self.reproductive_threshold)
        self.add_individual(sm)
        # self.environment.add_individual(sm)  # not sure if this is needed
        self.n += 1

    def add_individual(self, individual):
        if individual.ind_id not in self.individuals:
            self.individuals.update({individual.ind_id: individual})

    def add_individuals(self, individuals):
        _ = [self.add_individual(individual) for individual in individuals]

    def rm_individual(self, individual):
        self.individuals.pop(individual.ind_id)

    def rm_individuals(self, individuals):
        _ = [self.rm_individual(individual) for individual in individuals]

    def sim_time_step(self):
        _new_individuals, _dead_individuals, found_target = [], [], 0
        for sm in self.individuals.itervalues():
            child = sm.sim_time_step()
            if sm.is_dead:
                _dead_individuals.append(sm)
            if child:
                _new_individuals.append(child)
            found_target += 1 if sm.found_target() else 0
        self.__update_population_size({'existing': len(self.individuals), 'born': len(_new_individuals),
                                       'found_target': found_target, 'died': len(_dead_individuals)})
        self.add_individuals(_new_individuals)
        self.rm_individuals(_dead_individuals)
        self.t += 1

    def sim_time_steps(self, max_iter=10):
        while self.t < max_iter:
            n = 0
            # print "step: %d" %self.t
            for ind in self.individuals.itervalues():
                n += 1 if ind.is_living else 0
            trg = ' '.join([ind.ind_id for ind in self.individuals.itervalues() if ind.found_target()])
            self.sim_time_step()

    def __update_population_size(self, pop_dict):
        for key, val in pop_dict.iteritems():
            self._population_size[key].append(val)

    def __pandas_df_from_series(self, attr):
        return pd.DataFrame({sm_id: getattr(sm.internal.simdata, attr) for sm_id, sm in self.individuals.iteritems()})

    def test_attr(self, attr):
        return {sm_id: getattr(sm.internal.simdata, attr) for sm_id, sm in self.individuals.iteritems()}

    @property
    def population_size(self):
        return pd.DataFrame(self._population_size)

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
        self.network_attr(attr).plot(ax=ax, **kwargs)

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

    def individual_efficiency(self, to='Sensory'):
        """Return the efficiency of the individuals over the iterations"""
        return pd.DataFrame({sm_id: sm.efficiency(to=to) for sm_id, sm in self.individuals.iteritems()})

    def top_efficiencies(self, top=5, at_time=None):
        row = -1 if at_time is None else at_time
        df = pd.DataFrame({sm_id: sm.efficiency() for sm_id, sm in self.individuals.iteritems()})
        sm_ids = df.cumsum().iloc[row].sort_values(ascending=False).index.values.tolist()
        return sm_ids

    def plot_efficiency(self, to='Sensory', ax=None, **kwargs):
        df = self.individual_efficiency(to=to)
        ax = df.cumsum().plot(style='o-', ax=ax)
        return ax

    def draw_top_networkx(self, top=5, at_time=None, fig=None):
        """Draw the top individual's internal network by efficiency."""
        max_cols, max_axs = 5, 20
        axs = self._get_axes(top, max_cols)
        sm_ids = self.top_efficiencies(top, at_time)
        row = -1 if at_time is None else at_time
        for i, sm_id in enumerate(sm_ids[:top]):
            axs[i] = draw_networkx(self.individuals[sm_id].internal.simdata[row], ax=axs[i])
            axs[i].axis('square')
            axs[i].set(xticklabels=[], yticklabels=[], title="%s\nEfficiency: %.2f" % (sm_id, self.individuals[sm_id].efficiency().cumsum().iloc[-1]))
        return axs

    def draw_top_trajectories(self, top=5, at_time=None, fig=None):
        axs = self._get_axes(top, 5)
        row = -1 if at_time is None else at_time
        sm_ids = self.top_efficiencies(top, at_time)
        for i, sm_id in enumerate(sm_ids[:top]):
            axs[i] = self.environment.plot_individual_trajectory(individual=sm_id, upsample_factor=20, ax=axs[i])
            # axs[i].axis('equal')
            axs[i].set(title="%s Trajectory\nEfficiency: %.2f"
                             % (sm_id, self.individuals[sm_id].efficiency().cumsum().iloc[-1]))
        return axs


    def _get_axes(self, num, max_cols):
        rows = int(np.floor(num / max_cols)) + 1
        cols = int(num) if num <= max_cols else max_cols
        axs = [plt.subplot2grid((rows, cols), (i / cols, i % cols)) for i in range(num)]
        return axs


class SensorMoverPopulation(SensorMoverPopulationBase):
    """

    """
    pass