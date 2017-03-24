
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
                 reproductive_mode='asexual', data_cutoff=None, *args, **kwargs):
        self.reproductive_cost, self.reproductive_threshold = reproductive_cost, reproductive_threshold
        self.reproduction_mode = reproductive_mode
        self.share_world = share_world
        self.environment = environment  # base environment
        self.smd_dist = sensor_mover_driver_distribution
        self.network_type = network_type
        self.initial_population_size = initial_population_size
        self._population_size = {'existing':[], 'born': [], 'found_target': [], 'died': []}
        self.individuals = {}
        self.t, self.n, self.data_cutoff = 0, 0, data_cutoff
        self._create_initial_population(*args, **kwargs)

    def _create_initial_population(self, *args, **kwargs):
        for i in range(self.initial_population_size):
            self.create_and_add_individual(*args, **kwargs)

    def create_and_add_individual(self, position=None, *args, **kwargs):
        network = self.network_type(self.smd_dist.create_digraph(), data_cutoff=self.data_cutoff, *args, **kwargs)
        env = self.environment if self.share_world else copy.copy(self.environment)
        env.rm_individuals()
        _id = 'G0_I%s' % self.n
        sm = SensorMover(env, position=env.generate_position(position), initial_network=network, ind_id=_id,
                         reproduction_cost=self.reproductive_cost, reproductive_threshold=self.reproductive_threshold,
                         reproduction_mode=self.reproduction_mode, data_cutoff=self.data_cutoff)
        self.add_individual(sm)
        # self.environment.add_individual(sm)  # not sure if this is needed
        self.n += 1

    def add_individual(self, individual):
        if individual.ind_id not in self.individuals:
            self.individuals.update({individual.ind_id: individual})

    def add_individuals(self, individuals):
        if individuals is None:
            return
        _ = [self.add_individual(individual) for individual in individuals]

    def rm_individual(self, individual):
        self.individuals.pop(individual.ind_id)

    def rm_individuals(self, individuals):
        _ = [self.rm_individual(individual) for individual in individuals]

    def get_reproducing_individuals(self):
        return [i for i in self.individuals.values() if i.to_reproduce]

    def reproduce(self, mode='asexual', partner=None, new_environment=True, error_rate=0., *args, **kwargs):
        children = None
        if mode == 'asexual':
            children = self.reproduce_asexually(new_environment=new_environment, error_rate=error_rate)
        elif mode == 'sexual':
            children = self.reproduce_sexually(new_environment=new_environment, error_rate=error_rate)
        return children

    def reproduce_asexually(self, error_rate=0., new_environment=True):
        rep, children = self.get_reproducing_individuals(), []
        for i in rep:
            children.append(i.reproduce(mode='asexual', new_environment=new_environment, error_rate=error_rate))
        return children, self

    def reproduce_sexually(self, error_rate=0., new_environment=True):
        """Combine individuals within the population.
        1. Loop through each individual that will reproduce
        2. Match the reproducing individual with another reproducing individual
            - Matching process done at random"""
        rep, children, parents = self.get_reproducing_individuals(), [], []
        for i in range(len(rep)//2):
            p1, p2 = rep[2*i], rep[2*i+1]
            children.append(p1.reproduce(mode='sexual', partner=p2, new_environment=new_environment, error_rate=error_rate))
            p1._reproduced, p2._reproduced = True, True
            parents.extend([p1, p2])
        return children, parents

    def sim_time_step(self):
        _asexual_children, _dead_individuals, found_target = [], [], 0

        for iid, i in self.individuals.iteritems():
            _ = i.sim_time_step()
            if i.is_dead:
                _dead_individuals.append(i)
            found_target += 1 if i.found_target() else 0
        # population-level method because it requires two individuals
        children, parents = self.reproduce(mode=self.reproduction_mode, new_environment=True)
        self.__update_population_size({'existing': len(self.individuals),
                                       'born': len(children) if children is not None else 0,
                                       'found_target': found_target,
                                       'died': len(_dead_individuals)})
        self.add_individuals(children)
        self.rm_individuals(_dead_individuals)
        self.t += 1

    def sim_time_steps(self, max_iter=10):
        while self.t < max_iter:
            n = 0
            for ind in self.individuals.itervalues():
                n += 1 if ind.is_living else 0
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
    def energy_balance(self):
        p = self.population_size
        df = pd.concat([p['existing'], self.energy_initial, self.energy_final, self.energy_consumed, -self.energy_lost,
                        self.energy_added, self.global_energy], axis=1)
        df.rename(columns={'existing':"# Agents", 0: "Initial E", 1: "Final E", 2: "Consumed E", 3: "Lost E",
                           4: "Added E", 5: "Global E"}, inplace=True)
        return df

    @property
    def balance(self):
        p = self.population_size
        df = pd.concat([p['existing'], p['born'], p['died'], p['existing'] + p['born']-p['died'],self.energy_balance]
                      , axis=1)
        df.rename(columns={'existing': "# Init Agents", 'born': "# Born", 'died': "# Died", 0: "# Final Agents"},
                  inplace=True)
        return df

    @property
    def global_energy(self):
        df = pd.concat([self.energy_final + self.energy_consumed.cumsum() - self.energy_lost.cumsum()], axis=1)
        df.rename(columns={0:"Global E"}, inplace=True)
        return df

    @property
    def total_energy(self):
        total_energy = pd.Series(0., index=range(self.t), name="Total Energy")
        for iid, i in self.individuals.iteritems():
            total_energy = total_energy.add(i.total_energy, fill_value=0.)
        return total_energy

    @property
    def energy_initial(self):
        e = pd.Series(0., index=range(self.t), name="Initial Energy")
        for iid, i in self.individuals.iteritems():
            e = e.add(i.energy_initial.sum(axis=1), fill_value=0.)
        return e

    @property
    def individual_energy_initial(self):
        e = []
        for iid, i in self.individuals.iteritems():
            te = i.energy_initial.sum(axis=1)
            te.name = iid
            e.append(te)
        return pd.concat(e, axis=1)

    @property
    def energy_final(self):
        e = pd.Series(0., index=range(self.t), name="Final Energy")
        for iid, i in self.individuals.iteritems():
            e = e.add(i.energy_final.sum(axis=1), fill_value=0.)
        return e

    @property
    def energy_consumed(self):
        e = pd.Series(0., index=range(self.t), name="Energy Consumed")
        for iid, i in self.individuals.iteritems():
            e = e.add(i.energy_consumed.sum(axis=1), fill_value=0.)
        return e

    @property
    def energy_lost(self):
        e = pd.Series(0., index=range(self.t), name="Energy Lost")
        for iid, i in self.individuals.iteritems():
            e = e.add(i.energy_lost.sum(axis=1), fill_value=0.)
        return e

    @property
    def energy_added(self):
        e = pd.Series(0., index=range(self.t), name="Energy Added")
        for iid, i in self.individuals.iteritems():
            e = e.add(i.energy_added, fill_value=0.)
        return e

    @property
    def num_nodes_fired(self):
        e = pd.Series(0., index=range(self.t), name="# Nodes Fired")
        for iid, i in self.individuals.iteritems():
            e = e.add(i.energy_added, fill_value=0.)
        return e

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
        return sm_ids, df


class SensorMoverPopulation(SensorMoverPopulationBase):
    pass