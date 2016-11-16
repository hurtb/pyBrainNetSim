import pyBrainNetSim.models.world as world
import pyBrainNetSim.generators.network as nw
import pyBrainNetSim.simulation.evolution as evo
import settings
import cPickle as pickle
import os

NUM_SIMS = 100
MAX_SIMS = 1000
FNAME = 'sim_data'

def create_new_sim():
    my_environment = world.Environment(origin=settings.ENVIRONMENT['origin'],
                                       max_point=settings.ENVIRONMENT['maximum_point'],
                                       field_permeability=settings.ENVIRONMENT['permeability'])
    food = world.Attractor(environment=my_environment,
                           position=settings.ATTRACTOR['position'],
                           strength=settings.ATTRACTOR['strength'])  # add "food"
    sm_pd = nw.SensorMoverProperties(internal=nw.InternalNodeProperties(**settings.NODE_PROPS['Internal']),
                                     motor=nw.MotorNodeProperties(**settings.NODE_PROPS['Motor']),
                                     sensors=nw.SensoryNodeProperties(**settings.NODE_PROPS['Sensory']),
                                     weights=nw.EdgeProperties(prop=settings.SYNAPSES))

    smp = evo.SensorMoverPopulation(my_environment, sm_pd,
                                    initial_population_size=settings.POPULATION['initial_size'],
                                    reproductive_cost=settings.POPULATION['reproductive_cost'],
                                    reproductive_threshold=settings.POPULATION['reproductive_threshold'])
    print "Working on sim %d of %d (init population:%s, T:%d, " \
              %(J, NUM_SIMS, settings.POPULATION['initial_size'], settings.ITERATIONS),

    smp.sim_time_steps(max_iter=settings.ITERATIONS)

    # Saving the objects:
    fpath = os.path.dirname(os.path.abspath(__file__))
    i, n = 0, 0
    while i < MAX_SIMS:
        fname = 'data/%s%d_P%dT%d.pkl' %(FNAME, i, settings.POPULATION['initial_size'], settings.ITERATIONS)
        filepath = os.path.join(fpath, fname)
        if not os.path.exists(filepath):
            break
        i += 1

    with open(filepath, 'wb') as f:
        # pickle.dump([smp, food, sm_pd, smp], f, -1)
        pickle.dump(smp, f, -1)
    f.close()
    print "final pop:%d) saved: %s" % (len(smp.individuals), fname)


def main():

    create_new_sim()


if __name__ == "__main__":
    J = 1
    while J <= NUM_SIMS:
        main()
        J += 1
