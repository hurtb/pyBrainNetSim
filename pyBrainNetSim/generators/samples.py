import pandas as pd
import pyBrainNetSim.models.world as world
import pyBrainNetSim.models.network as network
import pyBrainNetSim.simulation.simnetwork as sim
import pyBrainNetSim.generators.networks as rnd


def test1():
    internal_props = {'number_neurons': 20, 'excitatory_to_inhibitory':1., 'spontaneity': 0.00, 'inactive_period': 0.,
                      'energy_value':100., 'threshold': 5.}
    sensor_props = {'number_neurons': 20, 'excitatory_to_inhibitory':0.01, 'spontaneity': 0.00, 'inactive_period': 0.}
    motor_props = {'number_neurons': 20, 'excitatory_to_inhibitory':0.01, 'spontaneity': 0.00, 'inactive_period': 0.}
    weight_props = {'int_to_int': 1., 'int_to_motor': 1., 'sensor_to_int': 0.5,
                    'sensory_max_connections': 1., 'motor_max_connections': 1.}

    sm_prop_dist = rnd.SensorMoverProperties(internal=rnd.InternalNodeProperties(**internal_props),
                                             sensors=rnd.SensoryNodeProperties(**sensor_props),
                                             motor=rnd.MotorNodeProperties(**motor_props),
                                             weights=rnd.EdgeProperties(**weight_props))
    g = sm_prop_dist.create_digraph()
    return g



