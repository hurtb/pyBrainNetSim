"""
Routines for built-in networks.
"""


def simple_1D_sensor_mover(number_signal_discretization=4, sens_sensitivity=None, sens_mid=None, sens_minsignal=0.,
                           sens_maxsignal=1, sensor_inactive_period=0):
    N = int(number_signal_discretization)
    num_units_per_module = 3 * N + 1
    ym = 2 * N + 1
    xlim = 4
    em0, em1 = num_units_per_module - 1, 2 * num_units_per_module - 1
    items = {'stimuli_sensitivity': sens_sensitivity, 'sensory_mid': sens_mid, 'stimuli_min': sens_minsignal,
             'stimuli_max': sens_maxsignal}
    sens_dict = {u: v for u, v in items.iteritems() if v}
    sens_dict.update({'node_class': 'Sensory', 'node_type': 'E', 'threshold': 0.2,
                      'inactive_period': sensor_inactive_period})

    neurons = [('S0', dict(sensor_direction=[-1.], pos=[-xlim, 0], **sens_dict)),
               ('S1', dict(sensor_direction=[1.], pos=[xlim, 0], **sens_dict)),
               ('I%s' % em0, {'node_class': 'Internal', 'node_type': 'E', 'threshold': 0.9, 'pos': [-xlim + 1, ym]}),
               ('M0', {'node_class': 'Motor', 'node_type': 'E', 'threshold': 0.9, 'force_direction': [-1],
                       'pos': [-xlim, ym]}),
               ('I%s' % em1, {'node_class': 'Internal', 'node_type': 'E', 'threshold': 0.9, 'pos': [xlim - 1, ym]}),
               ('M1', {'node_class': 'Motor', 'node_type': 'E', 'threshold': 0.9, 'force_direction': [1],
                       'pos': [xlim, ym]}),
               ]

    edges = [('I%s' % (num_units_per_module - 1), 'M0', 1.), ('I%s' % (num_units_per_module * 2 - 1), 'M1', 1.)]

    for n in range(N):
        e01, e02, e11, e12 = n, n + N, n + num_units_per_module, n + num_units_per_module + N
        i0, i1 = n + 2 * N, n + num_units_per_module + 2 * N
        thresh = (n + 1.) / (N + 1.)
        pn01 = ('I%s' % e01, {'node_class': 'Internal', 'node_type': 'E', 'threshold': thresh, 'inactive_period': 0,
                              'pos': [-xlim + 2, n]})  # Primary projection excitatory neuron, for s0
        pn02 = ('I%s' % e02, {'node_class': 'Internal', 'node_type': 'E', 'threshold': thresh, 'inactive_period': 0,
                              'pos': [-xlim + 3, n + N]})  # 2nd projection excitatory neuron
        in0 = ('I%s' % i0, {'node_class': 'Internal', 'node_type': 'I', 'threshold': 0.25, 'inactive_period': 0,
                            'pos': [-xlim + 3, n]})  # inhibitory neuron, for s0, to M1
        pn11 = ('I%s' % e11, {'node_class': 'Internal', 'node_type': 'E', 'threshold': thresh, 'inactive_period': 0,
                              'pos': [xlim - 2, n]})  # Primary projection excitatory neuron, for s1
        pn12 = ('I%s' % e12, {'node_class': 'Internal', 'node_type': 'E', 'threshold': thresh, 'inactive_period': 0,
                              'pos': [xlim - 3, n + N]})  # 2nd projection excitatory neuron
        in1 = ('I%s' % i1, {'node_class': 'Internal', 'node_type': 'I', 'threshold': 0.25, 'inactive_period': 0,
                            'pos': [xlim - 3, n]})  # inhibitory neuron, for s1, to M0
        neurons.extend([pn01, pn02, in0, pn11, pn12, in1])
        edges.extend([('S0', 'I%s' % e01, 1.), ('I%s' % e01, 'I%s' % i0, 1), ('I%s' % e01, 'I%s' % e02, 1.),
                      ('I%s' % e02, 'I%s' % em0, 1.),('I%s' % i0, 'I%s' % em1, 1.),
                      ('S1', 'I%s' % e11, 1.), ('I%s' % e11, 'I%s' % i1, 1), ('I%s' % e11, 'I%s' % e12, 1.),
                      ('I%s' % e12, 'I%s' % em1, 1.), ('I%s' % i1, 'I%s' % em0, 1.),
                      ])

    return neurons, edges