import numpy as np
import pandas as pd
import networkx as nx

def get_grid_positions(num_neurons, origin=(0,0), max_pt=(1,1)):
    num_per_row = np.ceil(np.sqrt(num_neurons)) + 1.
    num_per_col = np.floor(np.sqrt(num_neurons))
    x, y = np.meshgrid(np.linspace(origin[0], max_pt[0], num_per_col), np.linspace(origin[1], max_pt[1], num_per_row))
    i_pos = np.vstack([x.ravel(), y.ravel()]).T
    return i_pos


def _segregate_nodes(g):
    num_internal = len(g.nodes('Sensory'))
    int_connected_to_m_or_s, internal_to_internal = [], g.nodes('Internal')
    for m_id in g.nodes('Motor'):
        int_connected_to_m_or_s.extend(g.incoming_nodes(m_id))
    for s_id in g.nodes('Sensory'):
        int_connected_to_m_or_s.extend(g.neighbors(s_id))
    for i_id in g.nodes('Internal'):
        if i_id in int_connected_to_m_or_s:
            internal_to_internal.remove(i_id)
    return int_connected_to_m_or_s, internal_to_internal


def grid_layout(g, size=10., sensory_offset=3, connected_internal_offset=3):
    """Create a layout where the motor and sensor units are organized logically"""

    # 1. Get nodes attached to motor and sensory units
    # 2. Get non-connected nodes

    int_connected_to_m_or_s, internal_to_internal = _segregate_nodes(g)

    for m_id in g.nodes('Motor'):
        g.node[m_id]['pos'] = size * np.array(g.node[m_id]['force_direction'])
        for i_id in g.incoming_nodes(m_id):
            g.node[i_id]['pos'] = g.node[m_id]['pos'] \
                                  - np.array(g.node[m_id]['force_direction']) * connected_internal_offset
    for s_id in g.nodes('Sensory'):
        g.node[s_id]['pos'] = size * np.array(g.node[s_id]['sensor_direction'])\
                              + (np.array(g.node[s_id]['sensor_direction']) == 0).astype(float) * sensory_offset
        for i_id in g.neighbors(s_id):
            g.node[i_id]['pos'] = g.node[s_id]['pos'] - \
                                  np.array(g.node[s_id]['sensor_direction']) * connected_internal_offset

    pos = get_grid_positions(len(internal_to_internal), origin= -size/2 * np.array([1,1]),
                             max_pt=size/2 * np.array([1,1]))
    for i, i_id in enumerate(internal_to_internal):
        g.node[i_id]['pos'] = pos[i]

    return g