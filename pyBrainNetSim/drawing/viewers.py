# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 09:37:35 2016

@author: brian
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext

RENDER_NODE_PROPS = {'Internal': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 200.},
                     'Motor': {'shape': 'h', 'min_node_size': 100., 'max_node_size': 300.},
                     'Sensory': {'shape': 's', 'min_node_size': 100., 'max_node_size': 300.},
                     'Default': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.}}


class vTrajectory(LineCollection):
    def __init__(self, points, base_clr=(0, 0, 0), *args, **kwargs):
        self.text = mtext.Text(0, 0, '') # update the position when the line data is set
        t = np.linspace(0, 1, points.shape[0]) # "time" variable
        # set up a list of segments
        segs = np.concatenate([points[:-1],points[1:]], axis=1)
        cmp1 = LinearSegmentedColormap.from_list("my_cmp", ((1, 1, 1), base_clr))
        super(vTrajectory, self).__init__(segs, cmap=cmp1, *args, **kwargs)
        self.set_array(t)

        # can't access the label attr until *after* the line is inited
        self.text.set_text(self.get_label())

    def set_figure(self, figure):
        self.text.set_figure(figure)
        super(vTrajectory, self).set_figure(figure)

    def set_axes(self, axes):
        self.text.set_axes(axes)
        super(vTrajectory, self).set_axes(axes)

    def set_transform(self, transform):
        # 2 pixel offset
        texttrans = transform + mtransforms.Affine2D().translate(2, 2)
        self.text.set_transform(texttrans)
        super(vTrajectory, self).set_transform(transform)

    def set_data(self, x, y):
        super(vTrajectory, self).set_data(x, y)
        if len(x):
            self.text.set_position((x[-1], y[-1]))

    def draw(self, renderer):
        # draw my label at the end of the line with 2 pixel offset
        super(vTrajectory, self).draw(renderer)
        self.text.draw(renderer)


def draw_networkx(G, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for node_class in RENDER_NODE_PROPS.iterkeys():
        if node_class == 'Default':
            continue
        node_pos, node_colors, node_shape, node_size, edge_width = _get_node_plot_props(G, node_class)
        subgraph = G.subgraph(G.nodes(node_class)).copy()
        nx.draw_networkx_nodes(subgraph, node_pos, node_color=node_colors,
                               node_shape=node_shape, node_size=node_size, ax=ax)

    node_pos, node_colors, node_shape, node_size, edge_width = _get_node_plot_props(G)
    nx.draw_networkx_edges(G, node_pos, width=edge_width, alpha=0.2)
    return ax


def _get_node_plot_props(G, node_class=None):
    cm = plt.get_cmap('RdYlGn')  # Shade from red (inhibitory) to green (excitatory)
    nodes = G.nodes(node_class)
    adj_matrix = nx.adjacency_matrix(G)

    node_pos = {n_id: G.node[n_id]['pos'] for n_id in nodes}
    edge_width = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if u in nodes])

    if node_class is not None:
        min_ns, max_ns = RENDER_NODE_PROPS[node_class]['min_node_size'], RENDER_NODE_PROPS[node_class]['max_node_size']
        node_colors = np.array([-float(G.node[n_id]['energy_value']) if G.node[n_id]['node_type'] == 'I'
                                else float(G.node[n_id]['energy_value']) for n_id in nodes])
        for i, n in enumerate(node_colors):
            node_colors[i] = n / node_colors.max() if n > 0. else n / np.abs(node_colors.min())
        node_colors = cm((node_colors + 1.) * 256. / 2.)  # normalize to 0-256 and get colors
        node_shape = RENDER_NODE_PROPS[node_class]['shape']
        node_size = np.array([np.maximum(adj_matrix[i].sum(), .01) for i, n_id in enumerate(G.nodes())
                              if
                              G.node[n_id]['node_class'] == node_class])  # proportional to the number of connections
    else:
        node_colors = np.array([G.node[n_id]['energy_value'] for n_id in nodes])
        node_colors = cm(256. * (0.5 + node_colors / (2 * node_colors.max())))  # normalize to 0-256 and get colors
        node_shape, node_size = RENDER_NODE_PROPS['Default']['shape'], adj_matrix.sum(axis=1)
        min_ns, max_ns = RENDER_NODE_PROPS['Default']['min_node_size'], RENDER_NODE_PROPS['Default']['max_node_size']

    node_size = min_ns + (max_ns - min_ns) * (node_size - node_size.min()) / (node_size.max() - node_size.min()) \
        if node_size.max() > node_size.min() else max_ns * np.ones_like(node_size)
    return node_pos, node_colors, node_shape, node_size, edge_width
