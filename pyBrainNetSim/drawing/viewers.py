# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 09:37:35 2016

@author: brian
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext
import networkx as nx
from scipy.spatial import ConvexHull
import pyBrainNetSim.models.network

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


def draw_networkx(G, ax=None, **kwargs):
    color, alpha = '#D9F2FA', 0.5
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
    nx.draw_networkx_edges(G, node_pos, width=edge_width, alpha=0.2, ax=ax)

    i_subg = G.subgraph(G.nodes('Internal'))
    m_subg = G.subgraph(G.nodes('Motor'))
    s_subg = G.subgraph(G.nodes('Sensory'))
    i_points = np.array([p for p in nx.get_node_attributes(i_subg, 'pos').itervalues()])
    i_hull = ConvexHull(i_points)
    ax.add_patch(patches.Polygon([i_points[k] for k in i_hull.vertices], color=color, alpha=alpha))
    for m_id, attr in m_subg.node.iteritems():
        ax.arrow(attr['pos'][0], attr['pos'][1], attr['force_direction'][0] / 2, attr['force_direction'][1] / 2,
                     head_width=0.05, head_length=0.1, fc='k', ec='k')

    plt.gca().set(**kwargs)
    return ax


def pcolormesh_edges(sim_net, at_time=-1, ax=None, **kwargs):
    if isinstance(sim_net, pyBrainNetSim.models.world.Individual):
        num_neurons = len(sim_net.internal.simdata)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net.internal.simdata[row]
        my_title = "%s Synapse Strength\nt=%d" % (sim_net.ind_id, row)
    elif hasattr(sim_net, 'simdata'):
        num_neurons = len(sim_net.simdata)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net.simdata[row]
        my_title = "Synapse Strength\nt=%d" % row
    elif isinstance(sim_net, pyBrainNetSim.models.network.NeuralNetSimData):
        num_neurons = len(sim_net)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net[row]
        my_title = "Synapse Strength\nt=%d" % row
    elif isinstance(sim_net, nx.DiGraph):
        num_neurons = len(sim_net)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net
        my_title = "Synapse Strength"
    else:
        return

    if ax is None:
        fig, ax = plt.subplots()
    sorted_neuron_names = sorted(graph.nodes())
    x, y = np.meshgrid(range(graph.number_of_nodes() + 1), range(graph.number_of_nodes() + 1))
    x = x - 0.5
    y = y - 0.5
    z = np.array(nx.to_numpy_matrix(graph, nodelist=sorted_neuron_names))
    z = np.ma.masked_where(z==0, z)
    # z = z[:-1, :-1]
    z_min, z_max = 0., np.abs(z).max()

    p = ax.pcolormesh(x, y, z, cmap='Reds', vmin=z_min, vmax=z_max)
    ax.set(title=my_title, xlabel="Post-synaptic Neurons", ylabel="Pre-synaptic Neurons",
           xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax.set_title(my_title, {'fontsize':10})
    ax.set_xticks(range(graph.number_of_nodes()))
    ax.set_yticks(range(graph.number_of_nodes()))
    ax.set_xticklabels(sorted_neuron_names, rotation='vertical')
    ax.set_yticklabels(sorted_neuron_names)
    ax.set_aspect('equal', adjustable='box')
    plt.gcf().colorbar(p, ax=ax)
    return ax


def pcolormesh_edge_changes(sim_net, initial_time=0, final_time=-1, ax=None, as_pct=False, vmax=None, vmin=None, **kwargs):

    my_title = "Synapse Strength Percentage Change" if as_pct else "Synapse Strength Change"
    if isinstance(sim_net, pyBrainNetSim.models.world.Individual):
        sim_data = sim_net.internal.simdata
        my_title = "%s Synapse Strength Percentage Change" % (sim_net.ind_id) \
            if as_pct else "%s Synapse Strength Change" % (sim_net.ind_id)
    elif hasattr(sim_net, 'simdata'):
        sim_data = sim_net.simdata
    elif isinstance(sim_net, pyBrainNetSim.models.network.NeuralNetSimData):
        sim_data = sim_net
    else:
        return
    num_neurons = len(sim_data)
    row0 = num_neurons - initial_time if initial_time < 0 else initial_time
    rowf = num_neurons + final_time if final_time < 0 else final_time
    initial_net = sim_data[row0]
    final_net = sim_data[rowf]
    my_title = "%s\nFrom t=%d to %d" % (my_title, row0, rowf)

    if ax is None:
        fig, ax = plt.subplots()
    sorted_neuron_names = sorted(initial_net.nodes())
    x, y = np.meshgrid(range(initial_net.number_of_nodes() + 1), range(initial_net.number_of_nodes() + 1))
    x, y = x - 0.5, y - 0.5
    initial_z = np.array(nx.to_numpy_matrix(initial_net, nodelist=sorted_neuron_names))
    final_z = np.array(nx.to_numpy_matrix(final_net, nodelist=sorted_neuron_names))

    if as_pct:
        z = 100 * np.divide(final_z - initial_z, initial_z)
        # print final_z - initial_z
        z = np.ma.masked_where(z == 0, z)
        z = np.ma.masked_where(np.isnan(z), z)
        z_max = 10. * np.ceil(np.abs(z).max()/10.)
        z_min = -z_max
    else:
        z = final_z - initial_z
        z = np.ma.masked_where(z == 0, z)
        z_min = -np.abs(z).max() if vmin is None else vmin
        z_max = np.abs(z).max() if vmax is None else vmax
    p = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
    ax.set(title=my_title, xlabel="Post-synaptic Neurons", ylabel="Pre-synaptic Neurons",
           xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax.set_title(my_title, {'fontsize': 10})
    plt.xticks(range(initial_net.number_of_nodes()), sorted_neuron_names, rotation='vertical', ha='center')
    plt.yticks(range(initial_net.number_of_nodes()), sorted_neuron_names, va='center')
    ax.set_aspect('equal', adjustable='box')
    if as_pct:
        plt.gcf().colorbar(p, ax=ax, format='%d%%')
    else:
        plt.gcf().colorbar(p, ax=ax)
    return ax


def _get_net_data(sim_net):
    if isinstance(sim_net, pyBrainNetSim.models.world.Individual):
        num_neurons = len(sim_net.internal.simdata)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net.internal.simdata[row]
        my_title = "%s Synapse Strength\nt=%d" % (sim_net.ind_id, row)
    elif hasattr(sim_net, 'simdata'):
        num_neurons = len(sim_net.simdata)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net.simdata[row]
        my_title = "Synapse Strength\nt=%d" % row
    elif isinstance(sim_net, pyBrainNetSim.models.network.NeuralNetSimData):
        num_neurons = len(sim_net)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net[row]
        my_title = "Synapse Strength\nt=%d" % row
    elif isinstance(sim_net, nx.DiGraph):
        num_neurons = len(sim_net)
        row = num_neurons + at_time if at_time < 0 else at_time
        graph = sim_net
        my_title = "Synapse Strength"
    else:
        return


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
