# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 09:37:35 2016

@author: brian
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext
import networkx as nx
from scipy.spatial import ConvexHull
import pandas as pd
import pyBrainNetSim.models.network
import pyBrainNetSim.drawing.layouts as lyt
import pyBrainNetSim
import pyBrainNetSim.utils as utils

RENDER_NODE_PROPS = {'Internal': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 200.},
                     'Motor': {'shape': 'h', 'min_node_size': 100., 'max_node_size': 300.},
                     'Sensory': {'shape': 's', 'min_node_size': 100., 'max_node_size': 300.},
                     'Default': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.},
                     'Firing': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.,
                                'node_face_color': '#D1D66B', 'node_edge_color': '#D1D66B'},
                     'Active': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.,
                                'node_face_color': '#D1D66B', 'node_edge_color': '#D1D66B'},
                     'Dead': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.,
                                'node_face_color': '#D1D66B', 'node_edge_color': '#D1D66B'},
                     }


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


def draw_networkx(G, layout='by_position', ax=None, max_e=None, plot_active=True, active_node_color=None, **kwargs):
    internal_color, internal_ecolor, internal_alpha = '#FCDC79', '#C79500', 0.5
    overall_color, overall_ecolor, overall_alpha = '#A1A1A1', '#050505', 0.2

    if ax is None:
        fig, ax = plt.subplots()
    if layout == 'grid':  # rarely use, the 'pos' attribute is defined
        G = lyt.grid_layout(G)
    for node_class in RENDER_NODE_PROPS.iterkeys():
        if node_class in ['Default', 'Active', 'Dead', 'Firing']:
            continue
        node_pos, node_colors, node_shape, node_size, edge_width = _get_node_plot_props(G, node_class, max_energy=max_e)
        nx.draw_networkx_nodes(G.subgraph(G.nodes(node_class)).copy(), node_pos,  # Draw nodes
                               node_color=node_colors, node_shape=node_shape, node_size=node_size, ax=ax, **kwargs)
    node_pos, node_colors, node_shape, node_size, edge_width = _get_node_plot_props(G, max_energy=max_e)
    nx.draw_networkx_edges(G, node_pos, width=edge_width, alpha=0.2, ax=ax)  # draw edges

    i_subg = G.subgraph(G.nodes('Internal'))
    m_subg = G.subgraph(G.nodes('Motor'))
    s_subg = G.subgraph(G.nodes('Sensory'))

    # Add patches for the entire network and internal nodes
    ax.add_patch(
        create_axes_patch(nx.get_node_attributes(G, 'pos').values(), scale=1.2, facecolor=overall_color,
                          edgecolor=overall_ecolor, alpha=overall_alpha))
    ax.add_patch(
        create_axes_patch(nx.get_node_attributes(i_subg, 'pos').values(), scale=1.2, facecolor=internal_color,
                                   edgecolor=internal_ecolor, alpha=internal_alpha))

    # Add arrows indicating force direction
    firing_nc = colors.hex2color(active_node_color) if active_node_color is not None \
        else list(colors.hex2color(RENDER_NODE_PROPS['Firing']['node_face_color']))
    arrow_scale = 1
    for m_id, attr in m_subg.node.iteritems():
        arr_cl = firing_nc if G.is_node_firing(m_id) else 'k'
        ax.arrow(attr['pos'][0], attr['pos'][1], attr['force_direction'][0] * arrow_scale, attr['force_direction'][1] * arrow_scale,
                     head_width=1, head_length=np.linalg.norm(attr['force_direction'])/2, fc='k', ec=arr_cl)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim([xlim[0]-arrow_scale, xlim[1]+arrow_scale])
    ax.set_ylim([ylim[0] - arrow_scale, ylim[1] + arrow_scale])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.set_title("%s Network @ t:%d" %("ID", 0), {'fontsize': 10})
    ax.set_aspect('equal')
    ax.set(**kwargs)
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
    # cb = ax.images[-1].colorbar
    # cb.remove()
    # cb = plt.gcf().colorbar(p, ax=ax)
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
    plt.gca()
    ax.set(title=my_title, xlabel="Post-synaptic Neurons", ylabel="Pre-synaptic Neurons",
           xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax.set_title(my_title, {'fontsize': 10})
    ax.set_xticks(range(len(initial_net.nodes())))
    ax.set_yticks(range(len(initial_net.nodes())))
    ax.set_xticklabels(sorted_neuron_names, rotation='vertical', ha='center')
    ax.set_yticklabels(sorted_neuron_names, va='center')
    ax.set_aspect('equal', adjustable='box')
    # if as_pct:
    #     plt.gcf().colorbar(p, ax=ax, format='%d%%')
    # else:
    #     plt.gcf().colorbar(p, ax=ax)
    return ax


# def _get_net_data(sim_net):
#     if isinstance(sim_net, pyBrainNetSim.models.world.Individual):
#         num_neurons = len(sim_net.internal.simdata)
#         row = num_neurons + at_time if at_time < 0 else at_time
#         graph = sim_net.internal.simdata[row]
#         my_title = "%s Synapse Strength\nt=%d" % (sim_net.ind_id, row)
#     elif hasattr(sim_net, 'simdata'):
#         num_neurons = len(sim_net.simdata)
#         row = num_neurons + at_time if at_time < 0 else at_time
#         graph = sim_net.simdata[row]
#         my_title = "Synapse Strength\nt=%d" % row
#     elif isinstance(sim_net, pyBrainNetSim.models.network.NeuralNetSimData):
#         num_neurons = len(sim_net)
#         row = num_neurons + at_time if at_time < 0 else at_time
#         graph = sim_net[row]
#         my_title = "Synapse Strength\nt=%d" % row
#     elif isinstance(sim_net, nx.DiGraph):
#         num_neurons = len(sim_net)
#         row = num_neurons + at_time if at_time < 0 else at_time
#         graph = sim_net
#         my_title = "Synapse Strength"
#     else:
#         return


def _get_node_plot_props(G, node_class=None, max_energy=None, active_node_color=None, active_edge_color=None,
                         dead_node_color=None):
    """
    `node_`
    `node_size` - proportional to the sum of the presynaptic connections it makes with other nodes.
    `node_colors` - function of excitatory/inhibitory, energy_value, firing/inactive

    """
    cm = plt.get_cmap('coolwarm')  # Shade from red (inhibitory) to green (excitatory)
    nodes = G.nodes(node_class)
    adj_matrix = nx.adjacency_matrix(G)
    node_pos = nx.get_node_attributes(G.subgraph(nodes), 'pos')
    edge_width = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if u in nodes])
    firing_nc = colors.hex2color(active_node_color) if active_node_color is not None \
        else list(colors.hex2color(RENDER_NODE_PROPS['Firing']['node_face_color']))
    dead_nc = colors.hex2color(dead_node_color) if dead_node_color is not None \
        else list(colors.hex2color(RENDER_NODE_PROPS['Dead']['node_face_color']))
    _ = firing_nc.append(1.)
    _ = dead_nc.append(1.)

    node_colors = _get_node_colors(G, cm, node_class=node_class, max_energy=max_energy, firing_node_color=firing_nc,
                                   dead_node_color=dead_nc)

    if node_class is not None:
        min_ns, max_ns = RENDER_NODE_PROPS[node_class]['min_node_size'], RENDER_NODE_PROPS[node_class]['max_node_size']
        node_shape = RENDER_NODE_PROPS[node_class]['shape']
        node_size = np.array([np.maximum(adj_matrix[i].sum(), .01) for i, n_id in enumerate(G.nodes())
                              if G.node[n_id]['node_class'] == node_class])  # proportional to the number of connections
    else:
        node_shape, node_size = RENDER_NODE_PROPS['Default']['shape'], adj_matrix.sum(axis=1)
        min_ns, max_ns = RENDER_NODE_PROPS['Default']['min_node_size'], RENDER_NODE_PROPS['Default']['max_node_size']

    node_size = min_ns + (max_ns - min_ns) * (node_size - node_size.min()) / (node_size.max() - node_size.min()) \
        if node_size.max() > node_size.min() else max_ns * np.ones_like(node_size)
    return node_pos, node_colors, node_shape, node_size, edge_width


def _get_node_colors(G, cmap, node_class=None, max_energy=None, firing_node_color=None, dead_node_color=None):
    """Color node by: node_class (E or I), firing or not (edgecolor), """
    if node_class in ['Active', 'Dead', 'Firing']:
        return []
    if node_class is not None:
        node_colors = np.array([-float(G.node[n_id]['energy_value']) if G.node[n_id]['node_type'] == 'I'
                                else float(G.node[n_id]['energy_value']) for n_id in G.nodes(node_class)])
        max_e = abs(node_colors.max()) if max_energy is None else max_energy
        node_colors = np.array([n / max_e for n in node_colors])
        node_colors = cmap((node_colors + 1.) * 256. / 2.)  # normalize to 0-256 and get colors

    else:
        node_colors = []
        for i, n_id in enumerate(G.nodes(node_class)):
            node_colors.append(G.node[n_id]['energy_value'] if G.node[n_id]['node_type'] == 'E'
                               else -G.node[n_id]['energy_value'])
        node_colors = np.array(node_colors).astype(float)
        node_colors = node_colors - node_colors.min()

        max_e = np.abs(node_colors).max() if max_energy is None else max_energy
        node_colors = cmap(node_colors / max_e)  # normalize to 0-256 and get colors

    node_colors = node_colors.astype(list)
    if firing_node_color is not None:
        for i, n_id in enumerate(G.nodes(node_class)):
            node_colors[i] = firing_node_color if G.is_node_firing(n_id) else node_colors[i]

    if dead_node_color is not None:
        for i, n_id in enumerate(G.nodes(node_class)):
            node_colors[i] = dead_node_color if G.is_node_dead(n_id) else node_colors[i]

    return node_colors


def create_axes_patch(pts, scale=1., **kwargs):
    hull = ConvexHull(pts)
    center = utils.centroid(hull.points[hull.vertices])  # centroid of the edge node positions
    pts = scale * (hull.points[hull.vertices] - center) + center
    return patches.Polygon(pts, **kwargs)


def plot_node_property_ts(sim_data, kind='vlines', prop='value', neuron_ids=None, ax=None, **kwargs):
    """Plot the time-series of the neuron's property."""
    def plt_one_ts(series, iax):
        if kind == 'line':
            _pts = iax.plot(t, series.values, 'ro--', ms=12, mec='r')
        elif kind == 'vlines':
            _lines = iax.vlines(t, ymin=0, ymax=series.values, colors='r', lw=2)
        iax.set(ylim=(series.min() - y_l_buff, series.max() + y_u_buff),
                xlim=(series.index.min() - xbuff, series.index.max() + xbuff))
        iax.set_ylabel(series.name)
        return iax
    xbuff, y_u_buff, y_l_buff = 0.1, 0.2, 0.0
    t = range(len(sim_data))
    data = pd.DataFrame(sim_data.neuron_ts(prop))
    if isinstance(neuron_ids,(list, np.ndarray)):  # plot list of neuron_ids
        if ax is None:
            fig, ax = plt.subplots(len(neuron_ids), sharex=True)
        for i, neuron_id in enumerate(neuron_ids):
            series = data[neuron_id]
            ax[i] = plt_one_ts(series, ax[i])
        fig.subplots_adjust(hspace=0)
        bottom_ax, top_ax = ax[-1], ax[0]
    elif neuron_ids is None:  # plot all
        if ax is None:
            fig, ax = plt.subplots(len(data.columns), sharex=True)
        for i, neuron_id in enumerate(data):
            series = data[neuron_id]
            ax[i] = plt_one_ts(series, ax[i])
        fig.subplots_adjust(hspace=0)
        bottom_ax, top_ax = ax[-1], ax[0]
    else:  # plot 1 series
        if ax is None:
            fig, ax = plt.subplots()
        series = data[neuron_ids]
        ax = plt_one_ts(series, ax)
        bottom_ax, top_ax = ax, ax
    bottom_ax.set_xlabel('time')
    top_ax.set_title('Neuronal "%s" Time Series' % prop)
    return ax


def degree_histogram(g, min_weight=0, as_probability=False, ax=None):
    # Create histogram of node connections
    mybars = g.degree()
    edges = g.edge
    cnts = []
    for n1 in g.nodes():
        cnt = 0
        for n_out, to_nodes in edges.iteritems():
            if n_out == n1:
                for n_in, props in to_nodes.iteritems():
                    cnt = cnt + 1 if props['weight'] > min_weight else cnt
        cnts.append(cnt)

    vals, bins = np.histogram(cnts, bins=range(int(max(cnts))))
    if as_probability:
        vals = vals/vals.sum()
    ax.bar(bins[:-1], vals)
    ax.set_title("Outgoing Synapses per Neuron Distribution\nMinimum Synaptic Cutoff: %s" % min_weight)
    ax.set_xticks(range(int(max(cnts))))
#    ax.show() # show histogram
    return ax
