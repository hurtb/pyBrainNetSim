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
import pyBrainNetSim
import pyBrainNetSim.utils as utils
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, HoverTool, BoxAnnotation

RENDER_NODE_PROPS = {'Internal': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 200.},
                     'Motor': {'shape': 'h', 'min_node_size': 100., 'max_node_size': 300.},
                     'Sensory': {'shape': 's', 'min_node_size': 100., 'max_node_size': 300.},
                     'Default': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.},
                     'Firing': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.,
                                'node_face_color': '#D1D66B', 'node_edge_color': '#D1D66B'},
                     'Active': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.,
                                'node_face_color': '#D1D66B', 'node_edge_color': '#D1D66B'},
                     'Dead': {'shape': 'o', 'min_node_size': 50., 'max_node_size': 300.,
                                'node_face_color': '#ADADAD', 'node_edge_color': '#D1D66B'},
                     }
CMAP_DIFF = plt.get_cmap('coolwarm')
CMAP_EDGE = plt.get_cmap('PuRd')


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


def draw_networkx(G, pos=None, ax=None, max_e=None, plot_active=True, active_node_color=None, **kwargs):
    internal_color, internal_ecolor, internal_alpha = '#FCDC79', '#C79500', 0.5
    overall_color, overall_ecolor, overall_alpha = '#A1A1A1', '#050505', 0.2
    if ax is None:
        fig, ax = plt.subplots()
    for node_class in RENDER_NODE_PROPS.iterkeys():
        if node_class in ['Default', 'Active', 'Dead', 'Firing']:
            continue
        gs = G.subgraph(G.nodes(node_class)).copy()
        node_pos, node_colors, node_shape, node_size, edge_width = _get_node_plot_props(gs, node_class, max_energy=max_e)
        nx.draw_networkx_nodes(gs, node_pos, node_color=node_colors, node_shape=node_shape, node_size=node_size, ax=ax, **kwargs)
    node_pos, node_colors, node_shape, node_size, edge_width = _get_node_plot_props(G, max_energy=max_e)
    if pos is not None:
        node_pos = pos
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
        if len(attr['force_direction']) == 1:
            dx, dy = attr['force_direction'][0], 0.
        else:
            dx, dy = attr['force_direction']
        ax.arrow(attr['pos'][0], attr['pos'][1], dx * arrow_scale, dy * arrow_scale,
                     head_width=1, head_length=np.linalg.norm(attr['force_direction'])/2, fc='k', ec=arr_cl)

    labels = nx.draw_networkx_labels(G, pos=node_pos, font_color='w')
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
    cmap = CMAP_EDGE
    sorted_neuron_names = sorted(graph.nodes())
    x, y = np.meshgrid(range(graph.number_of_nodes() + 1), range(graph.number_of_nodes() + 1))
    x = x - 0.5
    y = y - 0.5
    z = np.array(nx.to_numpy_matrix(graph, nodelist=sorted_neuron_names))
    z = np.ma.masked_where(z==0, z)
    # z = z[:-1, :-1]
    z_min, z_max = 0., np.abs(z).max()
    p = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
    ax.set(title=my_title, xlabel="Post-synaptic Neurons", ylabel="Pre-synaptic Neurons",
           xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax.set_title(my_title, {'fontsize':10})
    ax.set_xticks(range(graph.number_of_nodes()))
    ax.set_yticks(range(graph.number_of_nodes()))
    ax.set_xticklabels(sorted_neuron_names, rotation='vertical')
    ax.set_yticklabels(sorted_neuron_names)
    sorted_index = np.argsort(graph.nodes())
    nc = _get_node_colors(graph, cmap=CMAP_DIFF, fixed_by_node_type=True)
    _ = [l.set_color(nc[sorted_index[i]]) for i, l in enumerate(ax.get_xticklabels())]
    _ = [l.set_color(nc[sorted_index[i]]) for i, l in enumerate(ax.get_yticklabels())]
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
    cmap = CMAP_DIFF
    # cmap.set_bad(color='#C9C9C9')
    # cmap.set_under(color='k')
    p = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
    plt.gca()
    ax.set(title=my_title, xlabel="Post-synaptic Neurons", ylabel="Pre-synaptic Neurons",
           xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax.set_title(my_title, {'fontsize': 10})
    ax.set_xticks(range(len(initial_net.nodes())))
    ax.set_yticks(range(len(initial_net.nodes())))
    ax.set_xticklabels(sorted_neuron_names, rotation='vertical', ha='center')
    ax.set_yticklabels(sorted_neuron_names, va='center')
    sorted_index = np.argsort(final_net.nodes())
    nc = _get_node_colors(final_net, CMAP_DIFF, fixed_by_node_type=True)
    _ = [l.set_color(nc[sorted_index[i]]) for i, l in enumerate(ax.get_xticklabels())]
    _ = [l.set_color(nc[sorted_index[i]]) for i, l in enumerate(ax.get_yticklabels())]
    ax.set_aspect('equal', adjustable='box')
    # if as_pct:
    #     plt.gcf().colorbar(p, ax=ax, format='%d%%')
    # else:
    #     plt.gcf().colorbar(p, ax=ax)
    return ax


def _get_node_plot_props(G, node_class=None, max_energy=None, active_node_color=None, active_edge_color=None,
                         dead_node_color=None):
    """
    `node_class` - Generic | Internal | Sensory | Motor
    `node_size` - proportional to the sum of the presynaptic connections it makes with other nodes.
    `node_colors` - function of excitatory/inhibitory, energy_value, firing/inactive

    """
    cm = CMAP_DIFF  # Shade from red (inhibitory) to green (excitatory)
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


def _get_node_colors(G, cmap, node_class=None, max_energy=None, firing_node_color=None, dead_node_color=None,
                     fixed_by_node_type=False):
    """Color node by: node_class (E or I), firing or not (edgecolor), """
    if node_class in ['Active', 'Dead', 'Firing']:
        return []
    if fixed_by_node_type:
        node_color = []
        for nid, props in G.node.iteritems():
            c = CMAP_DIFF(255) if props['node_type'] == 'E' else CMAP_DIFF(0)
            if nid in G.dead_nodes:
                c = RENDER_NODE_PROPS['Dead']['node_face_color']
            node_color.append(c)
        return node_color
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
        iax.set(ylim=(series.min() * (1. - y_l_buff), series.max() * (1. + y_u_buff)),
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


def plot_sensory_node_sensitivity(n, ax=None):
    """

    :param np: node properties dict, from the NeuralNetwork.edge[node_id].
    :param ax:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()

    pmax, pmin, kappa, phi_m = n['stimuli_max'], n['stimuli_min'], n['stimuli_sensitivity'], n['sensory_mid']
    f = n['stimuli_fxn']
    max_s = 100 * phi_m
    s = np.logspace(-5, 1, num=200)
    p = [f(i, pmin, pmax, kappa, phi_m) for i in s]
    ax.plot(s, p, '.-g')
    ax.set_xlabel('Sensory Concentration')
    ax.set_ylabel('Probability of Action Potential')
    ax.set_xscale('log')
    return p


def generate_node_data(G, node_class=None, node_type=None):
    post_fire_name = 'post_fire'
    colors = {post_fire_name:'#fcff60', 'E':'#ff9260', 'I':'#60b7ff'}
    G = G.subgraph(G.nodes(node_class=node_class, node_type=node_type))
    layout = nx.get_node_attributes(G, 'pos')
    name, pos = zip(*sorted(layout.items()))
    p = {'line_color': [], 'fill_color': []}
    for i in name:
        p['line_color'].append(colors['I'] if G.node[i]['node_type'] == 'I' else colors['E'])
        p['fill_color'].append(colors[post_fire_name] if G.node[i]['value'] == 1. else p['line_color'][-1])

    xp, yp = list(zip(*pos))
    _, thresh = zip(*sorted(nx.get_node_attributes(G, 'threshold').items()))
    _, ntype = zip(*sorted(nx.get_node_attributes(G, 'node_type').items()))
    _, nclass = zip(*sorted(nx.get_node_attributes(G, 'node_class').items()))
    _, energy = zip(*sorted(nx.get_node_attributes(G, 'energy_value').items()))
    _, state = zip(*sorted(nx.get_node_attributes(G, 'state').items()))
    _, spontaneity = zip(*sorted(nx.get_node_attributes(G, 'spontaneity').items()))
    _, value = zip(*sorted(nx.get_node_attributes(G, 'value').items()))
    _, presyn = zip(*sorted({G.nIx_to_nID[i]: val for i, val in enumerate(G.presyn_vector)}))
    out = dict(name=name, x=xp, y=yp, threshold=thresh, node_type=ntype, node_class=nclass, energy=energy,
             state=state, spontaneity=spontaneity, value=value, afferent_signal=None, presyn=presyn,
             line_color=p['line_color'], fill_color=p['fill_color'] )
    if node_class == "Sensory":
        _, signal = zip(*sorted(nx.get_node_attributes(G, 'signal').items()))
        out.update({'signal': signal})

    return pd.DataFrame(out)


def generate_edge_data(G, layout=None):
    layout = nx.get_node_attributes(G, 'pos') if layout is None else layout
    d = dict(x=[], y=[], text=[])
    for u, v, data in G.edges(data=True):
        d['x'].append([(layout[u][0] + layout[v][0]) / 2.])
        d['y'].append([(layout[u][1] + layout[v][1]) / 2.])
        d['text'].append(np.around(data['weight'], decimals=2))
    return pd.DataFrame(d)


def generate_connection_data(G, layout=None, offset=0.):
    layout = nx.get_node_attributes(G, 'pos') if layout is None else layout
    ad = dict(x0=[], x1=[], y0=[], y1=[])
    for u, v, data in G.edges(data=True):
        ad['x0'].append(layout[u][0])
        ad['x1'].append(layout[v][0])
        ad['y0'].append(layout[u][1])
        ad['y1'].append(layout[v][1])
    return pd.DataFrame(ad)


def draw_network_bokeh(G, fig=None):
    from bokeh.models import ColumnDataSource, Arrow
    from bokeh.plotting import show, figure
    from bokeh.models import HoverTool, LabelSet, Arrow, OpenHead, NormalHead, VeeHead
    # from bokeh.resources import CDN

    layout = nx.get_node_attributes(G, 'pos')
    df = generate_node_data(G)
    ed = generate_edge_data(G, layout)
    cd = generate_connection_data(G, layout)
    hover = HoverTool(tooltips=[('name', '@name'), ('class', '@node_class & @node_type'), ('threshold', '@threshold'),
                                ('energy', '@energy'), ('spontaneity', '@spontaneity')])

    if fig is None:
        plot = figure(title="Internal Network", tools=['tap', 'pan', hover, 'reset', 'wheel_zoom'])
    else:
        plot = fig
    # Setup data sources
    sens_source = ColumnDataSource(df[df['node_class'] == 'Sensory'])
    motor_source = ColumnDataSource(df[df['node_class'] == 'Motor'])
    int_source = ColumnDataSource(df[df['node_class'] == 'Internal'])
    nlabel_source = ColumnDataSource(df)
    elabel_source = ColumnDataSource(ed)
    connection_source = ColumnDataSource(cd)
    slabel_source = ColumnDataSource(generate_node_data(G, node_class='Sensory'))

    # Plot
    plot.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=connection_source, line_width=1, line_color="black",
                 line_alpha=0.6, legend='Synapses')
    sens = plot.diamond('x', 'y', source=sens_source, size=20, color='fill_color', legend='Sensory Neuron')
    moto = plot.square('x', 'y', source=motor_source, size=20, color='fill_color', legend='Motor Neuron')
    inte = plot.circle('x', 'y', source=int_source, size=20, line_color='line_color', fill_color='fill_color', legend='Internal Neuron')
    proc_labels = LabelSet(x='x', y='y', text="name", y_offset=10, text_font_size="8pt", text_color="#555555",
                           source=nlabel_source, text_align='center')
    sensory_label = LabelSet(x='x', y='y', text="signal", y_offset=-10, text_font_size="8pt", text_color="#555555",
                           source=slabel_source, text_align='center')
    plot.add_layout(proc_labels)
    edge_labels = LabelSet(x='x', y='y', y_offset=10, x_offset=-5, text_font_size="8pt", text_color="#555555",
                           source=elabel_source, text_align='center')
    plot.add_layout(edge_labels)

    plot.axis.visible = False
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.legend.location = "center_right"
    # connection_source = None
    sources = {'Sensory': sens_source, 'Motor': motor_source, 'Internal': int_source, 'nlabel': nlabel_source,
               'elabel': elabel_source, 'connections': connection_source}
    return sources, plot


def draw_fields(W, sensory_neuron=None, fig=None):
    sn = sensory_neuron
    if sensory_neuron:
        p0, p1, slope, mid = sn['stimuli_min'], sn['stimuli_max'], sn['stimuli_sensitivity'], sn['sensory_mid']
        fxn = sn['stimuli_fxn']
    else:
        from pyBrainNetSim.generators.special_functions import sigmoid
        p0, p1, slope, mid = 0., 1., 1.5, 0.3
        fxn = sigmoid
    max_signal, min_signal = 10, 10 ** -2
    x1 = np.logspace(np.log10(min_signal), np.log10(max_signal), num=200, base=10.)
    y1 = [fxn(i, p0, p1, slope, mid) for i in x1]
    x2 = W.attractor_field(field_type="Sensory")
    y2 = range(int(W.origin[0]), int(W.max_point[0] + 1))
    food = W.attractors[W.attractors.keys()[0]]

    # Map data to Bokeh DataSources
    sensory_source = ColumnDataSource(data=dict(x=x1, y=y1))
    field_source = ColumnDataSource(data=dict(x0=min_signal * np.ones_like(y2), y0=y2, x1=x2, y1=y2))
    ind_source = ColumnDataSource(data=dict(xf=[min_signal], yf=[food.position], xi=[None], yi=[None]))

    hover = HoverTool(tooltips=[("Field", "$x"), ("Location", "$y"), ("Sensori-Neural Response", "@y1")])

    f_world = figure(title="Attractor Emmitted Field",
                     tools=["crosshair,pan,reset,save,wheel_zoom", hover], x_axis_label='Emitted Field',
                     y_axis_label='Location', y_range=Range1d(W.origin[0], W.max_point[0]),
                     x_range=Range1d(min_signal, max_signal), x_axis_type="log", toolbar_location="above")
    f_world.extra_y_ranges = {'y2': Range1d(0., 1)}
    f_world.add_layout(LinearAxis(y_range_name='y2', axis_label='Strength of Sensori-neural Output'), 'right')
    f_world.line('x', 'y', source=sensory_source, line_width=3, line_alpha=0.6, y_range_name='y2',
                 legend='Sensory Encoding')
    f_world.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=field_source, line_width=2, line_color="green",
                    line_alpha=0.6, legend='Food Field')
    f_world.circle('x1', 'y1', source=field_source, size=5, fill_color="green", line_color="green", line_width=2)
    f_world.circle('xf', 'yf', source=ind_source, size=15, fill_color="green", line_color="green", line_width=1,
                   legend='Food')
    # f_world.diamond('xi', 'yi', source=ind_source, size=15, fill_color="orange", line_color="green", line_width=1,
    #                 legend='Individual')
    f_world.legend.location = "bottom_right"
    return f_world


def draw_trajectory(I, fig=None):
    x = [i[0] for i in I.trajectory[:-1]]
    time = range(len(x))
    a = [i[0] for i in I.attractor_pos]
    sg = [i[0][0] for i in I.sensory_gradients[:-1]]
    sg0 = np.zeros_like(sg)
    # print len(x), len(t), len(a), len(sg)
    data = ColumnDataSource(dict(x=x, a=a, t=time, sg=sg, sg0=sg0))
    W = I.environment
    hover = HoverTool(tooltips=[("time", "@t"), ("Location", "@x")])
    if fig is None:
        plot = figure(title="Trajectory", x_axis_label='Time', y_axis_label='Location',
                      tools=["crosshair,pan,reset,save,wheel_zoom", hover],
                      y_range=Range1d(W.origin[0], W.max_point[0]))
    else:
        plot = fig

    plot.extra_y_ranges = {'y2': Range1d(-1, 1)}
    plot.add_layout(LinearAxis(y_range_name='y2', axis_label='Sensory Gradient'), 'right')
    plot.line('t', 'x', source=data)
    plot.circle('t', 'x', source=data,  color='blue', legend='Individual Location')
    plot.line('t', 'a', source=data, color='red')
    plot.circle('t', 'a', source=data, color='red', legend='Food Location')
    plot.segment(x0='t', y0='sg0', x1='t', y1='sg', source=data, color='green', legend='Sensory Gradient', y_range_name='y2')
    plot.legend.location = "bottom_center"
    if I.is_dead:
        plot.add_layout(BoxAnnotation(left= I.dead_at, right=t[-1],
                                      fill_alpha=0.1, line_color='olive', fill_color='DarkGrey'))
    return plot