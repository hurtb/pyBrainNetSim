""" Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
"""
import numpy as np
import networkx as nx
import pandas as pd
from functools import partial
from threading import Thread
import time
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, layout, column
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, HoverTool
import bokeh.models.widgets as wg
from bokeh.plotting import figure
from bokeh.document import without_document_lock
import pyBrainNetSim as pbns
import pyBrainNetSim.models.world as world
import pyBrainNetSim.models.individuals as ind
from pyBrainNetSim.generators import network, special_functions, builtin
import pyBrainNetSim.generators.settings.base as settings
import pyBrainNetSim.simulation.simnetwork as sim
# from bokeh.charts import Scatter
# import pyBrainNetSim.simulation.evolution as evo
import pyBrainNetSim.drawing.viewers as vw
# from pyBrainNetSim.generators import special_functions as sf
# from tornado import gen
# from concurrent.futures import ThreadPoolExecutor

np.set_printoptions(precision=1)

# Initial Settings
max_signal, min_signal, p0i, p1i, slopei, midi = 10, 10**-2, 0., 1., 1.5, 0.3
w_sizei, w_min, w_max, f_loci, i_loci, fig_width, perm, decay_fxn = (0,40), 0, 40, 5, 2, 800, 0.1, 'Exponential'
discretizationi = 2
fig_height = fig_width * 9 / 16
control_width = 200
N = 200
time_steps = 2
initial_number_of_individuals = 2


def create_world(x_range, decay_function, permeability, food_loc):
    if decay_function == "Exponential":
        funct = pbns.models.world.ExponentialDecayScalarField
    elif decay_function == "Linear":
        funct = pbns.models.world.LinearDecayScalarField
    w = pbns.models.world.Environment(origin=[x_range[0]], max_point=[x_range[1]], field_decay=funct,
                          field_permeability=permeability)
    f = pbns.models.world.Attractor(environment=w, position=[food_loc], strength=1.)  # add "food"
    return w, f


def create_individual(w, discretization_number, position=[0.], sensitivity=None, smid=None, sminsignal=0.,
                      smaxsignal=1.):
    neurons, edges = pbns.generators.builtin.simple_SM1D(number_signal_discretization=discretization_number,
                                                         sens_sensitivity=sensitivity, sens_mid=smid,
                                                         sens_minsignal=sminsignal, sens_maxsignal=smaxsignal)
    smpd = pbns.generators.network.SensorMoverProperties(nodes=neurons,
                                                         edges=edges,
                                                         internal=pbns.generators.network.InternalNodeProperties(settings.NODE_PROPS['Internal']))
    I = pbns.models.individuals.SensorMover(w,
                                            initial_network=pbns.simulation.simnetwork.HebbianNetworkBasic(smpd.create_digraph(),
                                                                   pos_synapse_growth=0., neg_synapse_growth=0.),
                                            position=position)
    return I


def generate_node_data(G, node_class=None, node_type=None):
    post_fire_name = 'post_fire'
    colors = {post_fire_name:'#fcff60', 'E':'#ff9260', 'I':'#60b7ff'}
    G = G.subgraph(G.nodes(node_class=node_class, node_type=node_type))
    layout = nx.get_node_attributes(G, 'pos')
    name, pos = zip(*sorted(layout.items()))
    xp, yp = list(zip(*pos))
    _, thresh = zip(*sorted(nx.get_node_attributes(G, 'threshold').items()))
    _, ntype = zip(*sorted(nx.get_node_attributes(G, 'node_type').items()))
    _, nclass = zip(*sorted(nx.get_node_attributes(G, 'node_class').items()))
    _, energy = zip(*sorted(nx.get_node_attributes(G, 'energy_value').items()))
    _, state = zip(*sorted(nx.get_node_attributes(G, 'state').items()))
    _, spontaneity = zip(*sorted(nx.get_node_attributes(G, 'spontaneity').items()))
    _, value = zip(*sorted(nx.get_node_attributes(G, 'value').items()))
    # print G.number_of_nodes()
    # _, afferent = zip(*sorted({G.nIx_to_nID[i]: val for i, val in enumerate(G.afferent_signal)}))
    _, presyn = zip(*sorted({G.nIx_to_nID[i]: val for i, val in enumerate(G.presyn_vector)}))
    p = {'line_color': [], 'fill_color': []}
    for i in name:
        p['line_color'].append(colors['I'] if G.node[i]['node_type'] == 'I' else colors['E'])
        p['fill_color'].append(colors[post_fire_name] if G.node[i]['value'] == 1. else p['line_color'][-1])

    return pd.DataFrame(
        dict(name=name, x=xp, y=yp, threshold=thresh, node_type=ntype, node_class=nclass, energy=energy,
             state=state, spontaneity=spontaneity, value=value, afferent_signal=None, presyn=presyn,
             line_color=p['line_color'], fill_color=p['fill_color'] ))


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


def draw_network(G, fig=None):
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
        plot = figure(plot_width=800, plot_height=400,
                      tools=['tap', 'pan', hover, 'reset', 'wheel_zoom'])
    else:
        plot = fig
    plot.add_tools(hover)
    # Setup data sources
    sens_source = ColumnDataSource(df[df['node_class'] == 'Sensory'])
    motor_source = ColumnDataSource(df[df['node_class'] == 'Motor'])
    int_source = ColumnDataSource(df[df['node_class'] == 'Internal'])
    nlabel_source = ColumnDataSource(df)
    elabel_source = ColumnDataSource(ed)
    connection_source = ColumnDataSource(cd)
    slabel_source = ColumnDataSource()

    # Plot
    plot.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=connection_source, line_width=1, line_color="black",
                 line_alpha=0.6, legend='Synapses')
    sens = plot.diamond('x', 'y', source=sens_source, size=20, color='fill_color', legend='Sensory Neuron')
    moto = plot.square('x', 'y', source=motor_source, size=20, color='fill_color', legend='Motor Neuron')
    inte = plot.circle('x', 'y', source=int_source, size=20, line_color='line_color', fill_color='fill_color', legend='Internal Neuron')
    proc_labels = LabelSet(x='x', y='y', text="name", y_offset=10, text_font_size="8pt", text_color="#555555",
                           source=nlabel_source, text_align='center')
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


# Create Initial Data
w, food = create_world(w_sizei, decay_fxn, perm, f_loci)
I = create_individual(w, discretizationi, sensitivity=slopei, smid=midi, sminsignal=p0i, smaxsignal=p1i)

x1 = np.logspace(np.log10(min_signal), np.log10(max_signal), num=N, base=10.)
y1 = [pbns.generators.special_functions.sigmoid(i, p0i, p1i, slopei, midi) for i in x1]
x2 = w.attractor_field(field_type="Sensory")
y2 = range(w_sizei[0], w_sizei[1]+1)

# Map data to Bokeh DataSources
sensory_source = ColumnDataSource(data=dict(x=x1, y=y1))
field_source = ColumnDataSource(data=dict(x0=min_signal * np.ones_like(y2), y0=y2, x1=x2, y1=y2))
ind_source = ColumnDataSource(data=dict(xf=[min_signal], yf=[f_loci], xi=[min_signal], yi=[i_loci]))

# @without_document_lock
def simulate_one_step():
    global I
    I.sim_time_step()
    redraw_network()
    i_loc.value = I.position[0]
    ind_source.data=dict(xf=[min_signal], yf=[f_loc.value], xi=[min_signal], yi=[i_loc.value])


# Create Widgets
w_size = wg.RangeSlider(start=w_min, end=w_max, range=w_sizei, step=1, title="World Size", width=control_width)
w_decay = wg.Select(title="Food Field Decay Type", value=decay_fxn, options=['Exponential', 'Linear'], width=control_width)
w_perm = wg.Slider(start=0.05, end=.5, value=perm, step=0.025, title="Food Field Decay Rate", width=control_width)
f_loc = wg.Slider(start=w_min, end=w_max, value=f_loci, step=1, title="Food Location", width=control_width)
i_loc = wg.Slider(start=w_min, end=w_max, value=i_loci, step=1, title="Individual Location", width=control_width)
p0 = wg.Slider(title="Minimum Probability", value=p0i, start=0.0, end=0.5, step=0.05, width=control_width)
p1 = wg.Slider(title="Maximum Probability", value=p1i, start=0.5, end=1, step=0.05, width=control_width)
slope = wg.Slider(title="Response Slope", value=slopei, start=0.01, end=10., step=0.01, width=control_width)
mid = wg.Slider(title="Signal Mid-Point", value=midi, start=0.05, end=0.99, step=0.05, width=control_width)
env_inputs = (w_size, w_decay, w_perm, f_loc, i_loc, p0, p1, slope, mid)

hover = HoverTool(tooltips=[("Field", "$x"), ("Location", "$y"), ("Sensori-Neural Response", "$y2")])

f_world = figure(plot_height=fig_height, plot_width=fig_width, title="Attractor Emmitted Field",
                 tools=["crosshair,pan,reset,save,wheel_zoom", hover], x_axis_label='Emitted Field', y_axis_label='Location',
                 y_range=Range1d(w_min, w_max), x_range=Range1d(min_signal, max_signal), x_axis_type="log",
                 toolbar_location="above")
f_world.extra_y_ranges = {'y2': Range1d(0., 1)}
f_world.add_layout(LinearAxis(y_range_name='y2', axis_label='Strength of Sensori-neural Output'), 'right')
f_world.line('x', 'y', source=sensory_source, line_width=3, line_alpha=0.6, y_range_name='y2',
             legend='Sensory Encoding')
f_world.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=field_source, line_width=2, line_color="green",
                line_alpha=0.6, legend='Food Field')
f_world.circle('x1', 'y1', source=field_source, size=5, fill_color="green", line_color="green", line_width=2)
f_world.circle('xf', 'yf', source=ind_source, size=15, fill_color="green", line_color="green", line_width=1,
               legend='Food')
f_world.diamond('xi', 'yi', source=ind_source, size=15, fill_color="orange", line_color="green", line_width=1,
                legend='Individual')
f_world.legend.location = "bottom_right"


network_type = wg.Select(title="Network Type:", value="Network Type", options=["1D Sensor Mover"], width=control_width)
num_layers = wg.Slider(title="Signal Discritization", value=2, start=1, end=10, step=1, width=control_width)
signal_map = wg.Select(title="Signal Mapping Growth:", value="Linear", options=["Linear"], width=control_width)
f_network = figure(plot_height=fig_height, plot_width=fig_width, title="Internal Network", toolbar_location="above")
net_source,  _ = draw_network(I.internal.simdata[-1], fig=f_network)
# f_network.renderers = []

network_inputs = (network_type, num_layers, signal_map)

create_network_button = wg.Button(label="Create Network", button_type="primary", width=150)
simulate_network_toggle = wg.Toggle(label="Simulate", button_type="primary", width=150)
simulate_one_button = wg.Button(label="Simulate 1 Step", button_type="primary", width=150)
reset_sim_button = wg.Button(label="Reset", button_type="warning", width=100)
f_trajectory = figure(plot_height=fig_height, plot_width=fig_width, title="Trajectory")
time_slider = wg.Slider(title="Time", value=0, start=0, end=1, step=1, width=fig_width)

# f_trajectory = figure(plot_height=fig_height, plot_width=fig_width, title="Attribute",
#                       tools=["crosshair,pan,reset,save,wheel_zoom", hover], x_axis_label='Time',y_axis_label='Location',
#                       y_range=Range1d(w_min, w_max), x_range=Range1d(min_signal, max_signal), toolbar_location="above")

data_cols = [wg.TableColumn(field='name', title='ID'), wg.TableColumn(field='presyn', title='Presyn Sig'),
             wg.TableColumn(field='state', title='State'), wg.TableColumn(field='threshold', title='Thresh'),
             wg.TableColumn(field='value', title='AP'), wg.TableColumn(field='energy', title='E')]
data_table = wg.DataTable(source=net_source['nlabel'], columns=data_cols, width=300, height=fig_height)


# Set up callbacks
def update_env(attrname, old, new):
    global w, food
    # Get the current slider values
    ws, wd, wp, fl, il, pm, pM, s, m = w_size.range, w_decay.value, float(w_perm.value), f_loc.value, i_loc.value,\
                                       p0.value, p1.value, slope.value, mid.value

    w, food = create_world(ws, wd, wp, fl)

    # Generate the new curve
    x_0 = w.attractor_field(field_type="Sensory")
    y_0 = range(ws[0], ws[1]+1)
    x_1 = np.logspace(np.log10(min_signal), np.log10(max_signal), num=N, base=10.)
    y_1 = [special_functions.sigmoid(i, pm, pM, s, m) for i in x_1]

    sensory_source.data = dict(x=x_1, y=y_1)
    field_source.data = dict(x0=min_signal * np.ones_like(x_0), y0=y_0, x1=x_0, y1=y_0)
    ind_source.data = dict(xf=[min_signal], yf=[fl], xi=[min_signal], yi=[il])

# @gen.coroutine
def redraw_network():
    global I, f_network, w, food
    w = I.environment
    print "Field: %s @ %s | source: %s" \
          %(I.position, w.attractor_field_at(I.position, field_type='Sensory')[0], w.attractors[w.attractors.keys()[0]].position)
    # print I.internal.simdata[-1].nodes()
    # print I.internal.simdata[-1].presyn_vector
    # print I.internal.simdata[-1].synapses
    # print I.internal.simdata[-2].postsyn_signal
    # print I.internal.simdata[-1].spont_signal
    # print I.internal.simdata[-1].driven_vector
    # print I.internal.simdata[-1].afferent_signal
    # print I.internal.simdata[-1].attr_vector('threshold')
    # print I.internal.simdata[-1].prop_vector
    # print I.internal.simdata[-1]
    # print I.internal.simdata[-1].prop_vector
    # print I.internal.simdata[-1].postsyn_nodes
    # print nx.get_node_attributes(I.internal.simdata[-1], 'signal')
    # print nx.get_node_attributes(I.internal.simdata[-1], 'sensory_response')
    # print nx.get_node_attributes(I.internal.simdata[-1], 'sensory_position')
    # print "----"
    nd = generate_node_data(I.internal.simdata[-1])
    ed = generate_edge_data(I.internal.simdata[-1])
    cd = generate_connection_data(I.internal.simdata[-1])
    net_source['Sensory'].data = nd[nd['node_class'] == 'Sensory'].to_dict(orient='list')
    net_source['Motor'].data = nd[nd['node_class'] == 'Motor'].to_dict(orient='list')
    net_source['Internal'].data = nd[nd['node_class'] == 'Internal'].to_dict(orient='list')
    net_source['nlabel'].data = nd.to_dict(orient='list')
    net_source['elabel'].data = ed.to_dict(orient='list')
    net_source['connections'].data = cd.to_dict(orient='list')
    f_network.title.text = "Internal Network @t%d" %(I.t+1)


def reset_network(attrname, old, new):
    global I
    nl = num_layers.value
    I = create_individual(w, nl, position=[i_loc.value], sensitivity=slope.value, smid=mid.value, sminsignal=p0.value,
                          smaxsignal=p1.value)
    redraw_network()


def reset_net():
    reset_network(None, None, None)

# Setup Callbacks
vals = ['value', 'range']
for item in env_inputs:
    for val in vals:
        if hasattr(item, val):
            item.on_change(val, update_env)

vals = ['value', 'range']
for item in [num_layers]:
    for val in vals:
        if hasattr(item, val):
            item.on_change(val, reset_network)

def toggle_sim(attr):
    if simulate_network_toggle.active:
        simulate_network_toggle.button_type = 'danger'
        simulate_network_toggle.label = 'Simulating...'

    else:
        simulate_network_toggle.button_type = 'primary'
        simulate_network_toggle.label = 'Simulate'


simulate_one_button.on_click(simulate_one_step)
reset_sim_button.on_click(reset_net)
simulate_network_toggle.on_click(toggle_sim)

doc = curdoc()
# executor = ThreadPoolExecutor(max_workers=2)

def blocking_task(i):
    time.sleep(1)
    return i

# @gen.coroutine
# @without_document_lock
def simulation():
    if simulate_network_toggle.active:
        # do some blocking computation
        # res = yield executor.submit(blocking_task, i)
        simulate_one_step()

        # but update the document from callback
        # doc.add_next_tick_callback(partial(update, x=x, y=y))
        # doc.add_next_tick_callback(partial(redraw_network))

# Set up layouts and add to document

design_widget = layout([[simulate_network_toggle, simulate_one_button, reset_sim_button],
                        [widgetbox(*env_inputs), f_world],
                        [widgetbox(*network_inputs), f_network, data_table],
                        ])

# sim_widget = column(sim_net_tog, f_networksim, f_trajectory, f_attribute, time_slider)
sim_net_tog = wg.Toggle(label="Simulate", button_type="primary")
sim_widget = column(sim_net_tog)

tab1 = wg.Panel(child=design_widget, title="Design Network")
tab2 = wg.Panel(child=sim_widget, title="Simulate")
tabs = wg.Tabs(tabs=[tab1, tab2])
doc.add_root(tabs)
doc.title = "Sensor Mover 1D"

# doc.add_periodic_callback(simulation, 1000)