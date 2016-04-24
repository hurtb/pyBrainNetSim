# -*- coding: utf-8 -*-
"""
simnet_test.py
Created on Thu Dec 31 12:19:04 2015

@author: brian
"""

import sys
sys.path.append('../')
from pyBrainNetSim.models import SimNetBasic
from pyBrainNetSim.generators import sensor_mover
import networkx as nx

#network = np.array([[0,1,0,1],[1,0,0,0],[1,1,0,1], [1,0,0,0]])
#network = nx.connected_watts_strogatz_graph(10, 3, 0.6)
sim_options = {'max_iter':8, 'dt': 1}
#network = excitatory_circular(3, threshold=0.9, in_per=0, spont_driver_prob=.5, spontaneity=.9)
network = sensor_mover(sensor_dir=[(0,1),(0,-1)], motor_dir=[(1,0),(-1,0),(0,1),(0,-1)], internal_neurons=9)
#network = sensor_mover(sensor_dir=[(0,1)], motor_dir=[(1,0)], internal_neurons=4)

class SimNetTest(SimNetBasic):
    
    def test1(self, **kwargs):
        self.simulate(**kwargs)

if __name__ == '__main__':  
    
    ts = SimNetTest(network, initial_fire='prescribed', prescribed=['S0'])    
    ts.test1(**sim_options)
    
    sd = ts.simdata
    col_order = sorted(ts.simdata[-1].nodes())
    col_order.reverse()
    nx.draw_networkx(ts.simdata[-1], pos=nx.get_node_attributes(ts.simdata[-1],'pos'))
    
    print ts.simdata.node_group_properties('presyn_vector')[col_order]