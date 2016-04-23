# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 08:57:44 2016

@author: brian
"""

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from pyBrainNetSim.models.world import Environment, Individual
from pyBrainNetSim.drawing.viewers import vTrajectory

# setup environment
scale , x0, y0 =100., 50., 50.
w1 = Environment(scale * np.array([1.,1.]))
i1 = Individual(environment=w1)

# create random, normally distrubuted trajectory
x_drift, x_var, y_drift, y_var, avg_step, num_steps = 0., 1., 0., 1., 3., 50
x = np.random.normal(x_drift, x_var, size=num_steps)
y = np.random.normal(y_drift, y_var, size=num_steps)
x, y = avg_step * x.cumsum() + x0, avg_step * y.cumsum() + y0
points = np.array([x, y]).transpose().reshape(-1,1,2)
i1.set_trajectory(points)
#traj_opt = {"facecolors":'red'}

# Setup plotting parameters and plot
fig, ax = plt.subplots()
lines = vTrajectory(points)
lines.text.set_fontsize(9)
ax.hold()
ax.scatter(*lines.get_segments()[-1][-1])
ax.add_collection(lines)
ax.set_xlim(left=0, right=100)
ax.set_ylim(bottom=0, top=100)
plt.show()