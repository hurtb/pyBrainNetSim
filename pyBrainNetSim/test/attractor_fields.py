# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 11:24:59 2016

@author: brian
"""

import sys
sys.path.append('../../')
import numpy as np
from pyBrainNetSim.models.world import ExponentialDecayScalarField

x = np.arange(0, 4, 1.)
y = np.arange(0, 4, 1.)
z = np.arange(0, 4, 1.)

c_grid = np.meshgrid(x, y)

edsf = ExponentialDecayScalarField('Sensory', c_grid)
edsf.add_point_source((1,1), 10.)
#edsf.add_point_source((3,3), 10.)
print edsf.field()
print edsf.gradient()

print ")_____"
print edsf.field_at((0,0))
print edsf.gradient_at((0,0))