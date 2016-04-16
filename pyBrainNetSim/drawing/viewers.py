# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 09:37:35 2016

@author: brian
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.transforms as mtransforms
import matplotlib.text as mtext


class vTrajectory(LineCollection):
    def __init__(self, points, base_clr=(0, 0, 0), *args, **kwargs):
        # we'll update the position when the line data is set
        self.text = mtext.Text(0, 0, '')
        t = np.linspace(0, 1, points.shape[0]) # your "time" variable
        # set up a list of segments
        segs = np.concatenate([points[:-1],points[1:]], axis=1)
        cmp1 = LinearSegmentedColormap.from_list("my_cmp", ((1, 1, 1), base_clr))
        
        super(vTrajectory, self).__init__(segs, cmap=cmp1, *args, **kwargs)
        self.set_array(t)

        # we can't access the label attr until *after* the line is
        # inited
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