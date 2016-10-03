# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:07:23 2016

@author: brian
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


def create_movie(plotter, numberOfFrames, fps=10, fpath='', fname='movie.mp4'):
    for i in range(numberOfFrames):
        plotter(i)
        ftname = '%s\_tmp%05d.png' %(fpath,i)
        plt.savefig(ftname)
        plt.clf()
    os.system("rm %s\%s" %(fpath,fname))
    os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie.mp4")
    os.system("rm _tmp*.png")


def merge_dicts(*dict_args):
    """Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts."""
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def cart2pol(c_grid, x0=(0., 0.)):
    """
    function to return a polar grid from a cartesian grid
    :param c_grid: grid of 2 or 3 dimensions. Use np.meshgrid function
    :param x0: origin. Default (0,0) or (0,0,0)
    :return: polar coordinates in 2 or 3 dimensions
    """
    _cg = [g.copy() for g in c_grid]
    if len(c_grid) not in [2, 3]:
        return []
    if len(x0) not in [2, 3]:
        x0 = np.zeros(len(_cg))
    for i in range(len(_cg)):
        _cg[i] = _cg[i] - x0[i]
    r = np.sqrt(_cg[0]**2 + _cg[1]**2)
    if len(c_grid) == 2:
        theta = np.arctan2(_cg[1], _cg[0])
        return r, theta
    elif len(c_grid) == 3:
        theta = np.arctan2(_cg[0], _cg[1])
        phi = np.arctan2(_cg[0]**2 + _cg[1]**2, _cg[2])
        return r, theta, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def points_to_segments(points):
    """Returns a list of [[x0,x1], [x1, x2], ...] from [x0, x1, x2 ...]"""
    segs = []
    for i in range(len(points)-1):
        segs.append([points[i], points[i+1]])
    return segs

def save_pickle(objs, file_name):
    with open(file_name, 'w') as f:
        pickle.dump(objs, f)
