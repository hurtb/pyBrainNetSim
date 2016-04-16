# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:07:23 2016

@author: brian
"""

import os
import matplotlib.pyplot as plt


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
