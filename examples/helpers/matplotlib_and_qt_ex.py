# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:58:41 2015

@author: brian
"""

import sys
from PyQt4 import QtGui

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

class MainWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Network Motif")
        # a figure instance to plot on
        self.figure = plt.figure()
        self.figure.subplots_adjust(left=0.05,right=0.95,
                            bottom=0.05,top=0.95,
                            hspace=0.1,wspace=0.1)
        

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create dropdowns for motifs button connected to `plot` method
        motif_types = [
            'Feedforward Excitatory',
            'Feedforward Inhibitory',
            'Feedback Excitatory',
            'Feedback Inhibitory',
            'Converging/Diverging',
        ]
        self.motif_dropdown = motif = QtGui.QComboBox()
        motif.addItems(motif_types)
        
        self.button = QtGui.QPushButton('Plot')
        self.button.clicked.connect(self.plot)
        
        # Number of Nodes
        self.num_nodes = QtGui.QSpinBox()
        
        # set the layout
        layout = QtGui.QVBoxLayout()
        controls = QtGui.QHBoxLayout()
#        controls.addWidget()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(controls)
        controls.addWidget(motif)
        controls.addWidget(self.button)
        
        self.setLayout(layout)

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        ax.hold(False)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = MainWindow()
    main.showMaximized()
#    main.show()

    sys.exit(app.exec_())