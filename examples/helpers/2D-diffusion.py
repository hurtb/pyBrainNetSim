# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:46:51 2016

@author: brian
"""

from fipy import *

nx, ny, dx, dy = 20, 20, 1., 1.
L = dx * nx
mesh = Grid2D(dx, dy, nx, ny)
phi = CellVariable(name="2D Diffusion", mesh=mesh, value=0.)
D = 1.
eq = TransientTerm() == DiffusionTerm(coeff=D)
valueTopLeft, valueBottomRight = 0., 1.

# Set Boundary Conditions
X, Y = mesh.faceCenters
facesTopLeft =((mesh.facesLeft & (Y > L / 2)) | (mesh.facesTop & (X < L / 2)))
facesBottomRight = ((mesh.facesRight & (Y < L / 2)) | (mesh.facesBottom & (X > L / 2)))
phi.constrain(valueTopLeft, facesTopLeft)
phi.constrain(valueBottomRight, facesBottomRight)

if __name__ == '__main__':
    viewer = Matplotlib2DViewer(vars=phi, datamin=0., datamax=1.)
    viewer.plot()
    
timeStepDuration = 10 * 0.9 * dx**2 / (2 * D)
steps = 10
for step in range(steps):
    eq.solve(var=phi, dt=timeStepDuration)
    if __name__ == '__main__':
        viewer.plot()