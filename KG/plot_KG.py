'''
Plot a saved solution of KG-type PDE.

'''

import numpy as np
from matplotlib import pyplot as plt

######################
# Specify which plots:
# Epsilon value:
eps = 0.01

######################
TT = 20
dt = 1e-4
t = np.arange(0.0, TT, dt)

Nx = 32
Lx = 2*np.pi
dx = Lx/Nx
x = np.arange(0,Lx,dx)  

X,T = np.meshgrid(x,t)
########################

# Load the solution:
sol = np.load(f'reference_solutions/KG/KG_ref_sol_eps{eps}.npy')

plt.figure()
plt.contourf(T,X,sol)
plt.xlabel('time')
plt.ylabel('x')
plt.colorbar()
plt.title(f'Reference solution, epsilon = {eps}')

plt.show()