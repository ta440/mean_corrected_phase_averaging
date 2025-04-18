'''
Generates a reference solution for the swinging spring ODE.
This is computed using standard RK4.

Errors are then computed relative to this for the phase-averaged
methods. A range of resonance factors can be defined and 
saved into a single numpy array.
'''

import numpy as np
from timestepping import *
from swing_spring_functions import *

##################
# Choose the range of resonance factors:
rhos = [2]#np.arange(1.5,2.51,0.010)

###################
#Parameter values
g = np.pi*np.pi
l_o = 1.2
l = 1
k = 4*g
m = 1

###################

#Calculate the natural frequencies of the normal modes::
omega_R = np.sqrt(g/l)

############################################
#Define the intitial conditions as three complex variables:
x0 = 0.04
y0 = 1j*0.03427/omega_R
z0 = 0.08

init = np.array([x0,y0,z0])
#######################################
# Define the time interval of the base solution
TT = 200 
dt = 1e-4
t = np.arange(0.0, TT, dt)

# Coarser time interval to save the solution
dt_coarse = 1e-2
t_coarse = np.arange(0.0, TT, dt_coarse)

inds = np.arange(0,len(t),int(np.rint(dt_coarse/dt)))
inds = np.asarray(inds)

# Array to store the solutions
ref_sols = np.zeros((len(rhos),3,len(t_coarse)))

#####################################
# Create the solutions:
for m in np.arange(len(rhos)):
    rho = rhos[m]
    print(f'rho = {rho}')

    omega_Z = rho*omega_R
    lamda = (l_o*omega_Z*omega_Z)/(l*l)

    sol_u = np.asarray(RK4(swing_spring,init,t,dt,[omega_R, rho, lamda]))
    sol_u = np.asarray(sol_u)

    ref_sols[m,0,:] = np.real(sol_u[inds,0])
    ref_sols[m,1,:] = np.real(sol_u[inds,1])
    ref_sols[m,2,:] = np.real(sol_u[inds,2])


# Save the solution
savename = f'reference_solutions/swing_spring/swing_spring_RF_range.npy'
np.save(savename, ref_sols)