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
# Choose either a range of resonance factors,
# or a single one.
single_sol = False

# Set a single rho
rho = 1.7

# Or a range:
rhos = np.arange(1.5,2.51,0.010)
rhos = np.round(rhos,2)

###################
#Parameter values
g = np.pi*np.pi
l_o = 1.2
l = 1
k = 4*g
m = 1

if single_sol:
    rhos = np.array([rho])

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
dt = 1e-2
t = np.arange(0.0, TT, dt)

# Array to store the solutions
ref_sols = np.zeros((len(rhos),3,len(t)))

#####################################
# Create the solutions:
for m in np.arange(len(rhos)):
    rho = rhos[m]
    print(f'rho = {rho}')

    omega_Z = rho*omega_R
    lamda = (l_o*omega_Z*omega_Z)/(l*l)

    sol_u = np.asarray(RK4(swing_spring,init,t,dt,[omega_R, rho, lamda]))
    sol_u = np.asarray(sol_u)

    ref_sols[m,0,:] = np.real(sol_u[:,0])
    ref_sols[m,1,:] = np.real(sol_u[:,1])
    ref_sols[m,2,:] = np.real(sol_u[:,2])

# Save the solution
if single_sol == True:
    savename = f'reference_solutions/swing_spring/swing_spring_rho{rho}.npy'
else:
    savename = f'reference_solutions/swing_spring/swing_spring_rho_range.npy'

np.save(savename, ref_sols)