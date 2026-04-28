'''
Solve the 1d rswes using the standard modulation variable
method and compute errors relative to a
reference solution, computed with a much
smaller timestep.
'''

import numpy as np
from matplotlib import pyplot as plt

######################################
# Import helper functions:
import sys
sys.path.append('../')

from functions.timestepping import *
from functions.rswes_functions import *

##########################################
# Define the level of time-scale separation.
# Values of 0.5,0.1,0.05,0.01, 0.001 are used in the paper.
epsilon = 0.001

ic_type = 'Gaussian_mean_shift'

longer_time = True

if longer_time:
    TT = 50
else:
    TT = 10

##########################################
# Setup parameters:

dt = 0.3
t = np.arange(0,TT,dt) 

Nx = 32
Lx = 2*np.pi
dx = Lx/Nx
x = np.arange(0,Lx,dx)  

X,T = np.meshgrid(x,t)
#####################################

k = np.fft.fftfreq(Nx,dx)*(2*np.pi)

# Create a de-aliasing array with top 2/3 of wavenumbers set to zero:
k_max = np.max(np.abs(k))

# Specify the strength of the 
# fourth-order hyperviscosity.
visc = 1e-4

# Nondimensional dispersion relation
psi = np.sqrt((1 + (k**2)))

# Time independent part of the matrix exponential
A = np.array([[psi/psi,(-1/psi),(1j*k/psi)],
             [(1/psi), (1/psi**2),(1j*k/(psi**2))],
             [(1j*k/psi),(1j*k/(psi**2)), (1/(psi**2))]])

###############################
# Initial condition
init = set_RSWE_initial_conditions(x, ic_type)

#print(np.max(init[2,:]))

plt.figure()
plt.plot(x, init[2,:])
plt.xlabel('x')
plt.title('Symmetrical geopotential height')
plt.title(f'{ic_type} initial condition')

#plt.show()

# Transform IC into spectral space
init_spec = np.fft.fft(init)

# Obtain the solution, in the modulation variable.
V_specs = np.asarray(RK4(etL_dot_N,init_spec,t,dt, [A, psi, k, visc, epsilon]))

# Components of the modulation variable solution
V_u = np.real(np.fft.ifft(V_specs[:,0,:]))
V_v = np.real(np.fft.ifft(V_specs[:,1,:]))
V_phi = np.real(np.fft.ifft(V_specs[:,2,:]))

# Convert back to the standard variable
#Now, get the proper solution variables.
U_specs = np.asarray(V_to_U(t, init_spec, V_specs, [A, psi, k, epsilon]))

U_u = np.real(np.fft.ifft(U_specs[:,0,:]))
U_v = np.real(np.fft.ifft(U_specs[:,1,:]))
U_phi = np.real(np.fft.ifft(U_specs[:,2,:]))


############################################################
# Base solution to compare errors relative to:

# Ref sol time step size
dt_b = 1e-2

if longer_time:
    U_analyt = np.load(f'../reference_solutions/rswes/rswes_{ic_type}_eps{epsilon}_TT{TT}.npy')
else:
    U_analyt = np.load(f'../reference_solutions/rswes/rswes_{ic_type}_eps{epsilon}.npy')

# Indices to compare with reference solution:
t_b = np.arange(0,TT+dt_b,dt_b)
inds = np.arange(0,len(t_b),int(np.rint(dt/dt_b)))
inds = np.asarray(inds)

if len(inds) != len(t):
    inds = inds[:-1]

U_analyt = U_analyt[:, inds, :]   

u_analyt = U_analyt[0,:]
v_analyt = U_analyt[1,:]
phi_analyt = U_analyt[2,:]

#######################

# Compute the L2 error in each field, then sum
u_err = np.sqrt(np.sum((U_u-u_analyt)**2,axis=1))
v_err = np.sqrt(np.sum((U_v-v_analyt)**2,axis=1))
phi_err = np.sqrt(np.sum((U_phi-phi_analyt)**2,axis=1))

# Sum the L2 field errors
tot_err = np.sum(u_err + v_err + phi_err)/len(t)


print(f'epsilon = {epsilon}, dt = {dt}, TT={TT}')
print(f'total error is {tot_err}')