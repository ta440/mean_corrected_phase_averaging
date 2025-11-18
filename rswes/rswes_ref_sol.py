'''
Generate reference solutions for the 1d rswes,
which can then be used to compute errors with
the phase-averaged methods.

This uses the modulation variable method.
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
epsilon = 0.1

ic_type = 'Gaussian_mean_shift'

save = False

longer_time = False
TT = 10

##########################################
# Setup parameters:

dt = 0.001
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


######################################
# Plots!

# Modulation variable:
fig, (ax1,ax2,ax3) = plt.subplots(3,1)
fig1 = ax1.contourf(T,X,V_u)
ax1.set_ylabel('u')
plt.colorbar(fig1,ax=ax1)
fig2 = ax2.contourf(T,X,V_v)
ax2.set_ylabel('v')
plt.colorbar(fig2,ax=ax2)
fig3 = ax3.contourf(T,X,V_phi)
ax3.set_ylabel('\u03d5')
ax3.set_xlabel('Time')
plt.colorbar(fig3,ax=ax3)
plt.suptitle('1D Gaussian, modulation variable space, \u03b5 = {}'.format(epsilon))

# Standard solution:
fig, (ax1,ax2,ax3) = plt.subplots(3,1)
fig1 = ax1.contourf(T,X,U_u)
ax1.set_ylabel('u')
plt.colorbar(fig1,ax=ax1)
fig2 = ax2.contourf(T,X,U_v)
ax2.set_ylabel('v')
plt.colorbar(fig2,ax=ax2)
fig3 = ax3.contourf(T,X,U_phi)
ax3.set_ylabel('\u03d5')
ax3.set_xlabel('Time')
plt.colorbar(fig3,ax=ax3)
plt.suptitle('1D Gaussian, \u03b5 = {}'.format(epsilon))


print(np.shape(U_u))

# Plot just u for graphical abstract:
fig, ax1 = plt.subplots(1,1, figsize=(10,4))
plot = ax1.contourf(T,X,U_u)
ax1.set_ylabel('$x \in [0, 2 \pi)$', size=14)
ax1.set_xlabel('Time', size=14)
plt.colorbar(plot)

fig, ax1 = plt.subplots(1,1, figsize=(10,4))
plot = ax1.contourf(T,X,V_u)
ax1.set_ylabel('$x \in [0, 2 \pi)$', size=14)
ax1.set_xlabel('Time', size=14)
plt.colorbar(plot)

plt.show()

#################################
# Save the solution array

# Coarser time interval to save the solution
dt_coarse = 1e-2
t_coarse = np.arange(0.0, TT, dt_coarse)

inds = np.arange(0,len(t),int(np.rint(dt_coarse/dt)))
inds = np.asarray(inds)

# Array to store the solutions
ref_sol = np.zeros((3,len(t_coarse), Nx))
ref_sol[0,:,:] = U_u[inds,:]
ref_sol[1,:,:] = U_v[inds, :]
ref_sol[2,:,:] = U_phi[inds, :]

if longer_time:
    savename = f'reference_solutions/rswes/rswes_{ic_type}_eps{epsilon}_TT{TT}.npy'
else:
    savename = f'reference_solutions/rswes/rswes_{ic_type}_eps{epsilon}.npy'

if save:
    print(f'Saving reference solution to {savename}')
    np.save(savename, ref_sol)