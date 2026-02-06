# -*- coding: utf-8 -*-
"""

Tests for the 1d RSWEs
Comparing:
1) standard phase-averaging
2) Mean corrected phase-averaging with a local mean correction

Pick the value of epsilon (the speed of linear oscillations)
and a single time step to compare at. 
Define the range of normalised averaging windows.
Then, store the results and make a Peddle Plot.

@author: timmo
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy

######################################
# Import helper functions:
import sys
sys.path.append('../')

from functions.timestepping import *
from functions.rswes_functions import *

##################################################################
# Define the level of time-scale separation.
# Values of 0.5,0.1,0.05,0.01, 0.001 are used in the paper.
epsilon = 0.1

# Set a single large timestep size. 
# Values of 0.05, 0.1, ... 0.35 are used in the paper.
dt = 0.3

ic_type = 'Gaussian_mean_shift'

longer_time = False

TT = 10

##########################################
# Setup parameters:
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

###############################################################
# Initial condition
init = set_RSWE_initial_conditions(x, ic_type)

# Transform IC into spectral space
init_spec = np.fft.fft(init)

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

# Parameters for the timestepping averaging kernel
K_min = 21 # Minimum number of sample points
ppp = 4 # averaging points per period
alpha = 4 # Kernel decay rate

# Range of phase-averaging windows for timestepping
zetas = np.arange(0.05, 2.5, 0.05)

# Solutions to save errors in 
C_errs = np.zeros(len(zetas))

print(U_analyt.shape)

U_analyt = U_analyt[:, inds, :]   

u_analyt = U_analyt[0,:]
v_analyt = U_analyt[1,:]
phi_analyt = U_analyt[2,:]

################################################################

###################################################################
# Loop over the phase-averaged methods with the different 
# averaging window sizes
# The mean corrected method uses eta=eta_C to
# reduce the size of the parameter space.

for j in np.arange(len(zetas)):
    zeta = zetas[j]
    zeta = np.round(zeta,3)
    print(zeta)    
    
    eta = zeta*dt
    eta = np.round(eta,3)
    
    #Set the required number of kernel points.
    K_an = np.ceil(ppp*eta*np.max(psi)/(epsilon*2*np.pi))
    K = int(max(K_an,K_min))
    print(f'zeta={zeta}, eta={eta}, K = {K} \n')

    # Construct the kernel
    s_vals, kernel = kernel_vals(K, eta, alpha)    

    V_specs = np.asarray(RK4(etL_dot_N_aved, init_spec, t, dt, [kernel, s_vals, K, A, psi, k, visc, epsilon]))

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

    #Compute errors relative to the saved analytical solution:
    u_err = np.sum(np.abs(U_u-u_analyt),axis=1)
    v_err = np.sum(np.abs(U_v-v_analyt),axis=1)
    phi_err = np.sum(np.abs(U_phi-phi_analyt),axis=1)

    #Sum of L2 errors:
    tot_err = np.sum(np.sqrt(u_err**2 + v_err**2 + phi_err**2))/len(t)

    C_errs[j] = tot_err

C_best = np.nanmin(C_errs)
C_zeta_opt = zetas[np.nanargmin(C_errs)]

print('C best error is, ', C_best)
print('C best normalised window is, ', C_zeta_opt)

#Or, a normalised version (eta/DT):
plt.figure()
plt.plot(zetas,C_errs)
plt.xlabel('Normalised averaging window, \u03b6 = \u03b7/\u0394t')
plt.ylabel('L2 Error')
plt.title('Peddle plot, Gaussian ICs, \u0394t = {}, \u03b5 = {}'.format(dt,epsilon))

plt.show()

print(C_errs)