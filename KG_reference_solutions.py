# -*- coding: utf-8 -*-
"""

For the KG-type equation.

Construct reference solutions using the modulation variable
method and a small time step. There is no phase-averaging here.

@author: timmo
"""

import numpy as np
from matplotlib import pyplot as plt
from functions.timestepping import *
from functions.KG_functions import *

#############################################################
def modvar_aved_non_d(t, v_vals):

    N_tot = np.zeros_like(v_vals)
    
    for j in np.arange(K):
        N_cur = k_gord_modvar_non_d(t+s_vals[j],v_vals)
        N_tot += kernel[j]*N_cur
    
    return N_tot

################################################################
# Setup parameters:

dt = 0.0001
TT = 20
t = np.arange(0,TT,dt) 

Nx = 32
Lx = 2*np.pi
dx = Lx/Nx
x = np.arange(0,Lx,dx)  

global k
k = np.fft.fftfreq(Nx,dx)*(2*np.pi)

#Create a de-aliasing array with top 2/3 of wavenumbers set to zero:
k_max = np.max(np.abs(k))

#Specify a hyperviscosity to apply:
global visc
visc = 1e-4

##################################################################
# Define the level of time-scale separation
global epsilon
epsilon = 0.01

# Calculate the linear dispersion relation:
global omega
omega = np.sqrt(1 + (k**2))

# Apply the time-scale separation for the linear term
omega = omega*(1/epsilon)

X,T = np.meshgrid(x,t)

###############################################################
# Initial conditions:

# Gaussian:
init_a = np.exp(-((x-(Lx/2))**2)/2) 
init_b = 0*x

# Now, we transform by a factor of omega:
init_hat = np.array([np.fft.fft(init_a)*omega,np.fft.fft(init_b)])

##################################################################
# Run the simulation!

# Modulation variable solution.
# This is in spectral space!
V = np.asarray(RK4(KG_modvar,init_hat,t,dt, [omega, visc, k]))

# Transform back to the standard variable
U = np.zeros_like(V)

for i in np.arange(0,len(t)):
    U[i,0,:] = np.cos(omega*t[i])*V[i,0,:] + np.sin(omega*t[i])*V[i,1,:]
    U[i,1,:] = -np.sin(omega*t[i])*V[i,0,:] + np.cos(omega*t[i])*V[i,1,:]
   
# Transform from spectral to physical space.
# Remember the factor of omega on the U solution.
U_a = np.real(np.fft.ifft(U[:,0,:]/omega))
U_b = np.real(np.fft.ifft(U[:,1,:]))

##############################################################
# Save the reference solution numpy array:
savename = f'reference_solutions/KG/KG_ref_sol_eps{epsilon}.npy'
np.save(savename, U_a)