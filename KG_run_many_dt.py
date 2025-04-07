# -*- coding: utf-8 -*-
"""

Tests for the KG-type equation.
Comparing:
1) standard phase-averaging
2) Mean corrected phase-averaging with an analytical mean correction
3) Mean corrected phase-averaging with a local mean correction

@author: timmo
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
from timestepping import *
from KG_functions import *

################################################################

TT = 20

Nx = 32
Lx = 2*np.pi
dx = Lx/Nx
x = np.arange(0,Lx,dx)  

global k
k = np.fft.fftfreq(Nx,dx)*(2*np.pi)

#Specify a hyperviscosity to apply:
global visc
#visc = 1e-4
visc = 0

##################################################################
# Specify a level of time-scale separation
global epsilon
epsilon = 0.1

#Calculate the linear dispersion relation:
global omega
omega = np.sqrt(1 + (k**2))

#Apply the time-scale separation for the linear term
omega = omega*(1/epsilon)

###############################################################
# Initial conditions:

# Gaussian:
init_a = np.exp(-((x-(Lx/2))**2)/2) 
init_b = 0*x

# Now, we transform by a factor of omega:
init_hat = np.array([np.fft.fft(init_a)*omega,np.fft.fft(init_b)])

# Tolerance for iterating the initial condition:
Ctol = 1e-10

############################################################
# Base solution to compare errors relative to:

# Ref sol time step size
dt_b = 1e-4

a_analytical = np.load(f'reference_solutions/KG/KG_ref_sol_eps{epsilon}.npy')

# Parameters for the averaging kernel
K_min = 21
P = 4

# Define the range of step sizes to examine:
DTs = np.array([1]) #np.array([1,1.5,2,2.5,3])

# Range of local mean correction averaging windows to examine
zetas = np.aarray([0.5, 1]) #np.arange(0.05,2.00,0.05)
zetas = np.round(zetas,3)

# Solutions to save errors in 
C_errs = np.zeros((len(DTs),len(zetas)))
D_an_errs = np.zeros((len(DTs),len(zetas)))
D_num_errs = np.zeros((len(DTs),len(zetas)))

for q in np.arange(len(DTs)):
    dt = DTs[q]
    print('dt = {}'.format(dt))
    

    t = np.arange(0,TT,dt) 

    ###########################################################
    #Base error for comparison:
    t_b = np.arange(0,TT+dt_b,dt_b)
    inds = np.arange(0,len(t_b),np.int(np.rint(dt/dt_b)))
    inds = np.asarray(inds)
    
    if len(inds) != len(t):
        inds = inds[:-1]
        
    a_analyt = a_analytical[inds,:]    
    ###################################################################
    #Now, loop over kernel methods:
    for j in np.arange(len(zetas)):
        zeta = zetas[j]
        eta = zeta*dt
        
        #Set the required number of kernel points.
        K_an = np.ceil(eta*P*16.03/(epsilon*2*np.pi))
        K = np.int(max(K_an,K_min))
        print('zeta={}, K = {}'.format(zeta,K))
        
        global kernel,s_vals
        s_vals, kernel = kernel_vals(K,eta) 

        ############################################################
        # 1) Standard phase-averaging
        
        vC = np.asarray(RK4(modvar_aved_non_d,init_hat,t,dt))
        
        uC = np.zeros_like(vC)
        
        for i in np.arange(0,len(t)):
            uC[i,0,:] = np.cos(omega*t[i])*vC[i,0,:] + np.sin(omega*t[i])*vC[i,1,:]
            
        #Remember the shift factor of omega on the u solution
        aC = np.real(np.fft.ifft(uC[:,0,:]/omega))
        
        C_err = np.sum(np.abs(aC-a_analyt),axis=1)
        C_errs[q,j] = np.sum(C_err)/len(t)

        #####################################################
        # 2) Analytical mean correction
        
        Cs_an, uD_an, wD_an = phase_imp_RK4_non_d_C_an(init_hat, t, dt)
        uD_an = np.asarray(uD_an)
        
        aD_an = np.real(np.fft.ifft(uD_an[:,0,:]/omega))
        
        D_an_err = np.sum(np.abs(aD_an-a_analyt),axis=1)
        D_an_errs[q,j] = np.sum(D_an_err)/len(t)
        
        #####################################################
        # 3) Numerical mean correction
        
        Cs_num, uD_num, wD_num = phase_imp_RK4_non_d_C_num(init_hat, t, dt)
        uD_num = np.asarray(uD_num)
        
        aD_num = np.real(np.fft.ifft(uD_num[:,0,:]/omega))
        
        D_num_err = np.sum(np.abs(aD_num-a_analyt),axis=1)
        D_num_errs[q,j] = np.sum(D_num_err)/len(t)

C_best = np.min(C_errs,axis=-1)
C_best_zeta = zetas[np.argmin(C_errs,axis=-1)]

D_an_best = np.min(D_an_errs,axis=-1)
D_an_best_zeta = zetas[np.argmin(D_an_errs,axis=-1)]

D_num_best = np.min(D_num_errs,axis=-1)
D_num_best_zeta = zetas[np.argmin(D_num_errs,axis=-1)]

# Save the error results in a text file.
