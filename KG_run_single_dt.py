# -*- coding: utf-8 -*-
"""

Tests for the KG-type equation.
Comparing:
1) standard phase-averaging
2) Mean corrected phase-averaging with an analytical mean correction
3) Mean corrected phase-averaging with a local mean correction

Pick the value of epsilon (the speed of linear oscillations)
and range of timesteps and averaging windows.
Then, store the results and make a Peddle Plot.

@author: timmo
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
from functions.timestepping import *
from functions.KG_functions import *

################################################
# Specify the level of time-scale separation
epsilon = 0.1

# Set a single large timestep size
dt = 1

###############################################
TT = 20
t = np.arange(0,TT,dt) 

Nx = 32
Lx = 2*np.pi
dx = Lx/Nx
x = np.arange(0,Lx,dx)  

global k
k = np.fft.fftfreq(Nx,dx)*(2*np.pi)

# Specify the hyperviscosity coefficient, if using
# Most cases are fine without hyperviscosity,
# so the default is 0.
visc = 0

#Calculate the linear dispersion relation:
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

# Tolerance for the mean correction initial condition iteration
Ctol = 1e-10

############################################################
# Base solution to compare errors relative to:

# Ref sol time step size
dt_b = 1e-4

a_ref = np.load(f'reference_solutions/KG/KG_ref_sol_eps{epsilon}.npy')

# Indices to compare with reference solution:
t_b = np.arange(0,TT+dt_b,dt_b)
inds = np.arange(0,len(t_b),int(np.rint(dt/dt_b)))
inds = np.asarray(inds)

if len(inds) != len(t):
    inds = inds[:-1]

# Parameters for the averaging kernel
K_min = 21 # Minimum number of sample points
ppp = 4 # averaging points per period
alpha = 4 # Kernel decay rate

# Range of local mean correction averaging windows to examine
#zetas = np.arange(0.05,2.00,0.05)
zetas = np.arange(1.0, 1.5, 0.05)
zetas = np.round(zetas,3)

# Solutions to save errors in 
C_errs = np.zeros(len(zetas))
D_an_errs = np.zeros(len(zetas))
D_num_errs = np.zeros(len(zetas))

a_analyt = a_ref[inds,:]    


###################################################################
# Loop over the phase-averaged methods with the different 
# averaging window sizes
# The mean corrected method uses eta=eta_C to
# reduce the size of the parameter space.

for j in np.arange(len(zetas)):
    zeta = zetas[j]
    eta = zeta*dt
    
    # Set the required number of kernel points.
    # 16.03 is the maximum oscillatory eigenvalue
    K_an = np.ceil(eta*ppp*16.03/(epsilon*2*np.pi))
    K = int(max(K_an,K_min))
    print('zeta={}, K = {}'.format(zeta,K))
    
    global kernel,s_vals
    s_vals, kernel = kernel_vals(K, eta, alpha) 

    ############################################################
    # 1) Standard phase-averaging
    
    vC = np.asarray(RK4(KG_modvar_phase_aved,init_hat,t,dt, [omega, visc, k, kernel, s_vals, K]))
    
    uC = np.zeros_like(vC)
    
    for i in np.arange(0,len(t)):
        uC[i,0,:] = np.cos(omega*t[i])*vC[i,0,:] + np.sin(omega*t[i])*vC[i,1,:]
        
    #Remember the shift factor of omega on the u solution
    aC = np.real(np.fft.ifft(uC[:,0,:]/omega))
    
    C_err = np.sum(np.abs(aC-a_analyt),axis=1)
    C_errs[j] = np.sum(C_err)/len(t)

    #####################################################
    # 2) Analytical mean correction
    
    Cs_an, uD_an, wD_an = classical_mean_corrected_KG_RK4(init_hat, t, dt, [omega, k, visc, kernel, s_vals, K, Nx, Ctol])
    uD_an = np.asarray(uD_an)
    
    aD_an = np.real(np.fft.ifft(uD_an[:,0,:]/omega))
    
    D_an_err = np.sum(np.abs(aD_an-a_analyt),axis=1)
    D_an_errs[j] = np.sum(D_an_err)/len(t)
    
    #####################################################
    # 3) Numerical mean correction
    
    Cs_num, uD_num, wD_num = local_mean_corrected_KG_RK4(init_hat, t, dt, [omega, k, visc, kernel, s_vals, K, Ctol])
    uD_num = np.asarray(uD_num)
    
    aD_num = np.real(np.fft.ifft(uD_num[:,0,:]/omega))
    
    D_num_err = np.sum(np.abs(aD_num-a_analyt),axis=1)
    D_num_errs[j] = np.sum(D_num_err)/len(t)


C_best = np.min(C_errs,axis=-1)
C_best_zeta = zetas[np.argmin(C_errs,axis=-1)]

D_an_best = np.min(D_an_errs,axis=-1)
D_an_best_zeta = zetas[np.argmin(D_an_errs,axis=-1)]

D_num_best = np.min(D_num_errs,axis=-1)
D_num_best_zeta = zetas[np.argmin(D_num_errs,axis=-1)]

###########################
# Quote minimum errors
print('\n')
print('Minimum error for standard phase-averaging is ', C_best)
print('Using zeta = ', C_best_zeta)
print('\n')

print('Minimum error for classically mean corrected phase-averaging is ', D_an_best)
print('Using zeta = ', D_an_best_zeta)
print('\n')

print('Minimum error for locally mean corrected phase-averaging is ', D_num_best)
print('Using zeta = ', D_num_best_zeta)
print('\n')

############################
# Make a Peddle plot:
plt.figure()
plt.plot(zetas, C_errs, label='Standard phase-averaging')
plt.plot(zetas, D_an_errs, label='Mean corrected (classical)')
plt.plot(zetas, D_num_errs, label='Mean corrected (local)')
plt.xlabel('Zeta')
plt.ylabel('Error')
plt.title(f'KG phase-averaging peddle plot, eps={epsilon}, dt = {dt}')
plt.legend()

plt.show()