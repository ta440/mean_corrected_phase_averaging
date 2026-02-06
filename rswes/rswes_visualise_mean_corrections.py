# -*- coding: utf-8 -*-
"""

Tests for the 1d RSWEs.

Run the mean corrected method with different 
eta_C, for the fixed zeta^*.

Plot these together to examine the differences
in the mean correction.

@author: timmo
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from functions.timestepping import *
from functions.rswes_functions import *

##################################################################
# Define the level of time-scale separation.
# Values of 0.5,0.1,0.05,0.01, 0.001 are used in the paper.
epsilon = 0.1

# Set a single large timestep size. 
# Values of 0.05, 0.1, ... 0.35 are used in the paper.
dt = 0.1

ic_type = 'Gaussian_mean_shift'

longer_time = False

TT = 10

# Phase-averaging window
zeta = 0.95

# Mean correction windows to examine:
# For the paper, use 0.7, 1.4, 2.1
etas_C = [0.7,1.4,2.1]
#etas_C=[2,3,4]

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

# Tolerance for the mean correction initial condition iteration
Ctol = 1e-10

############################################################
# Base solution to compare errors relative to:

# Ref sol time step size
dt_b = 1e-2

if longer_time:
    U_analyt = np.load(f'reference_solutions/rswes/rswes_{ic_type}_eps{epsilon}_TT{TT}.npy')
else:
    U_analyt = np.load(f'reference_solutions/rswes/rswes_{ic_type}_eps{epsilon}.npy')

# Indices to compare with reference solution:
t_b = np.arange(0,TT+dt_b,dt_b)
inds = np.arange(0,len(t_b),int(np.rint(dt/dt_b)))
inds = np.asarray(inds)

if len(inds) != len(t):
    inds = inds[:-1]

K_min = 21 # Minimum number of sample points
ppp = 4 # averaging points per period
alpha = 4 # Kernel decay rate

eta = np.round(zeta*dt,3)
K_an = np.ceil(ppp*eta*np.max(psi)/(epsilon*2*np.pi))
K = int(max(K_an,K_min))
KC_min = 21
s_vals, kernel = kernel_vals(K, eta, alpha)  

U_analyt = U_analyt[:, inds, :]   

u_analyt = U_analyt[0,:]
v_analyt = U_analyt[1,:]
phi_analyt = U_analyt[2,:]

##################################################
# Mean correction:

C_vals_rec = np.zeros((3,len(t),3,len(x)))
errs = [0,0,0]

for m in np.arange(len(etas_C)):
    
    eta_C = etas_C[m]
    eta_C = np.round(eta_C,3)
    
    #Set the required number of kernel points.
    KC_an = np.ceil(ppp*eta_C*np.max(psi)/(epsilon*2*np.pi))
    KC = int(max(KC_an,KC_min))
    
    #If KC is odd, make it even
    if (KC%2):
        KC = KC + 1

    print(f'eta_C={eta_C}, KC={KC} \n')

    #Construct the kernel
    global kernel_C, s_vals_C
    s_vals_C, kernel_C = kernel_vals(KC, eta_C, alpha)    

    #################################################
    # Hmm, double check this .... ... .. .
    e_sL_ave = np.zeros((3,3,Nx),dtype='complex128')
    eC_sL_ave = np.zeros((3,3,Nx),dtype='complex128')
    
    for q in np.arange(K):
        s = s_vals[q]
        B = B_mat(s, [psi, k, epsilon])
        e_sL_ave += kernel[q]*np.multiply(A,B)
        
    for q in np.arange(KC):
        s = s_vals_C[q]
        B = B_mat(s, [psi, k, epsilon])
        eC_sL_ave += kernel_C[q]*np.multiply(A,B)

    #############################################

    meancor_args = [kernel, s_vals, K, kernel_C, s_vals_C, KC, A, psi, k, visc, epsilon, e_sL_ave]
    C_vals, U_specs, W_specs = mean_cor_phase_aved_RK4(init_spec,t,dt, meancor_args)

    C_vals_rec[m,:,:,:] = np.real(np.fft.ifft(C_vals))

    U_specs = np.asarray(U_specs)
    
    U_u = np.real(np.fft.ifft(U_specs[:,0,:]))
    U_v = np.real(np.fft.ifft(U_specs[:,1,:]))
    U_phi = np.real(np.fft.ifft(U_specs[:,2,:]))

    #Compute errors relative to these analytical expressions:
    u_err = np.sum(np.abs(U_u-u_analyt),axis=1)
    v_err = np.sum(np.abs(U_v-v_analyt),axis=1)
    phi_err = np.sum(np.abs(U_phi-phi_analyt),axis=1)
    
    #Sum of L2 errors:
    tot_err = np.sum(np.sqrt(u_err**2 + v_err**2 + phi_err**2))/len(t)

    errs[m] = tot_err

# Plot the geopotential solutions to see if there is a phase shift 
fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True)
fig1 = ax1.contourf(T,X,C_vals_rec[0,:,0,:])
ax1.set_ylabel('$u')
plt.colorbar(fig1,ax=ax1)
fig2 = ax2.contourf(T,X,C_vals_rec[0,:,1,:])
ax2.set_ylabel('$v$')
plt.colorbar(fig2,ax=ax2)
fig3 = ax3.contourf(T,X,C_vals_rec[0,:,2,:])
ax3.set_ylabel('$\phi')
ax3.set_xlabel('Time')
plt.colorbar(fig3,ax=ax3)
plt.suptitle(f'$\eta_C={etas_C[0]}')

fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True)
fig1 = ax1.contourf(T,X,C_vals_rec[1,:,0,:])
ax1.set_ylabel('$u')
plt.colorbar(fig1,ax=ax1)
fig2 = ax2.contourf(T,X,C_vals_rec[1,:,1,:])
ax2.set_ylabel('$v$')
plt.colorbar(fig2,ax=ax2)
fig3 = ax3.contourf(T,X,C_vals_rec[1,:,2,:])
ax3.set_ylabel('$\phi')
ax3.set_xlabel('Time')
plt.colorbar(fig3,ax=ax3)
plt.suptitle(f'$\eta_C={etas_C[1]}')

fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True)
fig1 = ax1.contourf(T,X,C_vals_rec[2,:,0,:])
ax1.set_ylabel('$u')
plt.colorbar(fig1,ax=ax1)
fig2 = ax2.contourf(T,X,C_vals_rec[2,:,1,:])
ax2.set_ylabel('$v$')
plt.colorbar(fig2,ax=ax2)
fig3 = ax3.contourf(T,X,C_vals_rec[2,:,2,:])
ax3.set_ylabel('$\phi')
ax3.set_xlabel('Time')
plt.colorbar(fig3,ax=ax3)
plt.suptitle(f'$\eta_C={etas_C[2]}')


# Make a mega plot!

plt.rc('xtick', labelsize=12)  
plt.rc('ytick', labelsize=12) 

u_max = 0.04
v_max = 0.003
phi_max = 0.004

u_conts = np.linspace(-u_max,u_max,9)
v_conts = np.linspace(-v_max,v_max,9)
phi_conts = np.linspace(-phi_max,phi_max,9)

u_cont_ticks = np.linspace(-u_max,u_max,5)
v_cont_ticks = np.linspace(-v_max,v_max,5)
phi_cont_ticks = np.linspace(-phi_max,phi_max,5)

cmap = matplotlib.cm.viridis
#cmap.set_under('k')
#cmap.set_over('white')

fig, axs = plt.subplots(3,3, figsize=(10,6), sharex=True, sharey=True, constrained_layout=True)
(ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9) = axs
fig1 = ax1.contourf(T,X,C_vals_rec[0,:,0,:], levels=u_conts, cmap=cmap, extend='both')
fig2 = ax2.contourf(T,X,C_vals_rec[1,:,0,:], levels=u_conts, cmap=cmap, extend='both')
fig3 = ax3.contourf(T,X,C_vals_rec[2,:,0,:], levels=u_conts, cmap=cmap, extend='both')
fig4 = ax4.contourf(T,X,C_vals_rec[0,:,1,:], levels=v_conts, cmap=cmap, extend='both')
fig5 = ax5.contourf(T,X,C_vals_rec[1,:,1,:], levels=v_conts, cmap=cmap, extend='both')
fig6 = ax6.contourf(T,X,C_vals_rec[2,:,1,:], levels=v_conts, cmap=cmap, extend='both')
fig7 = ax7.contourf(T,X,C_vals_rec[0,:,2,:], levels=phi_conts, cmap=cmap, extend='both')
fig8 = ax8.contourf(T,X,C_vals_rec[1,:,2,:], levels=phi_conts, cmap=cmap, extend='both')
fig9 = ax9.contourf(T,X,C_vals_rec[2,:,2,:], levels=phi_conts, cmap=cmap, extend='both')

ax1.set_ylabel('$C_u$', size=14, rotation=0, labelpad=15)
ax4.set_ylabel('$C_v$', size=14, rotation=0, labelpad=15)
ax7.set_ylabel('$C_{\phi}$', size=14, rotation=0, labelpad=15)

ax1.set_title(f'$\eta_C={etas_C[0]}$ \n Total error is {errs[0]:.4f}', size=14)
ax2.set_title(f'$\eta_C={etas_C[1]}$ \n Total error is {errs[1]:.4f}', size=14)
ax3.set_title(f'$\eta_C={etas_C[2]}$ \n Total error is {errs[2]:.4f}', size=14)

cbar1 = fig.colorbar(fig3, ax=axs[0, :], pad=0.02, ticks=u_cont_ticks)
plt.colorbar(fig6, ax=axs[1, :], pad=0.02, ticks=v_cont_ticks)
plt.colorbar(fig9, ax=axs[2, :], pad=0.02, ticks=phi_cont_ticks)

#plt.colorbar(fig3,ax=ax3)
#plt.colorbar(fig6,ax=ax6)
#plt.colorbar(fig9,ax=ax9)

fig.supylabel(r"$x \in [0, 2 \pi)$", size=14)
fig.supxlabel('Time', size=14)

#plt.yticks(size=12)
#plt.xticks(size=12)
#plt.tick_params(axis='both', labelsize=12)
ax1.set_xticks([0,2,4,6,8])

plt.subplots_adjust(wspace=0.1, hspace=0.2)

#savename = f'figures/mean_cor_visual_eps{epsilon}_dt{dt}.png'
#print('saving figure to ', savename)
#plt.savefig(savename, bbox_inches="tight")

fig, ax1 = plt.subplots(1,1, figsize=(10,4.5))
fig = ax1.contourf(T, X, C_vals_rec[0,:,0,:])
ax1.set_ylabel('$x \in [0, 2 \pi)$', size=14)
ax1.set_xlabel('Time', size=14)
plt.colorbar(fig)

#######################
# What is dC/dt in each instance?
C_diffs = np.zeros((3,len(t)-1,3,len(x)))

for i in np.arange(len(t)-1):
    C_diffs[:,i,:,:] = C_vals_rec[:,i+1,:,:] - C_vals_rec[:,i,:,:]

plt.figure()
plt.plot(np.sum(np.abs(C_diffs[0,:,0,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[0]}')
plt.plot(np.sum(np.abs(C_diffs[1,:,0,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[1]}')
plt.plot(np.sum(np.abs(C_diffs[2,:,0,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[2]}')
plt.title('C u')
plt.legend()

plt.figure()
plt.plot(np.sum(np.abs(C_diffs[0,:,1,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[0]}')
plt.plot(np.sum(np.abs(C_diffs[1,:,1,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[1]}')
plt.plot(np.sum(np.abs(C_diffs[2,:,1,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[2]}')
plt.title('C v')
plt.legend()

plt.figure()
plt.plot(np.sum(np.abs(C_diffs[0,:,2,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[0]}')
plt.plot(np.sum(np.abs(C_diffs[1,:,2,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[1]}')
plt.plot(np.sum(np.abs(C_diffs[2,:,2,:]),axis=-1), label=r"$\eta_C$ = "+f'{etas_C[2]}')
plt.title('C phi')
plt.legend()

plt.show()


# Figure for the graphical abstract: