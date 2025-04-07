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

plt.close('all')


#############################################################
def modvar_aved_non_d(t, v_vals):

    N_tot = np.zeros_like(v_vals)
    
    for j in np.arange(K):
        N_cur = k_gord_modvar_non_d(t+s_vals[j],v_vals)
        N_tot += kernel[j]*N_cur
    
    return N_tot

def kernel_vals(K,eta):
    s = np.arange(0.5,K)/K
    
    #Adam's kernel has a decay rate of alpha=4
    k_vals = np.exp(-1/(s*(1-s))/4)
    k_vals = k_vals/sum(k_vals) #Normalise the kernel values
    
    #Shift the s axis:
    s_r = (s-0.5)*eta
    return s_r,k_vals

#####################################################################
def phase_imp_RK4_non_d_C_num(init,t,h):
    
    u_sol = [init]

    C0 = C_numerical(init,0)
    L_inv_C0 = np.zeros((2,len(k)),dtype='complex128')
    L_inv_C0[0,:] = (1/omega)*C0

    init_w = init - L_inv_C0

    w_sol = [init_w]
    
    w0 = init_w
    
    C_new = C_numerical(w0,0)
    
    # Iterate over the initial condition
    # with C_tol = 10^-10
    
    C_diff = np.sum(np.abs(C_new-C0))
    
    while  np.sum(C_diff) > Ctol:
        print('Numerical C_diff = ', C_diff)
        
        C0 = C_new
        L_inv_C0 = np.zeros((2,len(k)),dtype='complex128')
        L_inv_C0[0,:] = (1/omega)*C0
        init_w = init - L_inv_C0
        C_new = C_numerical(init_w,0)
        C_diff = np.sum(np.abs(C_new-C0))
        
        w0 = init_w
    
    Cs = [C0]
    C0 = C_new

    for i in np.arange(0,len(t)-1):
        #print(t[i])
        
        f0 = grad_ave(w0,t[i],C0)
        
        w1 = w0 + (h/2)*f0
        
        C1 = C_numerical(w1,t[i]+(h/2))
        f1 = grad_ave(w1,t[i]+(h/2),C1)
    
        w2 = w0 + (h/2)*f1
        
        C2 = C_numerical(w2,t[i]+(h/2))
        f2 = grad_ave(w2,t[i]+(h/2),C2)
        
        w3 = w0 + h*f2
        
        C3 = C_numerical(w3,t[i]+h)
        f3 = grad_ave(w3,t[i]+h,C3)
   
        w_new = w0 + (h/6)*(f0 + 2*f1 + 2*f2 + f3)
        w_sol.append(w_new)
            
        C_fin = C_numerical(w_new,t[i]+h)
        
        t_fin = t[i]+h
        
        a_new = np.cos(omega*t_fin)*w_new[0,:] + np.sin(omega*t_fin)*w_new[1,:] + (1/(omega))*C_fin
        b_new = -np.sin(omega*t_fin)*w_new[0,:] + np.cos(omega*t_fin)*w_new[1,:]
        
        u_new = np.array([a_new,b_new])
        
        u_sol.append(u_new)
        
        w0 = w_new
        
        Cs.append(C_fin)
        
        C0 = C_fin
        
    return Cs,u_sol,w_sol

#This function is for a numerical implementation of C:
def C_numerical(w_specs,t):
    C_vals = np.zeros_like(w_specs[1,:],dtype='complex128')
    
    for j in np.arange(K):
        ts = t+s_vals[j]
        C_cur = N_w_w(w_specs,ts)
        C_vals += kernel[j]*C_cur
    
    return C_vals   
    
def N_w_w(wc,t):
    
    c_hat = wc[0,:]
    d_hat = wc[1,:]
    
    a_hat  = c_hat*np.cos(omega*t) + d_hat*np.sin(omega*t)# + (1/(omega))*C
    
    a = np.fft.ifft(a_hat/omega) 
    
    return -np.fft.fft(a*a)

def grad_ave(wc,t,C):
    grad_tot = np.zeros_like(wc,dtype='complex128')
    
    for j in np.arange(K):
        grad_cur = N_minus_C(wc,t+s_vals[j],C)
        grad_tot += kernel[j]*grad_cur
    
    return grad_tot   

def N_minus_C(wc,t,C):
    #Evolve the modulation solution in Fourier space
    c_hat = wc[0,:]
    d_hat = wc[1,:]
    
    a_hat = c_hat*np.cos(omega*t) + d_hat*np.sin(omega*t) + (1/(omega))*C
    b_hat = -c_hat*np.sin(omega*t) + d_hat*np.cos(omega*t)
    a = np.fft.ifft(a_hat/omega) 
    
    a_square = np.fft.fft(a*a)
    
    #Compute N-C+Du
    rhs_a = -visc*(k**4)*a_hat
    rhs_b = -a_square -C - visc*(k**4)*b_hat
        
    e_tL_rhs = np.array([np.cos(omega*t)*rhs_a-np.sin(omega*t)*rhs_b,\
                          np.sin(omega*t)*rhs_a+np.cos(omega*t)*rhs_b])
    
    return e_tL_rhs 

##################################################################

def phase_imp_RK4_non_d_C_an(init,t,h):
    
    u_sol = [init]

    C0 = C_analytical(init,0)
    L_inv_C0 = np.zeros((2,len(k)),dtype='complex128')
    L_inv_C0[0,:] = (1/omega)*C0

    init_w = init - L_inv_C0

    w_sol = [init_w]
    
    Cs = [C0]
    
    w0 = init_w
    C_new = C_analytical(w0,0)
    
    # Iterate over the initial condition
    # with C_tol = 10^-10
    
    C_diff = np.sum(np.abs(C_new-C0))
    
    while  np.sum(C_diff) > Ctol:
        print('Analytical C_diff = ', C_diff)
        
        C0 = C_new
        L_inv_C0 = np.zeros((2,len(k)),dtype='complex128')
        L_inv_C0[0,:] = (1/omega)*C0
        
        init_w = init - L_inv_C0
        C_new = C_analytical(init_w,0)
        
        C_diff = np.sum(np.abs(C_new-C0))
        
    w0 = init_w    
        
    C0 = C_new
    Cs = [C0]

    for i in np.arange(0,len(t)-1):
        #print(t[i])
        
        f0 = grad_ave(w0,t[i],C0)
        
        w1 = w0 + (h/2)*f0
        
        C1 = C_analytical(w1,t[i]+(h/2))
        f1 = grad_ave(w1,t[i]+(h/2),C1)
    
        w2 = w0 + (h/2)*f1
        
        C2 = C_analytical(w2,t[i]+(h/2))
        f2 = grad_ave(w2,t[i]+(h/2),C2)
        
        w3 = w0 + h*f2
        
        C3 = C_analytical(w3,t[i]+h)
        f3 = grad_ave(w3,t[i]+h,C3)
   
        w_new = w0 + (h/6)*(f0 + 2*f1 + 2*f2 + f3)
        w_sol.append(w_new)
            
        C_fin = C_analytical(w_new,t[i]+h)
        
        t_fin = t[i]+h
        
        a_new = np.cos(omega*t_fin)*w_new[0,:] + np.sin(omega*t_fin)*w_new[1,:] + (1/(omega))*C_fin
        b_new = -np.sin(omega*t_fin)*w_new[0,:] + np.cos(omega*t_fin)*w_new[1,:]
        
        u_new = np.array([a_new,b_new])
        
        u_sol.append(u_new)
        
        w0 = w_new
        
        Cs.append(C_fin)
        
        C0 = C_fin
        
    return Cs,u_sol,w_sol

#This is the analytical evaluation of C 
def C_analytical(w_specs,t):
    
    c,d = w_specs
    
    cor1 = np.zeros(len(k),dtype='complex128')
    cor2 = np.zeros(len(k),dtype='complex128')
    
    for a in np.arange(len(k)):
        if a == 0:
            for i in np.arange(len(k)):
                #cor1[0] += c[i]*c[((-i)%len(k))]
                #cor2[0] += d[i]*d[((-i)%len(k))]
                cor1[0] += (1/omega[i])*c[i]*(1/omega[((-i)%len(k))])*c[((-i)%len(k))]
                cor2[0] += (1/omega[i])*d[i]*(1/omega[((-i)%len(k))])*d[((-i)%len(k))]
            
        elif (a%2) == 0:
            pos_ind = np.int(a/2)
            neg_ind = np.int((a+Nx)/2)
            
            for i in [pos_ind,neg_ind]:
                cor1[a] += (1/omega[i])*c[i]*(1/omega[i])*c[i]
                cor2[a] += (1/omega[i])*d[i]*(1/omega[i])*d[i]
    
    cor1 = cor1/(2*Nx)
    cor2 = cor2/(2*Nx)
    C_vals = cor1 + cor2
    C_vals = -C_vals
    
    return C_vals

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

a_analytical = np.load('KG_sol_TT_20_eps_0_1.npy')

# Parameters for the averaging kernel
K_min = 21
P = 4

# Define the range of step sizes to examine:
DTs = np.array([1,1.5,2,2.5,3])

# Range of local mean correction averaging windows to examine
zetas = np.arange(0.05,2.00,0.05)
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
