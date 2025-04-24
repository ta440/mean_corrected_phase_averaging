'''
Functions for the KG-type equation.
'''

import numpy as np

##################################
# Standard modulation variable
def KG_modvar(t,v, args):
    omega, visc, k = args
    # Evolve the modulation solution in Fourier space
    c_hat = v[0,:]
    d_hat = v[1,:]
    
    a_hat  = c_hat*np.cos(omega*t) + d_hat*np.sin(omega*t)
    a = np.real(np.fft.ifft(a_hat/omega))
    
    a_square = np.fft.fft(a*a)
    
    e_tL_N = np.array([np.sin(omega*t)*a_square,-np.cos(omega*t)*a_square])
    
    #Apply hyperviscosity straight to the modulation variables:
    Dv = np.array([-visc*(k**4)*c_hat,-visc*(k**4)*d_hat])
    
    return e_tL_N + Dv

###################################
# Locally mean corrected

def local_mean_corrected_KG_RK4(init, t, h):
    
    u_sol = [init]

    C0 = C_numerical(init,0)
    L_inv_C0 = np.zeros((2,len(k)),dtype='complex128')
    L_inv_C0[0,:] = (1/omega)*C0

    init_w = init - L_inv_C0

    w_sol = [init_w]
    
    w0 = init_w
    
    C_new = C_local(w0,0)
    
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

# Compute the local mean correction
def C_local(w_specs,t):
    C_vals = np.zeros_like(w_specs[1,:],dtype='complex128')
    
    for j in np.arange(K):
        ts = t+s_vals[j]
        C_cur = N_w_w(w_specs,ts)
        C_vals += kernel[j]*C_cur
    
    return C_vals   
    
# Compute the nonlinearity in terms of the modulation variable
def N_w_w(wc,t, omega):
    
    c_hat = wc[0,:]
    d_hat = wc[1,:]
    
    a_hat  = c_hat*np.cos(omega*t) + d_hat*np.sin(omega*t)
    
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

# Classically mean corrected

def classical_mean_corrected_KG_RK4(init,t,h):
    
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

# The classical mean correction
def C_classical(w_specs,t):
    
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