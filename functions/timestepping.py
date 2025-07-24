'''
Functions for timestepping.
'''

import numpy as np

def RK4_step(f,x,t,h,*pars):
    f0 = f(t,x,*pars)
    f1 = f(t + h/2, x + (h/2)*f0,*pars)
    f2 = f(t + h/2, x + (h/2)*f1,*pars)
    f3 = f(t + h, x + h*f2,*pars)
    #Linearly combine the four gradient evaulations:
    return x + (h/6)*(f0 + 2*f1 + 2*f2 + f3)

def RK4(f,init,t,h,*pars):
    #number of time steps to take:
    n = len(t) - 1 #Disregard last time
    
    #A list to append future solution values to
    x = [init]
    Ns = [init]
    for i in range(n):
        #print(t[i])
        x_new = RK4_step(f,x[i],t[i],h,*pars)
        x.append(x_new)
    return x

# Construct values for the averaging kernel
def kernel_vals(K, eta, alpha):
    s = np.arange(0.5,K)/K
    
    # alpha is the kernel decay rate 
    k_vals = np.exp(-1/(s*(1-s))/alpha)
    k_vals = k_vals/sum(k_vals) #Normalise the kernel values
    
    #Shift the s axis:
    s_r = (s-0.5)*eta
    return s_r, k_vals

# Matrix exponential operation on a vector
def etL_dot(E,V):
    c0 = E[0,0,:]*V[0,:] + E[0,1,:]*V[1,:] + E[0,2,:]*V[2,:]
    c1 = E[1,0,:]*V[0,:] + E[1,1,:]*V[1,:] + E[1,2,:]*V[2,:]
    c2 = E[2,0,:]*V[0,:] + E[2,1,:]*V[1,:] + E[2,2,:]*V[2,:]
    return np.array([c0,c1,c2])
