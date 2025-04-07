'''
Functions for the KG-type equation.
'''

import numpy as np

# KG-type equation
def KG_modvar(t,v):
    #Evolve the modulation solution in Fourier space
    c_hat = v[0,:]
    d_hat = v[1,:]
    
    a_hat  = c_hat*np.cos(omega*t) + d_hat*np.sin(omega*t)
    a = np.real(np.fft.ifft(a_hat/omega))
    
    a_square = np.fft.fft(a*a)
    
    e_tL_N = np.array([np.sin(omega*t)*a_square,-np.cos(omega*t)*a_square])
    
    #Apply hyperviscosity straight to the modulation variables:
    Dv = np.array([-visc*(k**4)*c_hat,-visc*(k**4)*d_hat])
    
    return e_tL_N + Dv