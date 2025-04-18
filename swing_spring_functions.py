'''
Functions for the swinging spring.

'''

import numpy as np

def swing_spring(t,u, args): 

    omega_R, rho, lamda = args
    
    x,y,z = u
    
    rhs_x = -1j*omega_R*x + (1j*lamda/omega_R)*np.real(x)*np.real(z)
    rhs_y = -1j*omega_R*y + (1j*lamda/omega_R)*np.real(y)*np.real(z)
    rhs_z = -1j*rho*omega_R*z + (1j*lamda/(2*rho*omega_R))*(np.real(x)**2 + np.real(y)**2)
    
    return np.array([rhs_x,rhs_y,rhs_z])


def swing_spring_modvar(t,v, args): 

    omega_R, rho, kappa = args
    
    x,y,z = v
    
    A = np.exp(-1j*omega_R*rho*t)
    B = np.exp(1j*omega_R*rho*t)
    C = np.exp(-1j*omega_R*(rho-2)*t)
    D = np.exp(1j*omega_R*(rho+2)*t)
    E = np.exp(1j*omega_R*(rho-2)*t)
    
    return np.array([1j*kappa*(A*x*z + B*x*np.conj(z) + C*np.conj(x)*z + D*np.conj(x)*np.conj(z)),
                     1j*kappa*(A*y*z + B*y*np.conj(z) + C*np.conj(y)*z + D*np.conj(y)*np.conj(z)),
                     (1j*kappa/(2*rho))*(E*(x**2 + y**2) + 2*B*(x*np.conj(x) + y*np.conj(y)) 
                                         + D*((np.conj(x))**2 + (np.conj(y))**2)) ])