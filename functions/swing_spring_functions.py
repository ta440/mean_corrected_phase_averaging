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


####################
# For phase-averaging

def N(t,u,args):
    x,y,z = u 

    lamda, omega_R, rho = args
    
    return np.array([(1j*lamda/omega_R)*np.real(x)*np.real(z),
                     (1j*lamda/omega_R)*np.real(y)*np.real(z),
                     (1j*lamda/(2*rho*omega_R))*((np.real(x)**2) + (np.real(y)**2))])

def N_ave(t, v, args):

    mod_x,mod_y,mod_z = v
    omega_R, rho, lamda, kappa, kernel, K, e_sLs = args
    
    x = np.exp(-1j*omega_R*t)*mod_x
    y = np.exp(-1j*omega_R*t)*mod_y
    z = np.exp(-1j*omega_R*rho*t)*mod_z
    
    N_tot = 0
    for k in np.arange(0,K):
        e_s_p = e_sLs[k,:]
        e_s_m = e_sLs[int(K-k-1),:]
        
        u_s = e_s_m*np.array([x,y,z])
        
        N_cur = N(t,u_s, [lamda, omega_R, rho])
        N_tot += kernel[k]*(e_s_p*(N_cur))
    
    return np.array([np.exp(1j*omega_R*t)*N_tot[0],
                     np.exp(1j*omega_R*t)*N_tot[1],
                     np.exp(1j*omega_R*rho*t)*N_tot[2]])

##################################################
# For the mean correction

def classical_C(w, args):
    x,y,z = w 

    lamda,omega_R,rho = args
    
    return np.array([0,0,(1j*lamda/(4*rho*omega_R))*((x*np.conj(x) + y*np.conj(y)))])

def meancor_ave(t,w,args):
    grad_tot = np.zeros_like(w, dtype='complex128')

    lamda, omega_R, rho, K, s_vals, kernel = args

    C = classical_C(w, [lamda,omega_R,rho])

    L_inv = np.array([-1j/omega_R,-1j/omega_R,-1j/(omega_R*rho)])
    
    for j in np.arange(K):
        grad_cur = rhs_D(w,t+s_vals[j],[lamda, omega_R, rho, L_inv, C])
        grad_tot += kernel[j]*grad_cur
    
    return grad_tot
    
def rhs_D(w,t,args):
    w_x,w_y,w_z = w

    lamda, omega_R, rho, L_inv, C = args
    
    #Convert from u to w:
    x = np.exp(-1j*omega_R*t)*w_x + L_inv[0]*C[0]
    y = np.exp(-1j*omega_R*t)*w_y + L_inv[1]*C[1]
    z = np.exp(-1j*omega_R*rho*t)*w_z + L_inv[2]*C[2]
    
    u = np.array([x,y,z])
    
    N_cur = N(t,u, [lamda, omega_R, rho])
    
    #Compute N-C
    rhs = N_cur - C
    
    # return e^((t+s)/L) (N-C)
    return np.array([np.exp(1j*omega_R*t)*rhs[0],
                     np.exp(1j*omega_R*t)*rhs[1],
                     np.exp(1j*omega_R*rho*t)*rhs[2]])

