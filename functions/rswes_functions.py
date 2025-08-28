'''
Functions for the 1d, f-plane, RSWEs.
'''

import numpy as np
import scipy

########################
# Different initial conditions:
def set_RSWE_initial_conditions(x_grid, IC):

  if IC == 'Gaussian':
      #Gaussian Dam Break::
      #Zero initial velocties over the grid
      u_0 = 0*x_grid
      v_0 = 0*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  elif IC == 'Gaussian_mean_shift':
      u_0 = 0*x_grid
      v_0 = 0*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2)  - scipy.special.erf(np.pi/np.sqrt(2))/np.sqrt(2*np.pi)
  elif IC == 'Gaussian_mean_shift2':
      u_0 = 0*x_grid
      v_0 = 0*x_grid
      c0 = scipy.special.erf(np.pi/np.sqrt(2))/np.sqrt(2*np.pi)
      phi_0 = (1/(1-c0))*(np.exp(-((x_grid-np.pi)**2)/2)  - c0)
  elif IC == 'Gaussian_meanflow_uv':
      #Gaussian Dam Break::
      u_0 = 0.05*x_grid
      v_0 = 0.05*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  elif IC == 'Terry':
      # The initial condition from Haut and Wingate (2014)
      u_0 = 0*x_grid
      v_0 = 0*x_grid

      c_0 = 0
      c_1 = 2
      phi_0 = c_1*(np.exp(-4*(x_grid-(np.pi/2))**2)*np.sin(3*(x_grid-(np.pi/2)) + \
              np.exp(-2*(x_grid-np.pi)**2)*np.sin(8*(x_grid-np.pi)))) + c_0
  elif IC == 'Gaussian_vel_grad':
      u_0 = np.sin(x_grid)
      v_0 = 0*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  elif IC == 'Gaussian_phi_cos_u':
      u_0 = np.cos(x_grid)
      v_0 = 0*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  elif IC == 'GG':
      # Gaussian profiles for u and phi
      u_0 = np.exp(-((x_grid-np.pi)**2)/2) 
      v_0 = 0*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  else:
      print('Not valid ICs given')
      
  return np.array([u_0,v_0,phi_0])

#######################################
# Matrix functions:
def B_mat(t, args):

    psi, k, eps = args
    #Time dependent parts of matrix exponential:
    t = t/eps
    return np.array([[np.cos(psi*t),np.sin(psi*t),np.sin(psi*t)],
                    [np.sin(psi*t),((k**2) + np.cos(psi*t)),(1- np.cos(psi*t))],
                    [np.sin(psi*t),(np.cos(psi*t)-1), 1 + (k**2)*np.cos(psi*t)]])

def etL_dot(E,V):
    c0 = E[0,0,:]*V[0,:] + E[0,1,:]*V[1,:] + E[0,2,:]*V[2,:]
    c1 = E[1,0,:]*V[0,:] + E[1,1,:]*V[1,:] + E[1,2,:]*V[2,:]
    c2 = E[2,0,:]*V[0,:] + E[2,1,:]*V[1,:] + E[2,2,:]*V[2,:]
    return np.array([c0,c1,c2])


#################################################
# Modulation variable method

def etL_dot_N(t, v_vals, args):

    A, psi, k, visc, eps = args

    Bp = B_mat(t, [psi, k, eps])
    Bm = B_mat(-t, [psi, k, eps])
    
    Ep = np.multiply(A,Bp)
    Em = np.multiply(A,Bm)
    
    #The v variables exist in Fourier Space
    #Compute u variables by the matrix exponential mapping
    u2,v2,phi2 = etL_dot(Em,v_vals)
    
    #Compute the derivatives and take IFFTs so we can create N:
    u = np.fft.ifft(u2)
    v = np.fft.ifft(v2)
    phi = np.fft.ifft(phi2)    
    
    u_x = np.real(np.fft.ifft(1j*k*u2))
    v_x = np.real(np.fft.ifft(1j*k*v2))
    phi_x = np.real(np.fft.ifft(1j*k*phi2))
    
    #Compute dissapative terms:
    u_px = np.fft.ifft((k**4)*u2)
    v_px = np.fft.ifft((k**4)*v2)
    phi_px = np.fft.ifft((k**4)*phi2)
    
    #Define N in the spatial domain so can perform multiplication
    N = np.array([-u*u_x,-u*v_x,-u_x*phi-u*phi_x])
    
    #Comptute dissapative terms:
    D = np.array([-visc*u_px,-visc*v_px,-visc*phi_px])
    
    #Transform N back into the Fourier Domain, now with dissapation applied
    N_hat = np.fft.fft(N+D)
    
    #Compute gradients
    grads = etL_dot(Ep,N_hat)
    
    return grads

def V_to_U(t, init, mod_spec, args):

    A, psi, k, eps = args

    #Converting the modulation variable solution into the usual one
    #Note, that we are working in Fourier space all the way here though.
    u_sol = [init]
    
    for i in np.arange(1,len(t)):
        Bm = B_mat(-t[i], [psi, k, eps])
        Em = A*Bm
        u_cur = etL_dot(Em,mod_spec[i,:,:])
        u_sol.append(u_cur)
    
    return np.array(u_sol) 

def etL_mats(t, args):

    A, psi, k, eps = args

    Bp = B_mat(t, [psi, k, eps])
    Bm = B_mat(-t, [psi, k, eps])
    
    Ep = np.multiply(A, Bp)
    Em = np.multiply(A, Bm)
    
    return Ep,Em

############################################
# Standard phase-averaging

def etL_dot_N_aved(t, v_vals, args):

    kernel, s_vals, K, A, psi, k, visc, eps = args

    N_tot = np.zeros_like(v_vals)
    
    for j in np.arange(K):
        N_cur = etL_dot_N(t+s_vals[j],v_vals, [A, psi, k, visc, eps])
        N_tot += kernel[j]*N_cur
    
    return N_tot

###############################################
# Mean corrected phase-averaging

def mean_cor_phase_aved_RK4(init, t, h, args):
    # RK4 timestepping routine for mean corrected
    # phase-averaged timestepping

    kernel, s_vals, K, kernel_C, s_vals_C, KC, A, psi, k, visc, eps, e_sL_ave = args
    
    C_eval_args = [kernel_C, s_vals_C, KC, A, psi, k, eps, visc]
    grad_ave_args = [kernel, s_vals, K, e_sL_ave, psi, k, eps, visc, A]

    u_sol = [init]

    C0 = C_eval(init, 0, C_eval_args)

    init_w = init - eps*L_inv_C(C0, psi, k)

    C_new = C_eval(init_w,0, C_eval_args)
    
    C_diff = np.sum(np.abs(C_new-C0))
    
    while  np.sum(C_diff) > 1e-10:
        #print(C_diff)
        
        C0 = C_new
        init_w = init - eps*L_inv_C(C0, psi, k)
        C_new = C_eval(init_w,0, C_eval_args)
        C_diff = np.sum(np.abs(C_new-C0))
    
    w_sol = [init_w]
    
    Cs = []

    w0 = init_w
    
    Cs.append(C_new)
    
    C0 = C_new

    for i in np.arange(0,len(t)-1):
        
        f0 = grad_ave(w0,t[i],C0, grad_ave_args)
    
        w1 = w0 + (h/2)*f0
        
        C1 = C_eval(w1, t[i]+(h/2), C_eval_args)
        f1 = grad_ave(w1, t[i]+(h/2), C1, grad_ave_args)
    
        w2 = w0 + (h/2)*f1
        
        C2 = C_eval(w2, t[i]+(h/2), C_eval_args)
        f2 = grad_ave(w2, t[i]+(h/2), C2, grad_ave_args)
        
        w3 = w0 + h*f2
        
        C3 = C_eval(w3, t[i]+h, C_eval_args)
        f3 = grad_ave(w3, t[i]+h, C3, grad_ave_args)
     
        w_new = w0 + (h/6)*(f0 + 2*f1 + 2*f2 + f3)
        w_sol.append(w_new)
            
        C_fin = C_eval(w_new, t[i]+h, C_eval_args)
        Cs.append(C_fin)
        
        Ep_fin, Em_fin = etL_mats(t[i]+h, [A, psi, k, eps])

        u_new = etL_dot(Em_fin, w_new) + eps*L_inv_C(C_fin, psi, k)
        u_sol.append(u_new)
        
        w0 = w_new
        
        C0 = C_fin
        
    return Cs,u_sol,w_sol

def C_eval(wc, t, args):
    # Compute the local mean correction, which 
    # is a phase-average of N(e^-tL W)

    kernel_C, s_vals_C, KC, A, psi, k, eps, visc = args

    C_vals = np.zeros_like(wc, dtype='complex128')
    
    for j in np.arange(KC):
        Ep, Em = etL_mats(t+s_vals_C[j], [A, psi, k, eps])
        C_cur = N_w_w(wc, [Ep, Em, k, visc])
        C_vals += kernel_C[j]*C_cur
    
    return C_vals  

def N_w_w(wc, args):
    # N(e^-tL W, e^-tL W), which is phase-averaged
    # to compute the local mean correction

    Ep, Em, k, visc = args
    uc = etL_dot(Em, wc) 

    u2, v2, phi2 = uc

    u = np.fft.ifft(u2)
    v = np.fft.ifft(v2)
    phi = np.fft.ifft(phi2)    
    
    u_x = np.real(np.fft.ifft(1j*k*u2))
    v_x = np.real(np.fft.ifft(1j*k*v2))
    phi_x = np.real(np.fft.ifft(1j*k*phi2))
    
    # Define N in the spatial domain so can perform multiplication
    N = np.array([-u*u_x, -u*v_x, -u_x*phi - u*phi_x])
    
    # Compute dissapative terms:
    u_px = np.fft.ifft((k**4)*u2)
    v_px = np.fft.ifft((k**4)*v2)
    phi_px = np.fft.ifft((k**4)*phi2)
    D = np.array([-visc*u_px,-visc*v_px,-visc*phi_px])
    
    N_hat = np.fft.fft(N + D)
    
    return N_hat
    
        
def rhs_func(wc, args):
    # Computes e^tL N for the mean corrected method,
    # so uses W

    psi, k, eps, C, visc, s_val, A, Ep, Em = args

    EpC, EmC = etL_mats(s_val, [A, psi, k, eps])

    uc = etL_dot(Em,wc) + etL_dot(EmC, eps*L_inv_C(C, psi, k))

    u2,v2,phi2 = uc

    u = np.fft.ifft(u2)
    v = np.fft.ifft(v2)
    phi = np.fft.ifft(phi2)    
    
    u_x = np.real(np.fft.ifft(1j*k*u2))
    v_x = np.real(np.fft.ifft(1j*k*v2))
    phi_x = np.real(np.fft.ifft(1j*k*phi2))
    
    #Compute dissapative terms:
    u_px = np.fft.ifft((k**4)*u2)
    v_px = np.fft.ifft((k**4)*v2)
    phi_px = np.fft.ifft((k**4)*phi2)
    
    #Define N in the spatial domain so can perform multiplication
    N = np.array([-u*u_x,-u*v_x,-u_x*phi-u*phi_x])
    
    #Compute dissapative terms:
    D = np.array([-visc*u_px,-visc*v_px,-visc*phi_px])
    
    N_hat = np.fft.fft(N + D)
    
    #Original implementation
    rhs = N_hat 
   
    #Compute gradients
    grads = etL_dot(Ep,rhs)
    
    return grads

def L_inv_C(C, psi, k):
    return np.array([(1/(psi**2))*C[1,:]+(-1j*k/(psi**2))*C[2,:],
                    (-1/(psi**2))*C[0,:],
                    (-1j*k/(psi**2))*C[0,:]])     

def L_Linv_C(C, psi, k):
    return np.array([C[0,:],
                    (1/(psi**2))*(C[1,:]-(1j*k)*C[2,:]),
                    (1/(psi**2))*((1j*k)*C[1,:]+(k**2)*C[2,:]) ])


def grad_ave(wc, t, C, args):
    # Compute the phase-averaged timestepping equation
    # So phase-ave

    kernel, s_vals, K, e_sL_ave, psi, k, eps, visc, A = args

    grad_tot = np.zeros_like(wc, dtype='complex128')
    
    for j in np.arange(K):
        Ep, Em = etL_mats(t+s_vals[j], [A, psi, k, eps])
        grad_cur = rhs_func(wc, [psi, k, eps, C, visc, s_vals[j], A, Ep, Em])
        grad_tot += kernel[j]*grad_cur
    
    aved_L_Linv_C = etL_dot(e_sL_ave, L_Linv_C(C, psi, k))
    
    Ep, Em = etL_mats(t, [A, psi, k, eps])
    
    rhs = grad_tot - etL_dot(Ep, aved_L_Linv_C)
    
    return rhs   