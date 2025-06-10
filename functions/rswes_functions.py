'''
Functions for the 1d, f-plane, RSWEs.
'''

import numpy as np

########################
# Different initial conditions:
def set_RSWE_initial_conditions(x_grid, IC_type):
  #Define initial conditions:

  IC = 'Gaussian'

  if IC == 'Gaussian':
      #Gaussian Dam Break::
      #Zero initial velocties over the grid
      u_0 = 0*x_grid
      v_0 = 0*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  elif IC == 'Gaussian_meanflow_uv':
      #Gaussian Dam Break::
      u_0 = 0.05*x_grid
      v_0 = 0.05*x_grid
      phi_0 = np.exp(-((x_grid-np.pi)**2)/2) 
  elif IC == 'Terry':
      #Zero initial velocties over the grid
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
# Phase-averaged

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

######################################
# Phase-averaged

def etL_dot_N_aved(t, v_vals):

    N_tot = np.zeros_like(v_vals)
    
    for j in np.arange(K):
        N_cur = etL_dot_N(t+s_vals[j],v_vals)
        N_tot += kernel[j]*N_cur
    
    return N_tot