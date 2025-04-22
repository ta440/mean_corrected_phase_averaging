'''

Compute errors for the phase-averaged methods for a range of
averaging windows and resonance factors.


'''

import numpy as np
from timestepping import *
from swing_spring_functions import *

######################

# Specify the range of rho:
#rhos = np.arange(1.5,2.51,0.010)

rhos = np.linspace(1.5,2.5,11)

ext_name = 'larger_range'

###################
#Parameter values
g = np.pi*np.pi
l_o = 1.2
l = 1
k = 4*g
m = 1

rhos = np.round(rhos,2)

###################

#Calculate the natural frequencies of the normal modes::
omega_R = np.sqrt(g/l)

############################################
#Define the intitial conditions as three complex variables:
x0 = 0.04
y0 = 1j*0.03427/omega_R
z0 = 0.08

init = np.array([x0,y0,z0])

#####################

# Time set up for phase-averaging
TT = 200 
dt = 0.5
t = np.arange(0.0, TT, dt)

# Time from the reference solution:
dt_b = 0.01
t_b = np.arange(0.0, TT, dt_b)

inds = np.arange(0,len(t_b),int(np.rint(dt/dt_b)))
inds = np.asarray(inds)

#####################
# Read in the reference solution as a range
an_sol = np.load(f'reference_solutions/swing_spring/swing_spring_rho_range.npy')

an_sol = an_sol[:,:,inds]

#######################
K = 31  #Number of points for averaging
zetas = np.linspace(0.1,4.0,41) #Length of averaging window
zetas = np.round(zetas,1)
alpha = 4

C_best = np.zeros(len(rhos))
D_best = np.zeros(len(rhos))

C_zeta_best = np.zeros(len(rhos))
D_zeta_best = np.zeros(len(rhos))

for m in np.arange(len(rhos)):
    rho = rhos[m]
    print('\n')
    print('rho = ', rho)
    print('\n')

    rho_no = int((rho-1.5)*100)
    u_x = an_sol[rho_no,0,:]
    u_y = an_sol[rho_no,1,:]
    u_z = an_sol[rho_no,2,:]

    omega_Z = rho*omega_R
    lamda = (l_o*omega_Z*omega_Z)/(l*l)
    kappa = lamda/(4*omega_R)

    C_errs = np.zeros(len(zetas))
    D_errs = np.zeros(len(zetas))   

    for j in np.arange(len(zetas)):
        zeta = zetas[j]

        print(f'zeta = {zeta}')

        eta = zeta*dt

        s_vals, kernel = kernel_vals(K, eta, alpha)

        e_sLs = np.zeros((K,3),dtype='complex128')

        for i in np.arange(0,K):
            e_sLs[i,0] = np.exp(1j*omega_R*s_vals[i])
            e_sLs[i,1] = np.exp(1j*omega_R*s_vals[i])
            e_sLs[i,2] = np.exp(1j*omega_Z*s_vals[i])

        ##################
        # Standard phase-averaging C_:

        C_v = np.asarray(RK4(N_ave,init,t,dt,[omega_R, rho, lamda, kappa, kernel, K, e_sLs]))

        C_u = np.array([np.exp(-1j*omega_R*t)*C_v[:,0],
                        np.exp(-1j*omega_R*t)*C_v[:,1],
                        np.exp(-1j*omega_Z*t)*C_v[:,2] ])

        C_u = np.matrix.transpose(C_u)

        C_u_x = np.real(C_u[:,0])
        C_u_y = np.real(C_u[:,1])
        C_u_z = np.real(C_u[:,2])

        C_err = np.sqrt((C_u_x-u_x)**2 + (C_u_y-u_y)**2 + (C_u_z-u_z)**2)
        C_tot_err = np.sum(C_err)/len(t)

        C_errs[j] = C_tot_err

        ################################
        # Mean corrected D_
        D_w = np.asarray(RK4(meancor_ave,init,t,dt,[lamda, omega_R, rho, K, s_vals, kernel]))
                        
        D_u_x = np.exp(-1j*omega_R*t)*D_w[:,0]
        D_u_y = np.exp(-1j*omega_R*t)*D_w[:,1]
        D_u_z = np.exp(-1j*omega_Z*t)*D_w[:,2]+(lamda/(4*rho*rho*omega_R*omega_R))*((D_w[:,0]*np.conj(D_w[:,0]) + D_w[:,1]*np.conj(D_w[:,1])))

        D_u_x = np.real(D_u_x)
        D_u_y = np.real(D_u_y)
        D_u_z = np.real(D_u_z)

        D_err = np.sqrt((D_u_x-u_x)**2 + (D_u_y-u_y)**2 + (D_u_z-u_z)**2)
        D_tot_err = np.sum(D_err)/len(t)

        D_errs[j] = D_tot_err
        #######################
    
    # Record the bests:
    C_best[m] = np.min(C_errs)
    C_zeta_best[m] = zetas[np.argmin(C_errs)]

    D_best[m] = np.min(D_errs)
    D_zeta_best[m] = zetas[np.argmin(D_errs)]


# Save output:
filename = f'results/swinging_spring_results_{ext_name}'
delimiter = ', '

with open(filename, "w") as file:
    file.write('range of resonance factors is')
    file.write('\n')
    file.write(delimiter.join(map(str, rhos)))
    file.write('\n')
    file.write('Standard phase-averaging lowest errors are')
    file.write('\n')
    file.write(delimiter.join(map(str, C_best)))
    file.write('\n')
    file.write('Best standard phase-averaging zetas are')
    file.write('\n')
    file.write(delimiter.join(map(str, C_zeta_best)))
    file.write('\n')
    file.write('Mean corrected phase-averaging lowest errors are')
    file.write('\n')
    file.write(delimiter.join(map(str, D_best)))
    file.write('\n')
    file.write('Best mean corrected phase-averaging zetas are')
    file.write('\n')
    file.write(delimiter.join(map(str, D_zeta_best)))
    file.write('\n')