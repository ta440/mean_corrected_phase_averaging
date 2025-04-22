'''

Generates phase-averaged solutions to the swinging spring
for a single resonance factor.

Here we compare standard phase averaging
and mean corrected phase-averaging, using a classical
mean correction.

From convention in earlier studies (see Andrews 2024 PhD thesis)
standard phase-averaging is algorithm C
and mean corrected phase-averaging is algorithm D

'''

import numpy as np
from timestepping import *
from swing_spring_functions import *
from matplotlib import pyplot as plt

##################
# Specify the resonance factor
rho = 1.88

###################
#Parameter values
g = np.pi*np.pi
l_o = 1.2
l = 1
k = 4*g
m = 1

###################
# Frequencies:

omega_R = np.sqrt(g/l)
omega_Z = rho*omega_R

lamda = (l_o*omega_Z*omega_Z)/(l*l)
kappa = lamda/(4*omega_R)

############################################
#Define the intitial conditions as three complex variables:
x0 = 0.04
y0 = 1j*0.03427/omega_R
z0 = 0.08

init = np.array([x0,y0,z0])

# Initial condition for the mean correction:
#C0 = classical_C(init, [lamda,omega_R,rho])
#L_inv = np.array([-1j/omega_R,-1j/omega_R,-1j/(omega_R*rho)])
#init_w = init - L_inv*C0

# Or hard-coded:
init_w = np.array([x0,y0,z0-(lamda/(4*rho*rho*omega_R*omega_R))*(x0*np.conj(x0)+y0*np.conj(y0))])

#####################

# Time set up for phase-averaging
TT = 200 
dt = 0.5
t = np.arange(0.0, TT, dt)

# Time from the reference solution:
TT = 200 #Total time of the simulation
dt_b = 0.01
t_b = np.arange(0.0, TT, dt_b)

inds = np.arange(0,len(t_b),int(np.rint(dt/dt_b)))
inds = np.asarray(inds)

#####################
# Read in the reference solution:
an_sol = np.load(f'reference_solutions/swing_spring/swing_spring_rho{rho}.npy')

an_sol = an_sol[:,:,inds]

u_x = an_sol[:,0]
u_y = an_sol[:,1]
u_z = an_sol[:,2]

###################
K = 31  #Number of points for averaging
zetas = np.linspace(0.1,4.0,40) #Length of averaging window
zetas = np.round(zetas,1)
alpha = 4

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

###################################################
#Compute total L2 errors:
print('\n')

print('The resonance factor is', rho)
print('The averaging window size is', eta)
    
print('\n')
print('Best standard phase averaging error is', np.min(C_errs))
print('Using an averaging window of', zetas[np.argmin(C_errs)])

print('Best mean corrected phase averaging error is', np.min(D_errs))
print('Using an averaging window of', zetas[np.argmin(D_errs)])

#############
# Make a Peddle plot!

plt.figure()
plt.plot(zetas, C_errs, label='Standard phase-averaging', c='k')
plt.plot(zetas, D_errs, label='Mean corrected phase-averaging', c='r', linestyle='dashed')
plt.legend()
plt.title(f'Compare phase-averaged methods in the swinging spring, rho={rho}')
plt.xlabel('Zeta, normalised averaging window')
plt.ylabel('Solution error')
plt.show()