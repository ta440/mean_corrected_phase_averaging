'''

Swinging spring.
Just look at standard phase averaging,
and a fixed time step and resonance factor.

At each time step, examine the error obtained
by a range of different averaging windows,
and select the window of lowest error to step
forwards with.

Look at how this optimal averaging window 
varies over time. I hope it gives some insight into
the dynamics, and thus varies with the choice of rho!

'''

import numpy as np
from matplotlib import pyplot as plt

######################################
# Import helper functions:
import sys
sys.path.append('../')

from functions.timestepping import *
from functions.swing_spring_functions import *

##################
# Specify the resonance factor
rho = 2.05

rho_no = int((rho-1.5)*100)

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
# Read in the reference solution:
an_sol = np.load(f'../reference_solutions/swing_spring/swing_spring_rho_range.npy')

an_sol = an_sol[rho_no,:,inds]

print(np.shape(an_sol))

u_x = an_sol[:,0]
u_y = an_sol[:,1]
u_z = an_sol[:,2]

print(np.shape(an_sol))
print(np.shape(u_x))

print(len(t))
print(np.shape(an_sol))

#########################
K = 31  # Number of points for averaging
zetas = np.linspace(0.0,10.0,101) #Length of averaging window
zetas = np.round(zetas,1)

print(zetas)


alpha = 4

# Compute the kernel weights in advance
e_sLs = np.zeros((len(zetas), K, 3), dtype='complex128')
kernel_weights = np.zeros((len(zetas), K))
                 
for j in np.arange(len(zetas)):
    zeta = zetas[j]
    eta = zeta*dt

    s_vals, kernel_weights[j,:] = kernel_vals(K, eta, alpha)

    for i in np.arange(0,K):
        e_sLs[j,i,0] = np.exp(1j*omega_R*s_vals[i])
        e_sLs[j,i,1] = np.exp(1j*omega_R*s_vals[i])
        e_sLs[j,i,2] = np.exp(1j*omega_Z*s_vals[i])

print('Kernel weights are precomputed')
##########################

C_best_errs = np.zeros(len(t)-1)
C_best_zetas = np.zeros(len(t)-1)

# Go through one timestep at a time.
# Hence, call RK4_step() rather than RK4()
C_v_cur = init

for i in np.arange(len(t)-1):
    print(t[i])
    # Loop over all averaging windows.

    C_errs = np.zeros(len(zetas))

    # Extract analytical solution at next time step.
    # The analytical solution does contain in the inital state
    x_next = u_x[i+1]
    y_next = u_y[i+1]
    z_next = u_z[i+1]

    for j in np.arange(len(zetas)):
        # Take a phase-averaged time step
        C_v_next = RK4_step(N_ave,C_v_cur,t[i],dt,[omega_R, rho, lamda, kernel_weights[j,:], K, e_sLs[j,:,:]])
        C_v_next = np.real(C_v_next)

        # Compute error. 
        # recall we have moved forward a time index!
        C_u_x = np.real(np.exp(-1j*omega_R*t[i+1])*C_v_next[0])
        C_u_y = np.real(np.exp(-1j*omega_R*t[i+1])*C_v_next[1])
        C_u_z = np.real(np.exp(-1j*omega_Z*t[i+1])*C_v_next[2])

        C_errs[j] = np.sqrt((C_u_x-x_next)**2 + (C_u_y-y_next)**2 + (C_u_z-z_next)**2)

    # Now, compute the best window
    C_best_errs[i] = np.min(C_errs)
    j_best = np.argmin(C_errs)
    C_best_zetas[i] = zetas[j_best]

    # Hack... check implementation
    j_best = 18

    # Now, perform the actual timestep using the best averaging window
    C_v_new = RK4_step(N_ave,C_v_cur,t[i],dt,[omega_R, rho, lamda, kernel_weights[j_best,:], K, e_sLs[j_best,:,:]])
    C_v_cur = C_v_new
        
###################################################
# Quote the total error
C_tot_err = np.sum(C_best_errs)/len(t)
print('\n')
print('Best standard phase averaging error is', C_tot_err)

# Plot the best window for each timestep
plt.figure()
plt.plot(t[1:], C_best_zetas)
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.xlabel(f"Time (s)", size=14)
plt.ylabel(f"Normalised averaging window, $\zeta$", size=14)

plt.figure()
plt.hist(C_best_zetas, zetas)
plt.title(r'Best averaging window frequency, $\rho = $' f'{rho}')
plt.xlabel(f"Normalised averaging window, $\zeta$", size=14)
plt.ylabel('Frequency')

plt.figure()
plt.plot(t[1:], C_best_errs)

plt.show()