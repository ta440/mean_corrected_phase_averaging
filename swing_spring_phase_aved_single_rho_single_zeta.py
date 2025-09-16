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
from functions.timestepping import *
from functions.swing_spring_functions import *
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

##################
# Specify the resonance factor
rho = 2

# Specify single zeta to use
zeta = 2.6

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
init_w = np.array([x0,y0,z0-(lamda/(4*rho*rho*omega_R*omega_R))*(x0*np.conj(x0)+y0*np.conj(y0))])

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
an_sol = np.load(f'reference_solutions/swing_spring/swing_spring_rho{rho}.npy')

an_sol = an_sol[:,:,inds]

print(np.shape(an_sol))

u_x = an_sol[0,0,:]
u_y = an_sol[0,1,:]
u_z = an_sol[0,2,:]

print(np.shape(u_x))

# Convert reference solution into modvar space:
v_x = np.real(np.exp(1j*omega_R*t)*u_x)
v_y = np.real(np.exp(1j*omega_R*t)*u_y)
v_z = np.real(np.exp(1j*omega_Z*t)*u_z)

print(v_x)

############################
# Solve with the unaveraged modulation variable:

B_v = np.asarray(RK4(swing_spring_modvar,init, t_b ,dt_b,[omega_R, rho, kappa]))

B_v_x = np.real(B_v[inds,0])
B_v_y = np.real(B_v[inds,1])
B_v_z = np.real(B_v[inds,2])

###################
K = 31  #Number of points for averaging
alpha = 4

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

C_v_x = np.real(C_v[:,0])
C_v_y = np.real(C_v[:,1])
C_v_z = np.real(C_v[:,2])

C_u = np.array([np.exp(-1j*omega_R*t)*C_v[:,0],
                np.exp(-1j*omega_R*t)*C_v[:,1],
                np.exp(-1j*omega_Z*t)*C_v[:,2] ])

C_u = np.matrix.transpose(C_u)

C_u_x = np.real(C_u[:,0])
C_u_y = np.real(C_u[:,1])
C_u_z = np.real(C_u[:,2])

C_err = np.sqrt((C_u_x-u_x)**2 + (C_u_y-u_y)**2 + (C_u_z-u_z)**2)
C_tot_err = np.sum(C_err)/len(t)

################################
# Mean corrected D_
D_w = np.asarray(RK4(meancor_ave,init,t,dt,[lamda, omega_R, rho, K, s_vals, kernel]))

D_w_x = np.real(D_w[:,0])
D_w_y = np.real(D_w[:,1])
D_w_z = np.real(D_w[:,2])
                    
D_u_x = np.exp(-1j*omega_R*t)*D_w[:,0]
D_u_y = np.exp(-1j*omega_R*t)*D_w[:,1]
D_u_z = np.exp(-1j*omega_Z*t)*D_w[:,2]+(lamda/(4*rho*rho*omega_R*omega_R))*((D_w[:,0]*np.conj(D_w[:,0]) + D_w[:,1]*np.conj(D_w[:,1])))

D_u_x = np.real(D_u_x)
D_u_y = np.real(D_u_y)
D_u_z = np.real(D_u_z)

D_err = np.sqrt((D_u_x-u_x)**2 + (D_u_y-u_y)**2 + (D_u_z-u_z)**2)
D_tot_err = np.sum(D_err)/len(t)

###################################################
#Compute total L2 errors:
print('\n')

print('The resonance factor is', rho)
print('The normalised averaging window size is', zeta)

#############
# Visualise the modulation variables!


fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex=True)
ax1.plot(t,B_v_x,c='k')
ax1.plot(t,C_v_x,c='b',linestyle='dashed')
ax1.plot(t,D_w_x,c='r',linestyle='dotted')
ax1.set_ylabel(f"x",size=14)
ax2.plot(t,B_v_y,c='k')
ax2.plot(t,C_v_y,c='b',linestyle='dashed')
ax2.plot(t,D_w_y,c='r',linestyle='dotted')
ax2.set_ylabel(f"y",size=14)
ax3.plot(t,B_v_z,c='k')
ax3.plot(t,C_v_z,c='b',linestyle='dashed')
ax3.plot(t,D_w_z,c='r',linestyle='dotted')
ax3.set_ylabel(f"z",size=14)
fig.supxlabel('Time',size=14)

custom_lines = [Line2D([0], [0], c='k', lw=4),
                Line2D([0], [0], c='b', lw=4,linestyle='dashed'),
                Line2D([0], [0], c='r' , lw=4,linestyle='dotted')]
fig.legend(custom_lines, ['Modulation variable', 'Standard phase-averaging', 'Mean corrected'], loc='lower center',bbox_to_anchor=(0.5, -0.4), fontsize=12)


plt.savefig(f'figures/swing_spring_modvar_space_rho{rho}_zeta{zeta}.png', bbox_inches="tight")

plt.show()