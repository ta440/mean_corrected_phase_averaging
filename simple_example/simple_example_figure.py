'''
A simple example to show the mean correction.
Use timesteps of dt = 0.25.
The nonlinearity is treated as constant 
over each of these time intervals.
'''

import numpy as np
from os.path import abspath, dirname
import matplotlib
from matplotlib import pyplot as plt
import scipy

############################

# Import helper functions:
import sys
sys.path.append('../')

from functions.timestepping import *
from functions.rswes_functions import *

#############################
def phase_aved(v_sol, t, epsilon, chi, c_val):
    return chi*np.exp(1j*omega*t/epsilon)*c_val

##########################
# Construct an analytical solution
epsilon = 0.143
omega = 2*np.pi
TT = 1
u0 = 1 +1j

dt_ref = 0.001
t_ref = np.arange(0, TT + dt_ref, dt_ref)

c_val = 5

analyt_sol = (1j*c_val*epsilon/omega)*(np.exp(-1j*omega*t_ref/epsilon) - 1) + np.exp(-1j*omega*t_ref/epsilon)*u0
modvar_sol = u0 - (1j*c_val*epsilon/omega)*(np.exp(-1j*omega*t_ref/epsilon) - 1)

# Perform timestepping
dt = 0.2
t = np.arange(0, TT+dt, dt)
inds = np.arange(0,len(t_ref)+1,int(np.rint(dt/dt_ref)))
print(inds)

# Bonus, compute chi given a value of eta
zeta = 1

eta = zeta*dt
alpha = 4
K_min = 21
ppp = 4

K_an = np.ceil(ppp*eta*omega/(epsilon*2*np.pi))
K = int(max(K_an,K_min))

print(K)

s_vals, kernel = kernel_vals(K, eta, alpha)    

def L(t, epsilon):
    return np.exp(1j*omega*t/epsilon)

chi = 0

for k in np.arange(K):
    chi += kernel[k]*L(s_vals[k],epsilon)

chi = np.real(chi)


print(f'chi is {chi}')

v = np.zeros((len(t)), dtype='complex128')
v_bar = np.zeros((len(t)), dtype='complex128')
w_bar = np.zeros((len(t)), dtype='complex128')

v[0] = u0
v_bar[0] = u0
w_bar[0] = u0 + epsilon*1j*c_val/omega

for i in np.arange(len(t)-1):
    v[i+1] = v[i] + dt*phase_aved(v[i], t[i], epsilon, 1.0, c_val)
    v_bar[i+1] = v_bar[i] + dt*phase_aved(v_bar[i], t[i], epsilon, chi, c_val)
    w_bar[i+1] = w_bar[i]

discrete_modvar_sol = np.exp(-1j*omega*t/epsilon)*v
phase_aved_sol = np.exp(-1j*omega*t/epsilon)*v_bar

meancor_sol = np.exp(-1j*omega*t/epsilon)*w_bar - epsilon*1j*c_val/omega
w_sol = np.ones(len(t_ref))*w_bar[0]

# Print errors:
print('Phase-averaging error is:', np.sum(np.abs(phase_aved_sol - analyt_sol[inds]))/(len(t)-1))
print('Phase-averaging argerror is:', np.sum(np.abs(np.angle(phase_aved_sol) - np.angle(analyt_sol[inds])))/(len(t)-1))
print('Phase-averaging Re error is:', np.sum(np.abs(np.real(phase_aved_sol - analyt_sol[inds])))/(len(t)-1))
print('Phase-averaging Im error is:', np.sum(np.abs(np.imag(phase_aved_sol - analyt_sol[inds])))/(len(t)-1))

# Plot the analytical solution:
fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.real(analyt_sol), label=r"Re($u$)")
ax1.plot(t_ref, np.imag(analyt_sol), label=r"Im($u$)")
ax1.plot(t_ref, np.abs(analyt_sol), label=r"abs($u$)", c='k')
ax2.plot(t_ref, np.real(modvar_sol), label=r"Re($v$)")
ax2.plot(t_ref, np.imag(modvar_sol), label=r"Im($v$)")
ax2.plot(t_ref, np.abs(modvar_sol), label=r"abs($v$)", c='k')
ax1.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
ax2.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
plt.xlim([0,TT])
plt.xlabel('Time', size=14)
ax1.set_ylabel('Standard Solution', size=14)
ax2.set_ylabel('Modulation Variable', size=14)

fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.real(analyt_sol), label=r"Re($u$)")
ax1.plot(t_ref, np.imag(analyt_sol), label=r"Im($u$)")
ax1.plot(t_ref, np.abs(analyt_sol), label=r"|$u$|", c='k')
ax2.plot(t_ref, np.real(modvar_sol), label=r"Re($v$)")
ax2.plot(t_ref, np.imag(modvar_sol), label=r"Im($v$)")
ax2.plot(t_ref, np.abs(modvar_sol), label=r"|$v$|", c='k')
ax2.plot(t_ref, np.abs(w_sol), label=r"|$w$|", c='lime')
ax1.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
ax2.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
plt.xlim([0,TT])
plt.xlabel('Time', size=14)
ax1.set_ylabel('Standard Solution', size=14)
ax2.set_ylabel('Modulation Variable', size=14)

fig.savefig(f'{abspath(dirname(__file__))}/figures/simp_example_orig_sols.jpg', bbox_inches='tight')


# Just real components of the solution
fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.real(analyt_sol), label=r"Re($u$)", c='k')
ax1.plot(t, np.real(phase_aved_sol), c='b', linestyle='dashed')
ax1.plot(t, np.real(meancor_sol), c='r', linestyle='dashed')
ax1.scatter(t, np.real(phase_aved_sol), label=r"Re($u_{PA}$)", c='b')
ax1.scatter(t, np.real(meancor_sol), label=r"Re($u_{MC}$)", c='r')
ax2.plot(t_ref, np.real(modvar_sol), label=r"Re($v$)", c='k')
ax2.plot(t, np.real(v_bar), c='b', linestyle='dashed')
ax2.plot(t, np.real(w_bar), c='r', linestyle='dashed')
ax2.scatter(t, np.real(v_bar), label=r"Re($\overline{v}$)", c='b')
ax2.scatter(t, np.real(w_bar), label=r"Re($\overline{w}$)", c='r')
ax1.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
ax2.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
plt.xlim([0,TT])
plt.xlabel('Time', size=14)
ax1.set_ylabel('Standard Solution', size=14)
ax2.set_ylabel('Modulation Variable', size=14)

# Just imaginary components of the solution.
fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.imag(analyt_sol), label='imag', c='k')
ax1.plot(t, np.imag(phase_aved_sol), c='b', linestyle='dashed')
ax1.plot(t, np.imag(meancor_sol), c='r', linestyle='dashed')
ax1.scatter(t, np.imag(phase_aved_sol), label=r"Im($u_{PA}$)", c='b')
ax1.scatter(t, np.imag(meancor_sol), label=r"Im($u_{MC}$)", c='r')
ax2.plot(t_ref, np.imag(modvar_sol), label=r"Im($v$)", c='k')
ax2.plot(t, np.imag(v_bar), c='b', linestyle='dashed')
ax2.plot(t, np.imag(w_bar), c='r', linestyle='dashed')
ax2.scatter(t, np.imag(v_bar), label=r"Im($\overline{v}$)", c='b')
ax2.scatter(t, np.imag(w_bar), label=r"Im($\overline{w}$)", c='r')
ax1.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
ax2.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
plt.xlim([0,TT])
plt.xlabel('Time', size=14)
ax1.set_ylabel('Standard Solution', size=14)
ax2.set_ylabel('Modulation Variable', size=14)


fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.abs(analyt_sol), label=r"|$u$|", c='k')
ax1.plot(t, np.abs(phase_aved_sol), c='b', linestyle='dashed')
ax1.plot(t, np.abs(meancor_sol), c='r', linestyle='dotted')
ax1.scatter(t, np.abs(phase_aved_sol), label=r"|$u_{PA}$|", c='b')
ax1.scatter(t, np.abs(meancor_sol), label=r"|$u_{MC}$|", c='r')
ax2.plot(t_ref, np.abs(modvar_sol), label=r"|$v$|", c='k')
ax2.plot(t_ref, np.abs(w_sol), label=r"|$w$|", c='lime')
ax2.plot(t, np.abs(v_bar),c='b', linestyle='dashed')
ax2.plot(t, np.abs(w_bar), c='r', linestyle='dotted')
ax2.scatter(t, np.abs(v_bar), label=r"|$\overline{v}$|", c='b')
ax2.scatter(t, np.abs(w_bar), label=r"|$\overline{w}$|", c='r')
ax1.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.25, 0.5))
ax2.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.25, 0.5))
plt.xlim([0,TT])
plt.xlabel('Time', size=14)
ax1.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)
ax1.set_ylabel('Standard Solution', size=14)
ax2.set_ylabel('Modulation Variable', size=14)

fig.savefig(f'{abspath(dirname(__file__))}/figures/simp_example_phase_aved_abs.jpg', bbox_inches='tight')

plt.show()


'''
fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.abs(analyt_sol), label=r"abs($u$)", c='k')
ax1.plot(t, np.abs(discrete_modvar_sol), c='g', linestyle='dashed')
ax1.plot(t, np.abs(phase_aved_sol), c='b', linestyle='dashed')
ax1.plot(t, np.abs(meancor_sol), c='r', linestyle='dashed')
ax1.scatter(t, np.abs(discrete_modvar_sol), label=r"abs($u_{MV}$)", c='g')
ax1.scatter(t, np.abs(phase_aved_sol), label=r"abs($u_{PA}$)", c='b')
ax1.scatter(t, np.abs(meancor_sol), label=r"abs($u_{MC}$)", c='r')
ax2.plot(t_ref, np.abs(modvar_sol), label=r"abs($v$)", c='k')
ax2.plot(t, np.abs(v),c='g', linestyle='dashed')
ax2.plot(t, np.abs(v_bar),c='b', linestyle='dashed')
ax2.plot(t, np.abs(w_bar), c='r', linestyle='dashed')
ax2.scatter(t, np.abs(v), label=r"Abs($v$)", c='g')
ax2.scatter(t, np.abs(v_bar), label=r"Abs($\overline{v}$)", c='b')
ax2.scatter(t, np.abs(w_bar), label=r"Abs($\overline{w}$)", c='r')
ax1.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.35, 0.5))
ax2.legend(loc='center right', prop={'size': 12}, bbox_to_anchor=(1.3, 0.5))
plt.xlim([0,TT])
plt.xlabel('Time', size=14)
ax1.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)
ax1.set_ylabel('Standard Solution', size=14)
ax2.set_ylabel('Modulation Variable', size=14)

fig, axes = plt.subplots(2,1,sharex=True, constrained_layout=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.angle(analyt_sol), label=r"arg($u$)", c='k')
ax1.plot(t, np.angle(phase_aved_sol), c='b', linestyle='dashed')
ax1.plot(t, np.angle(meancor_sol), c='r', linestyle='dashed')
ax1.scatter(t, np.angle(phase_aved_sol), label=r"arg($u_{PA}$)", c='b')
ax1.scatter(t, np.angle(meancor_sol), label=r"arg($u_{MC}$)", c='r')
ax2.plot(t_ref, np.angle(modvar_sol), label=r"arg($v$)", c='k')
ax2.plot(t, np.angle(v_bar),c='b', linestyle='dashed')
ax2.plot(t, np.angle(w_bar), c='r', linestyle='dashed')
ax2.scatter(t, np.angle(v_bar), label=r"arg($\overline{v}$)", c='b')
ax2.scatter(t, np.angle(w_bar), label=r"arg($\overline{w}$)", c='r')
ax1.legend()
ax2.legend()

ax2.set_xlabel('Time')
ax1.set_ylabel('Standard Solution')
ax2.set_ylabel('Modulation Variable')

plt.figure()
plt.plot(t_ref, np.real(analyt_sol), label='real')
plt.plot(t_ref, np.imag(analyt_sol), label='imag')
plt.scatter(t, np.real(phase_aved_sol), label=r"Re($u_{PA}$)")
plt.scatter(t, np.imag(phase_aved_sol), label=r"Im($u_{PA}$)")
plt.scatter(t, np.real(meancor_sol), label=r"Re($u_{MC}$)")
plt.scatter(t, np.imag(meancor_sol), label=r"Im($u_{MC}$)")

plt.legend()
plt.xlabel('Time')
plt.ylabel('Original solution')

plt.figure()
plt.plot(t_ref, np.real(analyt_sol), label="Re($u$)", c='k')
plt.plot(t, np.real(phase_aved_sol), label=r"Re($u_{PA}$)", c='b')
plt.plot(t, np.real(meancor_sol), label=r"Re($u_{MC}$)", c='r')
plt.scatter(t, np.real(phase_aved_sol), label=r"Re($u_{PA}$)", c='b')
plt.scatter(t, np.real(meancor_sol), label=r"Re($u_{MC}$)", c='r')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Original solution')

plt.figure()
plt.plot(t_ref, np.imag(analyt_sol), label="Im($u$)", c='k')
plt.plot(t, np.imag(phase_aved_sol), label=r"Im($u_{PA}$)", c='b')
plt.plot(t, np.imag(meancor_sol), label=r"Im($u_{MC}$)", c='r')
plt.scatter(t, np.imag(phase_aved_sol), label=r"Im($u_{PA}$)", c='b')
plt.scatter(t, np.imag(meancor_sol), label=r"Im($u_{MC}$)", c='r')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Original solution')


plt.figure()
plt.plot(t_ref, np.abs(analyt_sol), label="Abs($u$)", c='k')
plt.plot(t, np.abs(phase_aved_sol), label=r"Abs($u_{PA}$)", c='b')
plt.plot(t, np.abs(meancor_sol), label=r"Abs($u_{MC}$)", c='r')
plt.scatter(t, np.abs(phase_aved_sol), label=r"Abs($u_{PA}$)", c='b')
plt.scatter(t, np.abs(meancor_sol), label=r"Abs($u_{MC}$)", c='r')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Abs of Original solution')

plt.figure()
plt.plot(t_ref, np.abs(modvar_sol), label="Abs($v$)", c='k')
plt.plot(t, np.abs(v_bar), label=r"Abs($\overline{v}$)", c='b')
plt.plot(t, np.abs(w_bar), label=r"Abs($\overline{v}$)", c='r')
plt.scatter(t, np.abs(v_bar), label=r"Abs($\overline{w}$)", c='b')
plt.scatter(t, np.abs(w_bar), label=r"Abs($\overline{w}$)", c='r')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Abs of modvar solution')

plt.figure()
plt.plot(t_ref, np.real(modvar_sol), label=r"Re($v$)")
plt.plot(t_ref, np.imag(modvar_sol), label=r"Im($v$)")
plt.plot(t, np.real(v_bar), label=r"Re($\overline{v}$)")
plt.plot(t, np.imag(v_bar), label=r"Im($\overline{v}$)")
plt.plot(t, np.real(w_bar), label=r"Re($\overline{w}$)")
plt.plot(t, np.imag(w_bar), label=r"Im($\overline{w}$)")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Modvar solution')

'''






