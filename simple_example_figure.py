'''
A simple example to show the mean correction.
Use timesteps of dt = 0.25.
The nonlinearity is treated as constant 
over each of these time intervals.


'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy
from functions.timestepping import *
from functions.rswes_functions import *

#############################
def phase_aved(v_sol, t, epsilon, chi, c_val):
    return chi*np.exp(1j*t/epsilon)*c_val

##########################
# Construct an analytical solution
epsilon = 0.14/2/np.pi
omega = 1
TT = 1
u0 = 2 +1j

dt_ref = 0.001
t_ref = np.arange(0, TT, dt_ref)

c_val = 5

analyt_sol = 1j*c_val*epsilon*(np.exp(-1j*t_ref/epsilon) - 1) + np.exp(-1j*t_ref/epsilon)*u0

modvar_sol = u0 - 1j*c_val*epsilon*(np.exp(-1j*t_ref/epsilon) - 1)

# Perform timestepping
dt = 0.2
t = np.arange(0, TT+dt, dt)

# Bonus, compute chi with zeta = 1:


chi = 0.5



v_bar = np.zeros((len(t)), dtype='complex128')
w_bar = np.zeros((len(t)), dtype='complex128')

v_bar[0] = u0
w_bar[0] = u0 + epsilon*1j*c_val

for i in np.arange(len(t)-1):
    v_bar[i+1] = v_bar[i] + dt*phase_aved(v_bar[i], t[i], epsilon, chi, c_val)
    w_bar[i+1] = w_bar[i]

phase_aved_sol = np.exp(-1j*t/epsilon)*v_bar
meancor_sol = np.exp(-1j*t/epsilon)*w_bar - epsilon*1j*c_val


# Plot the analytical solution:
fig, axes = plt.subplots(2,1,sharex=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.real(analyt_sol), label=r"Re($u$)")
ax1.plot(t_ref, np.imag(analyt_sol), label=r"Im($u$)")
ax2.plot(t_ref, np.real(modvar_sol), label=r"Re($v$)")
ax2.plot(t_ref, np.imag(modvar_sol), label=r"Im($v$)")
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
ax1.set_ylabel('Standard Solution')
ax2.set_ylabel('Modulation Variable')

fig, axes = plt.subplots(2,1,sharex=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.real(analyt_sol), label='real', c='k')
ax1.scatter(t, np.real(phase_aved_sol), label=r"Re($u_{PA}$)")
ax1.scatter(t, np.real(meancor_sol), label=r"Re($u_{MC}$)")
ax2.plot(t_ref, np.real(modvar_sol), label=r"Re($v$)")
ax2.plot(t, np.real(v_bar), label=r"Re($\overline{v}$)")
ax2.plot(t, np.real(w_bar), label=r"Re($\overline{w}$)")
ax1.legend()
ax2.legend()
plt.xlabel('Time')
plt.ylabel('Modvar solution')

fig, axes = plt.subplots(2,1,sharex=True)
ax1, ax2 = axes
ax1.plot(t_ref, np.imag(analyt_sol), label='imag', c='k')
ax1.scatter(t, np.imag(phase_aved_sol), label=r"Im($u_{PA}$)", c='b')
ax1.scatter(t, np.imag(meancor_sol), label=r"Im($u_{MC}$)", c='r')
ax2.plot(t_ref, np.imag(modvar_sol), label=r"Im($v$)", c='k')
ax2.plot(t, np.imag(v_bar), label=r"Im($\overline{v}$)", c='b')
ax2.plot(t, np.imag(w_bar), label=r"Im($\overline{w}$)", c='r')
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
ax1.set_ylabel('Im(Standard Solution)')
ax2.set_ylabel('Im(Modulation Variable)')


fig, axes = plt.subplots(2,1)
ax1, ax2 = axes
ax1.plot(t_ref, np.abs(analyt_sol), label=r"abs($u$)", c='k')
ax1.plot(t, np.abs(phase_aved_sol), c='b', linestyle='dashed')
ax1.plot(t, np.abs(meancor_sol), c='r', linestyle='dashed')
ax1.scatter(t, np.abs(phase_aved_sol), label=r"abs($u_{PA}$)", c='b')
ax1.scatter(t, np.abs(meancor_sol), label=r"abs($u_{MC}$)", c='r')
ax1.scatter(t, np.abs(phase_aved_sol), label=r"Re($u_{PA}$)", c='b')
ax1.scatter(t, np.abs(meancor_sol), label=r"Re($u_{MC}$)", c='r')
ax2.plot(t_ref, np.abs(modvar_sol), label=r"abs($v$)", c='k')
ax2.plot(t, np.abs(v_bar),c='b')
ax2.plot(t, np.abs(w_bar), c='r')
ax2.scatter(t, np.abs(v_bar), label=r"Abs($\overline{w}$)", c='b')
ax2.scatter(t, np.abs(w_bar), label=r"Abs($\overline{w}$)", c='r')
ax1.legend()
ax2.legend()
ax2.set_xlabel('Time')
ax1.set_ylabel('Standard Solution')
ax2.set_ylabel('Modulation Variable')

plt.show()

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

# Perform timestepping

plt.show()





