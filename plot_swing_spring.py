'''
Plot a saved solution of the swinging spring.

'''

import numpy as np
from matplotlib import pyplot as plt

######################

TT = 200 
dt = 1e-2
t = np.arange(0.0, TT, dt)

# Load the solution:
sol_u = np.load('reference_solutions/swing_spring/swing_spring_rho2.npy')

u_x = sol_u[0,0,:]
u_y = sol_u[0,1,:]
u_z = sol_u[0,2,:]


fig, (ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(t,u_x,c = 'b') #Plot the x co-ordinate over time
ax1.set_title('x co-ordinate')
ax2.plot(t,u_y,c = 'r') #Same for y
ax2.set_title('y co-ordinate')
ax3.plot(t,u_z,c = 'g') #Same for z
ax3.set_title('z co-ordinate')
fig.suptitle('Positions over time in each co-ordinate')

plt.figure()
plt.plot(u_x,u_y)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Trajectory in the x-y plane')
plt.show

plt.show()