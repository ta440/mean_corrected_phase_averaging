'''
Plots of the RSWEs error to show in the paper.


'''

import numpy as np
from matplotlib import pyplot as plt

##################
# Arrays for the errors:

eps = np.array([0.5,0.1,0.05,0.01,0.001])
dts = np.array([0.1,0.15,0.2,0.25,0.3,0.35])
max_err = np.array([0.05,0.1,0.08, 0.03, 0.003 ])

C_errors = np.array([[0.000767954, 0.005154496, 0.016130026, 0.026020931, 0.031826044, 0.04603805],
                     [0.014775239, 0.028725904, 0.053827323, 0.062178052, 0.087322762, 0.083196265],
                      [0.02961126, 0.044819545, 0.034183755, 0.069930618, 0.070447276, 0.076862223],
                       [0.017884614, 0.02651243, 0.023202593, 0.022738414, 0.026399006, 0.027858809],
                        [0.002745749, 0.002066761, 0.002384491, 0.002824295, 0.001944068, 0.002169136] ])

D_errors = np.array([[0.006203093, 0.008090515, 0.016327843, 0.025524479, 0.028925111, 0.040894121],
                     [0.013705835, 0.022867737, 0.046030305, 0.055413984, 0.072388052, 0.076446545], 
                     [0.024174699, 0.036381742, 0.027892114, 0.062260689, 0.064688747, 0.070390372], 
                     [0.017743233, 0.024655036, 0.022945796, 0.021589109, 0.024773883, 0.026076], 
                     [0.002581567, 0.002004682, 0.002208687, 0.00265178, 0.001899345, 0.002056785]])

for i in np.arange(5):
    plt.figure(i)
    plt.plot(dts, C_errors[i,:], label='Standard phase-averaging', c='b', marker='x', linewidth=2, linestyle='--')
    plt.plot(dts, D_errors[i,:], label='Mean corrected', c='r', marker='o', linewidth=2)
    plt.ylim([0.0, max_err[i]])
    plt.xlabel('\u0394t', size=12)
    plt.ylabel('Solution error', size=12)
    plt.legend()
    plt.savefig(f'figures/rswes_error_{eps[i]}.png', bbox_inches="tight")

plt.show()