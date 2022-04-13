#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1,2,0.001)
binary_threshold = 0.5

f_act =  1 * (x > binary_threshold)
f_surrogate = np.minimum(np.maximum(x, 0),1)
f_prime = np.zeros_like(f_act)
f_prime[np.where(x>binary_threshold)[0][0]]=1
H_of_x = 1*(x>0)
# H_of_1mx = 1*((1-x)>0)
H_of_xm1 = 1*((x-1)>0)
f_surr_prime= H_of_x - H_of_xm1
# f_surr_prime= 1*(x>0)*((1-x)>0) # This is the same

fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
axs[0,0].plot(x, f_act, label='$f$')
axs[0,1].plot(x, f_surrogate, label='$f_\mathrm{surrogate}$')
axs[0,2].plot(x, H_of_x, label='$H(x)$')
axs[1,0].plot(x, f_prime, label="$f'$")
axs[1,1].plot(x, f_surr_prime, label="$f'_\mathrm{surrogate}$")
# axs[1,2].plot(x, H_of_1mx, label='$H(1-x)$')
axs[1,2].plot(x, H_of_xm1, label='$H(x-1)$')
for row,axs1 in enumerate(axs):
    for col, ax in enumerate(axs1):
        ax.legend(loc='center left')
        ax.set_xticks([-1,0,1,2])
        ax.set_yticks([0, 1])
        if row == 1:
            ax.set_xlabel('x')
        if col == 0:
            ax.set_ylabel('y')
plt.tight_layout()
plt.savefig('activation_functions_separate.svg')


fig, ax = plt.subplots()
ax.plot(x, f_act, label='$f$')
ax.plot(x, f_surrogate, label='$f_\mathrm{surrogate}$')
# ax.plot(x, H_of_x, label='$H(x)$')
# ax.plot(x, f_prime, label="$f'$")
ax.plot(x, f_surr_prime, label="$f'_\mathrm{surrogate}$")
# ax.plot(x, H_of_xm1, label='$H(x-1)$')
ax.legend(loc='lower left')
ax.set_xticks([-1,0,1,2])
ax.set_yticks([0, 1])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig('activation_functions.svg')
