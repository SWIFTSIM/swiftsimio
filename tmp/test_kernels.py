#!/usr/bin/env python3

#---------------------------------------------------------------------------------
# Create a plot of kernels and kernel derivatives to check their validity
#---------------------------------------------------------------------------------


import numpy as np
from matplotlib import pyplot as plt
from swiftsimio import initial_conditions as ic


kernellist = [
                'cubic spline',
            ]



r = np.linspace(0, 1.1, 200)
H = 1.0


fig = plt.figure()

ax1 = fig.add_subplot(231)
ax1.set_ylabel(r"$W(r)$")
ax1.set_title("ndim = 1")

ax2 = fig.add_subplot(232)
ax2.set_ylabel(r"$W(r)$")
ax2.set_title("ndim = 2")

ax3 = fig.add_subplot(233)
ax3.set_ylabel(r"$W(r)$")
ax3.set_title("ndim = 3")

ax4 = fig.add_subplot(234)
ax4.set_ylabel(r"$\frac{dW}{dr}$")

ax5 = fig.add_subplot(235)
ax5.set_ylabel(r"$\frac{dW}{dr}$")

ax6 = fig.add_subplot(236)
ax6.set_ylabel(r"$\frac{dW}{dr}$")


for kernel in kernellist:
    
    Wf1, dWdrf1, _, = ic.get_kernel_data(kernel, 1)
    Wf2, dWdrf2, _, = ic.get_kernel_data(kernel, 2)
    Wf3, dWdrf3, _, = ic.get_kernel_data(kernel, 3)
    W1 = Wf1(r, H)
    W2 = Wf2(r, H)
    W3 = Wf3(r, H)
    dWdr1 = dWdrf1(r, H)
    dWdr2 = dWdrf2(r, H)
    dWdr3 = dWdrf3(r, H)

    ax1.plot(r, W1, label = kernel)
    ax2.plot(r, W2, label = kernel)
    ax3.plot(r, W3, label = kernel)
    ax4.plot(r, dWdr1, label = kernel)
    ax5.plot(r, dWdr2, label = kernel)
    ax6.plot(r, dWdr3, label = kernel)

for ax in fig.axes:
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$r$")

plt.show()
