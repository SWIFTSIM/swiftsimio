#!/usr/bin/env python3

"""
Kernel functions and constants following the Dehnen & Aly 2012 conventions
"""

from swiftsimio.accelerated import jit
from numpy import empty_like
from math import pow


@jit(nopython=True, fastmath=True)
def W_cubic_spline(r: float, H: float):
    """
    Cubic spline kernel.


    Parameters
    ------------------

    r:  float
        distance between particles.

    H:  float
        compact support radius of kernel.


    Returns
    ------------------
    W:  float
        evaluated kernel functions


    Note
    ------------------
    + The return value is not normalized. It needs to be divided by ``H^ndim`` 
      and multiplied by the kernel norm to get the proper values. 
    """

    q = r / H
    W = 0.0

    if q <= 1.0:
        W += (1.0 - q) ** 3
    if q <= 0.5:
        W -= 4 * (0.5 - q) ** 3

    return W


@jit(nopython=True, fastmath=True)
def dWdr_cubic_spline(r: float, H: float):
    """
    Cubic spline kernel derivative.


    Parameters
    ------------------

    r:  float
        distance between particles.

    H:  float
        compact support radius of kernel.


    Returns
    ------------------

    dWdr:  float
        evaluated kernel derivative function


    Note
    ------------------
    + The return value is not normalized. It needs to be divided by 
      ``H^(ndim+1)`` and multiplied by the kernel norm to get the proper values
    """

    q = r / H

    dW = 0.0

    if q <= 1.0:
        dW -= 3 * (1.0 - q) ** 2
    if q <= 0.5:
        dW += 12 * (0.5 - q) ** 2

    return dW


# dictionary "pointing" to the correct functions to call
kernel_funcs = {"cubic spline": W_cubic_spline}

kernel_derivatives = {"cubic spline": dWdr_cubic_spline}

# Constants are from Dehnen & Aly 2012

kernel_gamma_1D = {"cubic spline": 1.732051}

kernel_norm_1D = {"cubic spline": 2.666667}

kernel_gamma_2D = {"cubic spline": 1.778002}

kernel_norm_2D = {"cubic spline": 3.637827}

kernel_gamma_3D = {"cubic spline": 1.825742}

kernel_norm_3D = {"cubic spline": 5.092958}


def get_kernel_data(kernel: str, ndim: int):
    """
    Picks the correct kernel functions and constants for you.


    Parameters
    ------------------

    kernel: string {'cubic spline'}
        which kernel you want to use

    ndim: int
        dimensionality of the kernel that you want


    Returns
    --------------------

    W(r, H): callable
        normalized kernel function with two positional arguments:

        - r:  float
          distance between particles.

        - H:  float
          compact support radius of kernel.


    dWdr(r, H): callable
        normalized kernel derivative function with two positional arguments:

        - r:  float
          distance between particles.

        - H:  float
          compact support radius of kernel.

    kernel_gamma: float
        H/h (compact support radius / smoothing length) for given kernel and 
        dimension
    
    """
    kernel_function = kernel_funcs[kernel]
    kernel_derivative = kernel_derivatives[kernel]

    if ndim == 1:
        norm = kernel_norm_1D[kernel]
        kernel_gamma = kernel_gamma_1D[kernel]
    elif ndim == 2:
        norm = kernel_norm_2D[kernel]
        kernel_gamma = kernel_gamma_2D[kernel]
    elif ndim == 3:
        norm = kernel_norm_3D[kernel]
        kernel_gamma = kernel_gamma_3D[kernel]

    @jit(nopython=True, fastmath=True)
    def W(r, H):
        out = empty_like(r)

        for index in range(len(out)):
            this_kernel = kernel_function(r[index], H[index])
            out[index] = this_kernel * norm / pow(H[index], ndim)

        return out

    @jit(nopython=True, fastmath=True)
    def dWdr(r, H):
        out = empty_like(r)

        for index in range(len(out)):
            this_kernel = kernel_derivative(r[index], H[index])
            out[index] = this_kernel * norm / pow(H[index], ndim + 1)

        return out

    return W, dWdr, kernel_gamma
