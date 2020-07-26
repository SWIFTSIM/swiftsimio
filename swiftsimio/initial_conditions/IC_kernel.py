#!/usr/bin/env python3

"""
Kernel functions and constants following the Dehnen & Aly 2012 conventions
 
"""

# ------------------------------------
# Kernel related stuffs
# ------------------------------------

import numpy as np
from typing import Union


def W_cubic_spline(r: np.ndarray, H: Union[float, np.ndarray]):
    """
    Cubic spline kernel.


    Parameters
    ------------------

    r:  np.ndarray
        array of distances between particles.

    H:  np.ndarray or float
        compact support radius of kernel. Scalar or numpy array of same shape as r.


    Returns
    ------------------
    W:  np.ndarray 
        evaluated kernel functions with same shape as ``r``.


    Note
    ------------------
    + The return value is not normalized. It needs to be divided by H^ndim and multiplied by the kernel norm to get the proper values. 
    """

    q = r / H
    W = np.zeros(q.shape)

    #  if q <= 1.:
    #      W += (1. - q)**3
    #  if q <= 0.5:
    #      W -= 4*(0.5 - q)**3
    W[q <= 1] += (1.0 - q[q <= 1]) ** 3
    W[q <= 0.5] -= 4 * (0.5 - q[q < 0.5]) ** 3

    return W


def dWdr_cubic_spline(r: np.ndarray, H: Union[float, np.ndarray]):
    """
    Cubic spline kernel derivative.


    Parameters
    ------------------

    r:  np.ndarray
        array of distances between particles.

    H:  np.ndarray or float
        compact support radius of kernel. Scalar or numpy array of same shape as r.


    Returns
    ------------------
    dWdr:  np.ndarray 
        evaluated kernel derivative functions with same shape as ``r``.


    Note
    ------------------
    + The return value is not normalized. It needs to be divided by H^(ndim+1) and multiplied by the kernel norm to get the proper values. 
    """

    q = r / H
    dW = np.zeros(q.shape)

    #  if q <= 1.:
    #      W -= 3 * (1. - q)**2
    #  if q <= 0.5:
    #      W += 12*(0.5 - q)**2
    dW[q <= 1] -= 3 * (1.0 - q[q <= 1]) ** 2
    dW[q <= 0.5] += 12 * (0.5 - q[q <= 0.5]) ** 2

    return dW


# dictionary "pointing" to the correct functions to call
kernel_funcs = {}
kernel_funcs["cubic spline"] = W_cubic_spline

kernel_derivatives = {}
kernel_derivatives["cubic spline"] = dWdr_cubic_spline


# Constants are from Dehnen & Aly 2012

kernel_gamma_1D = {}
kernel_gamma_1D["cubic spline"] = 1.732051

kernel_norm_1D = {}
kernel_norm_1D["cubic spline"] = 2.666667


kernel_gamma_2D = {}
kernel_gamma_2D["cubic spline"] = 1.778002

kernel_norm_2D = {}
kernel_norm_2D["cubic spline"] = 3.637827


kernel_gamma_3D = {}
kernel_gamma_3D["cubic spline"] = 1.825742

kernel_norm_3D = {}
kernel_norm_3D["cubic spline"] = 5.092958


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

        - r:  np.ndarray
            array of distances between particles.

        - H:  np.ndarray or float
            compact support radius of kernel. Scalar or numpy array of same shape as r.

    dWdr(r, H): callable
        normalized kernel derivative function with two positional arguments:

        - r:  np.ndarray
            array of distances between particles.

        - H:  np.ndarray or float
            compact support radius of kernel. Scalar or numpy array of same shape as r.

    kernel_gamma: float
        H/h (compact support radius / smoothing length) for given kernel and dimension
    
    """
    if ndim == 1:
        W = lambda r, H: kernel_norm_1D[kernel] / H * kernel_funcs[kernel](r, H)
        dWdr = (
            lambda r, H: kernel_norm_1D[kernel]
            / H ** 2
            * kernel_derivatives[kernel](r, H)
        )
        kernel_gamma = kernel_gamma_1D[kernel]
    elif ndim == 2:
        W = lambda r, H: kernel_norm_2D[kernel] / H ** 2 * kernel_funcs[kernel](r, H)
        dWdr = (
            lambda r, H: kernel_norm_2D[kernel]
            / H ** 3
            * kernel_derivatives[kernel](r, H)
        )
        kernel_gamma = kernel_gamma_2D[kernel]
    elif ndim == 3:
        W = lambda r, H: kernel_norm_3D[kernel] / H ** 3 * kernel_funcs[kernel](r, H)
        dWdr = (
            lambda r, H: kernel_norm_3D[kernel]
            / H ** 4
            * kernel_derivatives[kernel](r, H)
        )
        kernel_gamma = kernel_gamma_3D[kernel]
    return W, dWdr, kernel_gamma
