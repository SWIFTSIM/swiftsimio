"""
Projection kernels.
"""

import numpy as np
from swiftsimio.accelerated import jit

# Taken from Dehnen & Aly 2012
kernel_gamma = np.float32(1.897367)
kernel_constant = np.float32(7.0 / 3.14159)


@jit("float32(float32, float32)", nopython=True, fastmath=True)
def kernel_single_precision(r: np.float32, H: np.float32):
    """
    Single precision kernel implementation for swiftsimio.

    This is the Wendland-C2 kernel as shown in Denhen & Aly (2012) [1]_.

    Parameters
    ----------

    r : np.float32
        radius used in kernel computation

    H : np.float32
        kernel width (i.e. radius of compact support for the kernel)

    Returns
    -------

    np.float32
        Contribution to the density by the particle

    See Also
    --------

    kernel_double_precision

    References
    ----------

    .. [1] Dehnen W., Aly H., 2012, MNRAS, 425, 1068
    """
    kernel_constant = np.float32(2.22817109)

    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        one_minus_ratio = 1.0 - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2

        kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)

        kernel *= kernel_constant * inverse_H * inverse_H

    return kernel


@jit("float64(float64, float64)", nopython=True, fastmath=True)
def kernel_double_precision(r: np.float64, H: np.float64):
    """
    Single precision kernel implementation for swiftsimio.

    This is the Wendland-C2 kernel as shown in Denhen & Aly (2012) [2]_.

    Parameters
    ----------

    r : np.float32
        radius used in kernel computation
    H : np.float32
        kernel width (i.e. radius of compact support for the kernel)

    Returns
    -------

    np.float32
        Contribution to the density by the particle

    See Also
    --------

    kernel_single_precision

    References
    ----------

    .. [2] Dehnen W., Aly H., 2012, MNRAS, 425, 1068
    """
    kernel_constant = np.float64(2.22817109)

    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        one_minus_ratio = 1.0 - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2

        kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)

        kernel *= kernel_constant * inverse_H * inverse_H

    return kernel
