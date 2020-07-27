#!/usr/bin/env python3

# -------------------------------------------------------
# Minor example file to test out the iterative IC
# generation.
# -------------------------------------------------------


import numpy as np
from scipy import stats
from unyt import unyt_array

from swiftsimio import initial_conditions as IC


def rho_parabola(x, ndim):
    """
    Analytical function to be used
    WATCH OUT FOR PERIODICITY WHEN SETTING BOXLEN
    """

    res = 10.0 - (x[:, 0] - 0.5) ** 2 - (x[:, 1] - 0.5) ** 2 - (x[:, 2] - 0.5) ** 2

    return res


def rho_uniform(x, ndim):
    """
    Analytical function to be used
    """
    return np.ones(x.shape[0], dtype=np.float)


npart = 2000
ndim = 2
nx = int(npart ** (1.0 / ndim) + 0.5)


# you need to call this first
icSimParams = IC.IC_set_IC_params(
    boxsize=unyt_array([1.0, 1.0, 1.0], "cm"),
    periodic=True,
    nx=nx,
    ndim=ndim,
    unit_l="cm",
    unit_m="g",
)


# this too
icRunParams = IC.IC_set_run_params(
    iter_max=2000,
    convergence_threshold=1e-5,
    tolerance_part=1e-3,
    displacement_threshold=1e-4,
    delta_init=None,
    delta_reduction_factor=1.0,
    delta_min=1e-6,
    redistribute_frequency=20,
    redistribute_fraction=0.01,
    redistribute_fraction_reduction=1.0,
    no_redistribution_after=200,
    intermediate_dump_frequency=50,
)


# the party starts here
x, m, stats = IC.generate_IC_for_given_density(rho_parabola, icSimParams, icRunParams)

print("Stats of last iteration:")
print("Number of iterations: {0:6d}".format(stats["niter"]))
print("Smallest displacement: {0:20.3e}".format(stats["min_motion"]))
print("Average displacement:  {0:20.3e}".format(stats["avg_motion"]))
print("Maximal displacement:  {0:20.3e}".format(stats["max_motion"]))
