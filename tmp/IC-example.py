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


ic_sim_params = IC.ic_sim_params(
    boxsize=unyt_array([1.0, 1.0, 1.0], "cm"),
    periodic=True,
    nx=nx,
    ndim=ndim,
    kernel="cubic spline",
    eta=1.2348,
)


ic_run_params = IC.ic_run_params(
    iter_max=20,
    convergence_threshold=1e-5,
    tolerance_part=1e-3,
    displacement_threshold=1e-4,
    #  delta_init                  = 0.01,
    delta_reduction_factor=1.0,
    delta_min=1e-6,
    redistribute_frequency=5,
    redistribute_fraction=0.01,
    no_redistribution_after=200,
    intermediate_dump_frequency=5,
    dump_basename="iteration-",
)


# the party starts here
x, m, stats = IC.generate_IC_for_given_density(
    rho_parabola, ic_sim_params, ic_run_params
)

print("Stats of last iteration:")
print("Number of iterations: {0:6d}".format(stats["niter"]))
print("Smallest displacement: {0:20.3e}".format(stats["min_motion"]))
print("Average displacement:  {0:20.3e}".format(stats["avg_motion"]))
print("Maximal displacement:  {0:20.3e}".format(stats["max_motion"]))
