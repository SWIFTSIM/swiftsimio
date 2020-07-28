#!/usr/bin/env python3

# -------------------------------------------------------
# Minor example file to test out the iterative IC
# generation.
# -------------------------------------------------------


import numpy as np
from scipy import stats
import unyt
from unyt import unyt_array

from swiftsimio import initial_conditions as IC
from swiftsimio import Writer
from swiftsimio.optional_packages import KDTree, TREE_AVAILABLE


#TODO: I assume you want a uniform density distribution.
# if not, you can change this function.
# It needs to take x and ndim as parameters. x needs to be a numpy array,
# ndim needs to be an integer that represents how many dimensions you want
# your simulation in.
# Also the function needs to return a numpy array as the result.
def rho_uniform(x, ndim):
    """
    Analytical function to be used.
    Here: uniform density.
    """
    return np.ones(x.shape[0], dtype=np.float)



# There are two sets of parameters that you can set.
# The first one is for the simulation parameters, which is
# what you should be interested in


ic_sim_params = IC.ic_sim_params(
    boxsize=unyt_array([1.0, 1.0, 1.0], "cm"), # TODO: set box size and correct units here
    periodic=True,
    nx=16,  # will be 16^3 total particles
    ndim=3,
    kernel="cubic spline", # don't touch this, there are no other kernels available atm
    eta=1.2348,
    unit_m = unyt_array(10, "kg") # set here what unit mass you want to use.
)


# the runtime parameters take a bit to explain. They are the second
# set of parameters needed to be set.
# don't touch stuff here unless you're not happy with the end result.
# but contact me first. Hopefully I'll have the documentation finished by then.
ic_run_params = IC.ic_run_params(
    iter_max=10000,
    convergence_threshold=1e-5,
    tolerance_part=1e-3,
    displacement_threshold=1e-4,
    delta_init = None,
    delta_reduction_factor=1.0,
    delta_min=1e-6,
    redistribute_frequency=30,
    redistribute_fraction=0.01,
    no_redistribution_after = 210,
    intermediate_dump_frequency = 0,
    dump_basename="iteration-",
)



# the party starts here: this function generates the IC
x, m, stats = IC.generate_IC_for_given_density( rho_uniform, ic_sim_params, ic_run_params )

# x and m are unyt arrays of the initial conditions.
# The code generates particle positions and masses.
# The rest of the data you need to fill out yourself based on these positions and masses and what you want.



# generate approximate smoothing lengths and approximate densities (don't touch this)

kernel_func, _, kernel_gamma = IC.IC_kernel.get_kernel_data('cubic spline', 3)
Nngb = 4 / 3 * np.pi * (kernel_gamma * ic_sim_params.eta) ** 3 # different for 1D and 2D
Nngb = int(Nngb + 0.5)
npart = ic_sim_params.nx ** ic_sim_params.ndim
h = np.zeros(npart, dtype = np.float)
rho = np.zeros(npart, dtype = np.float)

tree = KDTree(x.value, boxsize=ic_sim_params.boxsize.to(x.units).value)
for p in range(npart):
    dist, neighs = tree.query(x[p].value, k=Nngb)
    h[p] = dist[-1] / kernel_gamma
    for i, n in enumerate(neighs):
        W = kernel_func(dist[i], dist[-1])
        rho[p] += W * m[n].value

h = unyt.unyt_array(h, x.units)
rho = unyt.unyt_array(rho, m.units/x.units**3)





ICunits = unyt.UnitSystem(
    "IC_generation", 
    ic_sim_params.unit_l,  # TODO: you need to put correct units here. This is for unit length
    ic_sim_params.unit_m,  # TODO: this is for unit mass
    unyt.s,                # TODO: this is for unit time
)

# alternatively, you can use already made unit systems.
# an example of how to do that is given in https://swiftsimio.readthedocs.io/en/latest/creating_initial_conditions/index.html



W = Writer(ICunits, ic_sim_params.boxsize)
W.gas.coordinates = x
W.gas.smoothing_length = h
W.gas.masses = m
W.gas.densities = unyt_array(
    rho, W.gas.masses.units / W.gas.coordinates.units ** 3
)

# TODO: You need to figure out what kind of initial conditions you want for internal energies and velocities!
# TODO: and also pay attention to the units!
W.gas.internal_energy = unyt_array(
    np.zeros(npart, dtype=np.float), unyt.m ** 2 / unyt.s ** 2
)
W.gas.velocities = unyt_array(np.zeros(npart, dtype=np.float), unyt.m / unyt.s)

# finally, write the file down.
fname = "my_ic_file.hdf5"
W.write(fname)

# and now you have a swift IC file! :)
