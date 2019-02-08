"""
Basic integration test!
"""

from swiftsimio import load
from swiftsimio import Writer
from swiftsimio.units import cosmo_units

import unyt
import numpy as np

import os

def test_write():
    """
    Creates a sample dataset.
    """
    # Box is 100 Mpc
    boxsize = 100 * unyt.Mpc

    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    x = Writer("galactic", boxsize)

    # 32^3 particles.
    n_p = 32**3

    # Randomly spaced coordinates from 0, 100 Mpc in each direction
    x.gas.coordinates = np.random.rand(n_p, 3) * (100 * unyt.Mpc)

    # Random velocities from 0 to 1 km/s
    x.gas.velocities = np.random.rand(n_p, 3) * (unyt.km / unyt.s)

    # Generate uniform masses as 10^6 solar masses for each particle
    x.gas.masses = np.ones(n_p, dtype=float) * (1e6 * unyt.msun)

    # Generate internal energy corresponding to 10^4 K
    x.gas.internal_energy = np.ones(n_p, dtype=float) * (1e4 * unyt.kb * unyt.K) / (1e6 * unyt.msun)

    # Generate initial guess for smoothing lengths based on MIPS
    x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

    # If IDs are not present, this automatically generates    
    x.write("test.hdf5")


def test_load():
    """
    Tries to load the dataset we just made.
    """
    x = load("test.hdf5")

    density = x.gas.internal_energy
    coordinates = x.gas.coordinates

    os.remove("test.hdf5")    