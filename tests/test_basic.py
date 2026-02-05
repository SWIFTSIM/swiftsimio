"""Basic integration test."""

from swiftsimio import load, Writer, cosmo_array

import unyt
import numpy as np

import os


def test_write():
    """Create a sample dataset. Should not crash."""
    a = 0.5
    # Box is 100 Mpc
    boxsize = cosmo_array(
        [100, 100, 100],
        unyt.Mpc,
        comoving=True,
        scale_factor=a,
        scale_exponent=1.0,
    )
    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    x = Writer("galactic", boxsize, scale_factor=a)

    # 32^3 particles.
    n_p = 32**3

    # Randomly spaced coordinates from 0, 100 Mpc in each direction
    x.gas.coordinates = cosmo_array(
        np.random.rand(n_p, 3) * 100,
        unyt.Mpc,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=1.0,
    )

    # Random velocities from 0 to 1 km/s
    x.gas.velocities = cosmo_array(
        np.random.rand(n_p, 3),
        unyt.km / unyt.s,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=0.0,
    )

    # Generate uniform masses as 10^6 solar masses for each particle
    x.gas.masses = cosmo_array(
        np.ones(n_p, dtype=float) * 1e6,
        unyt.solMass,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=0.0,
    )

    # Generate internal energy corresponding to 10^4 K
    x.gas.internal_energy = cosmo_array(
        np.ones(n_p, dtype=float) * 1e4 / 1e6,
        unyt.kb * unyt.K / unyt.solMass,
        comoving=True,
        scale_factor=x.scale_factor,
        scale_exponent=-2.0,
    )

    # Generate initial guess for smoothing lengths based on MIPS
    x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

    # If IDs are not present, this automatically generates
    x.write("test.hdf5")


def test_load():
    """Try to load the dataset we just made. Should not crash."""
    x = load("test.hdf5")

    x.gas.internal_energy
    x.gas.coordinates

    os.remove("test.hdf5")
