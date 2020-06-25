from swiftsimio import load
from swiftsimio import Writer
from swiftsimio.units import cosmo_units

import unyt
import numpy as np


def test_read():
    # Box is 100 Mpc
    boxsize = 100 * unyt.Mpc

    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    x = Writer(cosmo_units, boxsize)

    # 32^3 particles.
    n_p = 32 ** 3

    # Randomly spaced coordinates from 0, 100 Mpc in each direction
    x.gas.coordinates = np.random.rand(n_p, 3) * (100 * unyt.Mpc)

    # Random velocities from 0 to 1 km/s
    x.gas.velocities = np.random.rand(n_p, 3) * (unyt.km / unyt.s)

    # Generate uniform masses as 10^6 solar masses for each particle
    x.gas.masses = np.ones(n_p, dtype=float) * (1e6 * unyt.msun)

    # Generate internal energy corresponding to 10^4 K
    x.gas.internal_energy = (
        np.ones(n_p, dtype=float) * (1e4 * unyt.kb * unyt.K) / (1e6 * unyt.msun)
    )

    # Generate initial guess for smoothing lengths based on MIPS
    x.gas.generate_smoothing_lengths(boxsize=boxsize, dimension=3)

    # If IDs are not present, this automatically generates
    x.write("test.hdf5")

    data = load("test.hdf5")

    assert np.array_equal(data.gas.coordinates, x.gas.coordinates)
    assert np.array_equal(data.gas.velocities, x.gas.velocities)
    assert np.array_equal(data.gas.masses, x.gas.masses)
    assert np.array_equal(data.gas.internal_energy, x.gas.internal_energy)
    assert np.array_equal(data.gas.smoothing_length, x.gas.smoothing_length)

    return
