from swiftsimio import load
from swiftsimio import Writer
from swiftsimio.units import cosmo_units
from swiftsimio import cosmo_array

import unyt
import numpy as np
from os import remove

import pytest

# File we're writing to
test_filename = "test_write_output_units.hdf5"
test_file_fields = (
    "coordinates",
    "velocities",
    "masses",
    "internal_energy",
    "smoothing_length",
)


@pytest.fixture(scope="function")
def simple_snapshot_data():
    """
    Fixture to create and cleanup a simple snapshot for testing.
    """
    # Box is 100 Mpc
    boxsize = 100 * unyt.Mpc

    # Generate object. cosmo_units corresponds to default Gadget-oid units
    # of 10^10 Msun, Mpc, and km/s
    x = Writer(cosmo_units, boxsize)

    # 32^3 particles.
    n_p = 32**3

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
    x.write(test_filename)

    # Yield the test data
    yield x

    # The file is automatically cleaned up after the test.
    remove(test_filename)


@pytest.mark.parametrize("field", test_file_fields)
def test_reading_ic_units(simple_snapshot_data, field):
    """
    Test to ensure we are able to correctly read ICs created with swiftsimio
    """
    data = load(test_filename)

    assert isinstance(getattr(data.gas, field), cosmo_array)
    # np.allclose checks unit consistency
    assert np.allclose(
        getattr(data.gas, field), getattr(simple_snapshot_data.gas, field), rtol=1.0e-4
    )
    return
