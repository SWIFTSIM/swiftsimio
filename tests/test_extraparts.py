"""Test for extra particle types."""

from swiftsimio import load, Writer, cosmo_array
import unyt
import numpy as np
import os


def test_write(extra_part_type):
    """
    Tests whether swiftsimio can handle a new particle type.

    If the test doesn't crash this is a success.
    """
    # Use default units, i.e. cm, grams, seconds, Ampere, Kelvin
    unit_system = unyt.UnitSystem(
        name="default", length_unit=unyt.cm, mass_unit=unyt.g, time_unit=unyt.s
    )
    a = 0.5
    boxsize = cosmo_array(
        [10, 10, 10], unyt.cm, comoving=False, scale_factor=a, scale_exponent=1
    )

    x = Writer(
        unit_system,
        boxsize,
        scale_factor=a,
    )

    x.extratype.coordinates = cosmo_array(
        np.array([np.arange(10), np.zeros(10), np.zeros(10)]).astype(float).T,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )
    x.extratype.velocities = cosmo_array(
        np.zeros((10, 3), dtype=float),
        unyt.cm / unyt.s,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    x.extratype.masses = cosmo_array(
        np.ones(10, dtype=float),
        unyt.g,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=0,
    )

    x.extratype.smoothing_length = cosmo_array(
        np.ones(10, dtype=float) * 5.0,
        unyt.cm,
        comoving=False,
        scale_factor=x.scale_factor,
        scale_exponent=1,
    )
    x.write("extra_test.hdf5")
    os.remove("extra_test.hdf5")


def test_read(write_extra_part_type):
    """
    Test whether swiftsimio can handle a new particle type.

    Has a few asserts to check the data is read in correctly.
    """
    data = load("extra_test.hdf5")
    for i in range(0, 10):
        assert data.extratype.coordinates.value[i][0] == float(i)
