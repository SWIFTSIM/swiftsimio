from tests.helper import requires
import unyt as u
from swiftsimio import load
from numpy import allclose


@requires("cosmological_volume.hdf5")
def test_convert(filename):
    """
    Check that the conversion to physical units is done correctly.
    """
    data = load(filename)
    coords = data.gas.coordinates
    units = u.kpc
    assert units != coords.units  # ensure we make a non-trivial conversion
    assert data.metadata.a != 1.0  # ensure we make a non-trivial conversion
    coords_physical = coords.to_physical()

    # array_equal applied to cosmo_array's is aware of physical & comoving
    # make sure to compare bare arrays:
    assert allclose(
        coords.to_value(units) * data.metadata.a,
        coords_physical.to_value(units),
        rtol=1e-6,
    )
    return


@requires("cosmological_volume.hdf5")
def test_convert_to_value(filename):
    """
    Check that conversions to numerical values are correct.
    """
    data = load(filename)
    coords = data.gas.coordinates
    units = u.kpc
    assert units != coords.units  # ensure we make a non-trivial conversion
    assert data.metadata.a != 1.0  # ensure we make a non-trivial conversion
    coords_physical_values = coords.to_physical_value(units)
    coords_comoving_values = coords.to_comoving_value(units)
    print(coords_physical_values / (coords_comoving_values * data.metadata.a))
    assert allclose(
        coords_physical_values,
        coords_comoving_values * data.metadata.a,
        rtol=1e-6,
    )
