from tests.helper import requires
from swiftsimio import load, cosmo_array
from numpy import array_equal


@requires("cosmological_volume.hdf5")
def test_convert(filename):
    """
    Check that the conversion to physical units is done correctly
    """
    data = load(filename)
    coords = data.gas.coordinates
    units = coords.units
    coords_physical = coords.to_physical()

    # array_equal applied to cosmo_array's is aware of physical & comoving
    # make sure to compare bare arrays:
    assert array_equal(
        coords.to_value(units) * data.metadata.a, coords_physical.to_value(units)
    )
    return
