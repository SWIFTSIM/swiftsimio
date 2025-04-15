from swiftsimio import load
from numpy import array_equal


def test_convert(cosmological_volume):
    """
    Check that the conversion to physical units is done correctly
    """
    data = load(cosmological_volume)
    coords = data.gas.coordinates
    units = coords.units
    coords_physical = coords.to_physical()

    # array_equal applied to cosmo_array's is aware of physical & comoving
    # make sure to compare bare arrays:
    assert array_equal(
        coords.to_value(units) * data.metadata.a, coords_physical.to_value(units)
    )
    return
