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
    coords_physical = coords.to_physical()

    assert array_equal(coords * data.metadata.a, coords_physical)
    return
