from swiftsimio import load, cosmo_array
import numpy as np
import unyt as u


def test_convert(cosmological_volume_only_single):
    """
    Check that the conversion to physical units is done correctly.
    """
    data = load(cosmological_volume_only_single)
    coords = data.gas.coordinates
    units = u.kpc
    assert units != coords.units  # ensure we make a non-trivial conversion
    assert data.metadata.a != 1.0  # ensure we make a non-trivial conversion
    coords_physical = coords.to_physical()

    # allclose applied to cosmo_array's is aware of physical & comoving
    # make sure to compare bare arrays:
    assert np.allclose(
        coords.to_value(units) * data.metadata.a,
        coords_physical.to_value(units),
        rtol=1e-6,
    )
    return


def test_convert_to_value(cosmological_volume_only_single):
    """
    Check that conversions to numerical values are correct.
    """
    data = load(cosmological_volume_only_single)
    coords = data.gas.coordinates
    units = u.kpc
    assert units != coords.units  # ensure we make a non-trivial conversion
    assert data.metadata.a != 1.0  # ensure we make a non-trivial conversion
    coords_physical_values = coords.to_physical_value(units)
    coords_comoving_values = coords.to_comoving_value(units)
    print(coords_physical_values / (coords_comoving_values * data.metadata.a))
    assert np.allclose(
        coords_physical_values, coords_comoving_values * data.metadata.a, rtol=1e-6
    )


def test_combine_physical_comoving(cosmological_volume_only_single):
    """
    Check that we can combine physical and comoving quantities.
    """
    comoving_arr = cosmo_array(
        u.unyt_array(np.ones(5), units=u.kpc),
        scale_factor=1.0,
        scale_exponent=1,
        comoving=True,
    )
    physical_arr = cosmo_array(
        u.unyt_array(np.ones(5), units=u.kpc),
        scale_factor=1.0,
        scale_exponent=1,
        comoving=False,
        valid_transform=False,
    )
    physical_arr / comoving_arr
