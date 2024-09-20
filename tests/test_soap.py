"""
Tests that we can open SOAP files
"""

from tests.helper import requires

from swiftsimio import load, mask
import unyt


@requires("soap_example.hdf5")
def test_soap_can_load(filename):
    data = load(filename)

    return


@requires("soap_example.hdf5")
def test_soap_can_mask_spatial(filename):
    this_mask = mask(filename, spatial_only=True)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(filename, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


@requires("soap_example.hdf5")
def test_soap_can_mask_non_spatial(filename):
    this_mask = mask(filename, spatial_only=False)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(filename, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


@requires("soap_example.hdf5")
def test_soap_can_mask_spatial_and_non_spatial_actually_use(filename):
    this_mask = mask(filename, spatial_only=False)

    lower = unyt.unyt_quantity(1e5, "Msun")
    upper = unyt.unyt_quantity(1e13, "Msun")
    this_mask.constrain_mask("spherical_overdensity_200_mean", "total_mass", lower, upper)

    data = load(filename, mask=this_mask)

    masses = data.spherical_overdensity_200_mean.total_mass

    assert len(masses) > 0

    data2 = load(filename)

    masses2 = data2.spherical_overdensity_200_mean.total_mass

    # Manually mask
    custom_mask = (masses2 >= lower) & (masses2 <= upper)

    assert len(masses2[custom_mask]) == len(masses)


@requires("soap_example.hdf5")
def test_soap_single_row_mask(filename):
    this_mask = mask(filename, spatial_only=True)

    this_mask.constrain_index(21)

    data = load(filename, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == 1


@requires("soap_example.hdf5")
def test_soap_multiple_row_mask_non_spatial(filename):
    this_mask = mask(filename, spatial_only=False)

    IDC = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(IDC)

    data = load(filename, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == len(IDC)


@requires("soap_example.hdf5")
def test_soap_multiple_row_mask_spatial(filename):
    this_mask = mask(filename, spatial_only=False)

    IDC = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(IDC)

    data = load(filename, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == len(IDC)
