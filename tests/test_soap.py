"""
Tests that we can open SOAP files
"""

from tests.helper import requires

from swiftsimio import load, mask


@requires("soap_example.hdf5")
def test_soap_can_load(filename):
    data = load(filename)

    return


@requires("soap_example.hdf5")
def test_soap_can_mask_spatial(filename):
    this_mask = mask(filename, spatial_only=True)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial(
     [[0 * b, 0.5 * b] for b in bs]
    )

    data = load(filename, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


@requires("soap_example.hdf5")
def test_soap_can_mask_non_spatial(filename):
    this_mask = mask(filename, spatial_only=False)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial(
     [[0 * b, 0.5 * b] for b in bs]
    )

    data = load(filename, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]