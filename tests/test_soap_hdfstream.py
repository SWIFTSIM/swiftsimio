"""
Tests that we can open SOAP files using hdfstream
"""

import pytest
from swiftsimio import load, mask, cosmo_quantity

# Import hdfstream, if we can
try:
    import hdfstream
except ImportError:
    hdfstream = None

# For use with test server
if hdfstream is not None:
    hdfstream.verify_cert(False)
server = "https://localhost:8444/hdfstream"


@pytest.fixture
def soap_example():
    yield {"filename": "SWIFT/test_data/IOExamples/ssio_ci_04_2025/SoapExample.hdf5", "server" : server}


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_can_load(soap_example):
    load(**soap_example)
    return


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_can_mask_spatial(soap_example):
    this_mask = mask(**soap_example, spatial_only=True)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(**soap_example, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_can_mask_non_spatial(soap_example):
    this_mask = mask(**soap_example, spatial_only=False)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(**soap_example, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_can_mask_spatial_and_non_spatial_actually_use(soap_example):
    this_mask = mask(**soap_example, spatial_only=False)

    lower = cosmo_quantity(
        1e5,
        "Msun",
        comoving=True,
        scale_factor=this_mask.metadata.scale_factor,
        scale_exponent=0,
    )
    upper = cosmo_quantity(
        1e13,
        "Msun",
        comoving=True,
        scale_factor=this_mask.metadata.scale_factor,
        scale_exponent=0,
    )
    this_mask.constrain_mask(
        "spherical_overdensity_200_mean", "total_mass", lower, upper
    )

    data = load(**soap_example, mask=this_mask)

    masses = data.spherical_overdensity_200_mean.total_mass

    assert len(masses) > 0

    data2 = load(**soap_example)

    masses2 = data2.spherical_overdensity_200_mean.total_mass

    # Manually mask
    custom_mask = (masses2 >= lower) & (masses2 <= upper)

    assert len(masses2[custom_mask]) == len(masses)


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_single_row_mask(soap_example):
    this_mask = mask(**soap_example, spatial_only=True)

    this_mask.constrain_index(21)

    data = load(**soap_example, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == 1


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_multiple_row_mask_non_spatial(soap_example):
    this_mask = mask(**soap_example, spatial_only=False)

    IDC = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(IDC)

    data = load(**soap_example, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == len(IDC)


@pytest.mark.skipif(hdfstream is None, reason="hdfstream is not available")
def test_soap_multiple_row_mask_spatial(soap_example):
    this_mask = mask(**soap_example, spatial_only=False)

    IDC = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(IDC)

    data = load(**soap_example, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == len(IDC)
