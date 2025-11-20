"""Tests that we can open SOAP files."""

import pytest
import numpy as np
from swiftsimio import load, mask, cosmo_quantity
from swiftsimio.file_utils import open_path_or_handle


def test_soap_can_load(soap_example):
    """Just check we don't crash loading a SOAP file."""
    load(soap_example)

    return


@pytest.mark.parametrize("spatial_only", [True, False])
def test_soap_can_mask_spatial(soap_example, spatial_only):
    """
    Check we don't crash applying a mask to a SOAP file.

    Covers both the spatial only and non-spatial only cases.
    """
    this_mask = mask(soap_example, spatial_only=spatial_only)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(soap_example, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


def test_soap_can_mask_spatial_and_non_spatial_actually_use(soap_example):
    """Check that non-spatial masking is equivalent to loading all and masking by hand."""
    this_mask = mask(soap_example, spatial_only=False)

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

    data = load(soap_example, mask=this_mask)

    masses = data.spherical_overdensity_200_mean.total_mass

    assert len(masses) > 0

    data2 = load(soap_example)

    masses2 = data2.spherical_overdensity_200_mean.total_mass

    # Manually mask
    custom_mask = (masses2 >= lower) & (masses2 <= upper)

    assert len(masses2[custom_mask]) == len(masses)


def test_soap_single_row_mask(soap_example):
    """Check that we can mask down to a single row."""
    this_mask = mask(soap_example, spatial_only=True)

    this_mask.constrain_index(21)

    data = load(soap_example, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == 1


@pytest.mark.parametrize("spatial_only", [True, False])
def test_soap_multiple_row_mask_non_spatial(soap_example, spatial_only):
    """
    Check that we can mask multiple non-consecutive rows.

    Covers both spatial only and non-spatial only cases.
    """
    this_mask = mask(soap_example, spatial_only=spatial_only)

    indices = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(indices)

    data = load(soap_example, mask=this_mask)
    assert len(data.spherical_overdensity_200_mean.total_mass) == len(indices)

    # Check that we read the right values in the right order
    with open_path_or_handle(soap_example) as f:
        all_values = f["SO/200_mean/TotalMass"][...]

    # Expected ordering depends on whether we use spatial_only
    values_read = data.spherical_overdensity_200_mean.total_mass.value
    if spatial_only:
        assert np.all(values_read == all_values[indices])
    else:
        assert np.all(values_read == all_values[sorted(indices)])
