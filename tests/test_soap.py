"""
Tests that we can open SOAP files
"""

from swiftsimio import load, mask, cosmo_quantity


def test_soap_can_load(soap_example_params):
    load(**soap_example_params)

    return


def test_soap_can_mask_spatial(soap_example_params):
    this_mask = mask(**soap_example_params, spatial_only=True)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(**soap_example_params, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


def test_soap_can_mask_non_spatial(soap_example_params):
    this_mask = mask(**soap_example_params, spatial_only=False)

    bs = this_mask.metadata.boxsize
    this_mask.constrain_spatial([[0 * b, 0.5 * b] for b in bs])

    data = load(**soap_example_params, mask=this_mask)

    data.spherical_overdensity_200_mean.total_mass[0]


def test_soap_can_mask_spatial_and_non_spatial_actually_use(soap_example_params):
    this_mask = mask(**soap_example_params, spatial_only=False)

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

    data = load(**soap_example_params, mask=this_mask)

    masses = data.spherical_overdensity_200_mean.total_mass

    assert len(masses) > 0

    data2 = load(**soap_example_params)

    masses2 = data2.spherical_overdensity_200_mean.total_mass

    # Manually mask
    custom_mask = (masses2 >= lower) & (masses2 <= upper)

    assert len(masses2[custom_mask]) == len(masses)


def test_soap_single_row_mask(soap_example_params):
    this_mask = mask(**soap_example_params, spatial_only=True)

    this_mask.constrain_index(21)

    data = load(**soap_example_params, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == 1


def test_soap_multiple_row_mask_non_spatial(soap_example_params):
    this_mask = mask(**soap_example_params, spatial_only=False)

    IDC = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(IDC)

    data = load(**soap_example_params, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == len(IDC)


def test_soap_multiple_row_mask_spatial(soap_example_params):
    this_mask = mask(**soap_example_params, spatial_only=False)

    IDC = [0, 1, 2, 3, 6, 23, 94, 57]

    this_mask.constrain_indices(IDC)

    data = load(**soap_example_params, mask=this_mask)

    assert len(data.spherical_overdensity_200_mean.total_mass) == len(IDC)
