"""Test the helper to convert unyt quantities into cosmo quantities."""

import pytest
import numpy as np
import unyt as u
from swiftsimio import mask, cosmo_array, cosmo_quantity
from swiftsimio.objects import _AHelper


@pytest.mark.parametrize("scale_exponent", (-1, 0, 1, 2, None))
@pytest.mark.parametrize("data", (10, np.array([1, 2]), [1, 2], (1, 2)))
@pytest.mark.parametrize(
    "order",
    (
        "data_units_a",
        "data_a_units",
        "a_data_units",
        "units_data_a",
        "units_a_data",
        "a_units_data",
    ),
)
@pytest.mark.parametrize("com_or_phys", ("comoving", "physical"))
def test_multiply_unyt_with_ahelper(scale_exponent, data, order, com_or_phys):
    """
    Test cosmo-ifying unyt objects by multiplying with the helper object.

    Covers permutations of the multiplication order, scalar, list, tuple or array input,
    value and presence of the exponent, and comoving/physical.

    This should cover all of the fundamental functionality expected of the helper.
    """
    scale_factor = 0.5
    a = getattr(_AHelper(scale_factor=scale_factor), com_or_phys)
    if scale_exponent is not None:
        a = a**scale_exponent
    operands = {
        "a": a,
        "units": u.Mpc,
        "data": data,
    }
    first, second, third = [operands[i] for i in order.split("_")]
    cosmo = first * second * third
    if np.isscalar(data):
        assert isinstance(cosmo, cosmo_quantity)
    else:
        # careful, cosmo_quantity is a subclass and therefore counts as a cosmo_array
        assert isinstance(cosmo, cosmo_array) and not isinstance(cosmo, cosmo_quantity)
    if com_or_phys == "comoving":
        assert cosmo.comoving
    else:
        assert (
            cosmo.comoving is False
        )  # don't assert not cosmo.comoving in case is None
    assert cosmo.units == u.Mpc
    assert cosmo.cosmo_factor.a_factor == scale_factor**scale_exponent


def test_multiply_cosmo_with_ahelper():
    """Test that multiplying an already-cosmo object modifies the `cosmo_factor`."""
    raise NotImplementedError


@pytest.mark.parametrize("data", 10, np.array([1, 2]))
@pytest.mark.parametrize("com_or_phys", ("comoving", "physical"))
@pytest.mark.parametrize("scale_exponent", (1, None))
def test_assign_comoving_or_physical_units_to_name(data, com_or_phys, scale_exponent):
    """
    Test that we can make e.g. cMpc or pMpc units and then apply them to data.

    This is a documented feature so should make sure that it works.
    """
    scale_factor = 0.5
    # this is either a "cMpc" or "pMpc" depending on case, label "xMpc" here:
    a = getattr(_AHelper(scale_factor=scale_factor), com_or_phys)
    if scale_exponent is not None:
        a = a**scale_exponent
    xMpc = u.Mpc * a
    cosmo = data * xMpc
    if np.isscalar(data):
        assert isinstance(cosmo, cosmo_quantity)
    else:
        # careful, cosmo_quantity is a subclass and therefore counts as a cosmo_array
        assert isinstance(cosmo, cosmo_array) and not isinstance(cosmo, cosmo_quantity)
    if com_or_phys == "comoving":
        assert cosmo.comoving
    else:
        assert (
            cosmo.comoving is False
        )  # don't assert not cosmo.comoving in case is None
    assert cosmo.units == u.Mpc
    assert cosmo.cosmo_factor.a_factor == scale_factor**scale_exponent


def test_helper_available_from_metadata(cosmological_volume_only_single_local):
    """
    Test that the helper is available as a metadata attribute on datasets.

    This serves as an integration test by making sure that we can use the helper to define
    a region for masking, using it's comoving, physical and exponent features. We then
    check that defining the same region the tedious way gives the same result, and that
    the region defined with the helper is accepted by ``constrain_spatial``.
    """
    m = mask(cosmological_volume_only_single_local)
    a = mask.metadata.a
    scale_factor = mask.metadata.scale_factor
    region = [
        [1 * u.Mpc * a.comoving, 2 * u.Mpc * a.comoving],
        [1 * u.Mpc * a.physical, 2 * u.Mpc * a.physical],
        [1 * u.Mpc * a.comoving**1, 2 * u.Mpc * a.comoving**1],
    ]
    manual_region = [
        [
            cosmo_quantity(
                1, u.Mpc, comoving=True, scale_factor=scale_factor, scale_exponent=1
            ),
            cosmo_quantity(
                2, u.Mpc, comoving=True, scale_factor=scale_factor, scale_exponent=1
            ),
        ],
        [
            cosmo_quantity(
                1, u.Mpc, comoving=False, scale_factor=scale_factor, scale_exponent=1
            ),
            cosmo_quantity(
                2, u.Mpc, comoving=False, scale_factor=scale_factor, scale_exponent=1
            ),
        ],
        [
            cosmo_quantity(
                1, u.Mpc, comoving=True, scale_factor=scale_factor, scale_exponent=1
            ),
            cosmo_quantity(
                2, u.Mpc, comoving=True, scale_factor=scale_factor, scale_exponent=1
            ),
        ],
    ]
    # check that the regions are equivalent:
    for ax_region, ax_manual_region in zip(region, manual_region):
        for element, manual_element in zip(ax_region, ax_manual_region):
            assert element == manual_element
    # be a bit paranoid and make sure that constraining mask works:
    m.constrain_spatial(region)


def test_incorrect_usage():
    """Should also test expected failure modes..."""
    # `10 * a.comoving` should error, for example.
    # That `10 * a.comoving * u.Mpc` works and `10 * u.Mpc * a.comoving` also works
    # will rely on triggering `__multiply__` or `__rmultiply__` accordingly. Units but
    # no data yet is allowed to enable `cMpc = ...`, but we should not be able to end up
    # with data but no units.
