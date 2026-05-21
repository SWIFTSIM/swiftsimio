"""Test the helper to convert unyt quantities into cosmo quantities."""

import pytest
from copy import deepcopy
import numpy as np
import unyt as u
from unyt.exceptions import InvalidUnitOperation
from swiftsimio import mask, cosmo_array, cosmo_quantity, Writer
from swiftsimio.objects import _AHelper, InvalidCosmoUnit, cosmo_factor
from swiftsimio.metadata.writer.unit_systems import cosmo_units


@pytest.mark.parametrize("scale_exponent", (-1, 0, 1, 2, None))
@pytest.mark.parametrize("data", (10.0, np.array([1.0, 2.0]), [1.0, 2.0], (1.0, 2.0)))
@pytest.mark.parametrize(
    "order",
    (
        "data_units_a",
        "data_a_units",
        "units_a_data",
        "units_data_a",
        "a_units_data",
        "a_data_units",
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
    operands = {"a": a, "units": u.Mpc, "data": data}
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
        assert cosmo.comoving is False  # avoid `not cosmo.comoving` in case is None
    assert cosmo.units == u.Mpc
    assert scale_factor != 1  # else trivial
    if scale_exponent is not None:
        assert cosmo.cosmo_factor.a_factor == scale_factor**scale_exponent
    else:
        assert cosmo.cosmo_factor.a_factor == scale_factor


@pytest.mark.parametrize("scale_exponent", (-1, 0, 1, 2, None))
@pytest.mark.parametrize("data", (10.0, np.array([1.0, 2.0]), [1.0, 2.0], (1.0, 2.0)))
@pytest.mark.parametrize("order", ("data_a", "a_data"))
@pytest.mark.parametrize("com_or_phys", ("comoving", "physical"))
@pytest.mark.parametrize("in_com_or_phys", (True, False))
def test_multiply_cosmo_with_ahelper(
    scale_exponent, data, order, com_or_phys, in_com_or_phys
):
    """Test that multiplying an already-cosmo object modifies the `cosmo_factor`."""
    scale_factor = 0.5
    a = getattr(_AHelper(scale_factor=scale_factor), com_or_phys)
    if scale_exponent is not None:
        a = a**scale_exponent
    cosmo_in = (cosmo_quantity if np.isscalar(data) else cosmo_array)(
        deepcopy(data),  # beware modification in-place
        u.Mpc,
        comoving=in_com_or_phys,
        scale_factor=scale_factor,
        scale_exponent=1,
    )
    operands = {"data": cosmo_in, "a": a}
    first, second = [operands[i] for i in order.split("_")]
    cosmo = first * second
    if np.isscalar(data):
        assert isinstance(cosmo, cosmo_quantity)
    else:
        # careful, cosmo_quantity is a subclass and therefore counts as a cosmo_array
        assert isinstance(cosmo, cosmo_array) and not isinstance(cosmo, cosmo_quantity)
    if com_or_phys == "comoving":
        assert cosmo.comoving
    else:
        assert cosmo.comoving is False  # avoid `not cosmo.comoving` in case is None
    assert cosmo.units == u.Mpc
    assert scale_factor != 1  # else trivial
    expected_exponent = (scale_exponent if scale_exponent is not None else 1) + 1
    assert cosmo.cosmo_factor.a_factor == scale_factor**expected_exponent
    if com_or_phys == "comoving":
        # scale_factor**(scale_exponent) from the helper, plus another scale_factor**1
        # from cosmo_in if it was comoving, all needed to convert to physical
        conversion_exponent = (
            scale_exponent if scale_exponent is not None else 1
        ) + int(in_com_or_phys)
        assert np.allclose(
            cosmo.to_physical_value(u.Mpc),
            np.asarray(data) * scale_factor**conversion_exponent,
        )
    else:  # "physical"
        # just scale_factor**1 coming from converting cosmo_in to physical if it was
        # comoving, else scale_factor**0
        conversion_exponent = int(in_com_or_phys)
        assert np.allclose(
            cosmo.to_physical_value(u.Mpc),
            np.asarray(data) * scale_factor**conversion_exponent,
        )


@pytest.mark.parametrize("data", (10.0, np.array([1.0, 2.0])))
@pytest.mark.parametrize("com_or_phys", ("comoving", "physical"))
@pytest.mark.parametrize("scale_exponent", (1, None))
@pytest.mark.parametrize("order", ("a_unit", "unit_a"))
def test_assign_comoving_or_physical_units_to_name(
    data, com_or_phys, scale_exponent, order
):
    """
    Test that we can make e.g. cMpc or pMpc units and then apply them to data.

    This is a documented feature so should make sure that it works.
    """
    scale_factor = 0.5
    a = getattr(_AHelper(scale_factor=scale_factor), com_or_phys)
    if scale_exponent is not None:
        a = a**scale_exponent
    # this is either a "cMpc" or "pMpc" depending on case, label "xMpc" here:
    operands = {"a": a, "unit": u.Mpc}
    first, second = [operands[i] for i in order.split("_")]
    xMpc = first * second
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
    assert cosmo.cosmo_factor.a_factor == scale_factor ** (
        scale_exponent if scale_exponent is not None else 1
    )


@pytest.mark.parametrize(
    "data",
    (
        10,
        10.0,
        np.array([1.0, 2.0]),
        (1.0, 2.0),
        [1.0, 2.0],
        u.kpc,
        10.0 * u.kpc,
        [1.0, 2.0] * u.kpc,
        cosmo_quantity(10.0, u.kpc, comoving=True, scale_factor=0.5, scale_exponent=1),
        cosmo_array(
            [1.0, 2.0], u.kpc, comoving=True, scale_factor=0.5, scale_exponent=1
        ),
    ),
)
@pytest.mark.parametrize("order", ("a_data", "data_a"))
def test_division(data, order):
    """Test that the helper can handle division by and of all supported data types."""
    scale_factor = 0.5
    a = _AHelper(scale_factor=scale_factor).physical
    operands = {"a": a, "data": data}
    first, second = [operands[i] for i in order.split("_")]
    cosmo = first / second
    if isinstance(first, u.Unit):
        assert isinstance(cosmo, _AHelper)
    elif np.asarray(data).ndim == 0:
        assert isinstance(cosmo, cosmo_quantity)
    else:
        # careful, cosmo_quantity is a subclass and therefore counts as a cosmo_array
        assert isinstance(cosmo, cosmo_array) and not isinstance(cosmo, cosmo_quantity)
    if isinstance(cosmo, _AHelper):
        assert cosmo._comoving == a._comoving
    else:
        assert cosmo.comoving == getattr(data, "comoving", a._comoving)
    if isinstance(cosmo, _AHelper):
        assert cosmo.scale_factor == scale_factor
    else:
        assert cosmo.cosmo_factor.scale_factor == scale_factor
    assert scale_factor != 1  # else trivial
    if hasattr(data, "cosmo_factor"):
        expected_exponent = 0
    else:
        expected_exponent = 1 if isinstance(first, _AHelper) else -1
    if isinstance(cosmo, _AHelper):
        assert (
            cosmo._scale_factor**cosmo.scale_exponent == scale_factor**expected_exponent
        )
    else:
        assert cosmo.cosmo_factor.a_factor == scale_factor**expected_exponent


def test_helper_available_from_metadata(cosmological_volume_only_single_local):
    """
    Test that the helper is available as a metadata attribute on datasets.

    This serves as an integration test by making sure that we can use the helper to define
    a region for masking, using it's comoving, physical and exponent features. We then
    check that defining the same region the tedious way gives the same result, and that
    the region defined with the helper is accepted by ``constrain_spatial``.
    """
    m = mask(cosmological_volume_only_single_local)
    a = m.metadata.a
    scale_factor = m.metadata.scale_factor
    region = [
        [1.0 * u.Mpc * a.comoving, 2.0 * u.Mpc * a.comoving],
        [1.0 * u.Mpc * a.physical, 2.0 * u.Mpc * a.physical],
        [1.0 * u.Mpc * a.comoving**1, 2.0 * u.Mpc * a.comoving**1],
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


def test_helper_available_from_writer():
    """Test that the helper is available as an attribute on the ``Writer``."""
    scale_factor = 0.5
    w = Writer(
        unit_system=cosmo_units,
        boxsize=cosmo_array(
            [100, 100, 100],
            u.Mpc,
            comoving=True,
            scale_factor=scale_factor,
            scale_exponent=1,
        ),
        scale_factor=scale_factor,
    )
    a = w.a
    assert 10 * u.Mpc * a.comoving == cosmo_quantity(
        10, u.Mpc, comoving=True, scale_factor=scale_factor, scale_exponent=1
    )


def test_comoving_or_physical_missing():
    """Test that failing to specify ``a.comoving`` or ``a.physical`` is an error."""
    scale_factor = 0.5
    a = _AHelper(scale_factor=scale_factor)
    with pytest.raises(InvalidCosmoUnit, match="Cannot use scale factor helper as"):
        10 * u.Mpc * a
    with pytest.raises(InvalidCosmoUnit, match="Cannot use scale factor helper as"):
        cosmo_quantity(10, u.Mpc, comoving=True, scale_factor=0.5, scale_exponent=1) * a


def test_metadata_attribute_not_backwards_compatible(
    cosmological_volume_only_single_local,
):
    """
    Test that users get a clear error when metadata.a is used in old style.

    The introduction of the _AHelper is a breaking API change. We want to avoid users
    doing things like ``10 * metadata.a * u.kpc`` (even though they should probably be
    using :class:`~swiftsimio.objects.cosmo_quantity` instead) in the way that they used
    to and getting strange/incorrect results. Requiring that ``metadata.a.comoving``
    or ``metadata.a.physical`` is used before multiplication or division achieves this:
    ``10 * metadata.a * u.kpc`` is an error when ``metadata.a`` is a ``_AHelper``.

    The exception is ``cosmo_array(..., scale_factor=metadata.a)``. The ``cosmo_array``
    can handle this case safely and interpret the object as just a scale factor, offering
    some helpful backwards compatibility. See ``test_cosmo_factor_accepts_ahelper``.
    """
    m = mask(cosmological_volume_only_single_local)
    with pytest.raises(InvalidCosmoUnit, match="Cannot use scale factor helper as"):
        10 * m.metadata.a * u.kpc


def test_cosmo_factor_accepts_ahelper(cosmological_volume_only_single_local):
    """
    Check that cosmo array/quantity accepts metadata.a as a scale factor.

    This is an explicit exception to not being able to use ``metadata.a`` without using
    ``metadata.a.comoving`` or ``metadata.a.physical``. Actually, we specifically require
    it not to be used in this case (``metadata.a._comoving`` should be ``None``) to avoid
    ambiguity.
    """
    m = mask(cosmological_volume_only_single_local)
    scale_exponent = 2  # pick a non-default case
    with pytest.raises(InvalidCosmoUnit, match="Cannot use scale factor helper as"):
        # the __comoving attribute is supposed to be None so this should raise
        m.metadata.a._comoving
    cq = cosmo_quantity(
        1,
        u.kpc**2,
        comoving=True,
        scale_factor=m.metadata.a,  # can use it like this (backwards compatible)
        scale_exponent=scale_exponent,
    )
    assert cq.comoving
    assert cq.cosmo_factor == cosmo_factor.create(
        m.metadata.scale_factor, scale_exponent
    )
    with pytest.raises(InvalidCosmoUnit, match="Initialize cosmo_array"):
        cosmo_quantity(
            1,
            u.kpc**2,
            comoving=True,
            scale_factor=m.metadata.a.comoving,  # can't use it like this
            scale_exponent=scale_exponent,
        )


def test_inplace_operations():
    """
    Test that in-place operations work/don't as intended.

    The helper should not allow in-place operations as left argument.
    """
    a = _AHelper(scale_factor=0.5).comoving
    with pytest.raises(TypeError, match="In-place operations with"):
        a *= 1
    with pytest.raises(TypeError, match="In-place operations with"):
        a /= 1
    with pytest.raises(TypeError, match="In-place operations with"):
        a **= 1
    with pytest.raises(
        InvalidUnitOperation,
        match="in-place operations with unit objects are not allowed",
    ):
        u.kpc *= a
    x = 10
    x *= a
    assert isinstance(x, cosmo_quantity)
    assert x.units == u.Unit("")
    assert x.comoving
    assert x.cosmo_factor == cosmo_factor.create(0.5, 1)
    x = u.unyt_quantity(10, u.kpc)
    x *= a
    assert isinstance(x, cosmo_quantity)
    assert x.units == u.kpc
    assert x.comoving
    assert x.cosmo_factor == cosmo_factor.create(0.5, 1)
    x = cosmo_quantity(10, u.kpc, comoving=True, scale_factor=0.5, scale_exponent=1)
    x *= a
    assert isinstance(x, cosmo_quantity)
    assert x.units == u.kpc
    assert x.comoving
    assert x.cosmo_factor == cosmo_factor.create(0.5, 2)
    x = 10
    x /= a
    assert isinstance(x, cosmo_quantity)
    assert x.units == u.Unit("")
    assert x.comoving
    assert x.cosmo_factor == cosmo_factor.create(0.5, -1)
    x = u.unyt_quantity(10, u.kpc)
    x /= a
    assert isinstance(x, cosmo_quantity)
    assert x.units == u.kpc
    assert x.comoving
    assert x.cosmo_factor == cosmo_factor.create(0.5, -1)
    x = cosmo_quantity(10, u.kpc, comoving=True, scale_factor=0.5, scale_exponent=1)
    x /= a
    assert isinstance(x, cosmo_quantity)
    assert x.units == u.kpc
    assert x.comoving
    assert x.cosmo_factor == cosmo_factor.create(0.5, 0)


def test_aliases():
    """Test ``a.com``, ``a.c`` and ``a.phys``, ``a.p`` work as shorthands."""
    a = _AHelper(scale_factor=0.5)
    assert 10 * a.comoving == 10 * a.com
    assert 10 * a.physical == 10 * a.phys
    assert 10 * a.comoving != 10 * a.phys
    assert 10 * a.physical != 10 * a.com
    assert 10 * a.comoving == 10 * a.c
    assert 10 * a.physical == 10 * a.p
    assert 10 * a.comoving != 10 * a.p
    assert 10 * a.physical != 10 * a.c
