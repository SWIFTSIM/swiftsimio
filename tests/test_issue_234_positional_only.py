"""Regression test for issue #234.

``generate_smoothing_lengths`` is wrapped by
``_propagate_cosmo_array_attributes_to_result``. The decorator consumes the
first argument (``obj``) internally, but users call the function using its real
parameter name (``coordinates``). Passing ``coordinates`` as a keyword argument
previously hit the decorator before the real function was ever reached and
raised a cryptic error::

    _propagate_cosmo_array_attributes_to_result.<locals>.wrapped()
    missing 1 required positional argument: 'obj'

The decorator now forwards the user's call to the underlying function and
recovers the first argument whether it was passed positionally or by keyword,
so ``coordinates=...`` works as expected.
"""

import numpy as np
import pytest
import unyt as u

from swiftsimio import cosmo_array
from swiftsimio.visualisation.smoothing_length import generate_smoothing_lengths


def _make_inputs():
    """Build small coordinate / boxsize cosmo_arrays for fast local tests."""
    x = cosmo_array(
        np.arange(20), u.Mpc, comoving=False, scale_factor=0.5, scale_exponent=1
    )
    xgrid, ygrid, zgrid = np.meshgrid(x, x, x)
    coords = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).T
    lbox = cosmo_array(
        [20, 20, 20], u.Mpc, comoving=False, scale_factor=0.5, scale_exponent=1
    )
    return coords, lbox


def test_coordinates_as_keyword_argument_works():
    """Issue #234: ``coordinates=...`` must work, not raise a cryptic error."""
    pytest.importorskip("scipy.spatial")  # KDTree is an optional dependency

    coords, lbox = _make_inputs()
    out = generate_smoothing_lengths(
        coordinates=coords,
        boxsize=lbox,
        kernel_gamma=1.0,
    )

    assert out.shape[0] == coords.shape[0]
    assert out.units == coords.units


def test_positional_call_still_works():
    """Positional usage (with ``boxsize`` as a keyword) keeps working."""
    pytest.importorskip("scipy.spatial")

    coords, lbox = _make_inputs()
    out = generate_smoothing_lengths(coords, boxsize=lbox, kernel_gamma=1.0)

    assert out.shape[0] == coords.shape[0]
    assert out.units == coords.units
