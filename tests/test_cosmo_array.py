"""Tests the initialisation of a cosmo_array."""

from typing import Callable
import pytest
import os
import warnings
import numpy as np
import unyt as u
from copy import copy, deepcopy
import pickle
from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor, a
from importlib.metadata import version
from packaging.version import Version

NUMPY_VERSION = Version(version("numpy"))

savetxt_file = "saved_array.txt"


def getfunc(fname: str) -> Callable:
    """
    Test helper: get the function handle from a name.

    Possibly with attribute access.

    Parameters
    ----------
    fname : str
        Name of the numpy function.

    Returns
    -------
    Callable
        The function handle.
    """
    func = np
    for attr in fname.split("."):
        func = getattr(func, attr)
    return func


def ca(x: np.ndarray, unit: u.Unit = u.Mpc) -> cosmo_array:
    """
    Test helper: turn an array into a cosmo_array.

    Parameters
    ----------
    x : ndarray
        The numerical array.

    unit : Unit
        The units for the array.

    Returns
    -------
    cosmo_array
        A cosmo_array with the requested data and units.
    """
    return cosmo_array(x, unit, comoving=False, scale_factor=0.5, scale_exponent=1)


def cq(x: float, unit: u.Unit = u.Mpc) -> cosmo_quantity:
    """
    Test helper: turn a scalar into a cosmo_quantity.

    Parameters
    ----------
    x : float
        The numerical value.

    unit : Unit
        The units for the quantity.

    Returns
    -------
    cosmo_quantity
        A cosmo_quantity wih the requested data and units.
    """
    return cosmo_quantity(x, unit, comoving=False, scale_factor=0.5, scale_exponent=1)


def arg_to_ua(arg: cosmo_array) -> u.unyt_array:
    """
    Test helper: recursively convert cosmo_* to unyt_*.

    Recurseively converts cosmo_* items in an argument (possibly an
    iterable) to their unyt_* equivalents.

    Parameters
    ----------
    arg : cosmo_array
        The cosmo object to be converted.

    Returns
    -------
    unyt_array
        The unyt version(s) of the input.
    """
    if type(arg) in (list, tuple):
        return type(arg)([arg_to_ua(a) for a in arg])
    else:
        return to_ua(arg)


def to_ua(x: cosmo_array) -> u.unyt_array:
    """
    Test helper to turn a cosmo_* object into its unyt_* equivalent.

    Parameters
    ----------
    x : cosmo_array
        The cosmo object to be converted.

    Returns
    -------
    unyt_array
        The unyt version of the input.
    """
    return u.unyt_array(x) if hasattr(x, "comoving") else x


def check_result(
    x_c: cosmo_array, x_u: u.unyt_array, ignore_values: bool = False
) -> None:
    """
    Test helper to compare cosmo and unyt results.

    Check that a result with cosmo input matches what we expected based on the result with
    unyt input.

    We check:
     - that the type of the result makes sense, recursing if needed.
     - that the value of the result matches (unless ignore_values=False).
     - that the units match.

    Parameters
    ----------
    x_c : cosmo_array
        The cosmo_* object to be compared.

    x_u : unyt_array
        The unyt_* object to be compared.

    ignore_values : bool
        If ``True``, only compare non-data attributes.

    Raises
    ------
    AssertionError
        If the two inputs are not equivalent for attributes in common.
    """
    if x_u is None:
        assert x_c is None
        return
    if isinstance(x_u, str):
        assert isinstance(x_c, str)
        return
    if isinstance(x_u, type) or isinstance(x_u, np.dtype):
        assert x_u == x_c
        return
    if type(x_u) in (list, tuple):
        assert type(x_u) is type(x_c)
        assert len(x_u) == len(x_c)
        for x_c_i, x_u_i in zip(x_c, x_u):
            check_result(x_c_i, x_u_i)
            return
    # careful, unyt_quantity is a subclass of unyt_array:
    if isinstance(x_u, u.unyt_quantity):
        assert isinstance(x_c, cosmo_quantity)
    elif isinstance(x_u, u.unyt_array):
        assert isinstance(x_c, cosmo_array) and not isinstance(x_c, cosmo_quantity)
    else:
        assert not isinstance(x_c, cosmo_array)
        if not ignore_values:
            assert np.allclose(x_c, x_u)
        return
    assert x_c.units == x_u.units
    if not ignore_values:
        assert np.allclose(x_c.to_value(x_c.units), x_u.to_value(x_u.units))
    if isinstance(x_c, cosmo_array):  # includes cosmo_quantity
        assert x_c.comoving is False
        if x_c.units != u.dimensionless:
            assert x_c.cosmo_factor is not None
    return


class TestCosmoArrayInit:
    """Test different ways of initializing a cosmo_array."""

    def test_init_from_ndarray(self):
        """Check initializing from a bare numpy array."""
        arr = cosmo_array(
            np.ones(5), units=u.Mpc, scale_factor=1.0, scale_exponent=1, comoving=False
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)
        # also with a cosmo_factor argument instead of scale_factor & scale_exponent
        arr = cosmo_array(
            np.ones(5),
            units=u.Mpc,
            cosmo_factor=cosmo_factor(a**1, 1.0),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list(self):
        """Check initializing from a list of values."""
        arr = cosmo_array(
            [1, 1, 1, 1, 1],
            units=u.Mpc,
            scale_factor=1.0,
            scale_exponent=1,
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)
        # also with a cosmo_factor argument instead of scale_factor & scale_exponent
        arr = cosmo_array(
            [1, 1, 1, 1, 1],
            units=u.Mpc,
            cosmo_factor=cosmo_factor(a**1, 1.0),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_unyt_array(self):
        """Check initializing from a unyt_array."""
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=u.Mpc),
            scale_factor=1.0,
            scale_exponent=1,
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)
        # also with a cosmo_factor argument instead of scale_factor & scale_exponent
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=u.Mpc),
            cosmo_factor=cosmo_factor(a**1, 1.0),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list_of_unyt_arrays(self):
        """
        Check initializing from a list of unyt_array's.

        Note that unyt won't recurse deeper than one level on inputs, so we don't test
        deeper than one level of lists. This behaviour is documented in cosmo_array.
        """
        arr = cosmo_array(
            [u.unyt_array(1, units=u.Mpc) for _ in range(5)],
            scale_factor=1.0,
            scale_exponent=1,
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)
        # also with a cosmo_factor argument instead of scale_factor & scale_exponent
        arr = cosmo_array(
            [u.unyt_array(1, units=u.Mpc) for _ in range(5)],
            cosmo_factor=cosmo_factor(a**1, 1.0),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list_of_cosmo_arrays(self):
        """
        Check initializing from a list of cosmo_array's.

        Note that unyt won't recurse deeper than one level on inputs, so we don't test
        deeper than one level of lists. This behaviour is documented in cosmo_array.
        """
        arr = cosmo_array(
            [
                cosmo_array(
                    [1], units=u.Mpc, comoving=False, scale_factor=1.0, scale_exponent=1
                )
                for _ in range(5)
            ]
        )
        assert isinstance(arr, cosmo_array)
        assert hasattr(arr, "cosmo_factor") and arr.cosmo_factor == cosmo_factor(
            a**1, 1
        )
        assert hasattr(arr, "comoving") and arr.comoving is False
        # also with a cosmo_factor argument instead of scale_factor & scale_exponent
        arr = cosmo_array(
            [
                cosmo_array(
                    [1],
                    units=u.Mpc,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a**1, 1.0),
                )
                for _ in range(5)
            ]
        )
        assert isinstance(arr, cosmo_array)
        assert hasattr(arr, "cosmo_factor") and arr.cosmo_factor == cosmo_factor(
            a**1, 1
        )
        assert hasattr(arr, "comoving") and arr.comoving is False

    def test_expected_init_failures(self):
        """Test desired failure modes for initialising cosmo_array & cosmo_quantity."""
        for cls, inp in ((cosmo_array, [1]), (cosmo_quantity, 1)):
            # we refuse both cosmo_factor and scale_factor/scale_exponent provided:
            with pytest.raises(ValueError):
                cls(
                    inp,
                    units=u.Mpc,
                    comoving=False,
                    cosmo_factor=cosmo_factor.create(1.0, 1),
                    scale_factor=0.5,
                    scale_exponent=1,
                )
            # unless they match, that's fine:
            cls(
                inp,
                units=u.Mpc,
                comoving=False,
                cosmo_factor=cosmo_factor.create(1.0, 1),
                scale_factor=1.0,
                scale_exponent=1,
            )
            # we refuse scale_factor with missing scale_exponent and vice-versa:
            with pytest.raises(ValueError):
                cls(inp, units=u.Mpc, comoving=False, scale_factor=1.0)
            with pytest.raises(ValueError):
                cls(inp, units=u.Mpc, comoving=False, scale_exponent=1)
            # we refuse overriding an input cosmo_array's information:
            with pytest.raises(ValueError):
                cls(
                    cls(
                        inp,
                        units=u.Mpc,
                        comoving=False,
                        cosmo_factor=cosmo_factor.create(1.0, 1),
                    ),
                    units=u.Mpc,
                    comoving=False,
                    scale_factor=0.5,
                    scale_exponent=1,
                )
            # unless it matches, that's fine:
            cls(
                cls(
                    inp,
                    units=u.Mpc,
                    comoving=False,
                    cosmo_factor=cosmo_factor.create(1.0, 1),
                ),
                units=u.Mpc,
                comoving=False,
                scale_factor=1.0,
                scale_exponent=1,
            )

    def test_init_from_not_iterable_invalid(self):
        """Check initializing with a scalar raises (should use cosmo_quantity)."""
        with pytest.raises(ValueError, match="cosmo_array data must be iterable"):
            cosmo_array(0, u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1)
        # make sure inheriting class doesn't do anything silly:
        cosmo_quantity(0, u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1)


class TestNumpyFunctions:
    """Check that numpy functions recognize and handle our cosmo classes."""

    def test_explicitly_handled_funcs(self):
        """
        Test consistency with unyt for functions that we handle explicitly.

        Make sure we at least handle everything that unyt does, and anything that
        'just worked' for unyt but that we need to handle by hand.

        We don't try to be exhaustive here, but at give some basic input to every function
        that we expect to be able to take cosmo input. We then use our helpers defined
        above to convert the inputs to unyt equivalents and call the numpy function on
        both cosmo and unyt input. Then we use our helpers to check the results for
        consistency. For instnace, if with unyt input we got back a unyt_array, we
        should expect a cosmo_array.

        We are not currently explicitly testing that the results of any specific function
        are numerically what we expected them to be (seems like overkill), nor that the
        cosmo_factor's make sense given the input. The latter would be a useful addition,
        but I can't think of a sensible way to implement this besides writing in the
        expectation for every output value of every function by hand.

        As long as no functions outright crash, the test will report the list of functions
        that we should have covered that we didn't cover in tests, and/or the list of
        functions whose output values were not what we expected based on running them with
        unyt input. Otherwise we just get a stack trace of the first function that
        crashed.
        """
        from unyt._array_functions import _HANDLED_FUNCTIONS
        from unyt.tests.test_array_functions import NOOP_FUNCTIONS

        functions_to_check = {
            # FUNCTIONS UNYT HANDLES EXPLICITLY:
            "array2string": (ca(np.arange(3)),),
            "dot": (ca(np.arange(3)), ca(np.arange(3))),
            "vdot": (ca(np.arange(3)), ca(np.arange(3))),
            "inner": (ca(np.arange(3)), ca(np.arange(3))),
            "outer": (ca(np.arange(3)), ca(np.arange(3))),
            "kron": (ca(np.arange(3)), ca(np.arange(3))),
            "histogram_bin_edges": (ca(np.arange(3)),),
            "linalg.inv": (ca(np.eye(3)),),
            "linalg.tensorinv": (ca(np.eye(9).reshape((3, 3, 3, 3))),),
            "linalg.pinv": (ca(np.eye(3)),),
            "linalg.svd": (ca(np.eye(3)),),
            "histogram": (ca(np.arange(3)),),
            "histogram2d": (ca(np.arange(3)), ca(np.arange(3))),
            "histogramdd": (ca(np.arange(3)).reshape((1, 3)),),
            "concatenate": (ca(np.eye(3)),),
            "cross": (ca(np.arange(3)), ca(np.arange(3))),
            "intersect1d": (ca(np.arange(3)), ca(np.arange(3))),
            "union1d": (ca(np.arange(3)), ca(np.arange(3))),
            "linalg.norm": (ca(np.arange(3)),),
            "vstack": (ca(np.arange(3)),),
            "hstack": (ca(np.arange(3)),),
            "dstack": (ca(np.arange(3)),),
            "column_stack": (ca(np.arange(3)),),
            "stack": (ca(np.arange(3)),),
            "around": (ca(np.arange(3)),),
            "block": ([[ca(np.arange(3))], [ca(np.arange(3))]],),
            "fft.fft": (ca(np.arange(3)),),
            "fft.fft2": (ca(np.eye(3)),),
            "fft.fftn": (ca(np.arange(3)),),
            "fft.hfft": (ca(np.arange(3)),),
            "fft.rfft": (ca(np.arange(3)),),
            "fft.rfft2": (ca(np.eye(3)),),
            "fft.rfftn": (ca(np.arange(3)),),
            "fft.ifft": (ca(np.arange(3)),),
            "fft.ifft2": (ca(np.eye(3)),),
            "fft.ifftn": (ca(np.arange(3)),),
            "fft.ihfft": (ca(np.arange(3)),),
            "fft.irfft": (ca(np.arange(3)),),
            "fft.irfft2": (ca(np.eye(3)),),
            "fft.irfftn": (ca(np.arange(3)),),
            "fft.fftshift": (ca(np.arange(3)),),
            "fft.ifftshift": (ca(np.arange(3)),),
            "sort_complex": (ca(np.arange(3)),),
            "isclose": (ca(np.arange(3)), ca(np.arange(3))),
            "allclose": (ca(np.arange(3)), ca(np.arange(3))),
            "array_equal": (ca(np.arange(3)), ca(np.arange(3))),
            "array_equiv": (ca(np.arange(3)), ca(np.arange(3))),
            "linspace": (cq(1), cq(2)),
            "logspace": (cq(1, unit=u.dimensionless), cq(2, unit=u.dimensionless)),
            "geomspace": (cq(1), cq(1)),
            "copyto": (ca(np.arange(3)), ca(np.arange(3))),
            "prod": (ca(np.arange(3)),),
            "var": (ca(np.arange(3)),),
            "trace": (ca(np.eye(3)),),
            "percentile": (ca(np.arange(3)), 30),
            "quantile": (ca(np.arange(3)), 0.3),
            "nanpercentile": (ca(np.arange(3)), 30),
            "nanquantile": (ca(np.arange(3)), 0.3),
            "linalg.det": (ca(np.eye(3)),),
            "diff": (ca(np.arange(3)),),
            "ediff1d": (ca(np.arange(3)),),
            "ptp": (ca(np.arange(3)),),
            "cumprod": (ca(np.arange(3)),),
            "pad": (ca(np.arange(3)), 3),
            "choose": (np.arange(3), ca(np.eye(3))),
            "insert": (ca(np.arange(3)), 1, cq(1)),
            "linalg.lstsq": (ca(np.eye(3)), ca(np.eye(3))),
            "linalg.solve": (ca(np.eye(3)), ca(np.eye(3))),
            "linalg.tensorsolve": (
                ca(np.eye(24).reshape((6, 4, 2, 3, 4))),
                ca(np.ones((6, 4))),
            ),
            "linalg.eig": (ca(np.eye(3)),),
            "linalg.eigh": (ca(np.eye(3)),),
            "linalg.eigvals": (ca(np.eye(3)),),
            "linalg.eigvalsh": (ca(np.eye(3)),),
            "savetxt": (savetxt_file, ca(np.arange(3))),
            "fill_diagonal": (ca(np.eye(3)), ca(np.arange(3))),
            "apply_over_axes": (lambda x, axis: x, ca(np.eye(3)), (0, 1)),
            "place": (ca(np.arange(3)), np.arange(3) > 0, ca(np.arange(3))),
            "put": (ca(np.arange(3)), np.arange(3), ca(np.arange(3))),
            "put_along_axis": (ca(np.arange(3)), np.arange(3), ca(np.arange(3)), 0),
            "putmask": (ca(np.arange(3)), np.arange(3), ca(np.arange(3))),
            "searchsorted": (ca(np.arange(3)), ca(np.arange(3))),
            "select": (
                [np.arange(3) < 1, np.arange(3) > 1],
                [ca(np.arange(3)), ca(np.arange(3))],
                cq(1),
            ),
            "setdiff1d": (ca(np.arange(3)), ca(np.arange(3, 6))),
            "sinc": (ca(np.arange(3)),),
            "clip": (ca(np.arange(3)), cq(1), cq(2)),
            "where": (
                ca(np.arange(3)),
                ca(np.arange(3)),
                ca(np.arange(3)),
            ),
            "triu": (ca(np.ones((3, 3))),),
            "tril": (ca(np.ones((3, 3))),),
            "einsum": ("ii->i", ca(np.eye(3))),
            "convolve": (ca(np.arange(3)), ca(np.arange(3))),
            "correlate": (ca(np.arange(3)), ca(np.arange(3))),
            "tensordot": (ca(np.eye(3)), ca(np.eye(3))),
            "unwrap": (ca(np.arange(3)),),
            "interp": (
                ca(np.arange(3)),
                ca(np.arange(3)),
                ca(np.arange(3)),
            ),
            "array_repr": (ca(np.arange(3)),),
            "linalg.outer": (ca(np.arange(3)), ca(np.arange(3))),
            "trapezoid": (ca(np.arange(3)),),
            "isin": (ca(np.arange(3)), ca(np.arange(3))),
            "take": (ca(np.arange(3)), np.arange(3)),
            # FUNCTIONS THAT UNYT DOESN'T HANDLE EXPLICITLY (THEY "JUST WORK"):
            "all": (ca(np.arange(3)),),
            "amax": (ca(np.arange(3)),),  # implemented via max
            "amin": (ca(np.arange(3)),),  # implemented via min
            "angle": (cq(complex(1, 1)),),
            "any": (ca(np.arange(3)),),
            "append": (ca(np.arange(3)), cq(1)),
            "apply_along_axis": (lambda x: x, 0, ca(np.eye(3))),
            "argmax": (ca(np.arange(3)),),  # implemented via max
            "argmin": (ca(np.arange(3)),),  # implemented via min
            "argpartition": (
                ca(np.arange(3)),
                1,
            ),  # implemented via partition
            "argsort": (ca(np.arange(3)),),  # implemented via sort
            "argwhere": (ca(np.arange(3)),),
            "array_str": (ca(np.arange(3)),),
            "atleast_1d": (ca(np.arange(3)),),
            "atleast_2d": (ca(np.arange(3)),),
            "atleast_3d": (ca(np.arange(3)),),
            "average": (ca(np.arange(3)),),
            "can_cast": (ca(np.arange(3)), np.float64),
            "common_type": (ca(np.arange(3)), ca(np.arange(3))),
            "result_type": (ca(np.ones(3)), ca(np.ones(3))),
            "iscomplex": (ca(np.arange(3)),),
            "iscomplexobj": (ca(np.arange(3)),),
            "isreal": (ca(np.arange(3)),),
            "isrealobj": (ca(np.arange(3)),),
            "nan_to_num": (ca(np.arange(3)),),
            "nanargmax": (ca(np.arange(3)),),  # implemented via max
            "nanargmin": (ca(np.arange(3)),),  # implemented via min
            "nanmax": (ca(np.arange(3)),),  # implemented via max
            "nanmean": (ca(np.arange(3)),),  # implemented via mean
            "nanmedian": (ca(np.arange(3)),),  # implemented via median
            "nanmin": (ca(np.arange(3)),),  # implemented via min
            "trim_zeros": (ca(np.arange(3)),),
            "max": (ca(np.arange(3)),),
            "mean": (ca(np.arange(3)),),
            "median": (ca(np.arange(3)),),
            "min": (ca(np.arange(3)),),
            "ndim": (ca(np.arange(3)),),
            "shape": (ca(np.arange(3)),),
            "size": (ca(np.arange(3)),),
            "sort": (ca(np.arange(3)),),
            "sum": (ca(np.arange(3)),),
            "repeat": (ca(np.arange(3)), 2),
            "tile": (ca(np.arange(3)), 2),
            "shares_memory": (ca(np.arange(3)), ca(np.arange(3))),
            "nonzero": (ca(np.arange(3)),),
            "count_nonzero": (ca(np.arange(3)),),
            "flatnonzero": (ca(np.arange(3)),),
            "isneginf": (ca(np.arange(3)),),
            "isposinf": (ca(np.arange(3)),),
            "empty_like": (ca(np.arange(3)),),
            "full_like": (ca(np.arange(3)), cq(1)),
            "ones_like": (ca(np.arange(3)),),
            "zeros_like": (ca(np.arange(3)),),
            "copy": (ca(np.arange(3)),),
            "meshgrid": (ca(np.arange(3)), ca(np.arange(3))),
            "transpose": (ca(np.eye(3)),),
            "reshape": (ca(np.arange(3)), (3,)),
            "resize": (ca(np.arange(3)), 6),
            "roll": (ca(np.arange(3)), 1),
            "rollaxis": (ca(np.arange(3)), 0),
            "rot90": (ca(np.eye(3)),),
            "expand_dims": (ca(np.arange(3)), 0),
            "squeeze": (ca(np.arange(3)),),
            "flip": (ca(np.eye(3)),),
            "fliplr": (ca(np.eye(3)),),
            "flipud": (ca(np.eye(3)),),
            "delete": (ca(np.arange(3)), 0),
            "partition": (ca(np.arange(3)), 1),
            "broadcast_to": (ca(np.arange(3)), 3),
            "broadcast_arrays": (ca(np.arange(3)),),
            "split": (ca(np.arange(3)), 1),
            "array_split": (ca(np.arange(3)), 1),
            "dsplit": (ca(np.arange(27)).reshape(3, 3, 3), 1),
            "hsplit": (ca(np.arange(3)), 1),
            "vsplit": (ca(np.eye(3)), 1),
            "swapaxes": (ca(np.eye(3)), 0, 1),
            "moveaxis": (ca(np.eye(3)), 0, 1),
            "nansum": (ca(np.arange(3)),),  # implemented via sum
            "std": (ca(np.arange(3)),),
            "nanstd": (ca(np.arange(3)),),
            "nanvar": (ca(np.arange(3)),),
            "nanprod": (ca(np.arange(3)),),
            "diag": (ca(np.eye(3)),),
            "diag_indices_from": (ca(np.eye(3)),),
            "diagflat": (ca(np.eye(3)),),
            "diagonal": (ca(np.eye(3)),),
            "ravel": (ca(np.arange(3)),),
            "ravel_multi_index": (np.eye(2, dtype=int), (2, 2)),
            "unravel_index": (np.arange(3), (3,)),
            "fix": (ca(np.arange(3)),),
            "round": (ca(np.arange(3)),),  # implemented via around
            "may_share_memory": (
                ca(np.arange(3)),
                ca(np.arange(3)),
            ),
            "linalg.matrix_power": (ca(np.eye(3)), 2),
            "linalg.cholesky": (ca(np.eye(3)),),
            "linalg.multi_dot": ((ca(np.eye(3)), ca(np.eye(3))),),
            "linalg.matrix_rank": (ca(np.eye(3)),),
            "linalg.qr": (ca(np.eye(3)),),
            "linalg.slogdet": (ca(np.eye(3)),),
            "linalg.cond": (ca(np.eye(3)),),
            "gradient": (ca(np.arange(3)),),
            "cumsum": (ca(np.arange(3)),),
            "nancumsum": (ca(np.arange(3)),),
            "nancumprod": (ca(np.arange(3)),),
            "bincount": (ca(np.arange(3)),),
            "unique": (ca(np.arange(3)),),
            "min_scalar_type": (ca(np.arange(3)),),
            "extract": (0, ca(np.arange(3))),
            "setxor1d": (ca(np.arange(3)), ca(np.arange(3))),
            "lexsort": (ca(np.arange(3)),),
            "digitize": (ca(np.arange(3)), ca(np.arange(3))),
            "tril_indices_from": (ca(np.eye(3)),),
            "triu_indices_from": (ca(np.eye(3)),),
            "imag": (ca(np.arange(3)),),
            "real": (ca(np.arange(3)),),
            "real_if_close": (ca(np.arange(3)),),
            "einsum_path": (
                "ij,jk->ik",
                ca(np.eye(3)),
                ca(np.eye(3)),
            ),
            "cov": (ca(np.arange(3)),),
            "corrcoef": (ca(np.arange(3)),),
            "compress": (np.zeros(3), ca(np.arange(3))),
            "take_along_axis": (ca(np.arange(3)), np.ones(3, dtype=int), 0),
            "linalg.cross": (ca(np.arange(3)), ca(np.arange(3))),
            "linalg.diagonal": (ca(np.eye(3)),),
            "linalg.matmul": (ca(np.eye(3)), ca(np.eye(3))),
            "linalg.matrix_norm": (ca(np.eye(3)),),
            "linalg.matrix_transpose": (ca(np.eye(3)),),
            "linalg.svdvals": (ca(np.eye(3)),),
            "linalg.tensordot": (ca(np.eye(3)), ca(np.eye(3))),
            "linalg.trace": (ca(np.eye(3)),),
            "linalg.vecdot": (ca(np.arange(3)), ca(np.arange(3))),
            "linalg.vector_norm": (ca(np.arange(3)),),
            "astype": (ca(np.arange(3)), float),
            "matrix_transpose": (ca(np.eye(3)),),
            "unique_all": (ca(np.arange(3)),),
            "unique_counts": (ca(np.arange(3)),),
            "unique_inverse": (ca(np.arange(3)),),
            "unique_values": (ca(np.arange(3)),),
            "cumulative_sum": (ca(np.arange(3)),),
            "cumulative_prod": (ca(np.arange(3)),),
            "unstack": (ca(np.arange(3)),),
        }
        if NUMPY_VERSION < Version("2.4.1"):
            functions_to_check["in1d"] = (ca(np.arange(3)), ca(np.arange(3)))
        functions_checked = list()
        bad_funcs = dict()
        for fname, args in functions_to_check.items():
            ua_args = list()
            for arg in args:
                ua_args.append(arg_to_ua(arg))
            func = getfunc(fname)
            try:
                with warnings.catch_warnings():
                    if "savetxt" in fname:
                        warnings.filterwarnings(
                            action="ignore",
                            category=UserWarning,
                            message="numpy.savetxt does not preserve units",
                        )
                    try:
                        ua_result = func(*ua_args)
                    except:
                        print(f"Crashed in {fname} with unyt input.")
                        raise
            except u.exceptions.UnytError:
                raises_unyt_error = True
            else:
                raises_unyt_error = False
            if "savetxt" in fname and os.path.isfile(savetxt_file):
                os.remove(savetxt_file)
            functions_checked.append(func)
            if raises_unyt_error:
                with pytest.raises(u.exceptions.UnytError):
                    result = func(*args)
                continue
            with warnings.catch_warnings():
                if "savetxt" in fname:
                    warnings.filterwarnings(
                        action="ignore",
                        category=UserWarning,
                        message="numpy.savetxt does not preserve units or cosmo",
                    )
                if "unwrap" in fname:
                    # haven't bothered to pass a cosmo_quantity for period
                    warnings.filterwarnings(
                        action="ignore",
                        category=RuntimeWarning,
                        message="Mixing arguments with and without cosmo_factors",
                    )
                try:
                    result = func(*args)
                except:
                    print(f"Crashed in {fname} with cosmo input.")
                    raise
            if fname.split(".")[-1] in (
                "fill_diagonal",
                "copyto",
                "place",
                "put",
                "put_along_axis",
                "putmask",
            ):
                # treat inplace modified values for relevant functions as result
                result = args[0]
                ua_result = ua_args[0]
            if "savetxt" in fname and os.path.isfile(savetxt_file):
                os.remove(savetxt_file)
            ignore_values = fname in {"empty_like"}  # empty_like has arbitrary data
            try:
                check_result(result, ua_result, ignore_values=ignore_values)
            except AssertionError:
                bad_funcs["np." + fname] = result, ua_result
        if len(bad_funcs) > 0:
            raise AssertionError(
                "Some functions did not return expected types "
                "(obtained, obtained with unyt input): " + str(bad_funcs)
            )
        unchecked_functions = [
            f
            for f in set(_HANDLED_FUNCTIONS) | NOOP_FUNCTIONS
            if f not in functions_checked
        ]
        try:
            assert len(unchecked_functions) == 0
        except AssertionError:
            raise AssertionError(
                "Did not check functions",
                [
                    (".".join((f.__module__, f.__name__)).replace("numpy", "np"))
                    for f in unchecked_functions
                ],
            )

    # the combinations of units and cosmo_factors is nonsense but it's just for testing...
    @pytest.mark.parametrize(
        "func_args",
        (
            (
                np.histogram,
                (
                    cosmo_array(
                        [1, 2, 3],
                        u.m,
                        comoving=False,
                        scale_factor=1.0,
                        scale_exponent=1,
                    ),
                ),
            ),
            (
                np.histogram2d,
                (
                    cosmo_array(
                        [1, 2, 3],
                        u.m,
                        comoving=False,
                        scale_factor=1.0,
                        scale_exponent=1,
                    ),
                    cosmo_array(
                        [1, 2, 3],
                        u.K,
                        comoving=False,
                        scale_factor=1.0,
                        scale_exponent=2,
                    ),
                ),
            ),
            (
                np.histogramdd,
                (
                    [
                        cosmo_array(
                            [1, 2, 3],
                            u.m,
                            comoving=False,
                            scale_factor=1.0,
                            scale_exponent=1,
                        ),
                        cosmo_array(
                            [1, 2, 3],
                            u.K,
                            comoving=False,
                            scale_factor=1.0,
                            scale_exponent=2,
                        ),
                        cosmo_array(
                            [1, 2, 3],
                            u.kg,
                            comoving=False,
                            scale_factor=1.0,
                            scale_exponent=3,
                        ),
                    ],
                ),
            ),
        ),
    )
    @pytest.mark.parametrize(
        "weights",
        (
            None,
            cosmo_array(
                [1, 2, 3], u.s, comoving=False, scale_factor=1.0, scale_exponent=1
            ),
            np.array([1, 2, 3]),
        ),
    )
    @pytest.mark.parametrize("bins_type", ("int", "ca"))
    @pytest.mark.parametrize("density", (None, True))
    def test_histograms(self, func_args, weights, bins_type, density):
        """
        Test that histograms give sensible output.

        Histograms are tricky with possible density and weights arguments, and the way
        that attributes need validation and propagation between the bins and values.
        They are also commonly used. They therefore need a bespoke test.
        """
        func, args = func_args
        bins = {
            "int": 10,
            "ca": [
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.kpc,
                    comoving=False,
                    scale_factor=1.0,
                    scale_exponent=1,
                ),
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.K,
                    comoving=False,
                    scale_factor=1.0,
                    scale_exponent=2,
                ),
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.Msun,
                    comoving=False,
                    scale_factor=1.0,
                    scale_exponent=3,
                ),
            ],
        }[bins_type]
        bins = (
            bins[
                {
                    np.histogram: np.s_[0],
                    np.histogram2d: np.s_[:2],
                    np.histogramdd: np.s_[:],
                }[func]
            ]
            if bins_type == "ca"
            else bins
        )
        result = func(*args, bins=bins, density=density, weights=weights)
        ua_args = tuple(
            (
                to_ua(arg)
                if not isinstance(arg, tuple)
                else tuple(to_ua(item) for item in arg)
            )
            for arg in args
        )
        ua_bins = (
            to_ua(bins)
            if not isinstance(bins, tuple)
            else tuple(to_ua(item) for item in bins)
        )
        ua_result = func(
            *ua_args, bins=ua_bins, density=density, weights=to_ua(weights)
        )
        if isinstance(ua_result, tuple):
            assert isinstance(result, tuple)
            assert len(result) == len(ua_result)
            for r, ua_r in zip(result, ua_result):
                check_result(r, ua_r)
        else:
            check_result(result, ua_result)
        if not density and not isinstance(weights, cosmo_array):
            assert not isinstance(result[0], cosmo_array)
        else:
            assert result[0].comoving is False
        if density and not isinstance(weights, cosmo_array):
            assert (
                result[0].cosmo_factor
                == {
                    np.histogram: cosmo_factor(a**-1, 1.0),
                    np.histogram2d: cosmo_factor(a**-3, 1.0),
                    np.histogramdd: cosmo_factor(a**-6, 1.0),
                }[func]
            )
        elif density and isinstance(weights, cosmo_array):
            assert result[0].comoving is False
            assert (
                result[0].cosmo_factor
                == {
                    np.histogram: cosmo_factor(a**0, 1.0),
                    np.histogram2d: cosmo_factor(a**-2, 1.0),
                    np.histogramdd: cosmo_factor(a**-5, 1.0),
                }[func]
            )
        elif not density and isinstance(weights, cosmo_array):
            assert result[0].comoving is False
            assert (
                result[0].cosmo_factor
                == {
                    np.histogram: cosmo_factor(a**1, 1.0),
                    np.histogram2d: cosmo_factor(a**1, 1.0),
                    np.histogramdd: cosmo_factor(a**1, 1.0),
                }[func]
            )
        ret_bins = {
            np.histogram: [result[1]],
            np.histogram2d: result[1:],
            np.histogramdd: result[1],
        }[func]
        for b, expt_cf in zip(
            ret_bins,
            (
                [
                    cosmo_factor(a**1, 1.0),
                    cosmo_factor(a**2, 1.0),
                    cosmo_factor(a**3, 1.0),
                ]
            ),
        ):
            assert b.comoving is False
            assert b.cosmo_factor == expt_cf

    def test_getitem(self):
        """Make sure that we don't degrade to an ndarray on slicing."""
        assert isinstance(ca(np.arange(3))[0], cosmo_quantity)

    def test_reshape_to_scalar(self):
        """Make sure that we convert to a cosmo_quantity when we reshape to a scalar."""
        assert isinstance(ca(np.ones(1)).reshape(tuple()), cosmo_quantity)

    def test_iter(self):
        """Make sure that we get cosmo_quantity's when iterating over a cosmo_array."""
        for cq in ca(np.arange(3)):
            assert isinstance(cq, cosmo_quantity)

    def test_dot(self):
        """Make sure that we get a cosmo_array when we use array attribute dot."""
        res = ca(np.arange(3)).dot(ca(np.arange(3)))
        assert isinstance(res, cosmo_quantity)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**2, 0.5)
        assert res.valid_transform is True

    def test_average_with_returned(self):
        """Make sure that sum of weights in numpy's average gets cosmo attributes."""
        # regression test for https://github.com/SWIFTSIM/swiftsimio/issues/285
        x = cosmo_array(
            np.arange(9).reshape((3, 3)), u.kpc, scale_factor=1.0, scale_exponent=1.0
        )
        w = cosmo_array(np.arange(3), u.solMass, scale_factor=1.0, scale_exponent=0.0)
        avg, wsum = np.average(x, weights=w, axis=-1, returned=True)
        assert avg.cosmo_factor == x.cosmo_factor
        assert wsum.cosmo_factor == w.cosmo_factor


class TestCosmoQuantity:
    """
    Test that the cosmo_quantity class works as desired.

    Mostly test around issues converting back and forth with cosmo_array.
    """

    @pytest.mark.parametrize(
        "func, args",
        [
            ("astype", (float,)),
            ("in_units", (u.m,)),
            ("byteswap", tuple()),
            ("compress", ([True],)),
            ("flatten", tuple()),
            ("ravel", tuple()),
            ("repeat", (1,)),
            ("reshape", (1,)),
            ("take", ([0],)),
            ("transpose", tuple()),
            ("view", tuple()),
        ],
    )
    def test_propagation_func(self, func, args):
        """Test that functions that are supposed to propagate our attributes do so."""
        cq = cosmo_quantity(
            1,
            u.m,
            comoving=False,
            scale_factor=1.0,
            scale_exponent=1,
            valid_transform=True,
        )
        res = getattr(cq, func)(*args)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**1, 1.0)
        assert res.valid_transform is True

    def test_round(self):
        """Test that attributes propagate through the round builtin."""
        cq = cosmo_quantity(
            1.03,
            u.m,
            comoving=False,
            scale_factor=1.0,
            scale_exponent=1,
            valid_transform=True,
        )
        res = round(cq)
        assert res.value == 1.0
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**1, 1.0)
        assert res.valid_transform is True

    def test_scalar_return_func(self):
        """
        Make sure that default-wrapped functions convert to a cosmo_quantity.

        Many functions behave similarly. Check that the ones that take a
        cosmo_array and return a scalar convert to a cosmo_quantity.
        """
        ca = cosmo_array(
            np.arange(3),
            u.m,
            comoving=False,
            scale_factor=1.0,
            scale_exponent=1,
            valid_transform=True,
        )
        res = np.min(ca)
        assert isinstance(res, cosmo_quantity)

    @pytest.mark.parametrize("prop", ["T", "ua", "unit_array"])
    def test_propagation_props(self, prop):
        """Test that properties propagate our attributes as intended."""
        cq = cosmo_quantity(
            1,
            u.m,
            comoving=False,
            scale_factor=1.0,
            scale_exponent=1,
            valid_transform=True,
        )
        res = getattr(cq, prop)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**1, 1.0)
        assert res.valid_transform is True

    def test_multiply_quantities(self):
        """Test multiplying two quantities."""
        cq = cosmo_quantity(
            2,
            u.m,
            comoving=False,
            scale_factor=0.5,
            scale_exponent=1,
            valid_transform=True,
        )
        multiplied = cq * cq
        assert type(multiplied) is cosmo_quantity
        assert multiplied.comoving is False
        assert multiplied.cosmo_factor == cosmo_factor(a**2, 0.5)
        assert multiplied.to_value(u.m**2) == 4

    def test_squeeze_to_quantity(self):
        """Test that squeezing to a scalar returns a cosmo_quantity."""
        ca = cosmo_array(
            [1],
            u.m,
            comoving=True,
            scale_factor=0.5,
            scale_exponent=1,
            valid_transform=True,
        )
        assert ca.squeeze().ndim == 0
        assert isinstance(ca.squeeze(), cosmo_quantity)


class TestCosmoArrayCopy:
    """Tests of explicit (deep)copying of cosmo_array."""

    def test_copy(self):
        """Check that when we copy values and attributes are preserved."""
        units = u.Mpc
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=units),
            scale_factor=1.0,
            scale_exponent=1,
            comoving=False,
        )
        copy_arr = copy(arr)
        assert np.allclose(arr.to_value(units), copy_arr.to_value(units))
        assert arr.units == copy_arr.units
        assert arr.cosmo_factor == copy_arr.cosmo_factor
        assert arr.comoving == copy_arr.comoving

    def test_deepcopy(self):
        """Check that when we deepcopy values and attributes are preserved."""
        units = u.Mpc
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=units),
            scale_factor=1.0,
            scale_exponent=1,
            comoving=False,
        )
        copy_arr = deepcopy(arr)
        assert np.allclose(arr.to_value(units), copy_arr.to_value(units))
        assert arr.units == copy_arr.units
        assert arr.cosmo_factor == copy_arr.cosmo_factor
        assert arr.comoving == copy_arr.comoving

    def test_to_cgs(self):
        """Check that using to_cgs properly preserves attributes."""
        units = u.Mpc
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=units),
            scale_factor=1.0,
            scale_exponent=1,
            comoving=False,
        )
        cgs_arr = arr.in_cgs()
        assert np.allclose(arr.to_value(u.cm), cgs_arr.to_value(u.cm))
        assert cgs_arr.units == u.cm
        assert cgs_arr.cosmo_factor == arr.cosmo_factor
        assert cgs_arr.comoving == arr.comoving


class TestMultiplicationByUnyt:
    """Tests for multiplying cosmo_array by a unyt unit."""

    def test_multiplication_by_unyt(self):
        """
        Check that left-sided multiplication behaves itself.

        We desire consistent behaviour for example for `cosmo_array(...) * (1 * u.Mpc)` as
        for `cosmo_array(...) * u.Mpc`.
        """
        ca = cosmo_array(
            np.ones(3), u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1
        )
        # required so that can test right-sided division with the same assertions:
        assert np.allclose(ca.to_value(ca.units), 1)
        # the reference result:
        multiplied_by_quantity = ca * (1 * u.Mpc)  # parentheses very important here
        # get the same result twice through left-sided multiplication and division:
        lmultiplied_by_unyt = ca * u.Mpc
        ldivided_by_unyt = ca / u.Mpc**-1
        # and twice more through right-sided multiplication and division:
        rmultiplied_by_unyt = u.Mpc * ca
        rdivided_by_unyt = u.Mpc**3 / ca

        for multiplied_by_unyt in (
            lmultiplied_by_unyt,
            ldivided_by_unyt,
            rmultiplied_by_unyt,
            rdivided_by_unyt,
        ):
            assert isinstance(multiplied_by_quantity, cosmo_array)
            assert isinstance(multiplied_by_unyt, cosmo_array)
            assert np.allclose(
                multiplied_by_unyt.to_value(multiplied_by_quantity.units),
                multiplied_by_quantity.to_value(multiplied_by_quantity.units),
            )


class TestPickle:
    """Test that the cosmo_array attributes survive being pickled and unpickled."""

    def test_pickle(self):
        """Test that the cosmo_array attributes survive pickle/unpickle."""
        attrs = {
            "comoving": False,
            "cosmo_factor": cosmo_factor(a**1, 0.5),
            "compression": "FMantissa9",
            "valid_transform": True,
        }
        ca = cosmo_array([123, 456], u.Mpc, **attrs)
        try:
            with open("ca.pkl", "wb") as pickle_handle:
                pickle.dump(ca, pickle_handle)
            with open("ca.pkl", "rb") as pickle_handle:
                unpickled_ca = pickle.load(pickle_handle)
        finally:
            if os.path.isfile("ca.pkl"):
                os.remove("ca.pkl")
        for attr_name, attr_value in attrs.items():
            assert getattr(unpickled_ca, attr_name) == attr_value
