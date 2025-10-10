"""
Tests the initialisation of a cosmo_array.
"""

import pytest
import os
import warnings
import numpy as np
import unyt as u
from copy import copy, deepcopy
import pickle
from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor, a

savetxt_file = "saved_array.txt"


def getfunc(fname):
    """
    Helper for our tests: get the function handle from a name (possibly with attribute
    access).
    """
    func = np
    for attr in fname.split("."):
        func = getattr(func, attr)
    return func


def cosmo_object(x, unit=u.Mpc):
    """
    Helper for our tests: turn an array into a cosmo_array.
    """
    return cosmo_array(x, unit, comoving=False, scale_factor=0.5, scale_exponent=1)


def cq(x, unit=u.Mpc):
    """
    Helper for our tests: turn a scalar into a cosmo_quantity.
    """
    return cosmo_quantity(x, unit, comoving=False, scale_factor=0.5, scale_exponent=1)


def arg_to_ua(arg):
    """
    Helper for our tests: recursively convert cosmo_* in an argument (possibly an
    iterable) to their unyt_* equivalents.
    """
    if type(arg) in (list, tuple):
        return type(arg)([arg_to_ua(a) for a in arg])
    else:
        return to_ua(arg)


def to_ua(x):
    """
    Helper for our tests: turn a cosmo_* object into its unyt_* equivalent.
    """
    return u.unyt_array(x) if hasattr(x, "comoving") else x


def check_result(x_c, x_u, ignore_values=False):
    """
    Helper for our tests: check that a result with cosmo input matches what we
    expected based on the result with unyt input.

    We check:
     - that the type of the result makes sense, recursing if needed.
     - that the value of the result matches (unless ignore_values=False).
     - that the units match.
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
    """
    Test different ways of initializing a cosmo_array.
    """

    def test_init_from_ndarray(self):
        """
        Check initializing from a bare numpy array.
        """
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
        """
        Check initializing from a list of values.
        """
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
        """
        Check initializing from a unyt_array.
        """
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
        """
        cosmo_array data must be iterable (scalar handled by cosmo_quantity).
        """
        with pytest.raises(ValueError, match="cosmo_array data must be iterable"):
            cosmo_array(0, u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1)
        # make sure inheriting class doesn't do anything silly:
        cosmo_quantity(0, u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1)


class TestNumpyFunctions:
    """
    Check that numpy functions recognize our cosmo classes as input and handle them
    correctly.
    """

    def test_explicitly_handled_funcs(self):
        """
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
            "array2string": (cosmo_object(np.arange(3)),),
            "dot": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "vdot": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "inner": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "outer": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "kron": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "histogram_bin_edges": (cosmo_object(np.arange(3)),),
            "linalg.inv": (cosmo_object(np.eye(3)),),
            "linalg.tensorinv": (cosmo_object(np.eye(9).reshape((3, 3, 3, 3))),),
            "linalg.pinv": (cosmo_object(np.eye(3)),),
            "linalg.svd": (cosmo_object(np.eye(3)),),
            "histogram": (cosmo_object(np.arange(3)),),
            "histogram2d": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "histogramdd": (cosmo_object(np.arange(3)).reshape((1, 3)),),
            "concatenate": (cosmo_object(np.eye(3)),),
            "cross": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "intersect1d": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "union1d": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "linalg.norm": (cosmo_object(np.arange(3)),),
            "vstack": (cosmo_object(np.arange(3)),),
            "hstack": (cosmo_object(np.arange(3)),),
            "dstack": (cosmo_object(np.arange(3)),),
            "column_stack": (cosmo_object(np.arange(3)),),
            "stack": (cosmo_object(np.arange(3)),),
            "around": (cosmo_object(np.arange(3)),),
            "block": ([[cosmo_object(np.arange(3))], [cosmo_object(np.arange(3))]],),
            "fft.fft": (cosmo_object(np.arange(3)),),
            "fft.fft2": (cosmo_object(np.eye(3)),),
            "fft.fftn": (cosmo_object(np.arange(3)),),
            "fft.hfft": (cosmo_object(np.arange(3)),),
            "fft.rfft": (cosmo_object(np.arange(3)),),
            "fft.rfft2": (cosmo_object(np.eye(3)),),
            "fft.rfftn": (cosmo_object(np.arange(3)),),
            "fft.ifft": (cosmo_object(np.arange(3)),),
            "fft.ifft2": (cosmo_object(np.eye(3)),),
            "fft.ifftn": (cosmo_object(np.arange(3)),),
            "fft.ihfft": (cosmo_object(np.arange(3)),),
            "fft.irfft": (cosmo_object(np.arange(3)),),
            "fft.irfft2": (cosmo_object(np.eye(3)),),
            "fft.irfftn": (cosmo_object(np.arange(3)),),
            "fft.fftshift": (cosmo_object(np.arange(3)),),
            "fft.ifftshift": (cosmo_object(np.arange(3)),),
            "sort_complex": (cosmo_object(np.arange(3)),),
            "isclose": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "allclose": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "array_equal": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "array_equiv": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "linspace": (cq(1), cq(2)),
            "logspace": (cq(1, unit=u.dimensionless), cq(2, unit=u.dimensionless)),
            "geomspace": (cq(1), cq(1)),
            "copyto": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "prod": (cosmo_object(np.arange(3)),),
            "var": (cosmo_object(np.arange(3)),),
            "trace": (cosmo_object(np.eye(3)),),
            "percentile": (cosmo_object(np.arange(3)), 30),
            "quantile": (cosmo_object(np.arange(3)), 0.3),
            "nanpercentile": (cosmo_object(np.arange(3)), 30),
            "nanquantile": (cosmo_object(np.arange(3)), 0.3),
            "linalg.det": (cosmo_object(np.eye(3)),),
            "diff": (cosmo_object(np.arange(3)),),
            "ediff1d": (cosmo_object(np.arange(3)),),
            "ptp": (cosmo_object(np.arange(3)),),
            "cumprod": (cosmo_object(np.arange(3)),),
            "pad": (cosmo_object(np.arange(3)), 3),
            "choose": (np.arange(3), cosmo_object(np.eye(3))),
            "insert": (cosmo_object(np.arange(3)), 1, cq(1)),
            "linalg.lstsq": (cosmo_object(np.eye(3)), cosmo_object(np.eye(3))),
            "linalg.solve": (cosmo_object(np.eye(3)), cosmo_object(np.eye(3))),
            "linalg.tensorsolve": (
                cosmo_object(np.eye(24).reshape((6, 4, 2, 3, 4))),
                cosmo_object(np.ones((6, 4))),
            ),
            "linalg.eig": (cosmo_object(np.eye(3)),),
            "linalg.eigh": (cosmo_object(np.eye(3)),),
            "linalg.eigvals": (cosmo_object(np.eye(3)),),
            "linalg.eigvalsh": (cosmo_object(np.eye(3)),),
            "savetxt": (savetxt_file, cosmo_object(np.arange(3))),
            "fill_diagonal": (cosmo_object(np.eye(3)), cosmo_object(np.arange(3))),
            "apply_over_axes": (lambda x, axis: x, cosmo_object(np.eye(3)), (0, 1)),
            "isin": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "place": (
                cosmo_object(np.arange(3)),
                np.arange(3) > 0,
                cosmo_object(np.arange(3)),
            ),
            "put": (
                cosmo_object(np.arange(3)),
                np.arange(3),
                cosmo_object(np.arange(3)),
            ),
            "put_along_axis": (
                cosmo_object(np.arange(3)),
                np.arange(3),
                cosmo_object(np.arange(3)),
                0,
            ),
            "putmask": (
                cosmo_object(np.arange(3)),
                np.arange(3),
                cosmo_object(np.arange(3)),
            ),
            "searchsorted": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "select": (
                [np.arange(3) < 1, np.arange(3) > 1],
                [cosmo_object(np.arange(3)), cosmo_object(np.arange(3))],
                cq(1),
            ),
            "setdiff1d": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3, 6))),
            "sinc": (cosmo_object(np.arange(3)),),
            "clip": (cosmo_object(np.arange(3)), cq(1), cq(2)),
            "where": (
                cosmo_object(np.arange(3)),
                cosmo_object(np.arange(3)),
                cosmo_object(np.arange(3)),
            ),
            "triu": (cosmo_object(np.ones((3, 3))),),
            "tril": (cosmo_object(np.ones((3, 3))),),
            "einsum": ("ii->i", cosmo_object(np.eye(3))),
            "convolve": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "correlate": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "tensordot": (cosmo_object(np.eye(3)), cosmo_object(np.eye(3))),
            "unwrap": (cosmo_object(np.arange(3)),),
            "interp": (
                cosmo_object(np.arange(3)),
                cosmo_object(np.arange(3)),
                cosmo_object(np.arange(3)),
            ),
            "array_repr": (cosmo_object(np.arange(3)),),
            "linalg.outer": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "trapezoid": (cosmo_object(np.arange(3)),),
            "in1d": (
                cosmo_object(np.arange(3)),
                cosmo_object(np.arange(3)),
            ),  # np deprecated
            "take": (cosmo_object(np.arange(3)), np.arange(3)),
            # FUNCTIONS THAT UNYT DOESN'T HANDLE EXPLICITLY (THEY "JUST WORK"):
            "all": (cosmo_object(np.arange(3)),),
            "amax": (cosmo_object(np.arange(3)),),  # implemented via max
            "amin": (cosmo_object(np.arange(3)),),  # implemented via min
            "angle": (cq(complex(1, 1)),),
            "any": (cosmo_object(np.arange(3)),),
            "append": (cosmo_object(np.arange(3)), cq(1)),
            "apply_along_axis": (lambda x: x, 0, cosmo_object(np.eye(3))),
            "argmax": (cosmo_object(np.arange(3)),),  # implemented via max
            "argmin": (cosmo_object(np.arange(3)),),  # implemented via min
            "argpartition": (
                cosmo_object(np.arange(3)),
                1,
            ),  # implemented via partition
            "argsort": (cosmo_object(np.arange(3)),),  # implemented via sort
            "argwhere": (cosmo_object(np.arange(3)),),
            "array_str": (cosmo_object(np.arange(3)),),
            "atleast_1d": (cosmo_object(np.arange(3)),),
            "atleast_2d": (cosmo_object(np.arange(3)),),
            "atleast_3d": (cosmo_object(np.arange(3)),),
            "average": (cosmo_object(np.arange(3)),),
            "can_cast": (cosmo_object(np.arange(3)), np.float64),
            "common_type": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "result_type": (cosmo_object(np.ones(3)), cosmo_object(np.ones(3))),
            "iscomplex": (cosmo_object(np.arange(3)),),
            "iscomplexobj": (cosmo_object(np.arange(3)),),
            "isreal": (cosmo_object(np.arange(3)),),
            "isrealobj": (cosmo_object(np.arange(3)),),
            "nan_to_num": (cosmo_object(np.arange(3)),),
            "nanargmax": (cosmo_object(np.arange(3)),),  # implemented via max
            "nanargmin": (cosmo_object(np.arange(3)),),  # implemented via min
            "nanmax": (cosmo_object(np.arange(3)),),  # implemented via max
            "nanmean": (cosmo_object(np.arange(3)),),  # implemented via mean
            "nanmedian": (cosmo_object(np.arange(3)),),  # implemented via median
            "nanmin": (cosmo_object(np.arange(3)),),  # implemented via min
            "trim_zeros": (cosmo_object(np.arange(3)),),
            "max": (cosmo_object(np.arange(3)),),
            "mean": (cosmo_object(np.arange(3)),),
            "median": (cosmo_object(np.arange(3)),),
            "min": (cosmo_object(np.arange(3)),),
            "ndim": (cosmo_object(np.arange(3)),),
            "shape": (cosmo_object(np.arange(3)),),
            "size": (cosmo_object(np.arange(3)),),
            "sort": (cosmo_object(np.arange(3)),),
            "sum": (cosmo_object(np.arange(3)),),
            "repeat": (cosmo_object(np.arange(3)), 2),
            "tile": (cosmo_object(np.arange(3)), 2),
            "shares_memory": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "nonzero": (cosmo_object(np.arange(3)),),
            "count_nonzero": (cosmo_object(np.arange(3)),),
            "flatnonzero": (cosmo_object(np.arange(3)),),
            "isneginf": (cosmo_object(np.arange(3)),),
            "isposinf": (cosmo_object(np.arange(3)),),
            "empty_like": (cosmo_object(np.arange(3)),),
            "full_like": (cosmo_object(np.arange(3)), cq(1)),
            "ones_like": (cosmo_object(np.arange(3)),),
            "zeros_like": (cosmo_object(np.arange(3)),),
            "copy": (cosmo_object(np.arange(3)),),
            "meshgrid": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "transpose": (cosmo_object(np.eye(3)),),
            "reshape": (cosmo_object(np.arange(3)), (3,)),
            "resize": (cosmo_object(np.arange(3)), 6),
            "roll": (cosmo_object(np.arange(3)), 1),
            "rollaxis": (cosmo_object(np.arange(3)), 0),
            "rot90": (cosmo_object(np.eye(3)),),
            "expand_dims": (cosmo_object(np.arange(3)), 0),
            "squeeze": (cosmo_object(np.arange(3)),),
            "flip": (cosmo_object(np.eye(3)),),
            "fliplr": (cosmo_object(np.eye(3)),),
            "flipud": (cosmo_object(np.eye(3)),),
            "delete": (cosmo_object(np.arange(3)), 0),
            "partition": (cosmo_object(np.arange(3)), 1),
            "broadcast_to": (cosmo_object(np.arange(3)), 3),
            "broadcast_arrays": (cosmo_object(np.arange(3)),),
            "split": (cosmo_object(np.arange(3)), 1),
            "array_split": (cosmo_object(np.arange(3)), 1),
            "dsplit": (cosmo_object(np.arange(27)).reshape(3, 3, 3), 1),
            "hsplit": (cosmo_object(np.arange(3)), 1),
            "vsplit": (cosmo_object(np.eye(3)), 1),
            "swapaxes": (cosmo_object(np.eye(3)), 0, 1),
            "moveaxis": (cosmo_object(np.eye(3)), 0, 1),
            "nansum": (cosmo_object(np.arange(3)),),  # implemented via sum
            "std": (cosmo_object(np.arange(3)),),
            "nanstd": (cosmo_object(np.arange(3)),),
            "nanvar": (cosmo_object(np.arange(3)),),
            "nanprod": (cosmo_object(np.arange(3)),),
            "diag": (cosmo_object(np.eye(3)),),
            "diag_indices_from": (cosmo_object(np.eye(3)),),
            "diagflat": (cosmo_object(np.eye(3)),),
            "diagonal": (cosmo_object(np.eye(3)),),
            "ravel": (cosmo_object(np.arange(3)),),
            "ravel_multi_index": (np.eye(2, dtype=int), (2, 2)),
            "unravel_index": (np.arange(3), (3,)),
            "fix": (cosmo_object(np.arange(3)),),
            "round": (cosmo_object(np.arange(3)),),  # implemented via around
            "may_share_memory": (
                cosmo_object(np.arange(3)),
                cosmo_object(np.arange(3)),
            ),
            "linalg.matrix_power": (cosmo_object(np.eye(3)), 2),
            "linalg.cholesky": (cosmo_object(np.eye(3)),),
            "linalg.multi_dot": ((cosmo_object(np.eye(3)), cosmo_object(np.eye(3))),),
            "linalg.matrix_rank": (cosmo_object(np.eye(3)),),
            "linalg.qr": (cosmo_object(np.eye(3)),),
            "linalg.slogdet": (cosmo_object(np.eye(3)),),
            "linalg.cond": (cosmo_object(np.eye(3)),),
            "gradient": (cosmo_object(np.arange(3)),),
            "cumsum": (cosmo_object(np.arange(3)),),
            "nancumsum": (cosmo_object(np.arange(3)),),
            "nancumprod": (cosmo_object(np.arange(3)),),
            "bincount": (cosmo_object(np.arange(3)),),
            "unique": (cosmo_object(np.arange(3)),),
            "min_scalar_type": (cosmo_object(np.arange(3)),),
            "extract": (0, cosmo_object(np.arange(3))),
            "setxor1d": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "lexsort": (cosmo_object(np.arange(3)),),
            "digitize": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "tril_indices_from": (cosmo_object(np.eye(3)),),
            "triu_indices_from": (cosmo_object(np.eye(3)),),
            "imag": (cosmo_object(np.arange(3)),),
            "real": (cosmo_object(np.arange(3)),),
            "real_if_close": (cosmo_object(np.arange(3)),),
            "einsum_path": (
                "ij,jk->ik",
                cosmo_object(np.eye(3)),
                cosmo_object(np.eye(3)),
            ),
            "cov": (cosmo_object(np.arange(3)),),
            "corrcoef": (cosmo_object(np.arange(3)),),
            "compress": (np.zeros(3), cosmo_object(np.arange(3))),
            "take_along_axis": (cosmo_object(np.arange(3)), np.ones(3, dtype=int), 0),
            "linalg.cross": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "linalg.diagonal": (cosmo_object(np.eye(3)),),
            "linalg.matmul": (cosmo_object(np.eye(3)), cosmo_object(np.eye(3))),
            "linalg.matrix_norm": (cosmo_object(np.eye(3)),),
            "linalg.matrix_transpose": (cosmo_object(np.eye(3)),),
            "linalg.svdvals": (cosmo_object(np.eye(3)),),
            "linalg.tensordot": (cosmo_object(np.eye(3)), cosmo_object(np.eye(3))),
            "linalg.trace": (cosmo_object(np.eye(3)),),
            "linalg.vecdot": (cosmo_object(np.arange(3)), cosmo_object(np.arange(3))),
            "linalg.vector_norm": (cosmo_object(np.arange(3)),),
            "astype": (cosmo_object(np.arange(3)), float),
            "matrix_transpose": (cosmo_object(np.eye(3)),),
            "unique_all": (cosmo_object(np.arange(3)),),
            "unique_counts": (cosmo_object(np.arange(3)),),
            "unique_inverse": (cosmo_object(np.arange(3)),),
            "unique_values": (cosmo_object(np.arange(3)),),
            "cumulative_sum": (cosmo_object(np.arange(3)),),
            "cumulative_prod": (cosmo_object(np.arange(3)),),
            "unstack": (cosmo_object(np.arange(3)),),
        }
        functions_checked = list()
        bad_funcs = dict()
        for fname, args in functions_to_check.items():
            # ----- this is to be removed ------
            # ---- see test_block_is_broken ----
            if fname == "block":
                # we skip this function due to issue in unyt with unreleased fix
                functions_checked.append(np.block)
                continue
            # ----------------------------------
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

    @pytest.mark.xfail
    def test_block_is_broken(self):
        """
        There is an issue in unyt affecting np.block and fixed in
        https://github.com/yt-project/unyt/pull/571

        When this fix is released:
        - This test will unexpectedly pass (instead of xfailing).
        - Remove lines flagged with a comment in `test_explicitly_handled_funcs`.
        - Remove this test.
        """
        assert isinstance(
            np.block([[cosmo_object(np.arange(3))], [cosmo_object(np.arange(3))]]),
            cosmo_array,
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
    @pytest.mark.parametrize("bins_type", ("int", "np", "cosmo_object"))
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
            "np": [np.linspace(0, 5, 11)] * 3,
            "cosmo_object": [
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
            if bins_type in ("np", "cosmo_object")
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
        """
        Make sure that we don't degrade to an ndarray on slicing.
        """
        assert isinstance(cosmo_object(np.arange(3))[0], cosmo_quantity)

    def test_reshape_to_scalar(self):
        """
        Make sure that we convert to a cosmo_quantity when we reshape to a scalar.
        """
        assert isinstance(cosmo_object(np.ones(1)).reshape(tuple()), cosmo_quantity)

    def test_iter(self):
        """
        Make sure that we get cosmo_quantity's when iterating over a cosmo_array.
        """
        for cq in cosmo_object(np.arange(3)):
            assert isinstance(cq, cosmo_quantity)

    def test_dot(self):
        """
        Make sure that we get a cosmo_array when we use array attribute dot.
        """
        res = cosmo_object(np.arange(3)).dot(cosmo_object(np.arange(3)))
        assert isinstance(res, cosmo_quantity)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**2, 0.5)
        assert res.valid_transform is True


class TestCosmoQuantity:
    """
    Test that the cosmo_quantity class works as desired, mostly around issues converting
    back and forth with cosmo_array.
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
        """
        Test that functions that are supposed to propagate our attributes do so.
        """
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
        """
        Test that attributes propagate through the round builtin.
        """
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
        Make sure that default-wrapped functions that take a cosmo_array and return a
        scalar convert to a cosmo_quantity.
        """
        cosmo_object = cosmo_array(
            np.arange(3),
            u.m,
            comoving=False,
            scale_factor=1.0,
            scale_exponent=1,
            valid_transform=True,
        )
        res = np.min(cosmo_object)
        assert isinstance(res, cosmo_quantity)

    @pytest.mark.parametrize("prop", ["T", "ua", "unit_array"])
    def test_propagation_props(self, prop):
        """
        Test that properties propagate our attributes as intended.
        """
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
        """
        Test multiplying two quantities.
        """
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


class TestCosmoArrayCopy:
    """
    Tests of explicit (deep)copying of cosmo_array.
    """

    def test_copy(self):
        """
        Check that when we copy a cosmo_array it preserves its values and attributes.
        """
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
        """
        Check that when we deepcopy a cosmo_array it preserves its values and attributes
        """
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
        """
        Check that using to_cgs properly preserves attributes.
        """
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
    @pytest.mark.parametrize(
        "cosmo_object",
        [
            cosmo_array(
                np.ones(3), u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1
            ),
            cosmo_quantity(
                np.ones(1), u.Mpc, comoving=True, scale_factor=1.0, scale_exponent=1
            ),
        ],
    )
    def test_multiplication_by_unyt(self, cosmo_object):
        """
        We desire consistent behaviour for example for `cosmo_array(...) * (1 * u.Mpc)` as
        for `cosmo_array(...) * u.Mpc`.
        """

        lmultiplied_by_quantity = cosmo_object * (
            1 * u.Mpc
        )  # parentheses very important here
        lmultiplied_by_unyt = cosmo_object * u.Mpc
        assert isinstance(lmultiplied_by_quantity, cosmo_array)
        assert isinstance(lmultiplied_by_unyt, cosmo_array)
        assert lmultiplied_by_unyt.comoving == lmultiplied_by_quantity.comoving
        assert np.allclose(
            lmultiplied_by_unyt.to_value(lmultiplied_by_quantity.units),
            lmultiplied_by_quantity.to_value(lmultiplied_by_quantity.units),
        )

        ldivided_by_quantity = cosmo_object / (
            1 * u.Mpc
        )  # parentheses very important here
        ldivided_by_unyt = cosmo_object / u.Mpc
        assert isinstance(ldivided_by_quantity, cosmo_array)
        assert isinstance(ldivided_by_unyt, cosmo_array)
        assert ldivided_by_unyt.comoving == ldivided_by_quantity.comoving
        assert np.allclose(
            ldivided_by_unyt.to_value(ldivided_by_quantity.units),
            ldivided_by_quantity.to_value(ldivided_by_quantity.units),
        )

        rmultiplied_by_quantity = (
            1 * u.Mpc
        ) * cosmo_object  # parentheses very important here
        rmultiplied_by_unyt = u.Mpc * cosmo_object
        rdivided_by_quantity = (
            1 * u.Mpc
        ) / cosmo_object  # parentheses very important here
        rdivided_by_unyt = u.Mpc / cosmo_object
        assert rmultiplied_by_quantity.comoving
        for multiplied_by_unyt in (rmultiplied_by_unyt, rdivided_by_unyt):
            assert isinstance(multiplied_by_quantity, cosmo_array)
            assert isinstance(multiplied_by_unyt, cosmo_array)
            assert multiplied_by_unyt.comoving == multiplied_by_quantity.comoving
            assert np.allclose(
                multiplied_by_unyt.to_value(multiplied_by_quantity.units),
                multiplied_by_quantity.to_value(multiplied_by_quantity.units),
            )


class TestPickle:
    """
    Test that the cosmo_array attributes survive being pickled and unpickled.
    """

    def test_pickle(self):
        attrs = {
            "comoving": False,
            "cosmo_factor": cosmo_factor(a ** 1, 0.5),
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
