"""
Tests the initialisation of a cosmo_array.
"""

import pytest
import os
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor, a

savetxt_file = "saved_array.txt"


def getfunc(fname):
    func = np
    for attr in fname.split("."):
        func = getattr(func, attr)
    return func


def ca(x, unit=u.Mpc):
    return cosmo_array(x, unit, comoving=False, cosmo_factor=cosmo_factor(a ** 1, 0.5))


def to_ua(x):
    return u.unyt_array(x.to_value(x.units), x.units) if hasattr(x, "comoving") else x


def check_result(x_c, x_u):
    if isinstance(x_u, str):
        assert isinstance(x_c, str)
        return
    # careful, unyt_quantity is a subclass of unyt_array
    if isinstance(x_u, u.unyt_quantity):
        assert isinstance(x_c, cosmo_quantity)
    elif isinstance(x_u, u.unyt_array):
        assert isinstance(x_c, cosmo_array) and not isinstance(x_c, cosmo_quantity)
    else:
        assert not isinstance(x_c, cosmo_array)
        assert np.allclose(x_c, x_u)
        return
    assert x_c.units == x_u.units
    assert np.allclose(x_c.to_value(x_c.units), x_u.to_value(x_u.units))
    return


class TestCosmoArrayInit:
    def test_init_from_ndarray(self):
        arr = cosmo_array(
            np.ones(5),
            units=u.Mpc,
            cosmo_factor=cosmo_factor(a ** 1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list(self):
        arr = cosmo_array(
            [1, 1, 1, 1, 1],
            units=u.Mpc,
            cosmo_factor=cosmo_factor(a ** 1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_unyt_array(self):
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=u.Mpc),
            cosmo_factor=cosmo_factor(a ** 1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list_of_unyt_arrays(self):
        arr = cosmo_array(
            [u.unyt_array(1, units=u.Mpc) for _ in range(5)],
            cosmo_factor=cosmo_factor(a ** 1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)


class TestNumpyFunctions:
    """
    Functions specially handled by unyt risk silently casting to unyt_array/unyt_quantity.
    """

    def test_handled_funcs(self):
        from unyt._array_functions import _HANDLED_FUNCTIONS

        functions_to_check = {
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
            # "histogramdd": (ca(np.arange(3)),),
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
            # "linspace": (ca(1), ca(2), 10),
            # "logspace": (ca(1, unit=u.dimensionless), ca(2, unit=u.dimensionless), 10),
            # "geomspace": (ca(1), ca(1), 10),
            "prod": (ca(np.arange(3)),),
            # "var": (ca(np.arange(3)),),
            "trace": (ca(np.eye(3)),),
            # "percentile": (ca(np.arange(3)), 30),
            # "quantile": (ca(np.arange(3)), 0.3),
            # "nanpercentile": (ca(np.arange(3)), 30),
            # "nanquantile": (ca(np.arange(3)), 0.3),
            # "diff": (ca(np.arange(3)),),
            # "ediff1d": (ca(np.arange(3)),),
            # "ptp": (ca(np.arange(3)),),
            "cumprod": (ca(np.arange(3)),),
            # "pad": (ca(np.arange(3)), 3),
            # "choose": (np.arange(3), ca(np.eye(3))),
            # "insert": (ca(np.arange(3)), 1, ca(1)),
            # "isin": (ca(np.arange(3)), ca(np.arange(3))),
            # "in1d": (ca(np.arange(3)), ca(np.arange(3))),
            # "interp": (ca(np.arange(3)), ca(np.arange(3)), ca(np.arange(3))),
            # "searchsorted": (ca(np.arange(3)), ca(np.arange(3))),
            # "select": (
            #     [np.arange(3) < 1, np.arange(3) > 1],
            #     [ca(np.arange(3)), ca(np.arange(3))],
            #     ca(1),
            # ),
            # "setdiff1d": (ca(np.arange(3)), ca(np.arange(3, 6))),
            # "sinc": (ca(np.arange(3)),),
            # "clip": (ca(np.arange(3)), ca(1), ca(2)),
            # "where": (ca(np.arange(3)), ca(np.arange(3)), ca(np.arange(3))),
            # "triu": (ca(np.arange(3)),),
            # "tril": (ca(np.arange(3)),),
            # "einsum": ("ii->i", ca(np.eye(3))),
            # "convolve": (ca(np.arange(3)), ca(np.arange(3))),
            # "correlate": (ca(np.arange(3)), ca(np.arange(3))),
            # "tensordot": (ca(np.eye(3)), ca(np.eye(3))),
            # "unwrap": (ca(np.arange(3)),),
            # "linalg.det": (ca(np.eye(3)),),
            # "linalg.outer": (ca(np.arange(3)), ca(np.arange(3))),
            # "linalg.solve": (ca(np.eye(3)), ca(np.eye(3))),
            # "linalg.tensorsolve": (
            #     ca(np.eye(24).reshape((6, 4, 2, 3, 4))),
            #     ca(np.ones((6, 4))),
            # ),
            # "linalg.eigvals": (ca(np.eye(3)),),
            # "linalg.eigvalsh": (ca(np.eye(3)),),
            # "linalg.lstsq": (ca(np.eye(3)), ca(np.eye(3))),
            # "linalg.eig": (ca(np.eye(3)),),
            # "linalg.eigh": (ca(np.eye(3)),),
            # "copyto": (ca(np.arange(3)), ca(np.arange(3))),
            # "savetxt": (savetxt_file, ca(np.arange(3))),
            # "fill_diagonal": (ca(np.eye(3)), ca(np.arange(3))),
            # "apply_over_axes": (lambda x, axis: x, ca(np.eye(3)), (0, 1)),
            # "place": (ca(np.arange(3)), np.arange(3) > 0, ca(np.arange(3))),
            # "put": (ca(np.arange(3)), np.arange(3), ca(np.arange(3))),
            # "put_along_axis": (ca(np.arange(3)), np.arange(3), ca(np.arange(3)), 0),
            # "putmask": (ca(np.arange(3)), np.arange(3), ca(np.arange(3))),
            # "array_repr": (ca(np.arange(3)),),
            # "trapezoid": (ca(np.arange(3)),),
        }
        functions_checked = list()
        bad_funcs = dict()
        for fname, args in functions_to_check.items():
            ua_args = tuple(to_ua(arg) for arg in args)
            func = getfunc(fname)
            try:
                ua_result = func(*ua_args)
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
            result = func(*args)
            if "savetxt" in fname and os.path.isfile(savetxt_file):
                os.remove(savetxt_file)
            if ua_result is None:
                try:
                    assert result is None
                except AssertionError:
                    bad_funcs["np." + fname] = result, ua_result
            try:
                if isinstance(ua_result, tuple):
                    assert isinstance(result, tuple)
                    assert len(result) == len(ua_result)
                    for r, ua_r in zip(result, ua_result):
                        check_result(r, ua_r)
                else:
                    check_result(result, ua_result)
            except AssertionError:
                bad_funcs["np." + fname] = result, ua_result
        if len(bad_funcs) > 0:
            raise AssertionError(
                "Some functions did not return expected types "
                "(obtained, obtained with unyt input): " + str(bad_funcs)
            )
        unchecked_functions = [
            f for f in _HANDLED_FUNCTIONS if f not in functions_checked
        ]
        try:
            assert len(unchecked_functions) == 0
        except AssertionError:
            raise AssertionError(
                "Did not check functions",
                [
                    ".".join((f.__module__, f.__name__)).replace("numpy", "np")
                    for f in unchecked_functions
                ],
            )

    def test_getitem(self):
        assert isinstance(ca(np.arange(3))[0], cosmo_quantity)

    def test_reshape_to_scalar(self):
        assert isinstance(ca(np.ones(1)).reshape(tuple()), cosmo_quantity)

    def test_iter(self):
        for cq in ca(np.arange(3)):
            assert isinstance(cq, cosmo_quantity)
