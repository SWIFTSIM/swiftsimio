"""
Tests the initialisation of a cosmo_array.
"""

import pytest
import os
import warnings
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
    return cosmo_array(x, unit, comoving=False, cosmo_factor=cosmo_factor(a**1, 0.5))


def arg_to_ua(arg):
    if type(arg) in (list, tuple):
        return type(arg)([arg_to_ua(a) for a in arg])
    else:
        return to_ua(arg)


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
    if isinstance(x_c, cosmo_array):  # includes cosmo_quantity
        assert x_c.comoving is False
        if x_c.units != u.dimensionless:
            assert x_c.cosmo_factor is not None
    return


class TestCosmoArrayInit:
    def test_init_from_ndarray(self):
        arr = cosmo_array(
            np.ones(5),
            units=u.Mpc,
            cosmo_factor=cosmo_factor(a**1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list(self):
        arr = cosmo_array(
            [1, 1, 1, 1, 1],
            units=u.Mpc,
            cosmo_factor=cosmo_factor(a**1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_unyt_array(self):
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=u.Mpc),
            cosmo_factor=cosmo_factor(a**1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list_of_unyt_arrays(self):
        arr = cosmo_array(
            [u.unyt_array(1, units=u.Mpc) for _ in range(5)],
            cosmo_factor=cosmo_factor(a**1, 1),
            comoving=False,
        )
        assert hasattr(arr, "cosmo_factor")
        assert hasattr(arr, "comoving")
        assert isinstance(arr, cosmo_array)

    def test_init_from_list_of_cosmo_arrays(self):
        arr = cosmo_array(
            [
                cosmo_array(
                    1, units=u.Mpc, comoving=False, cosmo_factor=cosmo_factor(a**1, 1)
                )
                for _ in range(5)
            ]
        )
        assert isinstance(arr, cosmo_array)
        assert hasattr(arr, "cosmo_factor") and arr.cosmo_factor == cosmo_factor(
            a**1, 1
        )
        assert hasattr(arr, "comoving") and arr.comoving is False


class TestNumpyFunctions:

    def test_explicitly_handled_funcs(self):
        """
        Make sure we at least handle everything that unyt does, and anything that
        'just worked' for unyt but that we need to handle by hand.
        """
        from unyt._array_functions import _HANDLED_FUNCTIONS

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
            "linspace": (ca(1), ca(2)),
            "logspace": (ca(1, unit=u.dimensionless), ca(2, unit=u.dimensionless)),
            "geomspace": (ca(1), ca(1)),
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
            "insert": (ca(np.arange(3)), 1, ca(1)),
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
            "isin": (ca(np.arange(3)), ca(np.arange(3))),
            "place": (ca(np.arange(3)), np.arange(3) > 0, ca(np.arange(3))),
            "put": (ca(np.arange(3)), np.arange(3), ca(np.arange(3))),
            "put_along_axis": (ca(np.arange(3)), np.arange(3), ca(np.arange(3)), 0),
            "putmask": (ca(np.arange(3)), np.arange(3), ca(np.arange(3))),
            "searchsorted": (ca(np.arange(3)), ca(np.arange(3))),
            "select": (
                [np.arange(3) < 1, np.arange(3) > 1],
                [ca(np.arange(3)), ca(np.arange(3))],
                ca(1),
            ),
            "setdiff1d": (ca(np.arange(3)), ca(np.arange(3, 6))),
            "sinc": (ca(np.arange(3)),),
            "clip": (ca(np.arange(3)), ca(1), ca(2)),
            "where": (ca(np.arange(3)), ca(np.arange(3)), ca(np.arange(3))),
            "triu": (ca(np.ones((3, 3))),),
            "tril": (ca(np.ones((3, 3))),),
            "einsum": ("ii->i", ca(np.eye(3))),
            "convolve": (ca(np.arange(3)), ca(np.arange(3))),
            "correlate": (ca(np.arange(3)), ca(np.arange(3))),
            "tensordot": (ca(np.eye(3)), ca(np.eye(3))),
            "unwrap": (ca(np.arange(3)),),
            "interp": (ca(np.arange(3)), ca(np.arange(3)), ca(np.arange(3))),
            "array_repr": (ca(np.arange(3)),),
            "linalg.outer": (ca(np.arange(3)), ca(np.arange(3))),
            "trapezoid": (ca(np.arange(3)),),
            "in1d": (ca(np.arange(3)), ca(np.arange(3))),  # np deprecated
            "take": (ca(np.arange(3)), np.arange(3)),
            # FUNCTIONS THAT UNYT DOESN'T HANDLE EXPLICITLY (THEY "JUST WORK"):
            "all": (ca(np.arange(3)),),
            "amax": (ca(np.arange(3)),),
            "amin": (ca(np.arange(3)),),
            # angle,  # expects complex numbers
            # any,  # works out of the box (tested)
            # append,  # we get it for free with np.concatenate (tested)
            # apply_along_axis,  # works out of the box (tested)
            # argmax,  # returns pure numbers
            # argmin,  # returns pure numbers
            # argpartition,  # returns pure numbers
            # argsort,  # returns pure numbers
            # argwhere,  # returns pure numbers
            # array_str,  # hooks into __str__
            # atleast_1d,  # works out of the box (tested)
            # atleast_2d,  # works out of the box (tested)
            # atleast_3d,  # works out of the box (tested)
            # average,  # works out of the box (tested)
            # can_cast,  # works out of the box (tested)
            # common_type,  # works out of the box (tested)
            # result_type,  # works out of the box (tested)
            # iscomplex,  # works out of the box (tested)
            # iscomplexobj,  # works out of the box (tested)
            # isreal,  # works out of the box (tested)
            # isrealobj,  # works out of the box (tested)
            # nan_to_num,  # works out of the box (tested)
            # nanargmax,  # return pure numbers
            # nanargmin,  # return pure numbers
            # nanmax,  # works out of the box (tested)
            # nanmean,  # works out of the box (tested)
            # nanmedian,  # works out of the box (tested)
            # nanmin,  # works out of the box (tested)
            # trim_zeros,  # works out of the box (tested)
            # max,  # works out of the box (tested)
            # mean,  # works out of the box (tested)
            # median,  # works out of the box (tested)
            # min,  # works out of the box (tested)
            # ndim,  # return pure numbers
            # shape,  # returns pure numbers
            # size,  # returns pure numbers
            # sort,  # works out of the box (tested)
            # sum,  # works out of the box (tested)
            # repeat,  # works out of the box (tested)
            # tile,  # works out of the box (tested)
            # shares_memory,  # works out of the box (tested)
            # nonzero,  # works out of the box (tested)
            # count_nonzero,  # returns pure numbers
            # flatnonzero,  # works out of the box (tested)
            # isneginf,  # works out of the box (tested)
            # isposinf,  # works out of the box (tested)
            # empty_like,  # works out of the box (tested)
            # full_like,  # works out of the box (tested)
            # ones_like,  # works out of the box (tested)
            # zeros_like,  # works out of the box (tested)
            # copy,  # works out of the box (tested)
            "meshgrid": (ca(np.arange(3)), ca(np.arange(3))),
            # transpose,  # works out of the box (tested)
            # reshape,  # works out of the box (tested)
            # resize,  # works out of the box (tested)
            # roll,  # works out of the box (tested)
            # rollaxis,  # works out of the box (tested)
            # rot90,  # works out of the box (tested)
            # expand_dims,  # works out of the box (tested)
            # squeeze,  # works out of the box (tested)
            # flip,  # works out of the box (tested)
            # fliplr,  # works out of the box (tested)
            # flipud,  # works out of the box (tested)
            # delete,  # works out of the box (tested)
            # partition,  # works out of the box (tested)
            # broadcast_to,  # works out of the box (tested)
            # broadcast_arrays,  # works out of the box (tested)
            # split,  # works out of the box (tested)
            # array_split,  # works out of the box (tested)
            # dsplit,  # works out of the box (tested)
            # hsplit,  # works out of the box (tested)
            # vsplit,  # works out of the box (tested)
            # swapaxes,  # works out of the box (tested)
            # moveaxis,  # works out of the box (tested)
            # nansum,  # works out of the box (tested)
            # std,  # works out of the box (tested)
            # nanstd,  # works out of the box (tested)
            # nanvar,  # works out of the box (tested)
            # nanprod,  # works out of the box (tested)
            # diag,  # works out of the box (tested)
            # diag_indices_from,  # returns pure numbers
            # diagflat,  # works out of the box (tested)
            # diagonal,  # works out of the box (tested)
            # ravel,  # returns pure numbers
            # ravel_multi_index,  # returns pure numbers
            # unravel_index,  # returns pure numbers
            # fix,  # works out of the box (tested)
            # round,  # is implemented via np.around
            # may_share_memory,  # returns pure numbers (booleans)
            # linalg.matrix_power,  # works out of the box (tested)
            # linalg.cholesky,  # works out of the box (tested)
            # linalg.multi_dot,  # works out of the box (tested)
            # linalg.matrix_rank,  # returns pure numbers
            # linalg.qr,  # works out of the box (tested)
            # linalg.slogdet,  # undefined units
            # linalg.cond,  # works out of the box (tested)
            # gradient,  # works out of the box (tested)
            # cumsum,  # works out of the box (tested)
            # nancumsum,  # works out of the box (tested)
            # nancumprod,  # we get it for free with np.cumprod (tested)
            # bincount,  # works out of the box (tested)
            # unique,  # works out of the box (tested)
            # min_scalar_type,  # returns dtypes
            # extract,  # works out of the box (tested)
            # setxor1d,  # we get it for free with previously implemented functions (tested)
            # lexsort,  # returns pure numbers
            # digitize,  # returns pure numbers
            # tril_indices_from,  # returns pure numbers
            # triu_indices_from,  # returns pure numbers
            # imag,  # works out of the box (tested)
            # real,  # works out of the box (tested)
            # real_if_close,  # works out of the box (tested)
            # einsum_path,  # returns pure numbers
            # cov,  # returns pure numbers
            # corrcoef,  # returns pure numbers
            # compress,  # works out of the box (tested)
            # take_along_axis,  # works out of the box (tested)
            # he following all work out of the box (tested):
            # linalg.cross,
            # linalg.diagonal,
            # linalg.matmul,
            # linalg.matrix_norm,
            # linalg.matrix_transpose,
            # linalg.svdvals,
            # linalg.tensordot,
            # linalg.trace,
            # linalg.vecdot,
            # linalg.vector_norm,
            # astype,
            # matrix_transpose,
            # unique_all,
            # unique_counts,
            # unique_inverse,
            # unique_values,
            # vecdot,
        }
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
                            message="numpy.savetxt does not preserve units or cosmo",
                        )
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
            with warnings.catch_warnings():
                if "savetxt" in fname:
                    warnings.filterwarnings(
                        action="ignore",
                        category=UserWarning,
                        message="numpy.savetxt does not preserve units or cosmo",
                    )
                result = func(*args)
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
            if ua_result is None:
                try:
                    assert result is None
                except AssertionError:
                    bad_funcs["np." + fname] = result, ua_result
            else:
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
                        cosmo_factor=cosmo_factor(a**1, 1.0),
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
                        cosmo_factor=cosmo_factor(a**1, 1.0),
                    ),
                    cosmo_array(
                        [1, 2, 3],
                        u.K,
                        comoving=False,
                        cosmo_factor=cosmo_factor(a**2, 1.0),
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
                            cosmo_factor=cosmo_factor(a**1, 1.0),
                        ),
                        cosmo_array(
                            [1, 2, 3],
                            u.K,
                            comoving=False,
                            cosmo_factor=cosmo_factor(a**2, 1.0),
                        ),
                        cosmo_array(
                            [1, 2, 3],
                            u.kg,
                            comoving=False,
                            cosmo_factor=cosmo_factor(a**3, 1.0),
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
                [1, 2, 3], u.s, comoving=False, cosmo_factor=cosmo_factor(a**1, 1.0)
            ),
            np.array([1, 2, 3]),
        ),
    )
    @pytest.mark.parametrize("bins_type", ("int", "np", "ca"))
    @pytest.mark.parametrize("density", (None, True))
    def test_histograms(self, func_args, weights, bins_type, density):
        func, args = func_args
        bins = {
            "int": 10,
            "np": [np.linspace(0, 5, 11)] * 3,
            "ca": [
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.kpc,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a**1, 1.0),
                ),
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.K,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a**2, 1.0),
                ),
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.Msun,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a**3, 1.0),
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
            if bins_type in ("np", "ca")
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
        assert isinstance(ca(np.arange(3))[0], cosmo_quantity)

    def test_reshape_to_scalar(self):
        assert isinstance(ca(np.ones(1)).reshape(tuple()), cosmo_quantity)

    def test_iter(self):
        for cq in ca(np.arange(3)):
            assert isinstance(cq, cosmo_quantity)


class TestCosmoQuantity:
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
        cq = cosmo_quantity(
            1,
            u.m,
            comoving=False,
            cosmo_factor=cosmo_factor(a**1, 1.0),
            valid_transform=True,
        )
        res = getattr(cq, func)(*args)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**1, 1.0)
        assert res.valid_transform is True

    @pytest.mark.parametrize("prop", ["T", "ua", "unit_array"])
    def test_propagation_props(self, prop):
        cq = cosmo_quantity(
            1,
            u.m,
            comoving=False,
            cosmo_factor=cosmo_factor(a**1, 1.0),
            valid_transform=True,
        )
        res = getattr(cq, prop)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a**1, 1.0)
        assert res.valid_transform is True
