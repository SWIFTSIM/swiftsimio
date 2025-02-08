"""
Tests the initialisation of a cosmo_array.
"""

import pytest
import os
import warnings
import numpy as np
import unyt as u
from copy import copy, deepcopy
from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor, a

savetxt_file = "saved_array.txt"


def getfunc(fname):
    func = np
    for attr in fname.split("."):
        func = getattr(func, attr)
    return func


def ca(x, unit=u.Mpc):
    return cosmo_array(x, unit, comoving=False, cosmo_factor=cosmo_factor(a ** 1, 0.5))


def cq(x, unit=u.Mpc):
    return cosmo_quantity(
        x, unit, comoving=False, cosmo_factor=cosmo_factor(a ** 1, 0.5)
    )


def arg_to_ua(arg):
    if type(arg) in (list, tuple):
        return type(arg)([arg_to_ua(a) for a in arg])
    else:
        return to_ua(arg)


def to_ua(x):
    return u.unyt_array(x) if hasattr(x, "comoving") else x


def check_result(x_c, x_u, ignore_values=False):
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

    def test_init_from_list_of_cosmo_arrays(self):
        arr = cosmo_array(
            [
                cosmo_array(
                    [1],
                    units=u.Mpc,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a ** 1, 1),
                )
                for _ in range(5)
            ]
        )
        assert isinstance(arr, cosmo_array)
        assert hasattr(arr, "cosmo_factor") and arr.cosmo_factor == cosmo_factor(
            a ** 1, 1
        )
        assert hasattr(arr, "comoving") and arr.comoving is False


class TestNumpyFunctions:
    def test_explicitly_handled_funcs(self):
        """
        Make sure we at least handle everything that unyt does, and anything that
        'just worked' for unyt but that we need to handle by hand.
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
            "isin": (ca(np.arange(3)), ca(np.arange(3))),
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
            "amax": (ca(np.arange(3)),),  # implemented via max
            "amin": (ca(np.arange(3)),),  # implemented via min
            "angle": (cq(complex(1, 1)),),
            "any": (ca(np.arange(3)),),
            "append": (ca(np.arange(3)), cq(1)),
            "apply_along_axis": (lambda x: x, 0, ca(np.eye(3))),
            "argmax": (ca(np.arange(3)),),  # implemented via max
            "argmin": (ca(np.arange(3)),),  # implemented via min
            "argpartition": (ca(np.arange(3)), 1),  # implemented via partition
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
            "may_share_memory": (ca(np.arange(3)), ca(np.arange(3))),
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
            "einsum_path": ("ij,jk->ik", ca(np.eye(3)), ca(np.eye(3))),
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
                if "unwrap" in fname:
                    # haven't bothered to pass a cosmo_quantity for period
                    warnings.filterwarnings(
                        action="ignore",
                        category=RuntimeWarning,
                        message="Mixing arguments with and without cosmo_factors",
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
                        cosmo_factor=cosmo_factor(a ** 1, 1.0),
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
                        cosmo_factor=cosmo_factor(a ** 1, 1.0),
                    ),
                    cosmo_array(
                        [1, 2, 3],
                        u.K,
                        comoving=False,
                        cosmo_factor=cosmo_factor(a ** 2, 1.0),
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
                            cosmo_factor=cosmo_factor(a ** 1, 1.0),
                        ),
                        cosmo_array(
                            [1, 2, 3],
                            u.K,
                            comoving=False,
                            cosmo_factor=cosmo_factor(a ** 2, 1.0),
                        ),
                        cosmo_array(
                            [1, 2, 3],
                            u.kg,
                            comoving=False,
                            cosmo_factor=cosmo_factor(a ** 3, 1.0),
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
                [1, 2, 3], u.s, comoving=False, cosmo_factor=cosmo_factor(a ** 1, 1.0)
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
                    cosmo_factor=cosmo_factor(a ** 1, 1.0),
                ),
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.K,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a ** 2, 1.0),
                ),
                cosmo_array(
                    np.linspace(0, 5, 11),
                    u.Msun,
                    comoving=False,
                    cosmo_factor=cosmo_factor(a ** 3, 1.0),
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
                    np.histogram: cosmo_factor(a ** -1, 1.0),
                    np.histogram2d: cosmo_factor(a ** -3, 1.0),
                    np.histogramdd: cosmo_factor(a ** -6, 1.0),
                }[func]
            )
        elif density and isinstance(weights, cosmo_array):
            assert result[0].comoving is False
            assert (
                result[0].cosmo_factor
                == {
                    np.histogram: cosmo_factor(a ** 0, 1.0),
                    np.histogram2d: cosmo_factor(a ** -2, 1.0),
                    np.histogramdd: cosmo_factor(a ** -5, 1.0),
                }[func]
            )
        elif not density and isinstance(weights, cosmo_array):
            assert result[0].comoving is False
            assert (
                result[0].cosmo_factor
                == {
                    np.histogram: cosmo_factor(a ** 1, 1.0),
                    np.histogram2d: cosmo_factor(a ** 1, 1.0),
                    np.histogramdd: cosmo_factor(a ** 1, 1.0),
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
                    cosmo_factor(a ** 1, 1.0),
                    cosmo_factor(a ** 2, 1.0),
                    cosmo_factor(a ** 3, 1.0),
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
            cosmo_factor=cosmo_factor(a ** 1, 1.0),
            valid_transform=True,
        )
        res = getattr(cq, func)(*args)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a ** 1, 1.0)
        assert res.valid_transform is True

    def test_scalar_return_func(self):
        """
        Make sure that default-wrapped functions that take a cosmo_array and return a
        scalar convert to a cosmo_quantity.
        """
        ca = cosmo_array(
            np.arange(3),
            u.m,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, 1.0),
            valid_transform=True,
        )
        res = np.min(ca)
        assert isinstance(res, cosmo_quantity)

    @pytest.mark.parametrize("prop", ["T", "ua", "unit_array"])
    def test_propagation_props(self, prop):
        cq = cosmo_quantity(
            1,
            u.m,
            comoving=False,
            cosmo_factor=cosmo_factor(a ** 1, 1.0),
            valid_transform=True,
        )
        res = getattr(cq, prop)
        assert res.comoving is False
        assert res.cosmo_factor == cosmo_factor(a ** 1, 1.0)
        assert res.valid_transform is True


class TestCosmoArrayCopy:
    def test_copy(self):
        """
        Check that when we copy a cosmo_array it preserves its values and attributes.
        """
        units = u.Mpc
        arr = cosmo_array(
            u.unyt_array(np.ones(5), units=units),
            cosmo_factor=cosmo_factor(a ** 1, 1),
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
            cosmo_factor=cosmo_factor(a ** 1, 1),
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
            cosmo_factor=cosmo_factor(a ** 1, 1),
            comoving=False,
        )
        cgs_arr = arr.in_cgs()
        assert np.allclose(arr.to_value(u.cm), cgs_arr.to_value(u.cm))
        assert cgs_arr.units == u.cm
        assert cgs_arr.cosmo_factor == arr.cosmo_factor
        assert cgs_arr.comoving == arr.comoving
