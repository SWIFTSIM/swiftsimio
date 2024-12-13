import numpy as np
from unyt import unyt_quantity, unyt_array
from unyt._array_functions import implements
from .objects import (
    cosmo_array,
    cosmo_quantity,
    _prepare_array_func_args,
    _multiply_cosmo_factor,
    _preserve_cosmo_factor,
    _reciprocal_cosmo_factor,
)

_HANDLED_FUNCTIONS = dict()


def _return_helper(res, helper_result, ret_cf, out=None):
    if out is None:
        if isinstance(res, unyt_quantity):
            return cosmo_quantity(
                res,
                comoving=helper_result["comoving"],
                cosmo_factor=ret_cf,
                compression=helper_result["compression"],
            )
        elif isinstance(res, unyt_array):
            return cosmo_array(
                res,
                comoving=helper_result["comoving"],
                cosmo_factor=ret_cf,
                compression=helper_result["compression"],
            )
        else:
            # unyt returned a bare array
            return res
    if hasattr(out, "comoving"):
        out.comoving = helper_result["comoving"]
    if hasattr(out, "cosmo_factor"):
        out.cosmo_factor = ret_cf
    if hasattr(out, "compression"):
        out.compression = helper_result["compression"]
    return cosmo_array(  # confused, do we set out, or return?
        res.to_value(res.units),
        res.units,
        bypass_validation=True,
        comoving=helper_result["comoving"],
        cosmo_factor=ret_cf,
        compression=helper_result["compression"],
    )


@implements(np.array2string)
def array2string(
    a,
    max_line_width=None,
    precision=None,
    suppress_small=None,
    separator=" ",
    prefix="",
    style=np._NoValue,
    formatter=None,
    threshold=None,
    edgeitems=None,
    sign=None,
    floatmode=None,
    suffix="",
    *,
    legacy=None,
):
    from unyt._array_functions import array2string as unyt_array2string

    res = unyt_array2string(
        a,
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
        separator=separator,
        prefix=prefix,
        style=style,
        formatter=formatter,
        threshold=threshold,
        edgeitems=edgeitems,
        sign=sign,
        floatmode=floatmode,
        suffix=suffix,
        legacy=legacy,
    )
    if a.comoving:
        append = " (comoving)"
    elif a.comoving is False:
        append = " (physical)"
    elif a.comoving is None:
        append = ""
    return res + append


@implements(np.dot)
def dot(a, b, out=None):
    from unyt._array_functions import dot as unyt_dot

    helper_result = _prepare_array_func_args(a, b, out=out)
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_dot(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.vdot)
def vdot(a, b, /):
    from unyt._array_functions import vdot as unyt_vdot

    helper_result = _prepare_array_func_args(a, b)
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_vdot(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.inner)
def inner(a, b, /):
    from unyt._array_functions import inner as unyt_inner

    helper_result = _prepare_array_func_args(a, b)
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_inner(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.outer)
def outer(a, b, out=None):
    from unyt._array_functions import outer as unyt_outer

    helper_result = _prepare_array_func_args(a, b, out=out)
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_outer(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.kron)
def kron(a, b):
    from unyt._array_functions import kron as unyt_kron

    helper_result = _prepare_array_func_args(a, b)
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_kron(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    from unyt._array_functions import histogram_bin_edges as unyt_histogram_bin_edges

    helper_result = _prepare_array_func_args(a, bins=bins, range=range, weights=weights)
    if not isinstance(bins, str) and np.ndim(bins) == 1:
        # we got bin edges as input
        ret_cf = _preserve_cosmo_factor(helper_result["kw_ca_cfs"]["bins"])
    else:
        # bins based on values in a
        ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
        res = unyt_histogram_bin_edges(
            *helper_result["args"], **helper_result["kwargs"]
        )
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.inv)
def linalg_inv(a):
    from unyt._array_functions import linalg_inv as unyt_linalg_inv

    helper_result = _prepare_array_func_args(a)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_linalg_inv(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.tensorinv)
def linalg_tensorinv(a, ind=2):
    from unyt._array_functions import linalg_tensorinv as unyt_linalg_tensorinv

    helper_result = _prepare_array_func_args(a, ind=ind)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_linalg_tensorinv(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.pinv)
def linalg_pinv(a, rcond=None, hermitian=False, *, rtol=np._NoValue):
    from unyt._array_functions import linalg_pinv as unyt_linalg_pinv

    helper_result = _prepare_array_func_args(
        a, rcond=rcond, hermitian=hermitian, rtol=rtol
    )
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_linalg_pinv(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.svd)
def linalg_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    from unyt._array_functions import linalg_svd as unyt_linalg_svd

    helper_result = _prepare_array_func_args(
        a,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        hermitian=hermitian
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    ress = unyt_linalg_svd(*helper_result["args"], **helper_result["kwargs"])
    if compute_uv:
        return tuple(_return_helper(res, helper_result, ret_cf) for res in ress)
    else:
        return _return_helper(ress, helper_result, ret_cf)


@implements(np.histogram)
def histogram(a, bins=10, range=None, density=None, weights=None):
    from unyt._array_functions import histogram as unyt_histogram

    helper_result = _prepare_array_func_args(
        a,
        bins=bins,
        range=range,
        density=density,
        weights=weights
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    counts, bins = unyt_histogram(*helper_result["args"], **helper_result["kwargs"])
    return counts, _return_helper(bins, helper_result, ret_cf)


# ND HISTOGRAMS ARE TRICKY - EACH AXIS CAN HAVE DIFFERENT COSMO FACTORS

# @implements(np.histogram2d)
# def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
#     from unyt._array_functions import histogram2d as unyt_histogram2d

#     helper_result = _prepare_array_func_args(
#         x,
#         y,
#         bins=bins,
#         range=range,
#         density=density,
#         weights=weights
#     )
#     ret_cf_x = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
#     ret_cf_y = _preserve_cosmo_factor(helper_result["ca_cfs"][1])
#     counts, xbins, ybins = unyt_histogram2d(
#         *helper_result["args"], **helper_result["kwargs"]
#     )
#     return (
#         counts,
#         _return_helper(xbins, helper_result, ret_cf),
#         _return_helper(ybins, helper_result, ret_cf),
#     )


# @implements(np.histogramdd)
# def histogramdd(sample, bins=10, range=None, density=None, weights=None):
#     from unyt._array_functions import histogramdd as unyt_histogramdd

#     helper_result = _prepare_array_func_args(
#         sample,
#         bins=bins,
#         range=range,
#         density=density,
#         weights=weights
#     )
#     ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
#     counts, bins = unyt_histogramdd(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.concatenate)
def concatenate(tup, axis=0, out=None, dtype=None, casting="same_kind"):
    from unyt._array_functions import concatenate as unyt_concatenate

    helper_result = _prepare_array_func_args(
        tup, axis=axis, out=out, dtype=dtype, casting=casting
    )
    helper_result_concat_items = _prepare_array_func_args(*tup)
    ret_cf = _preserve_cosmo_factor(helper_result_concat_items["ca_cfs"][0])
    res = unyt_concatenate(
        helper_result_concat_items["args"],
        *helper_result["args"][1:],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result_concat_items, ret_cf, out=out)


# @implements(np.cross)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.intersect1d)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.union1d)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.norm)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.vstack)
def vstack(tup, *, dtype=None, casting="same_kind"):
    from unyt._array_functions import vstack as unyt_vstack

    helper_result = _prepare_array_func_args(
        tup, dtype=dtype, casting=casting
    )
    helper_result_concat_items = _prepare_array_func_args(*tup)
    ret_cf = _preserve_cosmo_factor(helper_result_concat_items["ca_cfs"][0])
    res = unyt_vstack(
        helper_result_concat_items["args"],
        *helper_result["args"][1:],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result_concat_items, ret_cf)


@implements(np.hstack)
def hstack(tup, *, dtype=None, casting="same_kind"):
    from unyt._array_functions import hstack as unyt_hstack

    helper_result = _prepare_array_func_args(
        tup, dtype=dtype, casting=casting
    )
    helper_result_concat_items = _prepare_array_func_args(*tup)
    ret_cf = _preserve_cosmo_factor(helper_result_concat_items["ca_cfs"][0])
    res = unyt_hstack(
        helper_result_concat_items["args"],
        *helper_result["args"][1:],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result_concat_items, ret_cf)


@implements(np.dstack)
def dstack(tup):
    from unyt._array_functions import dstack as unyt_dstack

    helper_result_concat_items = _prepare_array_func_args(*tup)
    ret_cf = _preserve_cosmo_factor(helper_result_concat_items["ca_cfs"][0])
    res = unyt_dstack(helper_result_concat_items["args"])
    return _return_helper(res, helper_result_concat_items, ret_cf)


@implements(np.column_stack)
def column_stack(tup):
    from unyt._array_functions import column_stack as unyt_column_stack

    helper_result_concat_items = _prepare_array_func_args(*tup)
    ret_cf = _preserve_cosmo_factor(helper_result_concat_items["ca_cfs"][0])
    res = unyt_column_stack(helper_result_concat_items["args"])
    return _return_helper(res, helper_result_concat_items, ret_cf)


@implements(np.stack)
def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    from unyt._array_functions import stack as unyt_stack

    helper_result = _prepare_array_func_args(
        arrays, axis=axis, out=out, dtype=dtype, casting=casting
    )
    helper_result_concat_items = _prepare_array_func_args(*arrays)
    ret_cf = _preserve_cosmo_factor(helper_result_concat_items["ca_cfs"][0])
    res = unyt_stack(
        helper_result_concat_items["args"],
        *helper_result["args"][1:],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result_concat_items, ret_cf, out=out)


# @implements(np.around)
# def around(...):
#     from unyt._array_functions import around as unyt_around

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_around(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.block)
# def block(...):
#     from unyt._array_functions import block as unyt_block

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_block(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# UNYT HAS A COPY-PASTED TYPO fft -> ftt

# @implements(np.fft.fft)
# def ftt_fft(...):
#     from unyt._array_functions import ftt_fft as unyt_fft_fft

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_fft(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.fft2)
# def ftt_fft2(...):
#     from unyt._array_functions import ftt_fft2 as unyt_fft_fft2

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_fft2(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.fftn)
# def ftt_fftn(...):
#     from unyt._array_functions import ftt_fftn as unyt_fft_fftn

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_fftn(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.hfft)
# def ftt_hfft(...):
#     from unyt._array_functions import ftt_hfft as unyt_fft_hfft

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_hfft(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.rfft)
# def ftt_rfft(...):
#     from unyt._array_functions import ftt_rfft as unyt_fft_rfft

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_rfft(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.rfft2)
# def fft_rfft2(...):
#     from unyt._array_functions import ftt_rfft2 as unyt_fft_rfft2

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_rfft2(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.rfftn)
# def fft_rfftn(...):
#     from unyt._array_functions import ftt_rfftn as unyt_fft_rfftn

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_rfftn(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.ifft)
# def fft_ifft(...):
#     from unyt._array_functions import ftt_ifft as unyt_fft_ifft

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_ifft(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.ifft2)
# def fft_ifft2(...):
#     from unyt._array_functions import ftt_ifft2 as unyt_fft_ifft2

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_ifft2(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.ifftn)
# def fft_ifftn(...):
#     from unyt._array_functions import ftt_ifftn as unyt_fft_ifftn

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_ifftn(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.ihfft)
# def fft_ihfft(...):
#     from unyt._array_functions import ftt_ihfft as unyt_fft_ihfft

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_ihfft(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.irfft)
# def fft_irfft(...):
#     from unyt._array_functions import ftt_irfft as unyt_fft_irfft

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_irfft(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.irfft2)
# def fft_irfft2(...):
#     from unyt._array_functions import ftt_irfft2 as unyt_fft_irfft2

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_irfft2(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.irfftn)
# def fft_irfftn(...):
#     from unyt._array_functions import ftt_irfftn as unyt_fft_irfftn

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_irfftn(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.fftshift)
# def fft_fftshift(...):
#     from unyt._array_functions import fft_fftshift as unyt_fft_fftshift

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_fftshift(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fft.ifftshift)
# def fft_ifftshift(...):
#     from unyt._array_functions import fft_ifftshift as unyt_fft_ifftshift

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fft_ifftshift(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.sort_complex)
# def sort_complex(...):
#     from unyt._array_functions import sort_complex as unyt_sort_complex

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_sort_complex(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.isclose)
# def isclose(...):
#     from unyt._array_functions import isclose as unyt_isclose

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_isclose(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.allclose)
# def allclose(...):
#     from unyt._array_functions import allclose as unyt_allclose

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_allclose(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.array_equal)
# def array_equal(...):
#     from unyt._array_functions import array_equal as unyt_array_equal

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_array_equal(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.array_equiv)
# def array_equiv(...):
#     from unyt._array_functions import array_equiv as unyt_array_equiv

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_array_equiv(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linspace)
# def linspace(...):
#     from unyt._array_functions import linspace as unyt_linspace

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linspace(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.logspace)
# def logspace(...):
#     from unyt._array_functions import logspace as unyt_logspace

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_logspace(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.geomspace)
# def geomspace(...):
#     from unyt._array_functions import geomspace as unyt_geomspace

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_geomspace(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.copyto)
# def copyto(...):
#     from unyt._array_functions import copyto as unyt_copyto

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_copyto(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.prod)
# def prod(...):
#     from unyt._array_functions import prod as unyt_prod

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_prod(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.var)
# def var(...):
#     from unyt._array_functions import var as unyt_var

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_var(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.trace)
# def trace(...):
#     from unyt._array_functions import trace as unyt_trace

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_trace(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.percentile)
# def percentile(...):
#     from unyt._array_functions import percentile as unyt_percentile

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_percentile(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.quantile)
# def quantile(...):
#     from unyt._array_functions import quantile as unyt_quantile

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_quantile(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.nanpercentile)
# def nanpercentile(...):
#     from unyt._array_functions import nanpercentile as unyt_nanpercentile

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_nanpercentile(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.nanquantile)
# def nanquantile(...):
#     from unyt._array_functions import nanquantile as unyt_nanquantile

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_nanquantile(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.det)
# def linalg_det(...):
#     from unyt._array_functions import linalg_det as unyt_linalg_det

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_det(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.lstsq)
# def linalg_lstsq(...):
#     from unyt._array_functions import linalg_lstsq as unyt_linalg_lstsq

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_lstsq(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.solve)
# def linalg_solve(...):
#     from unyt._array_functions import linalg_solve as unyt_linalg_solve

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_solve(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.tensorsolve)
# def linalg_tensorsolve(...):
#     from unyt._array_functions import linalg_tensorsolve as unyt_linalg_tensorsolve

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_tensorsolve(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.eig)
# def linalg_eig(...):
#     from unyt._array_functions import linalg_eig as unyt_linalg_eig

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_eig(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.eigh)
# def linalg_eigh(...):
#     from unyt._array_functions import linalg_eigh as unyt_linalg_eigh

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_eigh(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.eigvals)
# def linalg_eigvals(...):
#     from unyt._array_functions import linalg_eigvals as unyt_linalg_eigvals

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_eigvals(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.eigvalsh)
# def linalg_eigvalsh(...):
#     from unyt._array_functions import linalg_eigvalsh as unyt_linalg_eigvalsh

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_eigvalsh(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.savetxt)
# def savetxt(...):
#     from unyt._array_functions import savetxt as unyt_savetxt

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_savetxt(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.apply_over_axes)
# def apply_over_axes(...):
#     from unyt._array_functions import apply_over_axes as unyt_apply_over_axes

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_apply_over_axes(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.diff)
# def diff(...):
#     from unyt._array_functions import diff as unyt_diff

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_diff(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.ediff1d)
# def ediff1d(...):
#     from unyt._array_functions import ediff1d as unyt_ediff1d

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_ediff1d(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.ptp)
# def ptp(...):
#     from unyt._array_functions import ptp as unyt_ptp

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_ptp(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.cumprod)
# def cumprod(...):
#     from unyt._array_functions import cumprod as unyt_cumprod

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_cumprod(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.pad)
# def pad(...):
#     from unyt._array_functions import pad as unyt_pad

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_pad(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.choose)
# def choose(...):
#     from unyt._array_functions import choose as unyt_choose

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_choose(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.fill_diagonal)
# def fill_diagonal(...):
#     from unyt._array_functions import fill_diagonal as unyt_fill_diagonal

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_fill_diagonal(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.insert)
# def insert(...):
#     from unyt._array_functions import insert as unyt_insert

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_insert(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.isin)
# def isin(...):
#     from unyt._array_functions import isin as unyt_isin

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_isin(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.place)
# def place(...):
#     from unyt._array_functions import place as unyt_place

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_place(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.put)
# def ...(...):
#     from unyt._array_functions import put as unyt_put

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_put(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.put_along_axis)
# def put_along_axis(...):
#     from unyt._array_functions import put_along_axis as unyt_put_along_axis

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_put_along_axis(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.putmask)
# def putmask(...):
#     from unyt._array_functions import putmask as unyt_putmask

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_putmask(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.searchsorted)
# def searchsorted(...):
#     from unyt._array_functions import searchsorted as unyt_searchsorted

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_searchsorted(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.select)
# def select(...):
#     from unyt._array_functions import select as unyt_select

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_select(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.setdiff1d)
# def setdiff1d(...):
#     from unyt._array_functions import setdiff1d as unyt_setdiff1d

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_setdiff1d(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.sinc)
# def sinc(...):
#     from unyt._array_functions import sinc as unyt_sinc

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_sinc(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.clip)
# def clip(...):
#     from unyt._array_functions import clip as unyt_clip

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_clip(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.where)
# def where(...):
#     from unyt._array_functions import where as unyt_where

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_where(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.triu)
# def triu(...):
#     from unyt._array_functions import triu as unyt_triu

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_triu(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.tril)
# def tril(...):
#     from unyt._array_functions import tril as unyt_tril

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_tril(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.einsum)
# def einsum(...):
#     from unyt._array_functions import einsum as unyt_einsum

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_einsum(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.convolve)
# def convolve(...):
#     from unyt._array_functions import convolve as unyt_convolve

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_convolve(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.correlate)
# def correlate(...):
#     from unyt._array_functions import correlate as unyt_correlate

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_correlate(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.tensordot)
# def tensordot(...):
#     from unyt._array_functions import tensordot as unyt_tensordot

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_tensordot(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.unwrap)
# def unwrap(...):
#     from unyt._array_functions import unwrap as unyt_unwrap

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_unwrap(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.interp)
# def interp(...):
#     from unyt._array_functions import interp as unyt_interp

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_interp(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.array_repr)
# def array_repr(...):
#     from unyt._array_functions import array_repr as unyt_array_repr

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_array_repr(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.linalg.outer)
# def linalg_outer(...):
#     from unyt._array_functions import linalg_outer as unyt_linalg_outer

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_linalg_outer(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.trapezoid)
# def trapezoid(...):
#     from unyt._array_functions import trapezoid as unyt_trapezoid

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_trapezoid(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.in1d)
# def in1d(...):
#     from unyt._array_functions import in1d as unyt_in1d

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_in1d(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)
