import numpy as np
from unyt import unyt_quantity, unyt_array
from unyt._array_functions import implements
from .objects import (
    cosmo_array,
    cosmo_quantity,
    _prepare_array_func_args,
    _multiply_cosmo_factor,
    _divide_cosmo_factor,
    _preserve_cosmo_factor,
    _reciprocal_cosmo_factor,
    _comparison_cosmo_factor,
    _power_cosmo_factor,
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
        a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
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
        a, bins=bins, range=range, density=density, weights=weights
    )
    ret_cf_bins = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    ret_cf_dens = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    counts, bins = unyt_histogram(*helper_result["args"], **helper_result["kwargs"])
    if weights is not None:
        ret_cf_w = _preserve_cosmo_factor(helper_result["kw_ca_cfs"]["weights"])
        ret_cf_counts = (
            _multiply_cosmo_factor(
                (ret_cf_w is not None, ret_cf_w), (ret_cf_dens is not None, ret_cf_dens)
            )
            if density
            else ret_cf_w
        )
    else:
        ret_cf_counts = ret_cf_dens if density else None
    if isinstance(counts, unyt_array):
        counts = cosmo_array(
            counts.to_value(counts.units),
            counts.units,
            comoving=helper_result["comoving"],
            cosmo_factor=ret_cf_counts,
            compression=helper_result["compression"],
        )
    return counts, _return_helper(bins, helper_result, ret_cf_bins)


@implements(np.histogram2d)
def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    from unyt._array_functions import histogram2d as unyt_histogram2d

    if range is not None:
        xrange, yrange = range
    else:
        xrange, yrange = None, None

    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 2:
        xbins = ybins = bins
    elif N == 2:
        xbins, ybins = bins
    helper_result_x = _prepare_array_func_args(
        x,
        bins=xbins,
        range=xrange,
    )
    helper_result_y = _prepare_array_func_args(
        y,
        bins=ybins,
        range=yrange,
    )
    if not density:
        helper_result_w = _prepare_array_func_args(weights=weights)
        ret_cf_x = _preserve_cosmo_factor(helper_result_x["ca_cfs"][0])
        ret_cf_y = _preserve_cosmo_factor(helper_result_y["ca_cfs"][0])
        if (helper_result_x["kwargs"]["range"] is None) and (
            helper_result_y["kwargs"]["range"] is None
        ):
            safe_range = None
        else:
            safe_range = (
                helper_result_x["kwargs"]["range"],
                helper_result_y["kwargs"]["range"],
            )
        counts, xbins, ybins = unyt_histogram2d(
            helper_result_x["args"][0],
            helper_result_y["args"][0],
            bins=(helper_result_x["kwargs"]["bins"], helper_result_y["kwargs"]["bins"]),
            range=safe_range,
            density=density,
            weights=helper_result_w["kwargs"]["weights"],
        )
        if weights is not None:
            ret_cf_w = _preserve_cosmo_factor(helper_result_w["kw_ca_cfs"]["weights"])
            if isinstance(counts, unyt_array):
                counts = cosmo_array(
                    counts.to_value(counts.units),
                    counts.units,
                    comoving=helper_result_w["comoving"],
                    cosmo_factor=ret_cf_w,
                    compression=helper_result_w["compression"],
                )
    else:  # density=True
        # now x, y and weights must be compatible because they will combine
        # we unpack input to the helper to get everything checked for compatibility
        helper_result = _prepare_array_func_args(
            x,
            y,
            xbins=xbins,
            ybins=ybins,
            xrange=xrange,
            yrange=yrange,
            weights=weights,
        )
        ret_cf_x = _preserve_cosmo_factor(helper_result_x["ca_cfs"][0])
        ret_cf_y = _preserve_cosmo_factor(helper_result_y["ca_cfs"][0])
        if (helper_result["kwargs"]["xrange"] is None) and (
            helper_result["kwargs"]["yrange"] is None
        ):
            safe_range = None
        else:
            safe_range = (
                helper_result["kwargs"]["xrange"],
                helper_result["kwargs"]["yrange"],
            )
        counts, xbins, ybins = unyt_histogram2d(
            helper_result["args"][0],
            helper_result["args"][1],
            bins=(helper_result["kwargs"]["xbins"], helper_result["kwargs"]["ybins"]),
            range=safe_range,
            density=density,
            weights=helper_result["kwargs"]["weights"],
        )
        ret_cf_xy = _multiply_cosmo_factor(
            helper_result["ca_cfs"][0],
            helper_result["ca_cfs"][1],
        )
        if weights is not None:
            ret_cf_w = _preserve_cosmo_factor(helper_result["kw_ca_cfs"]["weights"])
            inv_ret_cf_xy = _reciprocal_cosmo_factor((ret_cf_xy is not None, ret_cf_xy))
            ret_cf_counts = _multiply_cosmo_factor(
                (ret_cf_w is not None, ret_cf_w),
                (inv_ret_cf_xy is not None, inv_ret_cf_xy),
            )
        else:
            ret_cf_counts = _reciprocal_cosmo_factor((ret_cf_xy is not None, ret_cf_xy))
        if isinstance(counts, unyt_array):
            counts = cosmo_array(
                counts.to_value(counts.units),
                counts.units,
                comoving=helper_result["comoving"],
                cosmo_factor=ret_cf_counts,
                compression=helper_result["compression"],
            )
    return (
        counts,
        _return_helper(xbins, helper_result_x, ret_cf_x),
        _return_helper(ybins, helper_result_y, ret_cf_y),
    )


@implements(np.histogramdd)
def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    from unyt._array_functions import histogramdd as unyt_histogramdd

    D = len(sample)
    if range is not None:
        ranges = range
    else:
        ranges = D * [None]

    try:
        len(bins)
    except TypeError:
        # bins is an integer
        bins = D * [bins]
    helper_results = [
        _prepare_array_func_args(
            s,
            bins=b,
            range=r,
        )
        for s, b, r in zip(sample, bins, ranges)
    ]
    if not density:
        helper_result_w = _prepare_array_func_args(weights=weights)
        ret_cfs = [
            _preserve_cosmo_factor(helper_result["ca_cfs"][0])
            for helper_result in helper_results
        ]
        if all(
            [
                helper_result["kwargs"]["range"] is None
                for helper_result in helper_results
            ]
        ):
            safe_range = None
        else:
            safe_range = [
                helper_result["kwargs"]["range"] for helper_result in helper_results
            ]
        counts, bins = unyt_histogramdd(
            [helper_result["args"][0] for helper_result in helper_results],
            bins=[helper_result["kwargs"]["bins"] for helper_result in helper_results],
            range=safe_range,
            density=density,
            weights=helper_result_w["kwargs"]["weights"],
        )
        if weights is not None:
            ret_cf_w = _preserve_cosmo_factor(helper_result_w["kw_ca_cfs"]["weights"])
            if isinstance(counts, unyt_array):
                counts = cosmo_array(
                    counts.to_value(counts.units),
                    counts.units,
                    comoving=helper_result_w["comoving"],
                    cosmo_factor=ret_cf_w,
                    compression=helper_result_w["compression"],
                )
    else:  # density=True
        # now sample and weights must be compatible because they will combine
        # we unpack input to the helper to get everything checked for compatibility
        helper_result = _prepare_array_func_args(
            *sample,
            bins=bins,
            range=range,
            weights=weights,
        )
        ret_cfs = D * [_preserve_cosmo_factor(helper_result["ca_cfs"][0])]
        counts, bins = unyt_histogramdd(
            helper_result["args"],
            bins=helper_result["kwargs"]["bins"],
            range=helper_result["kwargs"]["range"],
            density=density,
            weights=helper_result["kwargs"]["weights"],
        )
        if len(helper_result["ca_cfs"]) == 1:
            ret_cf_sample = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
        else:
            ret_cf_sample = _multiply_cosmo_factor(*helper_result["ca_cfs"])
        if weights is not None:
            ret_cf_w = _preserve_cosmo_factor(helper_result["kw_ca_cfs"]["weights"])
            inv_ret_cf_sample = _reciprocal_cosmo_factor(
                (ret_cf_sample is not None, ret_cf_sample)
            )
            ret_cf_counts = _multiply_cosmo_factor(
                (ret_cf_w is not None, ret_cf_w),
                (inv_ret_cf_sample is not None, inv_ret_cf_sample),
            )
        else:
            ret_cf_counts = _reciprocal_cosmo_factor(
                (ret_cf_sample is not None, ret_cf_sample)
            )
        if isinstance(counts, unyt_array):
            counts = cosmo_array(
                counts.to_value(counts.units),
                counts.units,
                comoving=helper_result["comoving"],
                cosmo_factor=ret_cf_counts,
                compression=helper_result["compression"],
            )
    return (
        counts,
        [
            _return_helper(b, helper_result, ret_cf)
            for b, helper_result, ret_cf in zip(bins, helper_results, ret_cfs)
        ],
    )


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


@implements(np.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    from unyt._array_functions import cross as unyt_cross

    helper_result = _prepare_array_func_args(
        a,
        b,
        axisa=axisa,
        axisb=axisb,
        axisc=axisc,
        axis=axis,
    )
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_cross(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.intersect1d)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    from unyt._array_functions import intersect1d as unyt_intersect1d

    helper_result = _prepare_array_func_args(
        ar1, ar2, assume_unique=assume_unique, return_indices=return_indices
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_intersect1d(*helper_result["args"], **helper_result["kwargs"])
    if return_indices:
        return res
    else:
        return _return_helper(res, helper_result, ret_cf)


@implements(np.union1d)
def union1d(ar1, ar2):
    from unyt._array_functions import union1d as unyt_union1d

    helper_result = _prepare_array_func_args(ar1, ar2)
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_union1d(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.norm)
def linalg_norm(x, ord=None, axis=None, keepdims=False):
    # they didn't use linalg_norm, doesn't follow usual pattern:
    from unyt._array_functions import norm as unyt_linalg_norm

    helper_result = _prepare_array_func_args(x, ord=ord, axis=axis, keepdims=keepdims)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_linalg_norm(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.vstack)
def vstack(tup, *, dtype=None, casting="same_kind"):
    from unyt._array_functions import vstack as unyt_vstack

    helper_result = _prepare_array_func_args(tup, dtype=dtype, casting=casting)
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

    helper_result = _prepare_array_func_args(tup, dtype=dtype, casting=casting)
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


@implements(np.around)
def around(a, decimals=0, out=None):
    from unyt._array_functions import around as unyt_around

    helper_result = _prepare_array_func_args(a, decimals=decimals, out=out)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_around(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


def _recursive_to_comoving(lst):
    ret_lst = list()
    for item in lst:
        if isinstance(item, list):
            ret_lst.append(_recursive_to_comoving(item))
        else:
            ret_lst.append(item.to_comoving())
    return ret_lst


def _prepare_array_block_args(lst, recursing=False):
    """
    Block accepts only a nested list of array "blocks". We need to recurse on this.
    """
    helper_results = list()
    if isinstance(lst, list):
        for item in lst:
            if isinstance(item, list):
                helper_results += _prepare_array_block_args(item, recursing=True)
            else:
                helper_results.append(_prepare_array_func_args(item))
    if recursing:
        return helper_results
    cms = [hr["comoving"] for hr in helper_results]
    comps = [hr["compression"] for hr in helper_results]
    ca_cfs = [hr["ca_cfs"] for hr in helper_results]
    convert_to_cm = False
    if all(cms):
        ret_cm = True
    elif all([cm is None for cm in cms]):
        ret_cm = None
    elif any([cm is None for cm in cms]) and not all([cm is None for cm in cms]):
        raise ValueError(
            "Some input has comoving=None and others have "
            "comoving=True|False. Result is undefined!"
        )
    elif all([cm is False for cm in cms]):
        ret_cm = False
    else:
        # mix of True and False only
        ret_cm = True
        convert_to_cm = True
    if len(set(comps)) == 1:
        ret_comp = comps[0]
    else:
        ret_comp = None
    ret_cf = ca_cfs[0]
    for ca_cf in ca_cfs[1:]:
        if ca_cf != ret_cf:
            raise ValueError("Mixed cosmo_factor values in input.")
    if convert_to_cm:
        ret_lst = _recursive_to_comoving(lst)
    else:
        ret_lst = lst
    return dict(
        args=ret_lst,
        kwargs=dict(),
        comoving=ret_cm,
        cosmo_factor=ret_cf,
        compression=ret_comp,
    )


@implements(np.block)
def block(arrays):
    from unyt._array_functions import block as unyt_block

    helper_result_block = _prepare_array_block_args(arrays)
    ret_cf = helper_result_block["cosmo_factor"]
    res = unyt_block(helper_result_block["args"])
    return _return_helper(res, helper_result_block, ret_cf)


# UNYT HAS A COPY-PASTED TYPO fft -> ftt


@implements(np.fft.fft)
def ftt_fft(a, n=None, axis=-1, norm=None, out=None):
    from unyt._array_functions import ftt_fft as unyt_fft_fft

    helper_result = _prepare_array_func_args(a, n=n, axis=axis, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_fft(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.fft2)
def ftt_fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    from unyt._array_functions import ftt_fft2 as unyt_fft_fft2

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_fft2(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.fftn)
def ftt_fftn(a, s=None, axes=None, norm=None, out=None):
    from unyt._array_functions import ftt_fftn as unyt_fft_fftn

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_fftn(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.hfft)
def ftt_hfft(a, n=None, axis=-1, norm=None, out=None):
    from unyt._array_functions import ftt_hfft as unyt_fft_hfft

    helper_result = _prepare_array_func_args(a, n=n, axis=axis, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_hfft(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.rfft)
def ftt_rfft(a, n=None, axis=-1, norm=None, out=None):
    from unyt._array_functions import ftt_rfft as unyt_fft_rfft

    helper_result = _prepare_array_func_args(a, n=n, axis=axis, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_rfft(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.rfft2)
def fft_rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    from unyt._array_functions import ftt_rfft2 as unyt_fft_rfft2

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_rfft2(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.rfftn)
def fft_rfftn(a, s=None, axes=None, norm=None, out=None):
    from unyt._array_functions import ftt_rfftn as unyt_fft_rfftn

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_rfftn(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.ifft)
def fft_ifft(a, n=None, axis=-1, norm=None, out=None):
    from unyt._array_functions import ftt_ifft as unyt_fft_ifft

    helper_result = _prepare_array_func_args(a, n=n, axis=axis, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_ifft(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.ifft2)
def fft_ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    from unyt._array_functions import ftt_ifft2 as unyt_fft_ifft2

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_ifft2(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.ifftn)
def fft_ifftn(a, s=None, axes=None, norm=None, out=None):
    from unyt._array_functions import ftt_ifftn as unyt_fft_ifftn

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_ifftn(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.ihfft)
def fft_ihfft(a, n=None, axis=-1, norm=None, out=None):
    from unyt._array_functions import ftt_ihfft as unyt_fft_ihfft

    helper_result = _prepare_array_func_args(a, n=n, axis=axis, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_ihfft(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.irfft)
def fft_irfft(a, n=None, axis=-1, norm=None, out=None):
    from unyt._array_functions import ftt_irfft as unyt_fft_irfft

    helper_result = _prepare_array_func_args(a, n=n, axis=axis, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_irfft(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.irfft2)
def fft_irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    from unyt._array_functions import ftt_irfft2 as unyt_fft_irfft2

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_irfft2(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.irfftn)
def fft_irfftn(a, s=None, axes=None, norm=None, out=None):
    from unyt._array_functions import ftt_irfftn as unyt_fft_irfftn

    helper_result = _prepare_array_func_args(a, s=s, axes=axes, norm=norm, out=out)
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_irfftn(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.fft.fftshift)
def fft_fftshift(x, axes=None):
    from unyt._array_functions import fft_fftshift as unyt_fft_fftshift

    helper_result = _prepare_array_func_args(x, axes=axes)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_fftshift(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.fft.ifftshift)
def fft_ifftshift(x, axes=None):
    from unyt._array_functions import fft_ifftshift as unyt_fft_ifftshift

    helper_result = _prepare_array_func_args(x, axes=axes)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_fft_ifftshift(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.sort_complex)
def sort_complex(a):
    from unyt._array_functions import sort_complex as unyt_sort_complex

    helper_result = _prepare_array_func_args(a)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_sort_complex(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    from unyt._array_functions import isclose as unyt_isclose

    helper_result = _prepare_array_func_args(
        a,
        b,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    ret_cf = _comparison_cosmo_factor(
        helper_result["ca_cfs"][0],
        helper_result["ca_cfs"][1],
        inputs=(a, b),
    )
    res = unyt_isclose(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    from unyt._array_functions import allclose as unyt_allclose

    helper_result = _prepare_array_func_args(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    ret_cf = _comparison_cosmo_factor(
        helper_result["ca_cfs"][0],
        helper_result["ca_cfs"][1],
        inputs=(a, b),
    )
    res = unyt_allclose(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.array_equal)
def array_equal(a1, a2, equal_nan=False):
    from unyt._array_functions import array_equal as unyt_array_equal

    helper_result = _prepare_array_func_args(a1, a2, equal_nan=equal_nan)
    ret_cf = _comparison_cosmo_factor(
        helper_result["ca_cfs"][0],
        helper_result["ca_cfs"][1],
        inputs=(a1, a2),
    )
    res = unyt_array_equal(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.array_equiv)
def array_equiv(a1, a2):
    from unyt._array_functions import array_equiv as unyt_array_equiv

    helper_result = _prepare_array_func_args(a1, a2)
    ret_cf = _comparison_cosmo_factor(
        helper_result["ca_cfs"][0],
        helper_result["ca_cfs"][1],
        inputs=(a1, a2),
    )
    res = unyt_array_equiv(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linspace)
def linspace(
    start,
    stop,
    num=50,
    endpoint=True,
    retstep=False,
    dtype=None,
    axis=0,
    *,
    device=None,
):
    from unyt._array_functions import linspace as unyt_linspace

    helper_result = _prepare_array_func_args(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
        device=device,
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    ress = unyt_linspace(*helper_result["args"], **helper_result["kwargs"])
    if retstep:
        return tuple(_return_helper(res, helper_result, ret_cf) for res in ress)
    else:
        return _return_helper(ress, helper_result, ret_cf)


@implements(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    from unyt._array_functions import logspace as unyt_logspace

    helper_result = _prepare_array_func_args(
        start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis
    )
    ret_cf = _preserve_cosmo_factor(helper_result["kw_ca_cfs"]["base"])
    res = unyt_logspace(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.geomspace)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    from unyt._array_functions import geomspace as unyt_geomspace

    helper_result = _prepare_array_func_args(
        start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_geomspace(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.copyto)
def copyto(dst, src, casting="same_kind", where=True):
    from unyt._array_functions import copyto as unyt_copyto

    helper_result = _prepare_array_func_args(dst, src, casting=casting, where=where)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][1])
    # must pass dst directly here because it's modified in-place
    if isinstance(src, cosmo_array):
        comoving = getattr(dst, "comoving", None)
        if comoving:
            src.convert_to_comoving()
        elif comoving is False:
            src.convert_to_physical()
    unyt_copyto(dst, src, **helper_result["kwargs"])


@implements(np.prod)
def prod(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=np._NoValue,
    initial=np._NoValue,
    where=np._NoValue,
):
    from unyt._array_functions import prod as unyt_prod

    helper_result = _prepare_array_func_args(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )
    res = unyt_prod(*helper_result["args"], **helper_result["kwargs"])
    ret_cf = _power_cosmo_factor(
        helper_result["ca_cfs"][0],
        (False, None),
        power=a.size // res.size,
    )
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.var)
def var(
    a,
    axis=None,
    dtype=None,
    out=None,
    ddof=0,
    keepdims=np._NoValue,
    *,
    where=np._NoValue,
    mean=np._NoValue,
    correction=np._NoValue,
):
    from unyt._array_functions import var as unyt_var

    helper_result = _prepare_array_func_args(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        mean=mean,
        correction=correction,
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_var(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    from unyt._array_functions import trace as unyt_trace

    helper_result = _prepare_array_func_args(
        a,
        offset=offset,
        axis1=axis1,
        axis2=axis2,
        dtype=dtype,
        out=out,
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_trace(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.percentile)
def percentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    from unyt._array_functions import percentile as unyt_percentile

    helper_result = _prepare_array_func_args(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        weights=weights,
        interpolation=interpolation,
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_percentile(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.quantile)
def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    from unyt._array_functions import quantile as unyt_quantile

    helper_result = _prepare_array_func_args(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        weights=weights,
        interpolation=interpolation,
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_quantile(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.nanpercentile)
def nanpercentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    from unyt._array_functions import nanpercentile as unyt_nanpercentile

    helper_result = _prepare_array_func_args(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        weights=weights,
        interpolation=interpolation,
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_nanpercentile(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.nanquantile)
def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    from unyt._array_functions import nanquantile as unyt_nanquantile

    helper_result = _prepare_array_func_args(
        a,
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        weights=weights,
        interpolation=interpolation,
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_nanquantile(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.linalg.det)
def linalg_det(a):
    from unyt._array_functions import linalg_det as unyt_linalg_det

    helper_result = _prepare_array_func_args(a)
    ret_cf = _power_cosmo_factor(
        helper_result["ca_cfs"][0],
        (False, None),
        power=a.shape[0],
    )
    res = unyt_linalg_det(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.diff)
def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    from unyt._array_functions import diff as unyt_diff

    helper_result = _prepare_array_func_args(
        a, n=n, axis=axis, prepend=prepend, append=append
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_diff(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.ediff1d)
def ediff1d(ary, to_end=None, to_begin=None):
    from unyt._array_functions import ediff1d as unyt_ediff1d

    helper_result = _prepare_array_func_args(ary, to_end=to_end, to_begin=to_begin)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_ediff1d(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.ptp)
def ptp(a, axis=None, out=None, keepdims=np._NoValue):
    from unyt._array_functions import ptp as unyt_ptp

    helper_result = _prepare_array_func_args(a, axis=axis, out=out, keepdims=keepdims)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_ptp(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(np.cumprod)
# def cumprod(...):
#    Omitted because unyt just raises if called.


@implements(np.pad)
def pad(array, pad_width, mode="constant", **kwargs):
    from unyt._array_functions import pad as unyt_pad

    helper_result = _prepare_array_func_args(array, pad_width, mode=mode, **kwargs)
    # the number of options is huge, including user defined functions to handle data
    # let's just preserve the cosmo_factor of the input `array` and trust the user...
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_pad(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.choose)
def choose(a, choices, out=None, mode="raise"):
    from unyt._array_functions import choose as unyt_choose

    helper_result = _prepare_array_func_args(a, choices, out=out, mode=mode)
    helper_result_choices = _prepare_array_func_args(*choices)
    ret_cf = _preserve_cosmo_factor(*helper_result_choices["ca_cfs"])
    res = unyt_choose(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.insert)
def insert(arr, obj, values, axis=None):
    from unyt._array_functions import insert as unyt_insert

    helper_result = _prepare_array_func_args(arr, obj, values, axis=axis)
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][2]
    )
    res = unyt_insert(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.lstsq)
def linalg_lstsq(a, b, rcond=None):
    from unyt._array_functions import linalg_lstsq as unyt_linalg_lstsq

    helper_result = _prepare_array_func_args(a, b, rcond=rcond)
    ret_cf = _divide_cosmo_factor(
        helper_result["ca_cfs"][1], helper_result["ca_cfs"][0]
    )
    resid_cf = _power_cosmo_factor(helper_result["ca_cfs"][1], (False, None), power=2)
    sing_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    ress = unyt_linalg_lstsq(*helper_result["args"], **helper_result["kwargs"])
    return (
        _return_helper(ress[0], helper_result, ret_cf),
        _return_helper(ress[1], helper_result, resid_cf),
        ress[2],
        _return_helper(ress[3], helper_result, sing_cf),
    )


@implements(np.linalg.solve)
def linalg_solve(a, b):
    from unyt._array_functions import linalg_solve as unyt_linalg_solve

    helper_result = _prepare_array_func_args(a, b)
    ret_cf = _divide_cosmo_factor(
        helper_result["ca_cfs"][1], helper_result["ca_cfs"][0]
    )
    res = unyt_linalg_solve(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


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


@implements(np.fill_diagonal)
def fill_diagonal(a, val, wrap=False):
    from unyt._array_functions import fill_diagonal as unyt_fill_diagonal

    helper_result = _prepare_array_func_args(a, val, wrap=wrap)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][1])
    # must pass a directly here because it's modified in-place
    comoving = getattr(a, "comoving", None)
    if comoving:
        val.convert_to_comoving()
    elif comoving is False:
        val.convert_to_physical()
    unyt_fill_diagonal(a, val, **helper_result["kwargs"])


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


@implements(np.linalg.outer)
def linalg_outer(x1, x2, /):
    from unyt._array_functions import linalg_outer as unyt_linalg_outer

    helper_result = _prepare_array_func_args(x1, x2)
    ret_cf = _multiply_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
    )
    res = unyt_linalg_outer(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


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
