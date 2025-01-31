import warnings
import numpy as np
from unyt import unyt_quantity, unyt_array
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
    _return_without_cosmo_factor,
    _propagate_cosmo_array_attributes,
    _copy_cosmo_array_attributes,
)
from unyt._array_functions import (
    dot as unyt_dot,
    vdot as unyt_vdot,
    inner as unyt_inner,
    outer as unyt_outer,
    kron as unyt_kron,
    histogram_bin_edges as unyt_histogram_bin_edges,
    linalg_inv as unyt_linalg_inv,
    linalg_tensorinv as unyt_linalg_tensorinv,
    linalg_pinv as unyt_linalg_pinv,
    linalg_svd as unyt_linalg_svd,
    histogram as unyt_histogram,
    histogram2d as unyt_histogram2d,
    histogramdd as unyt_histogramdd,
    concatenate as unyt_concatenate,
    intersect1d as unyt_intersect1d,
    union1d as unyt_union1d,
    norm as unyt_linalg_norm,  # not linalg_norm, doesn't follow usual pattern
    vstack as unyt_vstack,
    hstack as unyt_hstack,
    dstack as unyt_dstack,
    column_stack as unyt_column_stack,
    stack as unyt_stack,
    around as unyt_around,
    block as unyt_block,
    ftt_fft as unyt_fft_fft,  # unyt has a copy-pasted typo fft -> ftt
    ftt_fft2 as unyt_fft_fft2,
    ftt_fftn as unyt_fft_fftn,
    ftt_hfft as unyt_fft_hfft,
    ftt_rfft as unyt_fft_rfft,
    ftt_rfft2 as unyt_fft_rfft2,
    ftt_rfftn as unyt_fft_rfftn,
    ftt_ifft as unyt_fft_ifft,
    ftt_ifft2 as unyt_fft_ifft2,
    ftt_ifftn as unyt_fft_ifftn,
    ftt_ihfft as unyt_fft_ihfft,
    ftt_irfft as unyt_fft_irfft,
    ftt_irfft2 as unyt_fft_irfft2,
    ftt_irfftn as unyt_fft_irfftn,
    fft_fftshift as unyt_fft_fftshift,
    fft_ifftshift as unyt_fft_ifftshift,
    sort_complex as unyt_sort_complex,
    isclose as unyt_isclose,
    allclose as unyt_allclose,
    array2string as unyt_array2string,
    cross as unyt_cross,
    array_equal as unyt_array_equal,
    array_equiv as unyt_array_equiv,
    linspace as unyt_linspace,
    logspace as unyt_logspace,
    geomspace as unyt_geomspace,
    copyto as unyt_copyto,
    prod as unyt_prod,
    var as unyt_var,
    trace as unyt_trace,
    percentile as unyt_percentile,
    quantile as unyt_quantile,
    nanpercentile as unyt_nanpercentile,
    nanquantile as unyt_nanquantile,
    linalg_det as unyt_linalg_det,
    diff as unyt_diff,
    ediff1d as unyt_ediff1d,
    ptp as unyt_ptp,
    pad as unyt_pad,
    choose as unyt_choose,
    insert as unyt_insert,
    linalg_lstsq as unyt_linalg_lstsq,
    linalg_solve as unyt_linalg_solve,
    linalg_tensorsolve as unyt_linalg_tensorsolve,
    linalg_eig as unyt_linalg_eig,
    linalg_eigh as unyt_linalg_eigh,
    linalg_eigvals as unyt_linalg_eigvals,
    linalg_eigvalsh as unyt_linalg_eigvalsh,
    savetxt as unyt_savetxt,
    fill_diagonal as unyt_fill_diagonal,
    isin as unyt_isin,
    place as unyt_place,
    put as unyt_put,
    put_along_axis as unyt_put_along_axis,
    putmask as unyt_putmask,
    searchsorted as unyt_searchsorted,
    select as unyt_select,
    setdiff1d as unyt_setdiff1d,
    sinc as unyt_sinc,
    clip as unyt_clip,
    where as unyt_where,
    triu as unyt_triu,
    tril as unyt_tril,
    einsum as unyt_einsum,
    convolve as unyt_convolve,
    correlate as unyt_correlate,
    tensordot as unyt_tensordot,
    unwrap as unyt_unwrap,
    interp as unyt_interp,
    array_repr as unyt_array_repr,
    linalg_outer as unyt_linalg_outer,
    trapezoid as unyt_trapezoid,
    isin as unyt_in1d,
    take as unyt_take,
)

_HANDLED_FUNCTIONS = {}

# first we define helper functions to handle repetitive operations in wrapping unyt &
# numpy functions (we will actually wrap the functions below):


def implements(numpy_function):
    """Register an __array_function__ implementation for cosmo_array objects."""

    # See NEP 18 https://numpy.org/neps/nep-0018-array-function-protocol.html
    def decorator(func):
        _HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def _return_helper(res, helper_result, ret_cf, out=None):
    if out is None:
        if isinstance(res, unyt_quantity) and not isinstance(res, cosmo_quantity):
            return cosmo_quantity(
                res,
                comoving=helper_result["comoving"],
                cosmo_factor=ret_cf,
                compression=helper_result["compression"],
            )
        elif isinstance(res, unyt_array) and not isinstance(res, cosmo_array):
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
    if res.shape == ():
        return cosmo_quantity(
            res.to_value(res.units),
            res.units,
            bypass_validation=True,
            comoving=helper_result["comoving"],
            cosmo_factor=ret_cf,
            compression=helper_result["compression"],
        )
    return cosmo_array(
        res.to_value(res.units),
        res.units,
        bypass_validation=True,
        comoving=helper_result["comoving"],
        cosmo_factor=ret_cf,
        compression=helper_result["compression"],
    )


def _default_unary_wrapper(unyt_func, cosmo_factor_wrapper):

    # assumes that we have one primary argument that will be handled
    # by the cosmo_factor_wrapper
    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        ret_cf = cosmo_factor_wrapper(helper_result["ca_cfs"][0])
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        if "out" in kwargs:
            return _return_helper(res, helper_result, ret_cf, out=kwargs["out"])
        else:
            return _return_helper(res, helper_result, ret_cf)

    return wrapper


def _default_binary_wrapper(unyt_func, cosmo_factor_wrapper):

    # assumes we have two primary arguments that will be handled
    # by the cosmo_factor_wrapper
    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        ret_cf = cosmo_factor_wrapper(
            helper_result["ca_cfs"][0], helper_result["ca_cfs"][1]
        )
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        if "out" in kwargs:
            return _return_helper(res, helper_result, ret_cf, out=kwargs["out"])
        else:
            return _return_helper(res, helper_result, ret_cf)

    return wrapper


def _default_comparison_wrapper(unyt_func):

    # assumes we have two primary arguments that will be handled with
    # _comparison_cosmo_factor with them as the inputs
    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        ret_cf = _comparison_cosmo_factor(
            helper_result["ca_cfs"][0],
            helper_result["ca_cfs"][1],
            inputs=args[:2],
        )
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        return _return_helper(res, helper_result, ret_cf)

    return wrapper


def _default_oplist_wrapper(unyt_func):

    # assumes first argument is a list of operands
    # assumes that we always preserve the cosmo factor of the first
    # element in the list of operands
    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        helper_result_oplist = _prepare_array_func_args(*args[0])
        ret_cf = _preserve_cosmo_factor(helper_result_oplist["ca_cfs"][0])
        res = unyt_func(
            helper_result_oplist["args"],
            *helper_result["args"][1:],
            **helper_result["kwargs"],
        )
        return _return_helper(res, helper_result_oplist, ret_cf)

    return wrapper


# Now we wrap functions that unyt handles explicitly (below that will be those not handled
# explicitly):


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


implements(np.dot)(_default_binary_wrapper(unyt_dot, _multiply_cosmo_factor))
implements(np.vdot)(_default_binary_wrapper(unyt_vdot, _multiply_cosmo_factor))
implements(np.inner)(_default_binary_wrapper(unyt_inner, _multiply_cosmo_factor))
implements(np.outer)(_default_binary_wrapper(unyt_outer, _multiply_cosmo_factor))
implements(np.kron)(_default_binary_wrapper(unyt_kron, _multiply_cosmo_factor))


@implements(np.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):

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


implements(np.linalg.inv)(
    _default_unary_wrapper(unyt_linalg_inv, _reciprocal_cosmo_factor)
)
implements(np.linalg.tensorinv)(
    _default_unary_wrapper(unyt_linalg_tensorinv, _reciprocal_cosmo_factor)
)
implements(np.linalg.pinv)(
    _default_unary_wrapper(unyt_linalg_pinv, _reciprocal_cosmo_factor)
)
implements(np.linalg.svd)(
    _default_unary_wrapper(unyt_linalg_svd, _preserve_cosmo_factor)
)


@implements(np.histogram)
def histogram(a, bins=10, range=None, density=None, weights=None):

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


implements(np.concatenate)(_default_oplist_wrapper(unyt_concatenate))
implements(np.cross)(_default_binary_wrapper(unyt_cross, _multiply_cosmo_factor))
implements(np.intersect1d)(
    _default_binary_wrapper(unyt_intersect1d, _preserve_cosmo_factor)
)
implements(np.union1d)(_default_binary_wrapper(unyt_union1d, _preserve_cosmo_factor))
implements(np.linalg.norm)(
    _default_unary_wrapper(unyt_linalg_norm, _preserve_cosmo_factor)
)
implements(np.vstack)(_default_oplist_wrapper(unyt_vstack))
implements(np.hstack)(_default_oplist_wrapper(unyt_hstack))
implements(np.dstack)(_default_oplist_wrapper(unyt_dstack))
implements(np.column_stack)(_default_oplist_wrapper(unyt_column_stack))
implements(np.stack)(_default_oplist_wrapper(unyt_stack))
implements(np.around)(_default_unary_wrapper(unyt_around, _preserve_cosmo_factor))


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
    # block is a special case since we need to recurse more than one level
    # down the list of arrays.
    helper_result_block = _prepare_array_block_args(arrays)
    ret_cf = helper_result_block["cosmo_factor"]
    res = unyt_block(helper_result_block["args"])
    return _return_helper(res, helper_result_block, ret_cf)


implements(np.fft.fft)(_default_unary_wrapper(unyt_fft_fft, _reciprocal_cosmo_factor))
implements(np.fft.fft2)(_default_unary_wrapper(unyt_fft_fft2, _reciprocal_cosmo_factor))
implements(np.fft.fftn)(_default_unary_wrapper(unyt_fft_fftn, _reciprocal_cosmo_factor))
implements(np.fft.hfft)(_default_unary_wrapper(unyt_fft_hfft, _reciprocal_cosmo_factor))
implements(np.fft.rfft)(_default_unary_wrapper(unyt_fft_rfft, _reciprocal_cosmo_factor))
implements(np.fft.rfft2)(
    _default_unary_wrapper(unyt_fft_rfft2, _reciprocal_cosmo_factor)
)
implements(np.fft.rfftn)(
    _default_unary_wrapper(unyt_fft_rfftn, _reciprocal_cosmo_factor)
)
implements(np.fft.ifft)(_default_unary_wrapper(unyt_fft_ifft, _reciprocal_cosmo_factor))
implements(np.fft.ifft2)(
    _default_unary_wrapper(unyt_fft_ifft2, _reciprocal_cosmo_factor)
)
implements(np.fft.ifftn)(
    _default_unary_wrapper(unyt_fft_ifftn, _reciprocal_cosmo_factor)
)
implements(np.fft.ihfft)(
    _default_unary_wrapper(unyt_fft_ihfft, _reciprocal_cosmo_factor)
)
implements(np.fft.irfft)(
    _default_unary_wrapper(unyt_fft_irfft, _reciprocal_cosmo_factor)
)
implements(np.fft.irfft2)(
    _default_unary_wrapper(unyt_fft_irfft2, _reciprocal_cosmo_factor)
)
implements(np.fft.irfftn)(
    _default_unary_wrapper(unyt_fft_irfftn, _reciprocal_cosmo_factor)
)
implements(np.fft.fftshift)(
    _default_unary_wrapper(unyt_fft_fftshift, _preserve_cosmo_factor)
)
implements(np.fft.ifftshift)(
    _default_unary_wrapper(unyt_fft_ifftshift, _preserve_cosmo_factor)
)

implements(np.sort_complex)(
    _default_unary_wrapper(unyt_sort_complex, _preserve_cosmo_factor)
)
implements(np.isclose)(_default_comparison_wrapper(unyt_isclose))
implements(np.allclose)(_default_comparison_wrapper(unyt_allclose))
implements(np.array_equal)(_default_comparison_wrapper(unyt_array_equal))
implements(np.array_equiv)(_default_comparison_wrapper(unyt_array_equiv))


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

    helper_result = _prepare_array_func_args(
        start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis
    )
    ret_cf = _preserve_cosmo_factor(helper_result["kw_ca_cfs"]["base"])
    res = unyt_logspace(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


implements(np.geomspace)(
    _default_binary_wrapper(unyt_geomspace, _preserve_cosmo_factor)
)


@implements(np.copyto)
def copyto(dst, src, casting="same_kind", where=True):

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


implements(np.var)(_default_unary_wrapper(unyt_var, _preserve_cosmo_factor))
implements(np.trace)(_default_unary_wrapper(unyt_trace, _preserve_cosmo_factor))
implements(np.percentile)(
    _default_unary_wrapper(unyt_percentile, _preserve_cosmo_factor)
)
implements(np.quantile)(_default_unary_wrapper(unyt_quantile, _preserve_cosmo_factor))
implements(np.nanpercentile)(
    _default_unary_wrapper(unyt_nanpercentile, _preserve_cosmo_factor)
)
implements(np.nanquantile)(
    _default_unary_wrapper(unyt_nanquantile, _preserve_cosmo_factor)
)


@implements(np.linalg.det)
def linalg_det(a):

    helper_result = _prepare_array_func_args(a)
    ret_cf = _power_cosmo_factor(
        helper_result["ca_cfs"][0],
        (False, None),
        power=a.shape[0],
    )
    res = unyt_linalg_det(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


implements(np.diff)(_default_unary_wrapper(unyt_diff, _preserve_cosmo_factor))
implements(np.ediff1d)(_default_unary_wrapper(unyt_ediff1d, _preserve_cosmo_factor))
implements(np.ptp)(_default_unary_wrapper(unyt_ptp, _preserve_cosmo_factor))
# implements(np.cumprod)(...) Omitted because unyt just raises if called.


@implements(np.pad)
def pad(array, pad_width, mode="constant", **kwargs):

    helper_result = _prepare_array_func_args(array, pad_width, mode=mode, **kwargs)
    # the number of options is huge, including user defined functions to handle data
    # let's just preserve the cosmo_factor of the input `array` and trust the user...
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_pad(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.choose)
def choose(a, choices, out=None, mode="raise"):

    helper_result = _prepare_array_func_args(a, choices, out=out, mode=mode)
    helper_result_choices = _prepare_array_func_args(*choices)
    ret_cf = _preserve_cosmo_factor(*helper_result_choices["ca_cfs"])
    res = unyt_choose(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.insert)
def insert(arr, obj, values, axis=None):

    helper_result = _prepare_array_func_args(arr, obj, values, axis=axis)
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0], helper_result["ca_cfs"][2]
    )
    res = unyt_insert(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.lstsq)
def linalg_lstsq(a, b, rcond=None):

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

    helper_result = _prepare_array_func_args(a, b)
    ret_cf = _divide_cosmo_factor(
        helper_result["ca_cfs"][1], helper_result["ca_cfs"][0]
    )
    res = unyt_linalg_solve(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.tensorsolve)
def linalg_tensorsolve(a, b, axes=None):

    helper_result = _prepare_array_func_args(a, b, axes=axes)
    ret_cf = _divide_cosmo_factor(
        helper_result["ca_cfs"][1], helper_result["ca_cfs"][0]
    )
    res = unyt_linalg_tensorsolve(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.eig)
def linalg_eig(a):

    helper_result = _prepare_array_func_args(a)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    ress = unyt_linalg_eig(*helper_result["args"], **helper_result["kwargs"])
    return (
        _return_helper(ress[0], helper_result, ret_cf),
        ress[1],
    )


@implements(np.linalg.eigh)
def linalg_eigh(a, UPLO="L"):

    helper_result = _prepare_array_func_args(a, UPLO=UPLO)
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][0])
    ress = unyt_linalg_eigh(*helper_result["args"], **helper_result["kwargs"])
    return (
        _return_helper(ress[0], helper_result, ret_cf),
        ress[1],
    )


implements(np.linalg.eigvals)(
    _default_unary_wrapper(unyt_linalg_eigvals, _preserve_cosmo_factor)
)
implements(np.linalg.eigvalsh)(
    _default_unary_wrapper(unyt_linalg_eigvalsh, _preserve_cosmo_factor)
)


@implements(np.savetxt)
def savetxt(
    fname,
    X,
    fmt="%.18e",
    delimiter=" ",
    newline="\n",
    header="",
    footer="",
    comments="# ",
    encoding=None,
):

    warnings.warn(
        "numpy.savetxt does not preserve units or cosmo_array information, "
        "and will only save the raw numerical data from the cosmo_array object.\n"
        "If this is the intended behaviour, call `numpy.savetxt(file, arr.d)` "
        "to silence this warning.\n",
        stacklevel=4,
    )
    helper_result = _prepare_array_func_args(
        fname,
        X,
        fmt=fmt,
        delimiter=delimiter,
        newline=newline,
        header=header,
        footer=footer,
        comments=comments,
        encoding=encoding,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="numpy.savetxt does not preserve units",
        )
        unyt_savetxt(*helper_result["args"], **helper_result["kwargs"])
    return


@implements(np.apply_over_axes)
def apply_over_axes(func, a, axes):
    res = func(a, axes[0])
    if len(axes) > 1:
        # this function is recursive by nature,
        # here we intentionally do not call the base _implementation
        return np.apply_over_axes(func, res, axes[1:])
    else:
        return res


@implements(np.fill_diagonal)
def fill_diagonal(a, val, wrap=False):

    helper_result = _prepare_array_func_args(a, val, wrap=wrap)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][1])
    # must pass a directly here because it's modified in-place
    comoving = getattr(a, "comoving", None)
    if comoving:
        val.convert_to_comoving()
    elif comoving is False:
        val.convert_to_physical()
    unyt_fill_diagonal(a, val, **helper_result["kwargs"])


implements(np.isin)(_default_comparison_wrapper(unyt_isin))


@implements(np.place)
def place(arr, mask, vals):

    helper_result = _prepare_array_func_args(arr, mask, vals)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(vals, cosmo_array):
        comoving = getattr(arr, "comoving", None)
        if comoving:
            vals.convert_to_comoving()
        elif comoving is False:
            vals.convert_to_physical()
    unyt_place(arr, mask, vals)


@implements(np.put)
def put(a, ind, v, mode="raise"):

    helper_result = _prepare_array_func_args(a, ind, v, mode=mode)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(v, cosmo_array):
        comoving = getattr(a, "comoving", None)
        if comoving:
            v.convert_to_comoving()
        elif comoving is False:
            v.convert_to_physical()
    unyt_put(a, ind, v, **helper_result["kwargs"])


@implements(np.put_along_axis)
def put_along_axis(arr, indices, values, axis):

    helper_result = _prepare_array_func_args(arr, indices, values, axis)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(values, cosmo_array):
        comoving = getattr(arr, "comoving", None)
        if comoving:
            values.convert_to_comoving()
        elif comoving is False:
            values.convert_to_physical()
    unyt_put_along_axis(arr, indices, values, axis)


@implements(np.putmask)
def putmask(a, mask, values):

    helper_result = _prepare_array_func_args(a, mask, values)
    _preserve_cosmo_factor(helper_result["ca_cfs"][0], helper_result["ca_cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(values, cosmo_array):
        comoving = getattr(a, "comoving", None)
        if comoving:
            values.convert_to_comoving()
        elif comoving is False:
            values.convert_to_physical()
    unyt_putmask(a, mask, values)


implements(np.searchsorted)(
    _default_binary_wrapper(unyt_searchsorted, _return_without_cosmo_factor)
)


@implements(np.select)
def select(condlist, choicelist, default=0):

    helper_result = _prepare_array_func_args(condlist, choicelist, default=default)
    helper_result_choicelist = _prepare_array_func_args(*choicelist)
    ret_cf = _preserve_cosmo_factor(*helper_result_choicelist["ca_cfs"])
    res = unyt_select(
        helper_result["args"][0],
        helper_result_choicelist["args"],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result, ret_cf)


implements(np.setdiff1d)(
    _default_binary_wrapper(unyt_setdiff1d, _preserve_cosmo_factor)
)


@implements(np.sinc)
def sinc(x):

    # unyt just casts to array and calls the numpy implementation
    # so let's just hand off to them
    return unyt_sinc(x)


@implements(np.clip)
def clip(
    a,
    a_min=np._NoValue,
    a_max=np._NoValue,
    out=None,
    *,
    min=np._NoValue,
    max=np._NoValue,
    **kwargs,
):

    # can't work out how to properly handle min and max,
    # just leave them in kwargs I guess (might be a numpy version conflict?)
    helper_result = _prepare_array_func_args(
        a,
        a_min=a_min,
        a_max=a_max,
        out=out,
        **kwargs,
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0],
        helper_result["kw_ca_cfs"]["a_min"],
        helper_result["kw_ca_cfs"]["a_max"],
    )
    res = unyt_clip(
        helper_result["args"][0],
        helper_result["kwargs"]["a_min"],
        helper_result["kwargs"]["a_max"],
        out=helper_result["kwargs"]["out"],
        **kwargs,
    )
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.where)
def where(condition, *args):

    helper_result = _prepare_array_func_args(condition, *args)
    if len(args) == 0:  # just condition
        ret_cf = _return_without_cosmo_factor(helper_result["ca_cfs"][0])
        res = unyt_where(*helper_result["args"], **helper_result["kwargs"])
    elif len(args) < 2:
        # error message borrowed from numpy 1.24.1
        raise ValueError("either both or neither of x and y should be given")
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][1], helper_result["ca_cfs"][2]
    )
    res = unyt_where(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


implements(np.triu)(_default_unary_wrapper(unyt_triu, _preserve_cosmo_factor))
implements(np.tril)(_default_unary_wrapper(unyt_tril, _preserve_cosmo_factor))


@implements(np.einsum)
def einsum(
    subscripts,
    *operands,
    out=None,
    dtype=None,
    order="K",
    casting="safe",
    optimize=False,
):

    helper_result = _prepare_array_func_args(
        subscripts,
        operands,
        out=out,
        dtype=dtype,
        order=order,
        casting=casting,
        optimize=optimize,
    )
    helper_result_operands = _prepare_array_func_args(*operands)
    ret_cf = _preserve_cosmo_factor(*helper_result_operands["ca_cfs"])
    res = unyt_einsum(
        helper_result["args"][0],
        *helper_result_operands["args"],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result_operands, ret_cf, out=out)


implements(np.convolve)(_default_binary_wrapper(unyt_convolve, _multiply_cosmo_factor))
implements(np.correlate)(
    _default_binary_wrapper(unyt_correlate, _multiply_cosmo_factor)
)
implements(np.tensordot)(
    _default_binary_wrapper(unyt_tensordot, _multiply_cosmo_factor)
)


@implements(np.unwrap)
def unwrap(p, discont=None, axis=-1, *, period=6.283185307179586):

    helper_result = _prepare_array_func_args(
        p, discont=discont, axis=axis, period=period
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["ca_cfs"][0],
        helper_result["kw_ca_cfs"]["discont"],
        helper_result["kw_ca_cfs"]["period"],
    )
    res = unyt_unwrap(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.interp)
def interp(x, xp, fp, left=None, right=None, period=None):

    helper_result = _prepare_array_func_args(
        x, xp, fp, left=left, right=right, period=period
    )
    ret_cf = _preserve_cosmo_factor(helper_result["ca_cfs"][2])
    res = unyt_interp(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.array_repr)
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):

    helper_result = _prepare_array_func_args(
        arr,
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )
    rep = unyt_array_repr(*helper_result["args"], **helper_result["kwargs"])[:-1]
    if hasattr(arr, "comoving"):
        rep += f", comoving='{arr.comoving}'"
    if hasattr(arr, "cosmo_factor"):
        rep += f", cosmo_factor='{arr.cosmo_factor}'"
    if hasattr(arr, "valid_transform"):
        rep += f", valid_transform='{arr.valid_transform}'"
    rep += ")"
    return rep


implements(np.linalg.outer)(
    _default_binary_wrapper(unyt_linalg_outer, _multiply_cosmo_factor)
)


@implements(np.trapezoid)
def trapezoid(y, x=None, dx=1.0, axis=-1):

    helper_result = _prepare_array_func_args(y, x=x, dx=dx, axis=axis)
    if x is None:
        ret_cf = _multiply_cosmo_factor(
            helper_result["ca_cfs"][0], helper_result["kw_ca_cfs"]["dx"]
        )
    else:
        ret_cf = _multiply_cosmo_factor(
            helper_result["ca_cfs"][0], helper_result["kw_ca_cfs"]["x"]
        )
    res = unyt_trapezoid(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


implements(np.in1d)(_default_comparison_wrapper(unyt_in1d))
implements(np.take)(_default_unary_wrapper(unyt_take, _preserve_cosmo_factor))

# Now we wrap functions that unyt does not handle explicitly:

implements(np.average)(_propagate_cosmo_array_attributes(np.average._implementation))
implements(np.max)(_propagate_cosmo_array_attributes(np.max._implementation))
implements(np.min)(_propagate_cosmo_array_attributes(np.min._implementation))
implements(np.mean)(_propagate_cosmo_array_attributes(np.mean._implementation))
implements(np.median)(_propagate_cosmo_array_attributes(np.median._implementation))
implements(np.sort)(_propagate_cosmo_array_attributes(np.sort._implementation))
implements(np.sum)(_propagate_cosmo_array_attributes(np.sum._implementation))
implements(np.partition)(
    _propagate_cosmo_array_attributes(np.partition._implementation)
)


@implements(np.meshgrid)
def meshgrid(*xi, **kwargs):
    # meshgrid is a unique case: arguments never interact with each other, so we don't
    # want to use our _prepare_array_func_args helper (that will try to coerce to
    # compatible comoving, cosmo_factor).
    # However we can't just use _propagate_cosmo_array_attributes because we need to
    # iterate over arguments.
    res = np.meshgrid._implementation(*xi, **kwargs)
    return tuple(_copy_cosmo_array_attributes(x, r) for (x, r) in zip(xi, res))
