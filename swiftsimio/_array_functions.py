import numpy as np
from unyt import unyt_quantity
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
        return cosmo_array(
            res,
            comoving=helper_result["comoving"],
            cosmo_factor=ret_cf,
            compression=helper_result["compression"],
        )
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
        res = unyt_histogram_bin_edges(*helper_result["args"], **helper_result["kwargs"])
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
        a,
        rcond=rcond,
        hermitian=hermitian,
        rtol=rtol
    )
    ret_cf = _reciprocal_cosmo_factor(helper_result["ca_cfs"][0])
    res = unyt_linalg_pinv(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)


# @implements(...)
# def ...(...):
#     from unyt._array_functions import ... as unyt_...

#     helper_result = _prepare_array_func_args(...)
#     ret_cf = ...()
#     res = unyt_...(*helper_result["args"], **helper_result["kwargs"])
#     return _return_helper(res, helper_result, ret_cf, out=out)
