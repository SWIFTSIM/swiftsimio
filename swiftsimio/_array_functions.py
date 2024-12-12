import numpy as np
from unyt import unyt_array, unyt_quantity
from unyt._array_functions import implements
from unyt._array_functions import _HANDLED_FUNCTIONS as _UNYT_HANDLED_FUNCTIONS
from .objects import _multiply_cosmo_factor, cosmo_array, cosmo_quantity

_HANDLED_FUNCTIONS = dict()


def arg_helper(*args, **kwargs):
    cms = [(hasattr(arg, "comoving"), getattr(arg, "comoving", None)) for arg in args]
    cfs = [
        (hasattr(arg, "cosmo_factor"), getattr(arg, "cosmo_factor", None))
        for arg in args
    ]
    comps = [
        (hasattr(arg, "compression"), getattr(arg, "compression", None)) for arg in args
    ]
    kw_cms = [
        (hasattr(arg, "comoving"), getattr(arg, "comoving", None)) for arg in args
    ]
    kw_cfs = [
        (hasattr(arg, "cosmo_factor"), getattr(arg, "cosmo_factor", None))
        for arg in args
    ]
    kw_comps = [
        (hasattr(arg, "compression"), getattr(arg, "compression", None)) for arg in args
    ]
    if all([cm[1] for cm in cms + kw_cms if cm[0]]):
        # all cosmo inputs are comoving
        ret_cm = True
    elif all([cm[1] is None for cm in cms + kw_cms if cm[0]]):
        # all cosmo inputs have comoving=None
        ret_cm = None
    elif any([cm[1] is None for cm in cms + kw_cms if cm[0]]):
        # only some cosmo inputs have comoving=None
        raise ValueError(
            "Some arguments have comoving=None and others have comoving=True|False. "
            "Result is undefined!"
        )
    elif not any([cm[1] for cm in cms + kw_cms if cm[0]]):
        # all cosmo_array inputs are physical
        ret_cm = False
    else:
        # mix of comoving and physical inputs
        args = [
            arg.to_comoving() if cm[0] and not cm[1] else arg
            for arg, cm in zip(args, cms)
        ]
        kwargs = [
            kwarg.to_comoving() if cm[0] and not cm[1] else kwarg
            for kwarg, cm in zip(kwargs, cms)
        ]
        ret_cm = True
    if len(set(comps + kw_comps)) == 1:
        # all compressions identical, preserve it
        ret_comp = comps[0]
    else:
        # mixed compressions, strip it off
        ret_comp = None
    return args, cfs, kwargs, kw_cfs, ret_cm, ret_comp


@implements(np.dot)
def dot(a, b, out=None):
    from unyt._array_functions import dot as unyt_dot

    args, cfs, kwargs, kw_cfs, ret_cm, ret_comp = arg_helper(a, b, out=out)
    ret_cf = (
        cfs[0][1] * cfs[1][1]
    )  # this needs helper function handling different cases
    res = unyt_dot(*args, **kwargs)
    if out is None:
        # should we be using __array_wrap__ somehow?
        if isinstance(res, unyt_quantity):
            return cosmo_quantity(
                res, comoving=ret_cm, cosmo_factor=ret_cf, compression=ret_comp
            )
        return cosmo_array(
            res, comoving=ret_cm, cosmo_factor=ret_cf, compression=ret_comp
        )
    if hasattr(out, "comoving"):
        out.comoving = ret_cm
    if hasattr(out, "cosmo_factor"):
        out.cosmo_factor = ret_cf
    if hasattr(out, "compression"):
        out.compression = ret_comp
    return cosmo_array(  # confused, do we set out, or return?
        res.to_value(res.units),
        res.units,
        bypass_validation=True,
        comoving=ret_cm,
        cosmo_factor=ret_cf,
        compression=ret_comp,
    )
