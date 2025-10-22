"""
Overloaded implementations of unyt and numpy functions to correctly handle
:class:`~swiftsimio.objects.cosmo_array` input.

This module also defines wrappers and helper functions to facilitate overloading
functions and handling the processing of our custom array attributes.

Nothing in this module is intended to be user-facing, but the helpers and wrappers
are documented to assist in maintenance and development of swiftsimio.
"""

import warnings
from functools import reduce
import numpy as np
from typing import Callable, Tuple, Optional
import unyt
from unyt import unyt_quantity, unyt_array
from swiftsimio import objects
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
    fft_fft as unyt_fft_fft,
    fft_fft2 as unyt_fft_fft2,
    fft_fftn as unyt_fft_fftn,
    fft_hfft as unyt_fft_hfft,
    fft_rfft as unyt_fft_rfft,
    fft_rfft2 as unyt_fft_rfft2,
    fft_rfftn as unyt_fft_rfftn,
    fft_ifft as unyt_fft_ifft,
    fft_ifft2 as unyt_fft_ifft2,
    fft_ifftn as unyt_fft_ifftn,
    fft_ihfft as unyt_fft_ihfft,
    fft_irfft as unyt_fft_irfft,
    fft_irfft2 as unyt_fft_irfft2,
    fft_irfftn as unyt_fft_irfftn,
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


def _copy_cosmo_array_attributes_if_present(
    from_ca: object, to_ca: object, copy_units=False
) -> object:
    """
    Copy :class:`~swiftsimio.objects.cosmo_array` attributes across two objects.

    Copies the ``cosmo_factor``, ``comoving``, ``valid_transform`` and ``compression``
    attributes across if both the source and destination objects are
    :class:`~swiftsimio.objects.cosmo_array` instances (else returns input).

    Parameters
    ----------
    from_ca : :obj:`object`
        The source object.

    to_ca : :obj:`object`
        The destination object.

    copy_units : bool
        If ``True`` also copy ``units`` attribute (usually let :mod:`unyt` handle this).

    Returns
    -------
    out : :obj:`object`
        The destination object (with attributes copied if copy occurred).
    """
    if not (
        isinstance(to_ca, objects.cosmo_array)
        and isinstance(from_ca, objects.cosmo_array)
    ):
        return to_ca
    if copy_units:
        to_ca.units = from_ca.units
    to_ca.cosmo_factor = from_ca.cosmo_factor
    to_ca.comoving = from_ca.comoving
    to_ca.valid_transform = from_ca.valid_transform
    to_ca.compression = from_ca.compression
    return to_ca


def _propagate_cosmo_array_attributes_to_result(func: Callable) -> Callable:
    """
    Wrapper that copies :class:`~swiftsimio.objects.cosmo_array` attributes from first
    input argument to first output.

    Many functions take one input (or have a first input that has a close correspondance
    to the output) and one output. This helper copies the ``cosmo_factor``, ``comoving``,
    ``valid_transform`` and ``compression`` attributes from the first input argument to
    the output. Can be used as a decorator on functions (the first argument is then the
    first argument of the function) or methods (the first argument is then ``self``).
    If the output is not a :class:`~swiftsimio.objects.cosmo_array` it is not promoted
    (and then no attributes are copied).

    Parameters
    ----------
    func : callable
        The function whose argument attributes will be copied to its result.

    Returns
    -------
    out : callable
        The wrapped function.
    """

    def wrapped(obj, *args, **kwargs):
        # omit docstring so that sphinx picks up docstring of wrapped function
        return _copy_cosmo_array_attributes_if_present(obj, func(obj, *args, **kwargs))

    return wrapped


def _promote_unyt_to_cosmo(input_object: object) -> object:
    """
    Upgrades the input unyt instance to its cosmo equivalent.

    In many cases we can obtain a unyt class instance and want to promote it to its cosmo
    equivalent to attach our cosmo attributes. This helper promotes an input
    :class:`~unyt.array.unyt_array` to a :class:`~swiftsimio.objects.cosmo_array` or an
    input :class:`~unyt.array.unyt_quantity` to a
    :class:`~swiftsimio.objects.cosmo_quantity`. If the input is neither type, it is just
    returned.

    Parameters
    ----------
    input_object : :obj:`object`
        Object to consider for promotion from unyt instance to cosmo instance.
    """
    if isinstance(input_object, unyt_quantity) and not isinstance(
        input_object, objects.cosmo_quantity
    ):
        return input_object.view(objects.cosmo_quantity)
    elif isinstance(input_object, unyt_array) and not isinstance(
        input_object, objects.cosmo_array
    ):
        return input_object.view(objects.cosmo_array)
    else:
        return input_object


def _ensure_array_or_quantity_matches_shape(input_object: object) -> object:
    """
    Convert scalars to :class:`~swiftsimio.objects.cosmo_quantity` and arrays to
    :class:`~swiftsimio.objects.cosmo_array`.

    Scalar quantities are meant to be contained in
    :class:`~swiftsimio.objects.cosmo_quantity` and arrays in
    :class:`~swiftsimio.objects.cosmo_array`. Many functions (e.g. from numpy) can change
    the data contents without changing the containing class. This helper checks the input
    to make sure the data match the container type and converts if not.

    Parameters
    ----------
    input_object : :obj:`object`
        The object whose data is to be checked against its type.

    Returns
    -------
    out : :obj:`object`
        A version of the input with container type matching data contents.
    """
    if (
        isinstance(input_object, objects.cosmo_array)
        and not isinstance(input_object, objects.cosmo_quantity)
        and input_object.shape == ()
    ):
        return input_object.view(objects.cosmo_quantity)
    elif isinstance(input_object, objects.cosmo_quantity) and input_object.shape != ():
        return input_object.view(objects.cosmo_array)
    else:
        return input_object


def _ensure_result_is_cosmo_array_or_quantity(func: Callable) -> Callable:
    """
    Wrapper that converts any :class:`~unyt.array.unyt_array` or
    :class:`~unyt.array.unyt_quantity` instances in function output to cosmo equivalents.

    If the wrapped function returns a :obj:`tuple` (as many numpy functions do) it is
    iterated over (but not recursively) and each element with a unyt class type is
    upgraded to its cosmo equivalent. If anything but a :obj:`tuple` is returned, that
    object is promoted to the cosmo equivalent if it is of a unyt class type.

    Parameters
    ----------
    func : Callable
        The function whose result(s) will be upgraded to
        :class:`~swiftsimio.objects.cosmo_array` or
        :class:`~swifsimio.objects.cosmo_quantity`.
    Returns
    -------
    out : Callable
        The wrapped function.
    """

    def wrapped(*args, **kwargs) -> object:
        # omit docstring so that sphinx picks up docstring of wrapped function
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(
                _ensure_array_or_quantity_matches_shape(_promote_unyt_to_cosmo(item))
                for item in result
            )
        else:
            return _ensure_array_or_quantity_matches_shape(
                _promote_unyt_to_cosmo(result)
            )

    return wrapped


def _sqrt_cosmo_factor(cf: "objects.cosmo_factor", **kwargs) -> "objects.cosmo_factor":
    """
    Take the square root of a :class:`~swiftsimio.objects.cosmo_factor`.

    Parameters
    ----------
    cf : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` whose square root should be taken.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The square root of the input :class:`~swiftsimio.objects.cosmo_factor`.
    """
    return _power_cosmo_factor(cf, None, power=0.5)


def _multiply_cosmo_factor(
    *cfs: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Recursively multiply :class:`~swiftsimio.objects.cosmo_factor`s.

    All arguments are expected to be of type :class:`~swiftsimio.objects.cosmo_factor`.
    They are cumumatively multipled together and the result returned.

    Parameters
    ----------
    cfs : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor`s to be multiplied.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The product of the input :class:`~swiftsimio.objects.cosmo_factor`s.
    """
    return reduce(__binary_multiply_cosmo_factor, cfs)


def __binary_multiply_cosmo_factor(
    cf1: "objects.cosmo_factor", cf2: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Multiply two :class:`~swiftsimio.objects.cosmo_factor`s.

    Not intended for direct use but only as a helper for
    :func:`~swiftsimio._array_functions._multiply_cosmo_factor`.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        The first :class:`~swiftsimio.objects.cosmo_factor`.

    cf2 : swiftsimio.objects.cosmo_factor
        The second :class:`~swiftsimio.objects.cosmo_factor`.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The product of the :class:`~swiftsimio.objects.cosmo_factor`s.
    """
    if (cf1 is None) and (cf2 is None):
        # neither has cosmo_factor information:
        return None
    elif (cf1 is None) and (cf2 is not None):
        # first has no cosmo information, allow e.g. multiplication by constants:
        return cf2
    elif (cf1 is not None) and (cf2 is None):
        # second has no cosmo information, allow e.g. multiplication by constants:
        return cf1
    elif (cf1 is not None) and (cf2 is not None):
        # both cosmo_array and both with cosmo_factor:
        return cf1 * cf2  # cosmo_factor.__mul__ raises if scale factors differ


def _preserve_cosmo_factor(
    *cfs: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Helper to preserve the :class:`~swiftsimio.objects.cosmo_factor` of input.

    If there is a single argument, return its ``cosmo_factor``. If there are multiple
    arguments, check that they all have matching ``cosmo_factor``. Any arguments that
    are not :class:`~swiftsimio.objects.cosmo_array`s are ignored for this purpose.

    Parameters
    ----------
    cfs : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor`s to be preserved.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The preserved :class:`~swiftsimio.objects.cosmo_factor`.
    """
    return reduce(__binary_preserve_cosmo_factor, cfs)


def __binary_preserve_cosmo_factor(
    cf1: "objects.cosmo_factor", cf2: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Given two :class:`~swiftsimio.objects.cosmo_factor`s, get it if they match.

    Not intended for direct use but only as a helper for
    :func:`~swiftsimio._array_functions._preserve_cosmo_factor`. If the two inputs
    are compatible, return the compatible :class:`~swiftsimio.objects.cosmo_factor`.
    If one of them is ``None``, produce a warning. If they are incompatible, raise.


    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        The first :class:`~swiftsimio.objects.cosmo_factor`.

    cf2 : swiftsimio.objects.cosmo_factor
        The second :class:`~swiftsimio.objects.cosmo_factor`.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The preserved :class:`~swiftsimio.objects.cosmo_factor`s.

    Raises
    ------
    ValueError
        Raised if the two arguments are :class:`~swiftsimio.objects.cosmo_factor`s
        that do not have matching attributes.
    """
    if (cf1 is None) and (cf2 is None):
        # neither has cosmo_factor information:
        return None
    elif (cf1 is not None) and (cf2 is None):
        # only one is cosmo_array
        warnings.warn(
            f"Mixing arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf1}) for all arguments.",
            RuntimeWarning,
        )
        return cf1
    elif (cf1 is None) and (cf2 is not None):
        # only one is cosmo_array
        warnings.warn(
            f"Mixing arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf2}) for all arguments.",
            RuntimeWarning,
        )
        return cf2
    elif cf1 != cf2:
        raise ValueError(f"Arguments have cosmo_factors that differ: {cf1} and {cf2}.")
    else:  # cf1 == cf2
        return cf1  # or cf2, they're equal


def _power_cosmo_factor(
    cf1: "objects.cosmo_factor",
    cf2: "objects.cosmo_factor",
    inputs: "Optional[Tuple[objects.cosmo_array]]" = None,
    power: Optional[float] = None,
) -> "objects.cosmo_factor":
    """
    Raise a :class:`~swiftsimio.objects.cosmo_factor` to a power of another
    :class:`~swiftsimio.objects.cosmo_factor`.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` attached to the base.

    cf2 : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` attached to the exponent.

    inputs : :obj:`tuple`
        The objects that ``cf1`` and ``cf2`` are attached to. Give ``inputs`` or
        ``power``, not both.

    power : float
        The power to raise ``cf1`` to. Give ``inputs`` or ``power``, not both.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The exponentiated :class:`~swiftsimio.objects.cosmo_factor`s.

    Raises
    ------
    ValueError
        If the exponent is not a dimensionless quantity, or the exponent has a
        ``cosmo_factor`` whose scaling with scale factor is not ``1.0``.
    """
    if (inputs is not None and power is not None) or (inputs is None and power is None):
        raise ValueError
    power = inputs[1] if inputs else power
    if hasattr(power, "units"):
        if not power.units.is_dimensionless:
            raise ValueError("Exponent must be dimensionless.")
        elif power.units is not unyt.dimensionless:
            power = power.to_value(unyt.dimensionless)
        # else power.units is unyt.dimensionless, do nothing
    exp_afactor = getattr(cf2, "a_factor", None)
    if (exp_afactor is not None) and (exp_afactor != 1.0):
        raise ValueError("Exponent has scaling with scale factor != 1.")
    if cf1 is None:
        return None
    return np.power(cf1, power)


def _square_cosmo_factor(
    cf: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Square a :class:`~swiftsimio.objects.cosmo_factor`.

    Parameters
    ----------
    cf : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` to square.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The squared :class:`~swiftsimio.objects.cosmo_factor`.
    """
    return _power_cosmo_factor(cf, None, power=2)


def _cbrt_cosmo_factor(cf: "objects.cosmo_factor", **kwargs) -> "objects.cosmo_factor":
    """
    Take the cube root of a :class:`~swiftsimio.objects.cosmo_factor`.

    Parameters
    ----------
    cf : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` whose cube root should be taken.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The cube root of the input :class:`~swiftsimio.objects.cosmo_factor`.
    """
    return _power_cosmo_factor(cf, None, power=1.0 / 3.0)


def _divide_cosmo_factor(
    cf1: "objects.cosmo_factor", cf2: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Divide two :class:`~swiftsimio.objects.cosmo_factor`s.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        Numerator :class:`~swiftsimio.objects.cosmo_factor`.

    cf1 : swiftsimio.objects.cosmo_factor
        Denominator :class:`~swiftsimio.objects.cosmo_factor`.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The ratio of the input :class:`~swiftsimio.objects.cosmo_factor`s.
    """
    return _multiply_cosmo_factor(cf1, _reciprocal_cosmo_factor(cf2))


def _reciprocal_cosmo_factor(
    cf: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Take the inverse of a :class:`~swiftsimio.objects.cosmo_factor`.

    Parameters
    ----------
    cf : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` to be inverted.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The inverted :class:`~swiftsimio.objects.cosmo_factor`.
    """
    return _power_cosmo_factor(cf, None, power=-1)


def _passthrough_cosmo_factor(
    cf: "objects.cosmo_factor", cf2: "Optional[objects.cosmo_factor]" = None, **kwargs
) -> "objects.cosmo_factor":
    """
    Preserve a :class:`~swiftsimio.objects.cosmo_factor`, optionally checking that it
    matches a second :class:`~swiftsimio.objects.cosmo_factor`.

    This helper is intended for e.g. numpy ufuncs with a second dimensionless argument
    so it's ok if ``cf2`` is ``None`` and ``cf1`` is not.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` to pass through.

    cf2 : swiftsimio.objects.cosmo_factor
        Optional second :class:`~swiftsimio.objects.cosmo_factor` to check matches.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The input :class:`~swiftsimio.objects.cosmo_factor`.

    Raises
    ------
    ValueError
        If ``cf2`` is provided, is not ``None`` and does not match ``cf1``.
    """
    if (cf2 is not None) and cf != cf2:
        # if both have cosmo_factor information and it differs this is an error
        raise ValueError(f"Arguments have cosmo_factors that differ: {cf} and {cf2}.")
    else:
        return cf


def _return_without_cosmo_factor(
    cf: "objects.cosmo_factor",
    cf2: "objects.cosmo_factor" = np._NoValue,
    zero_comparison: Optional[bool] = None,
    **kwargs,
) -> None:
    """
    Return ``None``, but first check that argument
    :class:`~swiftsimio.objects.cosmo_factor`s match, raising or warning if not.

    Comparisons are a special case that wraps around this wrapper, see
    :func:`~swiftsimio._array_functions._comparison_cosmo_factor`.

    We borrow ``np._NoValue`` as a default for ``cf2`` because ``None`` here
    represents the absence of a :class:`~swiftsimio.objects.cosmo_factor`, and
    we need to handle that case.

    Warnings are produced when one argument has no cosmo factor, relevant e.g.
    when comparing to constants. We handle comparison with zero, that is unambiguous,
    as a special case.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        The :class:`~swiftsimio.objects.cosmo_factor` to discard.

    cf2 : swiftsimio.objects.cosmo_factor
        Optional second :class:`~swiftsimio.objects.cosmo_factor` to check for a
        match with ``cf1``.

    zero_comparison : bool
        If ``True``, silences warnings when exactly one of ``cf1`` and ``cf2`` is
        ``None``. Enables comparing with zero without warning.

    Returns
    -------
    out : None
        The :class:`~swiftsimio.objects.cosmo_factor` is discarded.

    Raises
    ------
    ValueError
        If ``cf2`` is provided, is not ``None`` and does not match ``cf1``.
    """
    if cf2 is np._NoValue:
        # there was no second argument, return promptly
        return None
    if (cf is not None) and (cf2 is None):
        # one is not a cosmo_array, warn on e.g. comparison to constants:
        if not zero_comparison:
            warnings.warn(
                f"Mixing arguments with and without cosmo_factors, continuing"
                f" assuming provided cosmo_factor ({cf}) for all arguments.",
                RuntimeWarning,
            )
    elif (cf is None) and (cf2 is not None):
        # two is not a cosmo_array, warn on e.g. comparison to constants:
        if not zero_comparison:
            warnings.warn(
                f"Mixing arguments with and without cosmo_factors, continuing"
                f" assuming provided cosmo_factor ({cf2}) for all arguments.",
                RuntimeWarning,
            )
    elif (cf is not None) and (cf2 is not None) and (cf != cf2):
        # both have cosmo_factor, don't match:
        raise ValueError(f"Arguments have cosmo_factors that differ: {cf} and {cf2}.")
    elif (cf is not None) and (cf2 is not None) and (cf == cf2):
        # both have cosmo_factor, and they match:
        pass
    # return without cosmo_factor
    return None


def _arctan2_cosmo_factor(
    cf1: "objects.cosmo_factor", cf2: "objects.cosmo_factor", **kwargs
) -> "objects.cosmo_factor":
    """
    Helper specifically to handle the :class:`~swiftsimio.objects.cosmo_factor`s for the
    ``arctan2`` ufunc from numpy.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` for the first ``arctan2`` argument.

    cf2 : swiftsimio.objects.cosmo_factor
        :class:`~swiftsimio.objects.cosmo_factor` for the second ``arctan2`` argument.

    Returns
    -------
    out : swiftsimio.objects.cosmo_factor
        The :class:`~swiftsimio.objects.cosmo_factor` for the ``arctan2`` result.

    Raises
    ------
    ValueError
        If the input :class:`~swiftsimio.objects.cosmo_factor`s differ, they will
        not cancel out and this is an error.

    """
    if (cf1 is None) and (cf2 is None):
        return None
    elif (cf1 is None) and (cf2 is not None):
        warnings.warn(
            f"Mixing arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf2}) for all arguments.",
            RuntimeWarning,
        )
        return objects.cosmo_factor(objects.a**0, scale_factor=cf2.scale_factor)
    elif (cf1 is not None) and (cf2 is None):
        warnings.warn(
            f"Mixing arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf1}) for all arguments.",
            RuntimeWarning,
        )
        return objects.cosmo_factor(objects.a**0, scale_factor=cf1.scale_factor)
    elif (cf1 is not None) and (cf2 is not None) and (cf1 != cf2):
        raise ValueError(f"Arguments have cosmo_factors that differ: {cf1} and {cf2}.")
    elif (cf1 is not None) and (cf2 is not None) and (cf1 == cf2):
        return objects.cosmo_factor(objects.a**0, scale_factor=cf1.scale_factor)


def _comparison_cosmo_factor(
    cf1: "objects.cosmo_factor",
    cf2: "objects.cosmo_factor",
    inputs: "Optional[Tuple[objects.cosmo_array]]" = None,
) -> None:
    """
    Helper to enable comparisons involving :class:`~swiftsimio.objects.cosmo_factor`s.

    Warnings are emitted when the comparison is ambiguous, for instance if comparing to a
    bare :obj:`float` or similar. Comparison to zero is a special case where we suppress
    warnings.

    See also :func:`~swiftsimio._array_functions._return_without_cosmo_factor` that is
    used in implementing this function.

    Parameters
    ----------
    cf1 : swiftsimio.objects.cosmo_factor
        First :class:`~swiftsimio.objects.cosmo_factor` to compare.

    cf2 : swiftsimio.objects.cosmo_factor
        Second :class:`~swiftsimio.objects.cosmo_factor` to compare.

    inputs : :obj:`tuple`
        The objects that ``cf1`` and ``cf2`` are attached to.

    Returns
    -------
    out : None
        The :class:`~swiftsimio.objects.cosmo_factor` is discarded.
    """
    try:
        iter(inputs[0])
    except TypeError:
        input1_iszero = (
            not getattr(inputs[0], "value", inputs[0]) and inputs[0] is not False
        )
    else:
        input1_iszero = not getattr(inputs[0], "value", inputs[0]).any()
    try:
        iter(inputs[1])
    except IndexError:
        input2_iszero = None
    except TypeError:
        input2_iszero = (
            not getattr(inputs[1], "value", inputs[1]) and inputs[1] is not False
        )
    else:
        input2_iszero = not getattr(inputs[1], "value", inputs[1]).any()
    zero_comparison = input1_iszero or input2_iszero
    return _return_without_cosmo_factor(cf1, cf2=cf2, zero_comparison=zero_comparison)


def _prepare_array_func_args(*args, _default_cm: bool = True, **kwargs) -> dict:
    """
    Coerce args and kwargs to a common ``comoving`` and collect ``cosmo_factor``s.

    This helper function is mostly intended for writing wrappers for unyt and numpy
    functions. It checks for consistency for all args and kwargs, but is not recursive
    so mixed cosmo attributes could be passed in the first argument to
    :func:`numpy.concatenate`, for instance. This function can be used "recursively" in a
    limited way manually: in functions like :func:`numpy.concatenate` where a list of
    arrays is expected, it makes sense to pass the first argument to this function
    to check consistency and attempt to coerce to comoving if needed.

    Note that unyt allows creating a :class:`~unyt.array.unyt_array` from e.g. arrays with
    heterogenous units (it probably shouldn't...). Example:

    ::

        >>> u.unyt_array([np.arange(3), np.arange(3) * u.m])
        unyt_array([[0, 1, 2],
                    [0, 1, 2]], '(dimensionless)')

    It's impractical for us to try to cover all possible invalid user input without
    unyt being stricter.

    The best way to understand the usage of this helper function is to look at the many
    wrapped unyt and numpy functions in :mod:`~swiftsimio._array_functions`.

    Parameters
    ----------
    _default_cm: bool
        If mixed ``comoving`` attributes are found, their data are converted such that
        their ``comoving`` has the value of this argument. (Default: ``True``)

    Returns
    -------
    out : dict
        A dictionary containing the input `args`` and ``kwargs`` coerced to a common
        state, and lists of their ``cosmo_factor`` attributes, and ``comoving`` and
        ``compression`` values that can be used in return values for wrapped functions,
        when relevant.

    Raises
    ------
    ValueError
        If the input arrays cannot be coerced to a consistent state of ``comoving``.
    """
    cms = [(hasattr(arg, "comoving"), getattr(arg, "comoving", None)) for arg in args]
    cfs = [getattr(arg, "cosmo_factor", None) for arg in args]
    comps = [
        (hasattr(arg, "compression"), getattr(arg, "compression", None)) for arg in args
    ]
    kw_cms = {
        k: (hasattr(kwarg, "comoving"), getattr(kwarg, "comoving", None))
        for k, kwarg in kwargs.items()
    }
    kw_cfs = {k: getattr(kwarg, "cosmo_factor", None) for k, kwarg in kwargs.items()}
    kw_comps = {
        k: (hasattr(kwarg, "compression"), getattr(kwarg, "compression", None))
        for k, kwarg in kwargs.items()
    }
    if len([cm[1] for cm in cms + list(kw_cms.values()) if cm[0]]) == 0:
        # no cosmo inputs
        ret_cm = None
    elif all([cm[1] for cm in cms + list(kw_cms.values()) if cm[0]]):
        # all cosmo inputs are comoving
        ret_cm = True
    elif all([cm[1] is None for cm in cms + list(kw_cms.values()) if cm[0]]):
        # all cosmo inputs have comoving=None
        ret_cm = None
    elif any([cm[1] is None for cm in cms + list(kw_cms.values()) if cm[0]]):
        # only some cosmo inputs have comoving=None
        raise ValueError(
            "Some arguments have comoving=None and others have comoving=True|False. "
            "Result is undefined!"
        )
    elif all([cm[1] is False for cm in cms + list(kw_cms.values()) if cm[0]]):
        # all cosmo_array inputs are physical
        ret_cm = False
    else:
        # mix of comoving and physical inputs
        # better to modify inplace (convert_to_comoving)?
        if _default_cm:
            for arg in args:
                if not arg.valid_transform:
                    _default_cm = False
                    # TODO: Raise warning?
                    break
        if _default_cm:
            args = [
                arg.to_comoving() if cm[0] and not cm[1] else arg
                for arg, cm in zip(args, cms)
            ]
            kwargs = {
                k: kwarg.to_comoving() if kw_cms[k][0] and not kw_cms[k][1] else kwarg
                for k, kwarg in kwargs.items()
            }
            ret_cm = True
        else:
            args = [
                arg.to_physical() if cm[0] and not cm[1] else arg
                for arg, cm in zip(args, cms)
            ]
            kwargs = {
                k: kwarg.to_physical() if kw_cms[k][0] and not kw_cms[k][1] else kwarg
                for k, kwarg in kwargs.items()
            }
            ret_cm = False
    if len(set(comps + list(kw_comps.values()))) == 1:
        # all compressions identical, preserve it
        ret_comp = (comps + list(kw_comps.values()))[0]
    else:
        # mixed compressions, strip it off
        ret_comp = None
    return dict(
        args=args,
        kwargs=kwargs,
        cfs=cfs,
        kw_cfs=kw_cfs,
        comoving=ret_cm,
        compression=ret_comp,
    )


def implements(numpy_function: Callable) -> Callable:
    """
    Register an __array_function__ implementation for cosmo_array objects.

    Intended for use as a decorator.

    Parameters
    ----------
    numpy_function : Callable
        A function handle from numpy (not ufuncs).

    Returns
    -------
    out : Callable
        The wrapped function.
    """

    # See NEP 18 https://numpy.org/neps/nep-0018-array-function-protocol.html
    def decorator(func: Callable) -> Callable:
        """
        Actually register the specified function.

        Parameters
        ----------
        func: Callable
            The function wrapping the numpy equivalent.

        Returns
        -------
        out : Callable
            The input ``func``.
        """
        _HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def _return_helper(
    res: np.ndarray,
    helper_result: dict,
    ret_cf: "objects.cosmo_factor",
    out: Optional[np.ndarray] = None,
) -> "objects.cosmo_array":
    """
    Helper function to attach our cosmo attributes to return values of wrapped functions.

    The return value is first promoted to be a :class:`~swiftsimio.objects.cosmo_array`
    (or quantity) if necessary. If the return value is still not one of our cosmo
    types, we don't attach attributes.

    Parameters
    ----------
    res : numpy.ndarray
        The output array of a function to attach our attributes to.
    helper_result : dict
        A helper :obj:`dict` returned by
        :func:`~swiftsimio._array_functions._prepare_array_func_args`.
    ret_cf : swiftsimio.objects.cosmo_factor
        The :class:`~swiftsimio.objects.cosmo_factor` to attach to the result.
    out : numpy.ndarray
        For functions that can place output in an ``out`` argument, a reference to
        the output array (optional).

    Returns
    -------
    out : swiftsimio.objects.cosmo_array
        The input return value of a wrapped function with our cosmo attributes applied.
    """
    res = _promote_unyt_to_cosmo(res)
    if isinstance(res, objects.cosmo_array):  # also recognizes cosmo_quantity
        res.comoving = helper_result["comoving"]
        res.cosmo_factor = ret_cf
        res.compression = helper_result["compression"]
    if isinstance(out, objects.cosmo_array):  # also recognizes cosmo_quantity
        out.comoving = helper_result["comoving"]
        out.cosmo_factor = ret_cf
        out.compression = helper_result["compression"]
    return res


def _default_unary_wrapper(
    unyt_func: Callable, cosmo_factor_handler: Callable
) -> Callable:
    """
    Wrapper helper for unary functions with typical behaviour.

    For many numpy and unyt functions with one (main) input argument, the wrapping
    code that we need to apply is repetitive. Just prepare the arguments, apply
    the chosen processing to the ``cosmo_factor``, and attach our cosmo attributes
    to the return value. This function facilitates writing these wrappers.

    Can be used as a decorator.

    Parameters
    ----------
    unyt_func : Callable
        The unyt (or numpy) function to be wrapped.
    cosmo_factor_handler : Callable
        The function that handles the ``cosmo_factor``s, chosen from those defined
        in :mod:`~swiftsimio._array_functions`.

    Returns
    -------
    out : Callable
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        """
        Prepare arguments, handle ``cosmo_factor`` attriubtes, and attach attributes to
        output.

        Returns
        -------
        out : Callable
            The wrapped function.
        """
        helper_result = _prepare_array_func_args(*args, **kwargs)
        ret_cf = cosmo_factor_handler(helper_result["cfs"][0])
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        if "out" in kwargs:
            return _return_helper(res, helper_result, ret_cf, out=kwargs["out"])
        else:
            return _return_helper(res, helper_result, ret_cf)

    return wrapper


def _default_binary_wrapper(
    unyt_func: Callable, cosmo_factor_handler: Callable
) -> Callable:
    """
    Wrapper helper for binary functions with typical behaviour.

    For many numpy and unyt functions with two (main) input arguments, the wrapping
    code that we need to apply is repetitive. Just prepare the arguments, apply
    the chosen processing to the cosmo_factors, and attach our cosmo attributes
    to the return value. This function facilitates writing these wrappers.

    Can be used as a decorator.

    Parameters
    ----------
    unyt_func : Callable
        The unyt (or numpy) function to be wrapped.
    cosmo_factor_handler : Callable
        The function that handles the ``cosmo_factor``s, chosen from those defined
        in :mod:`~swiftsimio._array_functions`.

    Returns
    -------
    out : Callable
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        """
        Prepare arguments, handle ``cosmo_factor`` attributes, and attach attributes to
        output.

        Returns
        -------
        out : Callable
            The wrapped function.
        """
        helper_result = _prepare_array_func_args(*args, **kwargs)
        ret_cf = cosmo_factor_handler(helper_result["cfs"][0], helper_result["cfs"][1])
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        if "out" in kwargs:
            return _return_helper(res, helper_result, ret_cf, out=kwargs["out"])
        else:
            return _return_helper(res, helper_result, ret_cf)

    return wrapper


def _default_comparison_wrapper(unyt_func: Callable) -> Callable:
    """
    Wrapper helper for binary comparison functions with typical behaviour.

    For many numpy and unyt comparison functions with two (main) input arguments, the
    wrapping code that we need to apply is repetitive. Just prepare the arguments,
    process the ``cosmo_factors``, and attach our cosmo attributes
    to the return value. This function facilitates writing these wrappers.

    Can be used as a decorator.

    Parameters
    ----------
    unyt_func : Callable
        The unyt (or numpy) comparison function to be wrapped.

    Returns
    -------
    out : Callable
        The wrapped function.
    """

    # assumes we have two primary arguments that will be handled with
    # _comparison_cosmo_factor with them as the inputs
    def wrapper(*args, **kwargs):
        """
        Prepare arguments, handle ``cosmo_factor`` attributes, and attach attributes to
        output.

        Returns
        -------
        out : Callable
            The wrapped function.
        """
        helper_result = _prepare_array_func_args(*args, **kwargs)
        ret_cf = _comparison_cosmo_factor(
            helper_result["cfs"][0], helper_result["cfs"][1], inputs=args[:2]
        )
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        return _return_helper(res, helper_result, ret_cf)

    return wrapper


def _default_oplist_wrapper(unyt_func: Callable) -> Callable:
    """
    Wrapper helper for functions accepting a list of operands with typical behaviour.

    For many numpy and unyt functions taking a list of operands as an argument, the
    wrapping code that we need to apply is repetitive. Just prepare the arguments,
    preserve the ``cosmo_factor`` (after checking that it's common across the list), and
    attach our cosmo attributes to the return value. This function facilitates writing
    these wrappers.

    Can be used as a decorator.

    Parameters
    ----------
    unyt_func : Callable
        The unyt (or numpy) function to be wrapped.

    Returns
    -------
    out : Callable
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        """
        Prepare arguments, handle ``cosmo_factor`` attributes, and attach attributes to
        output.

        Returns
        -------
        out : Callable
            The wrapped function.
        """
        helper_result = _prepare_array_func_args(*args, **kwargs)
        helper_result_oplist = _prepare_array_func_args(*args[0])
        ret_cf = _preserve_cosmo_factor(helper_result_oplist["cfs"][0])
        res = unyt_func(
            helper_result_oplist["args"],
            *helper_result["args"][1:],
            **helper_result["kwargs"],
        )
        return _return_helper(res, helper_result_oplist, ret_cf)

    return wrapper


# Next we wrap functions from unyt and numpy. There's not much point in writing docstrings
# or type hints for all of these.

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
        ret_cf = _preserve_cosmo_factor(helper_result["kw_cfs"]["bins"])
    else:
        # bins based on values in a
        ret_cf = _preserve_cosmo_factor(helper_result["cfs"][0])
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
    ret_cf_bins = _preserve_cosmo_factor(helper_result["cfs"][0])
    ret_cf_dens = _reciprocal_cosmo_factor(helper_result["cfs"][0])
    counts, bins = unyt_histogram(*helper_result["args"], **helper_result["kwargs"])
    if weights is not None:
        ret_cf_w = _preserve_cosmo_factor(helper_result["kw_cfs"]["weights"])
        ret_cf_counts = (
            _multiply_cosmo_factor(ret_cf_w, ret_cf_dens) if density else ret_cf_w
        )
    else:
        ret_cf_counts = ret_cf_dens if density else None
    counts = _promote_unyt_to_cosmo(counts)
    if isinstance(counts, objects.cosmo_array):  # also recognizes cosmo_quantity
        counts.comoving = helper_result["comoving"]
        counts.cosmo_factor = ret_cf_counts
        counts.compression = helper_result["compression"]
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
    helper_result_x = _prepare_array_func_args(x, bins=xbins, range=xrange)
    helper_result_y = _prepare_array_func_args(y, bins=ybins, range=yrange)
    if not density:
        helper_result_w = _prepare_array_func_args(weights=weights)
        ret_cf_x = _preserve_cosmo_factor(helper_result_x["cfs"][0])
        ret_cf_y = _preserve_cosmo_factor(helper_result_y["cfs"][0])
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
            ret_cf_w = _preserve_cosmo_factor(helper_result_w["kw_cfs"]["weights"])
            counts = _promote_unyt_to_cosmo(counts)
            if isinstance(
                counts, objects.cosmo_array
            ):  # also recognizes cosmo_quantity
                counts.comoving = helper_result_w["comoving"]
                counts.cosmo_factor = ret_cf_w
                counts.compression = helper_result_w["compression"]
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
        ret_cf_x = _preserve_cosmo_factor(helper_result_x["cfs"][0])
        ret_cf_y = _preserve_cosmo_factor(helper_result_y["cfs"][0])
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
            helper_result["cfs"][0], helper_result["cfs"][1]
        )
        if weights is not None:
            ret_cf_w = _preserve_cosmo_factor(helper_result["kw_cfs"]["weights"])
            inv_ret_cf_xy = _reciprocal_cosmo_factor(ret_cf_xy)
            ret_cf_counts = _multiply_cosmo_factor(ret_cf_w, inv_ret_cf_xy)
        else:
            ret_cf_counts = _reciprocal_cosmo_factor(ret_cf_xy)
        counts = _promote_unyt_to_cosmo(counts)
        if isinstance(counts, objects.cosmo_array):  # also recognizes cosmo_quantity
            counts.comoving = helper_result["comoving"]
            counts.cosmo_factor = ret_cf_counts
            counts.compression = helper_result["compression"]
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
        _prepare_array_func_args(s, bins=b, range=r)
        for s, b, r in zip(sample, bins, ranges)
    ]
    if not density:
        helper_result_w = _prepare_array_func_args(weights=weights)
        ret_cfs = [
            _preserve_cosmo_factor(helper_result["cfs"][0])
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
            ret_cf_w = _preserve_cosmo_factor(helper_result_w["kw_cfs"]["weights"])
            counts = _promote_unyt_to_cosmo(counts)
            if isinstance(counts, objects.cosmo_array):
                counts.comoving = helper_result_w["comoving"]
                counts.cosmo_factor = ret_cf_w
                counts.compression = helper_result_w["compression"]
    else:  # density=True
        # now sample and weights must be compatible because they will combine
        # we unpack input to the helper to get everything checked for compatibility
        helper_result = _prepare_array_func_args(
            *sample, bins=bins, range=range, weights=weights
        )
        ret_cfs = D * [_preserve_cosmo_factor(helper_result["cfs"][0])]
        counts, bins = unyt_histogramdd(
            helper_result["args"],
            bins=helper_result["kwargs"]["bins"],
            range=helper_result["kwargs"]["range"],
            density=density,
            weights=helper_result["kwargs"]["weights"],
        )
        if len(helper_result["cfs"]) == 1:
            ret_cf_sample = _preserve_cosmo_factor(helper_result["cfs"][0])
        else:
            ret_cf_sample = _multiply_cosmo_factor(*helper_result["cfs"])
        if weights is not None:
            ret_cf_w = _preserve_cosmo_factor(helper_result["kw_cfs"]["weights"])
            inv_ret_cf_sample = _reciprocal_cosmo_factor(ret_cf_sample)
            ret_cf_counts = _multiply_cosmo_factor(ret_cf_w, inv_ret_cf_sample)
        else:
            ret_cf_counts = _reciprocal_cosmo_factor(ret_cf_sample)
        counts = _promote_unyt_to_cosmo(counts)
        if isinstance(counts, objects.cosmo_array):  # also recognizes cosmo_quantity
            counts.comoving = helper_result["comoving"]
            counts.cosmo_factor = ret_cf_counts
            counts.compression = helper_result["compression"]
    return (
        counts,
        tuple(
            _return_helper(b, helper_result, ret_cf)
            for b, helper_result, ret_cf in zip(bins, helper_results, ret_cfs)
        ),
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
    cfs = [hr["cfs"] for hr in helper_results]
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
    ret_cf = cfs[0]
    for cf in cfs[1:]:
        if cf != ret_cf:
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
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][1])
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
    ret_cf = _preserve_cosmo_factor(helper_result["kw_cfs"]["base"])
    res = unyt_logspace(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


implements(np.geomspace)(
    _default_binary_wrapper(unyt_geomspace, _preserve_cosmo_factor)
)


@implements(np.copyto)
def copyto(dst, src, casting="same_kind", where=True):

    helper_result = _prepare_array_func_args(dst, src, casting=casting, where=where)
    if isinstance(src, objects.cosmo_array) and isinstance(dst, objects.cosmo_array):
        # if we're copyting across two
        _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][1])
    # must pass dst directly here because it's modified in-place
    if isinstance(src, objects.cosmo_array):
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
        helper_result["cfs"][0], None, power=a.size // res.size
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
    ret_cf = _power_cosmo_factor(helper_result["cfs"][0], None, power=a.shape[0])
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
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][0])
    res = unyt_pad(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.choose)
def choose(a, choices, out=None, mode="raise"):

    helper_result = _prepare_array_func_args(a, choices, out=out, mode=mode)
    helper_result_choices = _prepare_array_func_args(*choices)
    ret_cf = _preserve_cosmo_factor(*helper_result_choices["cfs"])
    res = unyt_choose(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf, out=out)


@implements(np.insert)
def insert(arr, obj, values, axis=None):

    helper_result = _prepare_array_func_args(arr, obj, values, axis=axis)
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][2])
    res = unyt_insert(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.lstsq)
def linalg_lstsq(a, b, rcond=None):

    helper_result = _prepare_array_func_args(a, b, rcond=rcond)
    ret_cf = _divide_cosmo_factor(helper_result["cfs"][1], helper_result["cfs"][0])
    resid_cf = _power_cosmo_factor(helper_result["cfs"][1], None, power=2)
    sing_cf = _preserve_cosmo_factor(helper_result["cfs"][0])
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
    ret_cf = _divide_cosmo_factor(helper_result["cfs"][1], helper_result["cfs"][0])
    res = unyt_linalg_solve(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.tensorsolve)
def linalg_tensorsolve(a, b, axes=None):

    helper_result = _prepare_array_func_args(a, b, axes=axes)
    ret_cf = _divide_cosmo_factor(helper_result["cfs"][1], helper_result["cfs"][0])
    res = unyt_linalg_tensorsolve(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.linalg.eig)
def linalg_eig(a):

    helper_result = _prepare_array_func_args(a)
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][0])
    ress = unyt_linalg_eig(*helper_result["args"], **helper_result["kwargs"])
    return (_return_helper(ress[0], helper_result, ret_cf), ress[1])


@implements(np.linalg.eigh)
def linalg_eigh(a, UPLO="L"):

    helper_result = _prepare_array_func_args(a, UPLO=UPLO)
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][0])
    ress = unyt_linalg_eigh(*helper_result["args"], **helper_result["kwargs"])
    return (_return_helper(ress[0], helper_result, ret_cf), ress[1])


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
    _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][1])
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
    _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(vals, objects.cosmo_array):
        comoving = getattr(arr, "comoving", None)
        if comoving:
            vals.convert_to_comoving()
        elif comoving is False:
            vals.convert_to_physical()
    unyt_place(arr, mask, vals)


@implements(np.put)
def put(a, ind, v, mode="raise"):

    helper_result = _prepare_array_func_args(a, ind, v, mode=mode)
    _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(v, objects.cosmo_array):
        comoving = getattr(a, "comoving", None)
        if comoving:
            v.convert_to_comoving()
        elif comoving is False:
            v.convert_to_physical()
    unyt_put(a, ind, v, **helper_result["kwargs"])


@implements(np.put_along_axis)
def put_along_axis(arr, indices, values, axis):

    helper_result = _prepare_array_func_args(arr, indices, values, axis)
    _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(values, objects.cosmo_array):
        comoving = getattr(arr, "comoving", None)
        if comoving:
            values.convert_to_comoving()
        elif comoving is False:
            values.convert_to_physical()
    unyt_put_along_axis(arr, indices, values, axis)


@implements(np.putmask)
def putmask(a, mask, values):

    helper_result = _prepare_array_func_args(a, mask, values)
    _preserve_cosmo_factor(helper_result["cfs"][0], helper_result["cfs"][2])
    # must pass arr directly here because it's modified in-place
    if isinstance(values, objects.cosmo_array):
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
    ret_cf = _preserve_cosmo_factor(*helper_result_choicelist["cfs"])
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
        a, a_min=a_min, a_max=a_max, out=out, **kwargs
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["cfs"][0],
        helper_result["kw_cfs"]["a_min"],
        helper_result["kw_cfs"]["a_max"],
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
        ret_cf = _return_without_cosmo_factor(helper_result["cfs"][0])
        res = unyt_where(*helper_result["args"], **helper_result["kwargs"])
    elif len(args) < 2:
        # error message borrowed from numpy 1.24.1
        raise ValueError("either both or neither of x and y should be given")
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][1], helper_result["cfs"][2])
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
    ret_cf = _preserve_cosmo_factor(*helper_result_operands["cfs"])
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
def unwrap(p, discont=None, axis=-1, *, period=6.283_185_307_179_586):

    helper_result = _prepare_array_func_args(
        p, discont=discont, axis=axis, period=period
    )
    ret_cf = _preserve_cosmo_factor(
        helper_result["cfs"][0],
        helper_result["kw_cfs"]["discont"],
        helper_result["kw_cfs"]["period"],
    )
    res = unyt_unwrap(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


@implements(np.interp)
def interp(x, xp, fp, left=None, right=None, period=None):

    helper_result = _prepare_array_func_args(
        x, xp, fp, left=left, right=right, period=period
    )
    ret_cf = _preserve_cosmo_factor(helper_result["cfs"][2])
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
            helper_result["cfs"][0], helper_result["kw_cfs"]["dx"]
        )
    else:
        ret_cf = _multiply_cosmo_factor(
            helper_result["cfs"][0], helper_result["kw_cfs"]["x"]
        )
    res = unyt_trapezoid(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result, ret_cf)


implements(np.in1d)(_default_comparison_wrapper(unyt_in1d))
implements(np.take)(_default_unary_wrapper(unyt_take, _preserve_cosmo_factor))

# Now we wrap functions that unyt does not handle explicitly:

implements(np.average)(
    _propagate_cosmo_array_attributes_to_result(np.average._implementation)
)
implements(np.max)(_propagate_cosmo_array_attributes_to_result(np.max._implementation))
implements(np.min)(_propagate_cosmo_array_attributes_to_result(np.min._implementation))
implements(np.mean)(
    _propagate_cosmo_array_attributes_to_result(np.mean._implementation)
)
implements(np.median)(
    _propagate_cosmo_array_attributes_to_result(np.median._implementation)
)
implements(np.sort)(
    _propagate_cosmo_array_attributes_to_result(np.sort._implementation)
)
implements(np.sum)(_propagate_cosmo_array_attributes_to_result(np.sum._implementation))
implements(np.partition)(
    _propagate_cosmo_array_attributes_to_result(np.partition._implementation)
)


@implements(np.meshgrid)
def meshgrid(*xi, **kwargs):
    # meshgrid is a unique case: arguments never interact with each other, so we don't
    # want to use our _prepare_array_func_args helper (that will try to coerce to
    # compatible comoving, cosmo_factor).
    # However we can't just use _propagate_cosmo_array_attributes_to_result because we
    # need to iterate over arguments.
    res = np.meshgrid._implementation(*xi, **kwargs)
    return tuple(
        _copy_cosmo_array_attributes_if_present(x, r) for (x, r) in zip(xi, res)
    )
