"""
Contains global objects, e.g. the superclass version of the
unyt_array that we use, called cosmo_array.
"""

import warnings

import unyt
from unyt import unyt_array, unyt_quantity
from unyt.array import multiple_output_operators, _iterable
from numbers import Number as numeric_type

try:
    from unyt.array import POWER_MAPPING
except ImportError:
    raise ImportError("unyt >=2.9.0 required")

import sympy
import numpy as np
from numpy import (
    add,
    subtract,
    multiply,
    divide,
    logaddexp,
    logaddexp2,
    true_divide,
    floor_divide,
    negative,
    power,
    remainder,
    mod,
    absolute,
    rint,
    sign,
    conj,
    exp,
    exp2,
    log,
    log2,
    log10,
    expm1,
    log1p,
    sqrt,
    cbrt,
    square,
    reciprocal,
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    arctan2,
    hypot,
    sinh,
    cosh,
    tanh,
    arcsinh,
    arccosh,
    arctanh,
    deg2rad,
    rad2deg,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
    equal,
    logical_and,
    logical_or,
    logical_xor,
    logical_not,
    maximum,
    minimum,
    fmax,
    fmin,
    isreal,
    iscomplex,
    isfinite,
    isinf,
    isnan,
    signbit,
    copysign,
    nextafter,
    modf,
    frexp,
    fmod,
    floor,
    ceil,
    trunc,
    fabs,
    spacing,
    positive,
    divmod as divmod_,
    isnat,
    heaviside,
    matmul,
)
from numpy._core.umath import _ones_like

try:
    from numpy._core.umath import clip
except ImportError:
    clip = None

# The scale factor!
a = sympy.symbols("a")


class InvalidConversionError(Exception):
    def __init__(self, message="Could not convert to comoving coordinates"):
        self.message = message


def _copy_cosmo_array_attributes(from_ca, to_ca):
    if not isinstance(to_ca, cosmo_array):
        return to_ca
    if hasattr(from_ca, "cosmo_factor"):
        to_ca.cosmo_factor = from_ca.cosmo_factor
    if hasattr(from_ca, "comoving"):
        to_ca.comoving = from_ca.comoving
    if hasattr(from_ca, "valid_transform"):
        to_ca.valid_transform = from_ca.valid_transform
    return to_ca


def _propagate_cosmo_array_attributes(func):
    # can work on methods (obj is self) and functions (obj is first argument)
    def wrapped(obj, *args, **kwargs):
        ret = func(obj, *args, **kwargs)
        if not isinstance(ret, cosmo_array):
            return ret
        ret = _copy_cosmo_array_attributes(obj, ret)
        if ret.shape == ():
            return cosmo_quantity(ret)
        else:
            return ret

    return wrapped


def _sqrt_cosmo_factor(ca_cf, **kwargs):
    return _power_cosmo_factor(
        ca_cf, (False, None), power=0.5
    )  # ufunc sqrt not supported


def _multiply_cosmo_factor(*args, **kwargs):
    ca_cfs = args
    if len(ca_cfs) == 1:
        return __multiply_cosmo_factor(ca_cfs[0])
    retval = __multiply_cosmo_factor(ca_cfs[0], ca_cfs[1])
    for ca_cf in ca_cfs[2:]:
        retval = __multiply_cosmo_factor((retval is not None, retval), ca_cf)
    return retval


def __multiply_cosmo_factor(ca_cf1, ca_cf2, **kwargs):
    ca1, cf1 = ca_cf1
    ca2, cf2 = ca_cf2
    if (cf1 is None) and (cf2 is None):
        # neither has cosmo_factor information:
        return None
    elif not ca1 and ca2:
        # one is not a cosmo_array, allow e.g. multiplication by constants:
        return cf2
    elif ca1 and not ca2:
        # two is not a cosmo_array, allow e.g. multiplication by constants:
        return cf1
    elif (ca1 and ca2) and ((cf1 is None) or (cf2 is None)):
        # both cosmo_array but not both with cosmo_factor
        # (both without shortcircuited above already):
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors ({cf1} and {cf2}),"
            f" discarding cosmo_factor in return value.",
            RuntimeWarning,
        )
        return None
    elif (ca1 and ca2) and ((cf1 is not None) and (cf2 is not None)):
        # both cosmo_array and both with cosmo_factor:
        return cf1 * cf2  # cosmo_factor.__mul__ raises if scale factors differ
    else:
        raise RuntimeError("Unexpected state, please report this error on github.")


def _preserve_cosmo_factor(*args, **kwargs):
    ca_cfs = args
    if len(ca_cfs) == 1:
        return __preserve_cosmo_factor(ca_cfs[0])
    retval = __preserve_cosmo_factor(ca_cfs[0], ca_cfs[1])
    for ca_cf in ca_cfs[2:]:
        retval = __preserve_cosmo_factor((retval is not None, retval), ca_cf)
    return retval


def __preserve_cosmo_factor(ca_cf1, ca_cf2=None, **kwargs):
    ca1, cf1 = ca_cf1
    ca2, cf2 = ca_cf2 if ca_cf2 is not None else (None, None)
    if ca_cf2 is None:
        # single argument, return promptly
        return cf1
    elif (cf1 is None) and (cf2 is None):
        # neither has cosmo_factor information:
        return None
    elif ca1 and not ca2:
        # only one is cosmo_array
        return cf1
    elif ca2 and not ca1:
        # only one is cosmo_array
        return cf2
    elif (ca1 and ca2) and (cf1 is None and cf2 is not None):
        # both cosmo_array, but not both with cosmo_factor
        # (both without shortcircuited above already):
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf2}) for all arguments.",
            RuntimeWarning,
        )
        return cf2
    elif (ca1 and ca2) and (cf1 is not None and cf2 is None):
        # both cosmo_array, but not both with cosmo_factor
        # (both without shortcircuited above already):
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf1}) for all arguments.",
            RuntimeWarning,
        )
        return cf1
    elif (ca1 and ca2) and (cf1 != cf2):
        raise ValueError(
            f"Ufunc arguments have cosmo_factors that differ: {cf1} and {cf2}."
        )
    elif (ca1 and ca2) and (cf1 == cf2):
        return cf1  # or cf2, they're equal
    else:
        # not dealing with cosmo_arrays at all
        return None


def _power_cosmo_factor(ca_cf1, ca_cf2, inputs=None, power=None):
    if inputs is not None and power is not None:
        raise ValueError
    ca1, cf1 = ca_cf1
    ca2, cf2 = ca_cf2
    power = inputs[1] if inputs else power
    if hasattr(power, "units"):
        if not power.units.is_dimensionless:
            raise ValueError("Exponent must be dimensionless.")
        elif power.units is not unyt.dimensionless:
            power = power.to_value(unyt.dimensionless)
        # else power.units is unyt.dimensionless, do nothing
    if ca2 and cf2.a_factor != 1.0:
        raise ValueError("Exponent has scaling with scale factor != 1.")
    if cf1 is None:
        return None
    return np.power(cf1, power)


def _square_cosmo_factor(ca_cf, **kwargs):
    return _power_cosmo_factor(ca_cf, (False, None), power=2)


def _cbrt_cosmo_factor(ca_cf, **kwargs):
    return _power_cosmo_factor(ca_cf, (False, None), power=1.0 / 3.0)


def _divide_cosmo_factor(ca_cf1, ca_cf2, **kwargs):
    ca1, cf1 = ca_cf1
    ca2, cf2 = ca_cf2
    return _multiply_cosmo_factor(
        (ca1, cf1), (ca2, _reciprocal_cosmo_factor((ca2, cf2)))
    )


def _reciprocal_cosmo_factor(ca_cf, **kwargs):
    return _power_cosmo_factor(ca_cf, (False, None), power=-1)


def _passthrough_cosmo_factor(ca_cf, ca_cf2=None, **kwargs):
    ca, cf = ca_cf
    ca2, cf2 = ca_cf2 if ca_cf2 is not None else (None, None)
    if ca_cf2 is None:
        # no second argument, return promptly
        return cf
    elif (cf2 is not None) and cf != cf2:
        # if both have cosmo_factor information and it differs this is an error
        raise ValueError(
            f"Ufunc arguments have cosmo_factors that differ: {cf} and {cf2}."
        )
    else:
        # passthrough is for e.g. ufuncs with a second dimensionless argument,
        # so ok if cf2 is None and cf1 is not
        return cf


def _return_without_cosmo_factor(ca_cf, ca_cf2=None, inputs=None, zero_comparison=None):
    ca, cf = ca_cf
    ca2, cf2 = ca_cf2 if ca_cf2 is not None else (None, None)
    if ca_cf2 is None:
        # no second argument
        pass
    elif ca and not ca2:
        # one is not a cosmo_array, warn on e.g. comparison to constants:
        if not zero_comparison:
            warnings.warn(
                f"Mixing ufunc arguments with and without cosmo_factors, continuing"
                f" assuming provided cosmo_factor ({cf}) for all arguments.",
                RuntimeWarning,
            )
    elif not ca and ca2:
        # two is not a cosmo_array, warn on e.g. comparison to constants:
        if not zero_comparison:
            warnings.warn(
                f"Mixing ufunc arguments with and without cosmo_factors, continuing"
                f" assuming provided cosmo_factor ({cf2}) for all arguments.",
                RuntimeWarning,
            )
    elif (ca and ca2) and (cf is not None and cf2 is None):
        # one has no cosmo_factor information, warn:
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf}) for all arguments.",
            RuntimeWarning,
        )
    elif (ca and ca2) and (cf is None and cf2 is not None):
        # two has no cosmo_factor information, warn:
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf2}) for all arguments.",
            RuntimeWarning,
        )
    elif (cf is not None) and (cf2 is not None) and (cf != cf2):
        # both have cosmo_factor, don't match:
        raise ValueError(
            f"Ufunc arguments have cosmo_factors that differ: {cf} and {cf2}."
        )
    elif (cf is not None) and (cf2 is not None) and (cf == cf2):
        # both have cosmo_factor, and they match:
        pass
    else:
        # not dealing with cosmo_arrays at all
        pass
    # return without cosmo_factor
    return None


def _arctan2_cosmo_factor(ca_cf1, ca_cf2, **kwargs):
    ca1, cf1 = ca_cf1
    ca2, cf2 = ca_cf2
    if (cf1 is None) and (cf2 is None):
        return None
    if cf1 is None and cf2 is not None:
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf2}) for all arguments.",
            RuntimeWarning,
        )
    if cf1 is not None and cf2 is None:
        warnings.warn(
            f"Mixing ufunc arguments with and without cosmo_factors, continuing assuming"
            f" provided cosmo_factor ({cf1}) for all arguments.",
            RuntimeWarning,
        )
    if (cf1 is not None) and (cf2 is not None) and (cf1 != cf2):
        raise ValueError(
            f"Ufunc arguments have cosmo_factors that differ: {cf1} and {cf2}."
        )
    return cosmo_factor(a**0, scale_factor=cf1.scale_factor)


def _comparison_cosmo_factor(ca_cf1, ca_cf2=None, inputs=None):
    ca1, cf1 = ca_cf1
    ca2, cf2 = ca_cf2 if ca_cf2 is not None else (None, None)
    try:
        iter(inputs[0])
    except TypeError:
        if ca1:
            input1_iszero = not inputs[0].value and inputs[0] is not False
        else:
            input1_iszero = not inputs[0] and inputs[0] is not False
    else:
        if ca1:
            input1_iszero = not inputs[0].value.any()
        else:
            input1_iszero = not inputs[0].any()
    try:
        iter(inputs[1])
    except IndexError:
        input2_iszero = None
    except TypeError:
        if ca2:
            input2_iszero = not inputs[1].value and inputs[1] is not False
        else:
            input2_iszero = not inputs[1] and inputs[1] is not False
    else:
        if ca2:
            input2_iszero = not inputs[1].value.any()
        else:
            input2_iszero = not inputs[1].any()
    zero_comparison = input1_iszero or input2_iszero
    return _return_without_cosmo_factor(
        ca_cf1, ca_cf2=ca_cf2, inputs=inputs, zero_comparison=zero_comparison
    )


def _prepare_array_func_args(*args, _default_cm=True, **kwargs):
    # unyt allows creating a unyt_array from e.g. arrays with heterogenous units
    # (it probably shouldn't...).
    # Example:
    # >>> u.unyt_array([np.arange(3), np.arange(3) * u.m])
    # unyt_array([[0, 1, 2],
    #             [0, 1, 2]], '(dimensionless)')
    # It's impractical for cosmo_array to try to cover
    # all possible invalid user input without unyt being stricter.
    # This function checks for consistency for all args and kwargs, but is not recursive
    # so mixed cosmo attributes could be passed in the first argument to np.concatenate,
    # for instance. This function can be used "recursively" in a limited way manually:
    # in functions like np.concatenate where a list of arrays is expected, it makes sense
    # to pass the first argument (of np.concatenate - an iterable) to this function
    # to check consistency and attempt to coerce to comoving if needed.
    cms = [(hasattr(arg, "comoving"), getattr(arg, "comoving", None)) for arg in args]
    ca_cfs = [
        (hasattr(arg, "cosmo_factor"), getattr(arg, "cosmo_factor", None))
        for arg in args
    ]
    comps = [
        (hasattr(arg, "compression"), getattr(arg, "compression", None)) for arg in args
    ]
    kw_cms = {
        k: (hasattr(kwarg, "comoving"), getattr(kwarg, "comoving", None))
        for k, kwarg in kwargs.items()
    }
    kw_ca_cfs = {
        k: (hasattr(kwarg, "cosmo_factor"), getattr(kwarg, "cosmo_factor", None))
        for k, kwarg in kwargs.items()
    }
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
        ca_cfs=ca_cfs,
        kw_ca_cfs=kw_ca_cfs,
        comoving=ret_cm,
        compression=ret_comp,
    )


class InvalidScaleFactor(Exception):
    """
    Raised when a scale factor is invalid, such as when adding
    two cosmo_factors with inconsistent scale factors.
    """

    def __init__(self, message=None, *args):
        """
        Constructor for warning of invalid scale factor

        Parameters
        ----------

        message : str, optional
            Message to print in case of invalid scale factor
        """
        self.message = message

    def __str__(self):
        """
        Print warning message of invalid scale factor
        """
        return f"InvalidScaleFactor: {self.message}"


class InvalidSnapshot(Exception):
    """
    Generated when a snapshot is invalid (e.g. you are trying to partially load a
    sub-snapshot).
    """

    def __init__(self, message=None, *args):
        """
        Constructor for warning of invalid snapshot

        Parameters
        ----------

        message : str, optional
            Message to print in case of invalid snapshot
        """
        self.message = message

    def __str__(self):
        """
        Print warning message of invalid snapshot
        """
        return f"InvalidSnapshot: {self.message}"


class cosmo_factor:
    """
    Cosmology factor class for storing and computing conversion between
    comoving and physical coordinates.

    This takes the expected exponent of the array that can be parsed
    by sympy, and the current value of the cosmological scale factor a.

    This should be given as the conversion from comoving to physical, i.e.

    r = cosmo_factor * r' with r in physical and r' comoving

    Examples
    --------

    Typically this would make cosmo_factor = a for the conversion between
    comoving positions r' and physical co-ordinates r.

    To do this, use the a imported from objects multiplied as you'd like:

    ``density_cosmo_factor = cosmo_factor(a**3, scale_factor=0.97)``

    """

    def __init__(self, expr, scale_factor):
        """
        Constructor for cosmology factor class

        Parameters
        ----------

        expr : sympy.expr
            expression used to convert between comoving and physical coordinates
        scale_factor : float
            the scale factor of the simulation data
        """
        self.expr = expr
        self.scale_factor = scale_factor
        pass

    def __str__(self):
        """
        Print exponent and current scale factor

        Returns
        -------

        str
            string to print exponent and current scale factor
        """
        return str(self.expr) + f" at a={self.scale_factor}"

    @property
    def a_factor(self):
        """
        The a-factor for the unit.

        e.g. for density this is 1 / a**3.

        Returns
        -------

        float
            the a-factor for given unit
        """
        return float(self.expr.subs(a, self.scale_factor))

    @property
    def redshift(self):
        """
        Compute the redshift from the scale factor.

        Returns
        -------

        float
            redshift from the given scale factor

        Notes
        -----

        Returns the redshift
        ..math:: z = \\frac{1}{a} - 1,
        where :math: `a` is the scale factor
        """
        return (1.0 / self.scale_factor) - 1.0

    def __add__(self, b):
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to add two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        if not self.expr == b.expr:
            raise InvalidScaleFactor(
                "Attempting to add two cosmo_factors with different scale factor "
                f"dependence, {self.expr} and {b.expr}"
            )

        return cosmo_factor(expr=self.expr, scale_factor=self.scale_factor)

    def __sub__(self, b):
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to subtract two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        if not self.expr == b.expr:
            raise InvalidScaleFactor(
                "Attempting to subtract two cosmo_factors with different scale factor "
                f"dependence, {self.expr} and {b.expr}"
            )

        return cosmo_factor(expr=self.expr, scale_factor=self.scale_factor)

    def __mul__(self, b):
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to multiply two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        return cosmo_factor(expr=self.expr * b.expr, scale_factor=self.scale_factor)

    def __truediv__(self, b):
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to divide two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        return cosmo_factor(expr=self.expr / b.expr, scale_factor=self.scale_factor)

    def __radd__(self, b):
        return self.__add__(b)

    def __rsub__(self, b):
        return self.__sub__(b)

    def __rmul__(self, b):
        return self.__mul__(b)

    def __rtruediv__(self, b):
        return b.__truediv__(self)

    def __pow__(self, p):
        return cosmo_factor(expr=self.expr**p, scale_factor=self.scale_factor)

    def __lt__(self, b):
        return self.a_factor < b.a_factor

    def __gt__(self, b):
        return self.a_factor > b.a_factor

    def __le__(self, b):
        return self.a_factor <= b.a_factor

    def __ge__(self, b):
        return self.a_factor >= b.a_factor

    def __eq__(self, b):
        # Doesn't handle some corner cases, e.g. cosmo_factor(a ** 1, scale_factor=1)
        # is considered equal to cosmo_factor(a ** 2, scale_factor=1) because
        # 1 ** 1 == 1 ** 2. Should check self.expr vs b.expr with sympy?
        return (self.scale_factor == b.scale_factor) and (self.a_factor == b.a_factor)

    def __ne__(self, b):
        return not self.__eq__(b)

    def __repr__(self):
        """
        Print exponent and current scale factor

        Returns
        -------

        str
            string to print exponent and current scale factor
        """
        return self.__str__()


class cosmo_array(unyt_array):
    """
    Cosmology array class.

    This inherits from the unyt.unyt_array, and adds
    four variables: compression, cosmo_factor, comoving, and valid_transform.

    Parameters
    ----------

    unyt_array : unyt.unyt_array
        the inherited unyt_array

    Attributes
    ----------

    comoving : bool
        if True then the array is in comoving co-ordinates, and if
        False then it is in physical units.

    cosmo_factor : float
        Object to store conversion data between comoving and physical coordinates

    compression : string
        String describing any compression that was applied to this array in the
        hdf5 file.

    valid_transform: bool
       if True then the array can be converted from physical to comoving units

    """

    # TODO:
    _cosmo_factor_ufunc_registry = {
        add: _preserve_cosmo_factor,
        subtract: _preserve_cosmo_factor,
        multiply: _multiply_cosmo_factor,
        divide: _divide_cosmo_factor,
        logaddexp: _return_without_cosmo_factor,
        logaddexp2: _return_without_cosmo_factor,
        true_divide: _divide_cosmo_factor,
        floor_divide: _divide_cosmo_factor,
        negative: _passthrough_cosmo_factor,
        power: _power_cosmo_factor,
        remainder: _preserve_cosmo_factor,
        mod: _preserve_cosmo_factor,
        fmod: _preserve_cosmo_factor,
        absolute: _passthrough_cosmo_factor,
        fabs: _passthrough_cosmo_factor,
        rint: _return_without_cosmo_factor,
        sign: _return_without_cosmo_factor,
        conj: _passthrough_cosmo_factor,
        exp: _return_without_cosmo_factor,
        exp2: _return_without_cosmo_factor,
        log: _return_without_cosmo_factor,
        log2: _return_without_cosmo_factor,
        log10: _return_without_cosmo_factor,
        expm1: _return_without_cosmo_factor,
        log1p: _return_without_cosmo_factor,
        sqrt: _sqrt_cosmo_factor,
        square: _square_cosmo_factor,
        cbrt: _cbrt_cosmo_factor,
        reciprocal: _reciprocal_cosmo_factor,
        sin: _return_without_cosmo_factor,
        cos: _return_without_cosmo_factor,
        tan: _return_without_cosmo_factor,
        sinh: _return_without_cosmo_factor,
        cosh: _return_without_cosmo_factor,
        tanh: _return_without_cosmo_factor,
        arcsin: _return_without_cosmo_factor,
        arccos: _return_without_cosmo_factor,
        arctan: _return_without_cosmo_factor,
        arctan2: _arctan2_cosmo_factor,
        arcsinh: _return_without_cosmo_factor,
        arccosh: _return_without_cosmo_factor,
        arctanh: _return_without_cosmo_factor,
        hypot: _preserve_cosmo_factor,
        deg2rad: _return_without_cosmo_factor,
        rad2deg: _return_without_cosmo_factor,
        # bitwise_and: not supported for unyt_array
        # bitwise_or: not supported for unyt_array
        # bitwise_xor: not supported for unyt_array
        # invert: not supported for unyt_array
        # left_shift: not supported for unyt_array
        # right_shift: not supported for unyt_array
        greater: _comparison_cosmo_factor,
        greater_equal: _comparison_cosmo_factor,
        less: _comparison_cosmo_factor,
        less_equal: _comparison_cosmo_factor,
        not_equal: _comparison_cosmo_factor,
        equal: _comparison_cosmo_factor,
        logical_and: _comparison_cosmo_factor,
        logical_or: _comparison_cosmo_factor,
        logical_xor: _comparison_cosmo_factor,
        logical_not: _return_without_cosmo_factor,
        maximum: _passthrough_cosmo_factor,
        minimum: _passthrough_cosmo_factor,
        fmax: _preserve_cosmo_factor,
        fmin: _preserve_cosmo_factor,
        isreal: _return_without_cosmo_factor,
        iscomplex: _return_without_cosmo_factor,
        isfinite: _return_without_cosmo_factor,
        isinf: _return_without_cosmo_factor,
        isnan: _return_without_cosmo_factor,
        signbit: _return_without_cosmo_factor,
        copysign: _passthrough_cosmo_factor,
        nextafter: _preserve_cosmo_factor,
        modf: _passthrough_cosmo_factor,
        # ldexp: not supported for unyt_array
        frexp: _return_without_cosmo_factor,
        floor: _passthrough_cosmo_factor,
        ceil: _passthrough_cosmo_factor,
        trunc: _passthrough_cosmo_factor,
        spacing: _passthrough_cosmo_factor,
        positive: _passthrough_cosmo_factor,
        divmod_: _passthrough_cosmo_factor,
        isnat: _return_without_cosmo_factor,
        heaviside: _preserve_cosmo_factor,
        _ones_like: _preserve_cosmo_factor,
        matmul: _multiply_cosmo_factor,
        clip: _passthrough_cosmo_factor,
    }

    def __new__(
        cls,
        input_array,
        units=None,
        registry=None,
        dtype=None,
        bypass_validation=False,
        input_units=None,
        name=None,
        cosmo_factor=None,
        comoving=None,
        valid_transform=True,
        compression=None,
    ):
        """
        Essentially a copy of the __new__ constructor.

        Parameters
        ----------
        input_array : iterable
            A tuple, list, or array to attach units to
        units : str, unyt.unit_symbols or astropy.unit, optional
            The units of the array. Powers must be specified using python syntax
            (cm**3, not cm^3).
        registry : unyt.unit_registry.UnitRegistry, optional
            The registry to create units from. If input_units is already associated with a
            unit registry and this is specified, this will be used instead of the registry
            associated with the unit object.
        dtype : np.dtype or str, optional
            The dtype of the array data. Defaults to the dtype of the input data, or, if
            none is found, uses np.float64
        bypass_validation : bool, optional
            If True, all input validation is skipped. Using this option may produce
            corrupted, invalid units or array data, but can lead to significant speedups
            in the input validation logic adds significant overhead. If set, input_units
            must be a valid unit object. Defaults to False.
        input_units : str, optional
            deprecated in favour of units option
        name : str, optional
            The name of the array. Defaults to None. This attribute does not propagate
            through mathematical operations, but is preserved under indexing and unit
            conversions.
        cosmo_factor : cosmo_factor
            cosmo_factor object to store conversion data between comoving and physical
            coordinates
        comoving : bool
            flag to indicate whether using comoving coordinates
        valid_transform : bool
            flag to indicate whether this array can be converted to comoving
        compression : string
            description of the compression filters that were applied to that array in the
            hdf5 file
        """

        cosmo_factor: cosmo_factor

        if isinstance(input_array, cosmo_array):
            if comoving:
                input_array.convert_to_comoving()
            elif comoving is False:
                input_array.convert_to_physical()
            else:
                comoving = input_array.comoving
            cosmo_factor = _preserve_cosmo_factor(
                (cosmo_factor is not None, cosmo_factor),
                (input_array.cosmo_factor is not None, input_array.cosmo_factor),
            )
            if not valid_transform:
                input_array.convert_to_physical()
            if compression != input_array.compression:
                compression = None  # just drop it
        elif isinstance(input_array, np.ndarray):
            pass  # guard np.ndarray so it doesn't get caught by _iterable in next case
        elif _iterable(input_array) and input_array:
            if isinstance(input_array[0], cosmo_array):
                default_cm = comoving if comoving is not None else True
                helper_result = _prepare_array_func_args(
                    *input_array, _default_cm=default_cm
                )
                if comoving is None:
                    comoving = helper_result["comoving"]
                input_array = helper_result["args"]
                cosmo_factor = _preserve_cosmo_factor(
                    (cosmo_factor is not None, cosmo_factor), *helper_result["ca_cfs"]
                )
                if not valid_transform:
                    input_array.convert_to_physical()
                if compression != helper_result["compression"]:
                    compression = None  # just drop it

        obj = super().__new__(
            cls,
            input_array,
            units=units,
            registry=registry,
            dtype=dtype,
            bypass_validation=bypass_validation,
            name=name,
        )

        if isinstance(obj, unyt_array) and not isinstance(obj, cls):
            obj = obj.view(cls)

        obj.cosmo_factor = cosmo_factor
        obj.comoving = comoving
        obj.compression = compression
        obj.valid_transform = valid_transform
        if not obj.valid_transform:
            assert (
                not obj.comoving
            ), "Cosmo arrays without a valid transform to comoving units must be physical"
        if obj.comoving:
            assert (
                obj.valid_transform
            ), "Comoving Cosmo arrays must be able to be transformed to physical"

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.cosmo_factor = getattr(obj, "cosmo_factor", None)
        self.comoving = getattr(obj, "comoving", None)
        self.compression = getattr(obj, "compression", None)
        self.valid_transform = getattr(obj, "valid_transform", True)

    def __str__(self):
        if self.comoving:
            comoving_str = "(Comoving)"
        else:
            comoving_str = "(Physical)"

        return super().__str__() + " " + comoving_str

    def __repr__(self):
        return super().__repr__()

    def __reduce__(self):
        """
        Pickle reduction method

        Here we add an extra element at the start of the unyt_array state
        tuple to store the cosmology info.
        """
        np_ret = super(cosmo_array, self).__reduce__()
        obj_state = np_ret[2]
        cosmo_state = (
            ((self.cosmo_factor, self.comoving, self.valid_transform),) + obj_state[:],
        )
        new_ret = np_ret[:2] + cosmo_state + np_ret[3:]
        return new_ret

    def __setstate__(self, state):
        """
        Pickle setstate method

        Here we extract the extra cosmology info we added to the object
        state and pass the rest to unyt_array.__setstate__.
        """
        super(cosmo_array, self).__setstate__(state[1:])
        self.cosmo_factor, self.comoving, self.valid_transform = state[0]

    # Wrap functions that return copies of cosmo_arrays so that our
    # attributes get passed through:
    astype = _propagate_cosmo_array_attributes(unyt_array.astype)
    in_units = _propagate_cosmo_array_attributes(unyt_array.in_units)
    byteswap = _propagate_cosmo_array_attributes(unyt_array.byteswap)
    compress = _propagate_cosmo_array_attributes(unyt_array.compress)
    diagonal = _propagate_cosmo_array_attributes(unyt_array.diagonal)
    flatten = _propagate_cosmo_array_attributes(unyt_array.flatten)
    ravel = _propagate_cosmo_array_attributes(unyt_array.ravel)
    repeat = _propagate_cosmo_array_attributes(unyt_array.repeat)
    swapaxes = _propagate_cosmo_array_attributes(unyt_array.swapaxes)
    transpose = _propagate_cosmo_array_attributes(unyt_array.transpose)
    view = _propagate_cosmo_array_attributes(unyt_array.view)

    @_propagate_cosmo_array_attributes
    def take(self, indices, **kwargs):
        taken = unyt_array.take(self, indices, **kwargs)
        if np.ndim(indices) == 0:
            return cosmo_quantity(taken)
        else:
            return cosmo_array(taken)

    @_propagate_cosmo_array_attributes
    def reshape(self, shape, /, *, order="C"):
        reshaped = unyt_array.reshape(self, shape, order=order)
        if shape == () or shape is None:
            return cosmo_quantity(reshaped)
        else:
            return reshaped

    @_propagate_cosmo_array_attributes
    def __getitem__(self, *args, **kwargs):
        item = unyt_array.__getitem__(self, *args, *kwargs)
        if item.shape == ():
            return cosmo_quantity(item)
        else:
            return item

    # Also wrap some array "properties":

    @property
    def T(self):
        return self.transpose()  # transpose is wrapped above.

    @property
    def ua(self):
        return _propagate_cosmo_array_attributes(np.ones_like)(self)

    @property
    def unit_array(self):
        return _propagate_cosmo_array_attributes(np.ones_like)(self)

    def convert_to_comoving(self) -> None:
        """
        Convert the internal data to be in comoving units.
        """
        if self.comoving:
            return
        if not self.valid_transform or self.comoving is None:
            raise InvalidConversionError
        # Best to just modify values as otherwise we're just going to have
        # to do a convert_to_units anyway.
        values = self.d
        values /= self.cosmo_factor.a_factor
        self.comoving = True

    def convert_to_physical(self) -> None:
        """
        Convert the internal data to be in physical units.
        """
        if self.comoving is None:
            raise InvalidConversionError
        elif not self.comoving:
            return
        else:  # self.comoving
            # Best to just modify values as otherwise we're just going to have
            # to do a convert_to_units anyway.
            values = self.d
            values *= self.cosmo_factor.a_factor
            self.comoving = False

    def to_physical(self):
        """
        Creates a copy of the data in physical units.

        Returns
        -------
        cosmo_array
            copy of cosmo_array in physical units
        """
        copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
        copied_data.convert_to_physical()

        return copied_data

    def to_comoving(self):
        """
        Creates a copy of the data in comoving units.

        Returns
        -------
        cosmo_array
            copy of cosmo_array in comoving units
        """
        if not self.valid_transform:
            raise InvalidConversionError
        copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
        copied_data.convert_to_comoving()

        return copied_data

    def compatible_with_comoving(self):
        """
        Is this cosmo_array compatible with a comoving cosmo_array?

        This is the case if the cosmo_array is comoving, or if the scale factor
        exponent is 0 (cosmo_factor.a_factor() == 1)
        """
        return self.comoving or (self.cosmo_factor.a_factor == 1.0)

    def compatible_with_physical(self):
        """
        Is this cosmo_array compatible with a physical cosmo_array?

        This is the case if the cosmo_array is physical, or if the scale factor
        exponent is 0 (cosmo_factor.a_factor == 1)
        """
        return (not self.comoving) or (self.cosmo_factor.a_factor == 1.0)

    @classmethod
    def from_astropy(
        cls,
        arr,
        unit_registry=None,
        comoving=None,
        cosmo_factor=None,
        compression=None,
        valid_transform=True,
    ):
        """
        Convert an AstroPy "Quantity" to a cosmo_array.

        Parameters
        ----------
        arr: AstroPy Quantity
            The Quantity to convert from.
        unit_registry: yt UnitRegistry, optional
            A yt unit registry to use in the conversion. If one is not supplied, the
            default one will be used.
        comoving : bool
            if True then the array is in comoving co-ordinates, and if False then it is in
            physical units.
        cosmo_factor : float
            Object to store conversion data between comoving and physical coordinates
        compression : string
            String describing any compression that was applied to this array in the hdf5
            file.
        valid_transform : bool
            flag to indicate whether this array can be converted to comoving

        Example
        -------
        >>> from astropy.units import kpc
        >>> cosmo_array.from_astropy([1, 2, 3] * kpc)
        cosmo_array([1., 2., 3.], 'kpc')
        """

        obj = super().from_astropy(arr, unit_registry=unit_registry).view(cls)
        obj.comoving = comoving
        obj.cosmo_factor = cosmo_factor
        obj.compression = compression
        obj.valid_trasform = valid_transform

        return obj

    @classmethod
    def from_pint(
        cls,
        arr,
        unit_registry=None,
        comoving=None,
        cosmo_factor=None,
        compression=None,
        valid_transform=True,
    ):
        """
        Convert a Pint "Quantity" to a cosmo_array.

        Parameters
        ----------
        arr : Pint Quantity
            The Quantity to convert from.
        unit_registry : yt UnitRegistry, optional
            A yt unit registry to use in the conversion. If one is not
            supplied, the default one will be used.
        comoving : bool
            if True then the array is in comoving co-ordinates, and if False then it is in
            physical units.
        cosmo_factor : float
            Object to store conversion data between comoving and physical coordinates
        compression : string
            String describing any compression that was applied to this array in the hdf5
            file.
        valid_transform : bool
            flag to indicate whether this array can be converted to comoving

        Examples
        --------
        >>> from pint import UnitRegistry
        >>> import numpy as np
        >>> ureg = UnitRegistry()
        >>> a = np.arange(4)
        >>> b = ureg.Quantity(a, "erg/cm**3")
        >>> b
        <Quantity([0 1 2 3], 'erg / centimeter ** 3')>
        >>> c = cosmo_array.from_pint(b)
        >>> c
        cosmo_array([0, 1, 2, 3], 'erg/cm**3')
        """
        obj = super().from_pint(arr, unit_registry=unit_registry).view(cls)
        obj.comoving = comoving
        obj.cosmo_factor = cosmo_factor
        obj.compression = compression
        obj.valid_trasform = valid_transform

        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        helper_result = _prepare_array_func_args(*inputs, **kwargs)
        ca_cfs = helper_result["ca_cfs"]

        # make sure we evaluate the cosmo_factor_ufunc_registry function:
        # might raise/warn even if we're not returning a cosmo_array
        if ufunc in (multiply, divide) and method == "reduce":
            power_map = POWER_MAPPING[ufunc]
            if "axis" in kwargs and kwargs["axis"] is not None:
                ret_cf = _power_cosmo_factor(
                    ca_cfs[0],
                    (False, None),
                    power=power_map(inputs[0].shape[kwargs["axis"]]),
                )
            else:
                ret_cf = _power_cosmo_factor(
                    ca_cfs[0], (False, None), power=power_map(inputs[0].size)
                )
        else:
            ret_cf = self._cosmo_factor_ufunc_registry[ufunc](*ca_cfs, inputs=inputs)

        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        # if we get a tuple we have multiple return values to deal with
        # if unyt returns a bare ndarray, do the same
        # otherwise we create a view and attach our attributes
        if isinstance(ret, tuple):
            ret = tuple(
                r.view(type(self)) if isinstance(r, unyt_array) else r for r in ret
            )
            for r in ret:
                if isinstance(r, type(self)):
                    r.comoving = helper_result["comoving"]
                    r.cosmo_factor = ret_cf
                    r.compression = helper_result["compression"]
        if isinstance(ret, unyt_array):
            ret = ret.view(type(self))
            ret.comoving = helper_result["comoving"]
            ret.cosmo_factor = ret_cf
            ret.compression = helper_result["compression"]
        if "out" in kwargs:
            out = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                out = out[0]
                if isinstance(out, cosmo_array):
                    out.comoving = helper_result["comoving"]
                    out.cosmo_factor = ret_cf
                    out.compression = helper_result["compression"]
            else:
                for o in out:
                    if isinstance(o, type(self)):
                        o.comoving = helper_result["comoving"]
                        o.cosmo_factor = ret_cf
                        o.compression = helper_result["compression"]

        return ret

    def __array_function__(self, func, types, args, kwargs):
        # Follow NEP 18 guidelines
        # https://numpy.org/neps/nep-0018-array-function-protocol.html
        from ._array_functions import _HANDLED_FUNCTIONS
        from unyt._array_functions import (
            _HANDLED_FUNCTIONS as _UNYT_HANDLED_FUNCTIONS,
            _UNSUPPORTED_FUNCTIONS as _UNYT_UNSUPPORTED_FUNCTIONS,
        )

        # Let's claim to support everything supported by unyt.
        # If we can't do this in future, follow their pattern of
        # defining out own _UNSUPPORTED_FUNCTIONS in a _array_functions.py file
        _UNSUPPORTED_FUNCTIONS = _UNYT_UNSUPPORTED_FUNCTIONS

        if func in _UNSUPPORTED_FUNCTIONS:
            # following NEP 18, return NotImplemented as a sentinel value
            # which will lead to raising a TypeError, while
            # leaving other arguments a chance to take the lead
            return NotImplemented

        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](*args, **kwargs)
        elif func not in _HANDLED_FUNCTIONS and func in _UNYT_HANDLED_FUNCTIONS:
            # first look for unyt's implementation
            return _UNYT_HANDLED_FUNCTIONS[func](*args, **kwargs)
        elif func not in _UNYT_HANDLED_FUNCTIONS:
            # otherwise default to numpy's private implementation
            return func._implementation(*args, **kwargs)
        # Note: this allows subclasses that don't override
        # __array_function__ to handle cosmo_array objects
        if not all(issubclass(t, cosmo_array) or t is np.ndarray for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


class cosmo_quantity(cosmo_array, unyt_quantity):
    """
    Cosmology scalar class.

    This inherits from both the cosmo_array and the unyt.unyt_array, and has the same four
    attributes as cosmo_array: compression, cosmo_factor, comoving, and valid_transform.

    Parameters
    ----------

    cosmo_array : cosmo_array
        the inherited cosmo_array

    unyt_quantity : unyt.unyt_quantity
        the inherited unyt_quantity

    Attributes
    ----------

    comoving : bool
        if True then the array is in comoving co-ordinates, and if
        False then it is in physical units.

    cosmo_factor : float
        Object to store conversion data between comoving and physical coordinates

    compression : string
        String describing any compression that was applied to this array in the
        hdf5 file.

    valid_transform: bool
       if True then the array can be converted from physical to comoving units

    """

    def __new__(
        cls,
        input_scalar,
        units=None,
        registry=None,
        dtype=None,
        bypass_validation=False,
        name=None,
        cosmo_factor=None,
        comoving=None,
        valid_transform=True,
        compression=None,
    ):
        """
        Essentially a copy of the unyt_quantity.__new__ constructor.

        Parameters
        ----------
        input_scalar : an integer of floating point scalar
            A scalar to attach units and cosmological transofrmations to.
        units : str, unyt.unit_symbols or astropy.unit, optional
            The units of the array. Powers must be specified using python syntax
            (cm**3, not cm^3).
        registry : unyt.unit_registry.UnitRegistry, optional
            The registry to create units from. If input_units is already associated with a
            unit registry and this is specified, this will be used instead of the registry
            associated with the unit object.
        dtype : np.dtype or str, optional
            The dtype of the array data. Defaults to the dtype of the input data, or, if
            none is found, uses np.float64
        bypass_validation : bool, optional
            If True, all input validation is skipped. Using this option may produce
            corrupted, invalid units or array data, but can lead to significant speedups
            in the input validation logic adds significant overhead. If set, input_units
            must be a valid unit object. Defaults to False.
        name : str, optional
            The name of the array. Defaults to None. This attribute does not propagate
            through mathematical operations, but is preserved under indexing and unit
            conversions.
        cosmo_factor : cosmo_factor
            cosmo_factor object to store conversion data between comoving and physical
            coordinates.
        comoving : bool
            Flag to indicate whether using comoving coordinates.
        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving.
        compression : string
            Description of the compression filters that were applied to that array in the
            hdf5 file.
        """
        if not (
            bypass_validation
            or isinstance(input_scalar, (numeric_type, np.number, np.ndarray))
        ):
            raise RuntimeError("unyt_quantity values must be numeric")

        units = getattr(input_scalar, "units", None) if units is None else units
        name = getattr(input_scalar, "name", None) if name is None else name
        cosmo_factor = (
            getattr(input_scalar, "cosmo_factor", None)
            if cosmo_factor is None
            else cosmo_factor
        )
        comoving = (
            getattr(input_scalar, "comoving", None) if comoving is None else comoving
        )
        valid_transform = (
            getattr(input_scalar, "valid_transform", None)
            if valid_transform is None
            else valid_transform
        )
        compression = (
            getattr(input_scalar, "compression", None)
            if compression is None
            else compression
        )
        ret = super().__new__(
            cls,
            np.asarray(input_scalar),
            units,
            registry,
            dtype=dtype,
            bypass_validation=bypass_validation,
            name=name,
            cosmo_factor=cosmo_factor,
            comoving=comoving,
            valid_transform=valid_transform,
            compression=compression,
        )
        if ret.size > 1:
            raise RuntimeError("cosmo_quantity instances must be scalars")
        return ret

    @_propagate_cosmo_array_attributes
    def reshape(self, shape, /, *, order="C"):
        reshaped = unyt_array.reshape(self, shape, order=order)
        if shape == () or shape is None:
            return reshaped
        else:
            return cosmo_array(reshaped)
