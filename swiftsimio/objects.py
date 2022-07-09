"""
Contains global objects, e.g. the superclass version of the
unyt_array that we use, called cosmo_array.
"""

import warnings

import unyt
from unyt import unyt_array

# Eventually import of POWER_SIGN_MAPPING will fail because it has been
# replaced with POWER_MAPPING in a bugfix (unyt PR#231). Then see segment
# of unyt.array.__array_ufunc__ where it is used and update
# cosmo_array.__array_ufunc__ accordingly. There is an xfailing test in
# test_cosmo_array_attrs to catch when this occurs with a test failure.
from unyt.array import multiple_output_operators, POWER_SIGN_MAPPING

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
from numpy.core.umath import _ones_like

try:
    from numpy.core.umath import clip
except ImportError:
    clip = None

# The scale factor!
a = sympy.symbols("a")


def _propagate_cosmo_array_attributes(func):
    def wrapped(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if not type(ret) is cosmo_array:
            return ret
        if hasattr(self, "cosmo_factor"):
            ret.cosmo_factor = self.cosmo_factor
        if hasattr(self, "comoving"):
            ret.comoving = self.comoving
        return ret

    return wrapped


def _sqrt_cosmo_factor(ca_cf, **kwargs):
    return _power_cosmo_factor(
        ca_cf, (False, None), power=0.5
    )  # ufunc sqrt not supported


def _multiply_cosmo_factor(ca_cf1, ca_cf2, **kwargs):
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


def _preserve_cosmo_factor(ca_cf1, ca_cf2=None, **kwargs):
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
        raise RuntimeError("Unexpected state, please report this error on github.")


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
        raise RuntimeError("Unexpected state, please report this error on github.")
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
    return cosmo_factor(a ** 0, scale_factor=cf1.scale_factor)


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
        return cosmo_factor(expr=self.expr ** p, scale_factor=self.scale_factor)

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


class cosmo_array(unyt_array):
    """
    Cosmology array class.

    This inherits from the unyt.unyt_array, and adds
    three variables: compression, cosmo_factor, and comoving.
    Data is assumed to be comoving when passed to the object but you
    can override this by setting the latter flag to be False.

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

    """

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
        comoving=True,
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
        compression : string
            description of the compression filters that were applied to that array in the
            hdf5 file
        """

        cosmo_factor: cosmo_factor

        try:
            obj = super().__new__(
                cls,
                input_array,
                units=units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                input_units=input_units,
                name=name,
            )
        except TypeError:
            # Older versions of unyt
            obj = super().__new__(
                cls,
                input_array,
                units=units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                input_units=input_units,
            )

        if isinstance(obj, unyt_array) and not isinstance(obj, cls):
            obj = obj.view(cls)

        obj.cosmo_factor = cosmo_factor
        obj.comoving = comoving
        obj.compression = compression

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.cosmo_factor = getattr(obj, "cosmo_factor", None)
        self.comoving = getattr(obj, "comoving", True)
        self.compression = getattr(obj, "compression", None)

    def __str__(self):
        if self.comoving:
            comoving_str = "(Comoving)"
        else:
            comoving_str = "(Physical)"

        return super().__str__() + " " + comoving_str

    def __reduce__(self):
        """
        Pickle reduction method

        Here we add an extra element at the start of the unyt_array state
        tuple to store the cosmology info.
        """
        np_ret = super(cosmo_array, self).__reduce__()
        obj_state = np_ret[2]
        cosmo_state = (((self.cosmo_factor, self.comoving),) + obj_state[:],)
        new_ret = np_ret[:2] + cosmo_state + np_ret[3:]
        return new_ret

    def __setstate__(self, state):
        """
        Pickle setstate method

        Here we extract the extra cosmology info we added to the object
        state and pass the rest to unyt_array.__setstate__.
        """
        super(cosmo_array, self).__setstate__(state[1:])
        self.cosmo_factor, self.comoving = state[0]

    # Wrap functions that return copies of cosmo_arrays so that our
    # attributes get passed through:
    __getitem__ = _propagate_cosmo_array_attributes(unyt_array.__getitem__)
    astype = _propagate_cosmo_array_attributes(unyt_array.astype)
    in_units = _propagate_cosmo_array_attributes(unyt_array.in_units)
    byteswap = _propagate_cosmo_array_attributes(unyt_array.byteswap)
    compress = _propagate_cosmo_array_attributes(unyt_array.compress)
    diagonal = _propagate_cosmo_array_attributes(unyt_array.diagonal)
    flatten = _propagate_cosmo_array_attributes(unyt_array.flatten)
    newbyteorder = _propagate_cosmo_array_attributes(unyt_array.newbyteorder)
    ravel = _propagate_cosmo_array_attributes(unyt_array.ravel)
    repeat = _propagate_cosmo_array_attributes(unyt_array.repeat)
    reshape = _propagate_cosmo_array_attributes(unyt_array.reshape)
    swapaxes = _propagate_cosmo_array_attributes(unyt_array.swapaxes)
    take = _propagate_cosmo_array_attributes(unyt_array.take)
    transpose = _propagate_cosmo_array_attributes(unyt_array.transpose)
    view = _propagate_cosmo_array_attributes(unyt_array.view)

    # Also wrap some array "attributes":

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
        else:
            # Best to just modify values as otherwise we're just going to have
            # to do a convert_to_units anyway.
            values = self.d
            values /= self.cosmo_factor.a_factor
            self.comoving = True

    def convert_to_physical(self) -> None:
        """
        Convert the internal data to be in physical units.
        """
        if self.comoving:
            # Best to just modify values as otherwise we're just going to have
            # to do a convert_to_units anyway.
            values = self.d
            values *= self.cosmo_factor.a_factor
            self.comoving = False
        else:
            return

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
        cls, arr, unit_registry=None, comoving=True, cosmo_factor=None, compression=None
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

        return obj

    @classmethod
    def from_pint(
        cls, arr, unit_registry=None, comoving=True, cosmo_factor=None, compression=None
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

        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        cms = [
            (hasattr(inp, "comoving"), getattr(inp, "comoving", None)) for inp in inputs
        ]
        cfs = [
            (hasattr(inp, "cosmo_factor"), getattr(inp, "cosmo_factor", None))
            for inp in inputs
        ]
        comps = [
            (hasattr(inp, "compression"), getattr(inp, "compression", None))
            for inp in inputs
        ]

        # if we're here at least one input must be a cosmo_array
        if all([cm[1] for cm in cms if cm[0]]):
            # all cosmo_array inputs are comoving
            ret_cm = True
        elif not any([cm[1] for cm in cms if cm[0]]):
            # all cosmo_array inputs are physical
            ret_cm = False
        else:
            # mix of comoving and physical inputs
            inputs = [
                inp.to_comoving() if cm[0] and not cm[1] else inp
                for inp, cm in zip(inputs, cms)
            ]
            ret_cm = True

        if len(set(comps)) == 1:
            # all compressions identical, preserve it
            ret_comp = comps[0]
        else:
            # mixed compressions, strip it off
            ret_comp = None

        # make sure we evaluate the cosmo_factor_ufunc_registry function:
        # might raise/warn even if we're not returning a cosmo_array
        if ufunc in (multiply, divide) and method == "reduce":
            power_sign = POWER_SIGN_MAPPING[ufunc]
            if "axis" in kwargs and kwargs["axis"] is not None:
                ret_cf = _power_cosmo_factor(
                    cfs[0],
                    (False, None),
                    power=power_sign * inputs[0].shape[kwargs["axis"]],
                )
            else:
                ret_cf = _power_cosmo_factor(
                    cfs[0], (False, None), power=power_sign * inputs[0].size
                )
        else:
            ret_cf = self._cosmo_factor_ufunc_registry[ufunc](*cfs, inputs=inputs)

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
                    r.comoving = ret_cm
                    r.cosmo_factor = ret_cf
                    r.compression = ret_comp
        if isinstance(ret, unyt_array):
            ret = ret.view(type(self))
            ret.comoving = ret_cm
            ret.cosmo_factor = ret_cf
            ret.compression = ret_comp
        if "out" in kwargs:
            out = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                out = out[0]
                if isinstance(out, cosmo_array):
                    out.comoving = ret_cm
                    out.cosmo_factor = ret_cf
                    out.compression = ret_comp
            else:
                for o in out:
                    if isinstance(o, type(self)):
                        o.comoving = ret_cm
                        o.cosmo_factor = ret_cf
                        o.compression = ret_comp

        return ret
