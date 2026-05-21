"""
Definitions of our custom cosmology array and quantity classes.

Defines :class:`~swiftsimio.objects.cosmo_array`,
:class:`~swiftsimio.objects.cosmo_quantity` and
:class:`~swiftsimio.objects.cosmo_factor` objects for cosmology-aware
arrays, extending the functionality of the :class:`~unyt.array.unyt_array`.

For developers, see also :mod:`swiftsimio._array_functions` containing
helpers, wrappers and implementations that enable most :mod:`numpy` and
:mod:`unyt` functions to work with our cosmology-aware arrays.
"""

import unyt
from unyt import unyt_array, unyt_quantity, Unit
from unyt.array import multiple_output_operators, _iterable, POWER_MAPPING
from typing import Iterable, Callable, Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self  # in python 3.10
from collections.abc import Collection
from functools import singledispatchmethod

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
    vecdot,
)
from ._array_functions import (
    _propagate_cosmo_array_attributes_to_result,
    _ensure_result_is_cosmo_array_or_quantity,
    _copy_cosmo_array_attributes,
    _sqrt_cosmo_factor,
    _multiply_cosmo_factor,
    _preserve_cosmo_factor,
    _power_cosmo_factor,
    _square_cosmo_factor,
    _cbrt_cosmo_factor,
    _divide_cosmo_factor,
    _reciprocal_cosmo_factor,
    _passthrough_cosmo_factor,
    _return_without_cosmo_factor,
    _arctan2_cosmo_factor,
    _comparison_cosmo_factor,
    _prepare_array_func_args,
    _default_binary_wrapper,
)

try:
    import pint
except ImportError:
    pass  # only for type hinting

try:
    import astropy.units
except ImportError:
    pass  # only for type hinting

# The scale factor!
a = sympy.symbols("a")

numeric_type = int | float | np.number | complex


def _verify_valid_transform_validity(obj: "cosmo_array") -> None:
    """
    Check that ``comoving`` and ``valid_transform`` attributes are compatible.

    Comoving arrays must be able to transform, while arrays that don't transform must
    be physical. Arrays with ``comoving`` set to ``None`` must not be allowed to
    transform. This function raises if this is not the case.

    Parameters
    ----------
    obj : swiftsimio.objects.cosmo_array
        The array whose validity is to be checked.

    Raises
    ------
    InvalidConversionError
        When an invalid combination of ``comoving`` and ``valid_transform`` is found.
    """
    if obj.comoving is None and obj.valid_transform:
        raise InvalidConversionError(
            "Cosmo arrays must have comoving!=None to have valid_transform==True"
        )
    elif obj.valid_transform is False and obj.comoving is True:
        raise InvalidConversionError(
            "Cosmo arrays without a valid transform to comoving units must be physical,"
            " and comoving cosmo_arrays must be able to be transformed to physcial."
        )


class InvalidConversionError(Exception):
    """
    Raise when attempting comoving/physical conversion when not allowed.

    Parameters
    ----------
    message : str, optional
        Message to print in case of invalid conversion.
    """

    def __init__(
        self, message: str = "Could not convert to comoving coordinates."
    ) -> None:
        self.message = message

    def __str__(self) -> str:
        """
        Print error message for invalid conversion.

        Returns
        -------
        str
            The error message.
        """
        return f"{self.__class__}: {self.message}"


class InvalidScaleFactor(Exception):
    """
    Raised when a scale factor is invalid.

    For example, when adding two :class:`~swiftsimio.objects.cosmo_factor` objects
    with inconsistent scale factors.

    Parameters
    ----------
    message : str, optional
        Message to print in case of invalid scale factor.

    *args : Any
        Arbitrary arguments.
    """

    def __init__(self, message: str | None = None, *args: Any) -> None:
        self.message = message

    def __str__(self) -> str:
        """
        Print error message for invalid scale factor.

        Returns
        -------
        str
            The error message.
        """
        return f"{self.__class__}: {self.message}"


class InvalidSnapshot(Exception):
    """
    Generated when a snapshot is invalid.

    For example, trying to partially load a sub-snapshot.

    Parameters
    ----------
    message : str, optional
        Message to print in case of invalid snapshot.

    *args : Any
        Arbitrary arguments.
    """

    def __init__(self, message: str | None = None, *args: Any) -> None:
        self.message = message

    def __str__(self) -> str:
        """
        Print error message for invalid snapshot.

        Returns
        -------
        str
            The error message.
        """
        return f"{self.__class__}: {self.message}"


class InvalidCosmoUnit(Exception):
    """
    Generated when a :class:`~swiftsimio.objects._AHelper` encounters an incompatibility.

    For example, trying to multiply with a bare scalar.

    Parameters
    ----------
    message : str, optional
        Message to print in case of incompatible input.

    *args : Any
        Arbitrary arguments.
    """

    def __init__(self, message: str | None = None, *args: Any) -> None:
        self.message = message

    def __str__(self) -> str:
        """
        Print error message for incompatible input.

        Returns
        -------
        str
            The error message.
        """
        return f"{self.__class__}: {self.message}"


class cosmo_factor:
    r"""
    Cosmology factor class.

    For storing and computing conversion between comoving and physical coordinates.

    This takes the expected exponent of the array that can be parsed
    by :mod:`sympy`, and the current value of the cosmological scale factor ``a``.

    This should be given as the conversion from comoving to physical, i.e.
    :math:`r = a^f \times r` where :math:`a` is the scale factor,
    :math:`r` is a physical quantity and :math`r'` a comoving quantity.

    Parameters
    ----------
    expr : sympy.Expr
        Expression used to convert between comoving and physical coordinates.
    scale_factor : int or float or ~swiftsimio.objects._AHelper
        The scale factor (a).

    Attributes
    ----------
    expr : sympy.Expr
        Expression used to convert between comoving and physical coordinates.

    scale_factor : int or float
        The scale factor (a).

    See Also
    --------
    swiftsimio.objects.cosmo_factor.create

    Examples
    --------
    Mass density transforms as :math:`a^3`. To set up a ``cosmo_factor``, supposing
    a current ``scale_factor=0.97``, we import the scale factor ``a`` and initialize
    as:

    .. code-block:: python

        from swiftsimio.objects import a  # the scale factor (a sympy symbol object)
        density_cosmo_factor = cosmo_factor(a**3, scale_factor=0.97)

    :class:`~swiftsimio.objects.cosmo_factor` supports arithmetic, for example:

    .. code-block:: python

        >>> cosmo_factor(a**2, scale_factor=0.5) * cosmo_factor(a**-1, scale_factor=0.5)
        cosmo_factor(expr=a, scale_factor=0.5)
    """

    def __init__(
        self, expr: sympy.Expr | None, scale_factor: numeric_type | None
    ) -> None:
        self.expr = expr
        self.scale_factor = scale_factor

    @classmethod
    def create(
        cls,
        scale_factor: "numeric_type | _AHelper | None",
        exponent: numeric_type | None,
    ) -> "cosmo_factor":
        """
        Create :class:`~swiftsimio.objects.cosmo_factor` from scale factor and exponent.

        Parameters
        ----------
        scale_factor : int or float or ~swiftsimio.objects._AHelper
            The scale factor.

        exponent : int or float
            The exponent defining the scaling with the scale factor.

        Examples
        --------
        .. code-block:: python

            >>> cosmo_factor.create(0.5, 2)
            cosmo_factor(expr=a**2, scale_factor=0.5)
        """
        try:
            getattr(scale_factor, "_comoving")
        except AttributeError:
            # we're just dealing with a float, this is fine
            pass
        except InvalidCosmoUnit:
            # for an _AHelper we require this, otherwise _comoving is True or False
            pass
        else:
            # for an _AHelper complain that .comoving or .physical was accessed
            raise InvalidCosmoUnit(
                "Initialize cosmo_array (or cosmo_quantity, or cosmo_factor) with e.g. "
                "`scale_factor=metadata.scale_factor`, not e.g. "
                "`scale_factor=metadata.a.comoving` or "
                "`scale_factor=metadata.a.physical`."
            )
        # If we got an `_AHelper`, get its `_scale_factor` attribute, bypassing the
        # checks that happen when accessing its `scale_factor` attribute: this is an
        # intentional exception to the rule.
        scale_factor = (
            scale_factor._scale_factor
            if isinstance(scale_factor, _AHelper)
            else scale_factor
        )
        obj = cls(a**exponent, scale_factor)

        return obj

    def __str__(self) -> str:
        """
        Print exponent and current scale factor.

        Returns
        -------
        str
            String with exponent and current scale factor.
        """
        return str(self.expr) + f" at a={self.scale_factor}"

    @property
    def a_factor(self) -> float | None:
        """
        The multiplicative factor for conversion from comoving to physical.

        For example, for density this is :math:`a^{-3}`.

        Returns
        -------
        float
            The multiplicative factor for conversion from comoving to physical.
        """
        if (self.expr is None) or (self.scale_factor is None):
            return None
        else:
            return float(self.expr.subs(a, self.scale_factor))

    @property
    def redshift(self) -> numeric_type | None:
        r"""
        The redshift computed from the scale factor.

        Returns the redshift :math:`z = \\frac{1}{a} - 1`, where :math:`a` is the scale
        factor.

        Returns
        -------
        float
            The redshift.
        """
        if self.scale_factor is None:
            return None
        else:
            return (1.0 / self.scale_factor) - 1.0

    def __add__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Add two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to add to this one.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The sum of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to be summed is not a :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only add cosmo_factor to another cosmo_factor.")
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

    def __sub__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Subtract two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to subtract from this one.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The difference of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to be subtracted is not a
            :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError(
                "Can only subtract cosmo_factor from another cosmo_factor."
            )
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

    def __mul__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Multiply two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to multiply this one.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The product of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to be multiplied is not a
            :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError(
                "Can only multiply cosmo_factor with another cosmo_factor."
            )
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to multiply two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )
        if (self.expr is None) and (b.expr is None):
            # let's be permissive and allow two uninitialized cosmo_factors through
            return cosmo_factor(expr=None, scale_factor=self.scale_factor)
        elif self.expr is None or b.expr is None:
            raise InvalidScaleFactor(
                "Attempting to multiply an initialized cosmo_factor with an "
                f"uninitialized cosmo_factor {self} and {b}."
            )
        else:
            return cosmo_factor(expr=self.expr * b.expr, scale_factor=self.scale_factor)

    def __truediv__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Divide two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to divide this one by.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The quotient of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to divide by is not a :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only divide cosmo_factor with another cosmo_factor.")
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to divide two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        if (self.expr is None) and (b.expr is None):
            # let's be permissive and allow two uninitialized cosmo_factors through
            return cosmo_factor(expr=None, scale_factor=self.scale_factor)
        elif self.expr is None or b.expr is None:
            raise InvalidScaleFactor(
                "Attempting to divide an initialized cosmo_factor with an "
                f"uninitialized cosmo_factor {self} and {b}."
            )
        else:
            return cosmo_factor(expr=self.expr / b.expr, scale_factor=self.scale_factor)

    def __radd__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Add two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to add to this one.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The sum of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to be summed is not a :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        return self.__add__(b)

    def __rsub__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Subtract two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to subtract from this one.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The difference of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to be subtracted is not a
            :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        return self.__sub__(b)

    def __rmul__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Multiply two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to multiply this one.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The product of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to be multiplied is not a
            :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        return self.__mul__(b)

    def __rtruediv__(self, b: "cosmo_factor") -> "cosmo_factor":
        """
        Divide two :class:`~swiftsimio.objects.cosmo_factor`s.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to divide this one by.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The quotient of the two :class:`~swiftsimio.objects.cosmo_factor`s.

        Raises
        ------
        ValueError
            If the object to divide by is not a :class:`~swiftsimio.objects.cosmo_factor`.

        swiftsimio.objects.InvalidScaleFactor
            If the :class:`~swiftsimio.objects.cosmo_factor` has a ``scale_factor`` that
            does not match this one's.
        """
        return b.__truediv__(self)

    def __pow__(self, p: float) -> "cosmo_factor":
        """
        Raise this :class:`~swiftsimio.objects.cosmo_factor` to an exponent.

        Parameters
        ----------
        p : float
            The exponent by which to raise this :class:`~swiftsimio.objects.cosmo_factor`.

        Returns
        -------
        swiftsimio.objects.cosmo_factor
            The exponentiated :class:`~swiftsimio.objects.cosmo_factor`s.
        """
        if self.expr is None:
            return cosmo_factor(expr=None, scale_factor=self.scale_factor)
        return cosmo_factor(expr=self.expr**p, scale_factor=self.scale_factor)

    def __lt__(self, b: "cosmo_factor") -> bool:
        """
        Compare the values of two :meth:`~swiftsimio.objects.cosmo_factor.a_factor`s.

        The :meth:`~swiftsimio.objects.cosmo_factor.a_factor` is the ``expr`` attribute
        evaluated given the ``scale_factor`` attribute.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to compare with this one.

        Returns
        -------
        bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            return NotImplemented
        if self.a_factor is None or b.a_factor is None:
            raise ValueError(
                "Cannot compare cosmo_factors when one has unset a_factor."
            )
        return self.a_factor < b.a_factor

    def __gt__(self, b: "cosmo_factor") -> bool:
        """
        Compare the values of two :meth:`~swiftsimio.objects.cosmo_factor.a_factor`s.

        The :meth:`~swiftsimio.objects.cosmo_factor.a_factor` is the ``expr`` attribute
        evaluated given the ``scale_factor`` attribute.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to compare with this one.

        Returns
        -------
        bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            return NotImplemented
        if self.a_factor is None or b.a_factor is None:
            raise ValueError(
                "Cannot compare cosmo_factors when one has unset a_factor."
            )
        return self.a_factor > b.a_factor

    def __le__(self, b: "cosmo_factor") -> bool:
        """
        Compare the values of two :meth:`~swiftsimio.objects.cosmo_factor.a_factor`s.

        The :meth:`~swiftsimio.objects.cosmo_factor.a_factor` is the ``expr`` attribute
        evaluated given the ``scale_factor`` attribute.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to compare with this one.

        Returns
        -------
        bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            return NotImplemented
        if self.a_factor is None or b.a_factor is None:
            raise ValueError(
                "Cannot compare cosmo_factors when one has unset a_factor."
            )
        return self.a_factor <= b.a_factor

    def __ge__(self, b: "cosmo_factor") -> bool:
        """
        Compare the values of two :meth:`~swiftsimio.objects.cosmo_factor.a_factor`s.

        The :meth:`~swiftsimio.objects.cosmo_factor.a_factor` is the ``expr`` attribute
        evaluated given the ``scale_factor`` attribute.

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to compare with this one.

        Returns
        -------
        bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            return NotImplemented
        if self.a_factor is None or b.a_factor is None:
            raise ValueError(
                "Cannot compare cosmo_factors when one has unset a_factor."
            )
        return self.a_factor >= b.a_factor

    def __eq__(self, b: object) -> bool:
        """
        Compare the expressions and values of two scale factor expressions for equality.

        The :meth:`~swiftsimio.objects.cosmo_factor.a_factor` is the ``expr`` attribute
        evaluated given the ``scale_factor`` attribute. Notice that unlike ``__gt__``,
        ``__ge__``, ``__lt__`` and ``__le__``, (in)equality comparisons check that the
        expression is (un)equal as well as the value. This is so that e.g. ``a**1`` and
        ``a**2``, both with ``scale_factor=1.0`` are not equal (both have
        ``a_factor==1``).

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to compare with this one.

        Returns
        -------
        bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            return NotImplemented
        scale_factor_match = self.scale_factor == b.scale_factor
        if self.a_factor is None and b.a_factor is None:
            # guards passing None to isclose
            a_factor_match = True
        elif self.a_factor is None or b.a_factor is None:
            # we know they're not both None from previous case
            a_factor_match = False
        elif np.isclose(self.a_factor, b.a_factor, rtol=1e-6):
            a_factor_match = True
        else:
            a_factor_match = False
        return scale_factor_match and a_factor_match

    def __ne__(self, b: object) -> bool:
        """
        Compare the expressions and values of two scale factor expressions for inequality.

        The :meth:`~swiftsimio.objects.cosmo_factor.a_factor` is the ``expr`` attribute
        evaluated given the ``scale_factor`` attribute. Notice that unlike ``__gt__``,
        ``__ge__``, ``__lt__`` and ``__le__``, (in)equality comparisons check that the
        expression is (un)equal as well as the value. This is so that e.g. ``a**1`` and
        ``a**2``, both with ``scale_factor=1.0`` are not equal (both have
        ``a_factor==1``).

        Parameters
        ----------
        b : swiftsimio.objects.cosmo_factor
            The :class:`~swiftsimio.objects.cosmo_factor` to compare with this one.

        Returns
        -------
        bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        return not self.__eq__(b)

    def __repr__(self) -> str:
        """
        Get a string representation of the scaling with the scale factor.

        Returns
        -------
        str
            String representation of the scaling with the scale factor.
        """
        return f"cosmo_factor(expr={self.expr}, scale_factor={self.scale_factor})"


def _parse_cosmo_factor_args(
    cf: cosmo_factor | None = None,
    scale_factor: numeric_type | None = None,
    scale_exponent: numeric_type | None = None,
) -> cosmo_factor | None:
    """
    Decide what provided cosmology information to use, or raise an error.

    If both a ``cosmo_factor`` and a (``scale_factor``, ``scale_exponent``) pair are
    given then this is an error. If only one of ``scale_factor`` and ``scale_exponent``
    is given this is an error. Otherwise we construct the
    :class:`~swiftsimio.objects.cosmo_factor`, unless it's going to be a ``NULL_CF`` with
    the information we have - in that case we return ``None`` to give a chance for it
    to be filled in elsewhere before assuming the ``NULL_CF`` default.

    Parameters
    ----------
    cf : swiftsimio.objects.cosmo_factor
        The :class:`~swiftsimio.objects.cosmo_factor` passed as an explicit argument.

    scale_factor : int or float
        The scale factor passed as a kwarg.

    scale_exponent : float
        The exponent for the scale factor to convert to/from comoving passed as a kwarg.

    Returns
    -------
    cosmo_factor or None
        The :class:`~swiftsimio.objects.cosmo_factor` to use, or ``None``.

    Raises
    ------
    ValueError
        If multiple values or incomplete information for the desired
        :class:`~swiftsimio.objects.cosmo_factor` are provided.
    """
    if cf is None and scale_factor is None and scale_exponent is None:
        # we can return promptly
        return None
    if cf is not None:
        if scale_factor is not None or scale_exponent is not None:
            if cosmo_factor.create(scale_factor, scale_exponent) != cf:
                raise ValueError(
                    "Provide either `cosmo_factor` or (`scale_factor` and "
                    "`scale_exponent`, not both (perhaps there was a `cosmo_factor` "
                    "attached to the input array or scalar?)."
                )
            else:
                # the duplicate information matches so let's allow it
                return cf
        else:
            return cf
    else:
        if (scale_factor is not None and scale_exponent is None) or (
            scale_factor is None and scale_exponent is not None
        ):
            raise ValueError(
                "Provide values for both `scale_factor` and `scale_exponent`."
            )
        if scale_factor is None and scale_exponent is None:
            return NULL_CF
        else:
            return cosmo_factor.create(scale_factor, scale_exponent)


NULL_CF = cosmo_factor(None, None)  # helps avoid name collisions with kwargs below


class cosmo_array(unyt_array):
    """
    Cosmology array class.

    This inherits from the :class:`~unyt.array.unyt_array`, and adds
    four attributes: ``compression``, ``cosmo_factor``, ``comoving``, and
    ``valid_transform``.

    .. note::

        :class:`~swiftsimio.objects.cosmo_array` and the related
        :class:`~swiftsimio.objects.cosmo_quantity` are now intended to support all
        :mod:`numpy` functions, propagating units (thanks to :mod:`unyt`) and
        cosmology information. There are a large number of functions, and a very
        large number of possible parameter combinations, so some corner cases may
        have been missed in testing. Please report any issues on github, they are
        usually easy to fix for future use! Currently :mod:`scipy` functions are
        not supported (although some might "just work"). Requests to fully support
        specific functions can also be submitted as github issues.

    Attributes
    ----------
    comoving : bool
        If ``True`` then the array is in comoving coordinates, if``False`` then it is in
        physical units.

    cosmo_factor : swiftsimio.objects.cosmo_factor
        Object to store conversion data between comoving and physical coordinates.

    compression : str
        String describing any compression that was applied to this array in the
        hdf5 file.

    valid_transform: bool
       If ``True`` then the array can be converted from physical to comoving units.

    See Also
    --------
    swiftsimio.objects.cosmo_quantity

    Notes
    -----
    This class will generally try to make sense of input and initialize an array-like
    object consistent with the input, and warn or raise if this cannot be done
    consistently. However, the way that :class:`~unyt.array.unyt_array` handles input
    imposes some limits to this. In particular, nested non-numpy containers given in
    input are not traversed recursively, but only one level deep. This means that
    while with this input the attributes are detected by the new array correctly:

    .. code-block:: python

        >>> from swiftsimio.objects import cosmo_array, cosmo_factor
        >>> x = cosmo_array(
        ...     np.arange(3),
        ...     u.kpc,
        ...     comoving=True,
        ...     scale_factor=1.0,
        ...     scale_exponent=1,
        ... )
        >>> cosmo_array([x, x])
        cosmo_array([[0, 1, 2],
               [0, 1, 2]], 'kpc', comoving='True', cosmo_factor='a at a=1.0',
               valid_transform='True')

    with this input they are lost:

    .. code-block:: python

        >>> cosmo_array([[x, x],[x, x]])
        cosmo_array([[[0, 1, 2],[0, 1, 2]],[[0, 1, 2],[0, 1, 2]]],
               '(dimensionless)', comoving='None', cosmo_factor='None at a=None',
               valid_transform='True')
    """

    _cosmo_factor_ufunc_registry: dict[np.ufunc, Callable] = {
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
        matmul: _multiply_cosmo_factor,
        vecdot: _multiply_cosmo_factor,
    }

    def __new__(
        cls,
        input_array: Iterable,
        units: "str | unyt.unit_object.Unit | astropy.units.core.Unit | None" = None,
        *,
        registry: unyt.unit_registry.UnitRegistry | None = None,
        dtype: np.dtype | str | None = None,
        bypass_validation: bool = False,
        name: str | None = None,
        cosmo_factor: cosmo_factor | None = None,
        scale_factor: numeric_type | None = None,
        scale_exponent: numeric_type | None = None,
        comoving: bool | None = None,
        valid_transform: bool = True,
        compression: str | None = None,
    ) -> "cosmo_array":
        """
        Prepare a new class instance.

        Closely inspired by the :meth:`unyt.array.unyt_array.__new__` constructor.

        Parameters
        ----------
        input_array : np.ndarray, unyt.array.unyt_array or iterable
            A tuple, list, or array to attach units and cosmology information to.

        units : str, unyt.unit_object.Unit or astropy.units.core.Unit, optional
            The units of the array. When using strings, powers must be specified using
            python syntax (``cm**3``, not ``cm^3``).

        registry : unyt.unit_registry.UnitRegistry, optional
            The registry to create units from. If ``units`` is already associated
            with a unit registry and this is specified, this will be used instead of the
            registry associated with the unit object.

        dtype : np.dtype or str, optional
            The dtype of the array data. Defaults to the dtype of the input data, or, if
            none is found, uses ``np.float64``.

        bypass_validation : bool, optional
            If ``True``, all input validation is skipped. Using this option may produce
            corrupted or invalid data, but can lead to significant speedups
            in the input validation logic adds significant overhead. If set, minimally
            pass valid values for units, comoving and cosmo_factor. Defaults to ``False``.

        name : str, optional
            The name of the array. Defaults to ``None``. This attribute does not propagate
            through mathematical operations, but is preserved under indexing and unit
            conversions.

        cosmo_factor : swiftsimio.objects.cosmo_factor
            Object to store conversion data between comoving and physical coordinates.

        scale_factor : int or float
            The scale factor associated to the data. Also provide a value for
            ``scale_exponent``.

        scale_exponent : int or float
            The exponent for the scale factor giving the scaling for conversion to/from
            comoving units. Also provide a value for ``scale_factor``.

        comoving : bool
            Flag to indicate whether using comoving coordinates.

        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving. If
            ``False``, then ``comoving`` must be ``False`` (or ``None``).

        compression : str
            Description of the compression filters that were applied to that array in the
            hdf5 file.
        """
        if bypass_validation is True:
            obj = super().__new__(
                cls,
                input_array,
                units=units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                name=name,
            )

            # dtype, units, registry & name are handled by unyt
            obj.comoving = comoving
            obj.cosmo_factor = cosmo_factor if cosmo_factor is not None else NULL_CF
            if scale_factor is not None:  # ambiguity, but this is `bypass_validation`
                obj.cosmo_factor = NULL_CF.__class__.create(
                    scale_factor, scale_exponent
                )
            obj.valid_transform = valid_transform
            obj.compression = compression

            return obj

        if isinstance(input_array, cosmo_array):
            obj = input_array.view(cls)
            if comoving is None:
                # we could be dealing with unyt creating an object for us so copy the
                # value across to avoid defaulting to None
                obj.comoving = input_array.comoving

            # let's trust what the input_array tells us about valid_transform.
            # copy explicitly in case unyt is creating an object for us.
            obj.valid_transform = input_array.valid_transform

            # do cosmo_factor first since it can be used in comoving/physical conversion:
            cosmo_factor = (
                input_array.cosmo_factor
                if input_array.cosmo_factor != NULL_CF
                else None
            )
            cosmo_factor = _parse_cosmo_factor_args(
                cf=cosmo_factor,
                scale_factor=scale_factor,
                scale_exponent=scale_exponent,
            )
            if cosmo_factor is not None:
                obj.cosmo_factor = cosmo_factor
            else:
                # we could be dealing with unyt creating an object for us so copy the
                # value across to avoid defaulting to None
                obj.cosmo_factor = input_array.cosmo_factor

            if comoving is True:
                obj.convert_to_comoving()
            elif comoving is False:
                obj.convert_to_physical()
            # else is already copied from input_array

            # only overwrite valid_transform after transforming so that invalid
            # transformations raise:
            obj.valid_transform = valid_transform
            _verify_valid_transform_validity(obj)
            obj.compression = (
                compression if compression is not None else obj.compression
            )

            return obj

        elif isinstance(input_array, np.ndarray) and input_array.dtype != object:
            # guard np.ndarray so it doesn't get caught by _iterable in next case

            # ndarray with object dtype goes to next case to properly handle e.g.
            # ndarrays containing cosmo_quantities
            cosmo_factor = _parse_cosmo_factor_args(
                cf=cosmo_factor,
                scale_factor=scale_factor,
                scale_exponent=scale_exponent,
            )
            valid_transform = valid_transform if comoving is not None else False

        elif _iterable(input_array):
            # if _prepare_array_func_args finds cosmo_array input it will convert to:
            default_cm = comoving if comoving is not None else True
            default_vt = comoving is not None

            # coerce any cosmo_array inputs to consistency:
            helper_result = _prepare_array_func_args(
                *input_array, _default_cm=default_cm
            )

            input_array = helper_result["args"]

            # default to comoving, cosmo_factor and compression given as kwargs
            comoving = helper_result["comoving"] if comoving is None else comoving
            cosmo_factor = _parse_cosmo_factor_args(
                cf=cosmo_factor,
                scale_factor=scale_factor,
                scale_exponent=scale_exponent,
            )
            cosmo_factor = (
                _preserve_cosmo_factor(*helper_result["cfs"])
                if cosmo_factor is None and len(helper_result["cfs"])
                else cosmo_factor
            )
            compression = (
                helper_result["compression"] if compression is None else compression
            )
            # valid_transform has a non-None default, so we have to decide to always
            # respect it, unless comoving is None
            valid_transform = valid_transform if comoving is not None else default_vt
        else:
            # if the input isn't even iterable, this is an error
            raise ValueError(
                "cosmo_array data must be iterable (for scalar input use cosmo_quantity)."
            )

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

        # attach our attributes:
        obj.comoving = comoving
        # unyt allows creating a unyt_array from e.g. arrays with heterogenous units
        # (it probably shouldn't...), so we don't recurse deeply and therefore
        # can't guarantee that cosmo_factor isn't None at this point, guard with default
        obj.cosmo_factor = cosmo_factor if cosmo_factor is not None else NULL_CF
        obj.compression = compression
        obj.valid_transform = valid_transform
        _verify_valid_transform_validity(obj)

        return obj

    def __array_finalize__(self, obj: "cosmo_array") -> None:
        """
        Complete initialization of a cosmology array.

        Parameters
        ----------
        obj : cosmo_array
            The array to finish initializing.
        """
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.cosmo_factor = getattr(obj, "cosmo_factor", NULL_CF)
        self.comoving = getattr(obj, "comoving", None)
        self.compression = getattr(obj, "compression", None)
        self.valid_transform = getattr(obj, "valid_transform", True)

    def __str__(self) -> str:
        """
        Represent array as a string with cosmology metadata.

        Returns
        -------
        str
            The array string representataion annotated with cosmology metadata.
        """
        if self.comoving:
            comoving_str = "(Comoving)"
        elif self.comoving is None:
            comoving_str = "(Physical/comoving not set)"
        else:
            comoving_str = "(Physical)"

        return super().__str__() + " " + comoving_str

    def __repr__(self) -> str:
        """
        Represent array as a string with cosmology metadata.

        Returns
        -------
        str
            The array string representataion annotated with cosmology metadata.
        """
        return super().__repr__()

    def __reduce__(self) -> tuple:
        """
        Pickle reduction method.

        Here we add an extra element at the start of the :class:`~unyt.array.unyt_array`
        state tuple to store the cosmology info.

        Returns
        -------
        tuple
            The state ready for pickling.
        """
        np_ret = super(cosmo_array, self).__reduce__()
        obj_state = np_ret[2]
        cosmo_state = (
            (
                (
                    self.cosmo_factor,
                    self.comoving,
                    self.compression,
                    self.valid_transform,
                ),
            )
            + obj_state[:],
        )
        new_ret = np_ret[:2] + cosmo_state + np_ret[3:]
        return new_ret

    def __setstate__(self, state: tuple) -> None:
        """
        Pickle setstate method.

        Here we extract the extra cosmology info we added to the object
        state and pass the rest to :meth:`unyt.array.unyt_array.__setstate__`.

        Parameters
        ----------
        state : tuple
            A :obj:`tuple` containing the extra state information.
        """
        super(cosmo_array, self).__setstate__(state[1:])
        self.cosmo_factor, self.comoving, self.compression, self.valid_transform = (
            state[0]
        )

    # Wrap functions that return copies of cosmo_arrays so that our
    # attributes get passed through:
    astype = _propagate_cosmo_array_attributes_to_result(unyt_array.astype)
    in_units = _propagate_cosmo_array_attributes_to_result(unyt_array.in_units)
    byteswap = _propagate_cosmo_array_attributes_to_result(unyt_array.byteswap)
    compress = _propagate_cosmo_array_attributes_to_result(unyt_array.compress)
    diagonal = _propagate_cosmo_array_attributes_to_result(unyt_array.diagonal)
    flatten = _propagate_cosmo_array_attributes_to_result(unyt_array.flatten)
    ravel = _propagate_cosmo_array_attributes_to_result(unyt_array.ravel)
    repeat = _propagate_cosmo_array_attributes_to_result(unyt_array.repeat)
    swapaxes = _propagate_cosmo_array_attributes_to_result(unyt_array.swapaxes)
    transpose = _propagate_cosmo_array_attributes_to_result(unyt_array.transpose)
    view = _propagate_cosmo_array_attributes_to_result(unyt_array.view)
    __copy__ = _propagate_cosmo_array_attributes_to_result(unyt_array.__copy__)
    copy = _propagate_cosmo_array_attributes_to_result(unyt_array.copy)
    __deepcopy__ = _propagate_cosmo_array_attributes_to_result(unyt_array.__deepcopy__)
    in_cgs = _propagate_cosmo_array_attributes_to_result(unyt_array.in_cgs)
    in_base = _propagate_cosmo_array_attributes_to_result(unyt_array.in_base)
    take = _propagate_cosmo_array_attributes_to_result(
        _ensure_result_is_cosmo_array_or_quantity(unyt_array.take)
    )
    reshape = _propagate_cosmo_array_attributes_to_result(
        _ensure_result_is_cosmo_array_or_quantity(unyt_array.reshape)
    )
    __getitem__ = _propagate_cosmo_array_attributes_to_result(
        _ensure_result_is_cosmo_array_or_quantity(unyt_array.__getitem__)
    )
    dot = _default_binary_wrapper(unyt_array.dot, _multiply_cosmo_factor)
    squeeze = _ensure_result_is_cosmo_array_or_quantity(unyt_array.squeeze)

    # Also wrap some array "properties":
    T = property(_propagate_cosmo_array_attributes_to_result(unyt_array.transpose))
    ua = property(_propagate_cosmo_array_attributes_to_result(np.ones_like))
    unit_array = property(_propagate_cosmo_array_attributes_to_result(np.ones_like))

    def convert_to(
        self,
        units: unyt.Unit | str | None = None,
        *,
        comoving: bool | None = None,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Convert the internal data in-place to desired units and comoving status.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        Parameters
        ----------
        units : ~unyt.Unit or str, optional
            The desired units for the converted array. If omitted the units are preserved.

        comoving : bool, optional
            The desired comoving state for the converted array. If omitted the comoving
            state is preserved.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            This array in the requested comoving units, transformed in place.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value
        """
        if comoving:
            self.convert_to_comoving(units, equivalence=equivalence, **kwargs)
        elif comoving is False:
            self.convert_to_physical(units, equivalence=equivalence, **kwargs)

    def convert_to_comoving(
        self,
        units: unyt.Unit | str | None = None,
        *,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Convert the internal data in-place to be in comoving units.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        Parameters
        ----------
        units : ~unyt.Unit or str, optional
            The desired units for the converted array. If omitted the units are preserved.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            This array in the requested comoving units, transformed in place.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value
        """
        if units is not None:
            self.convert_to_units(units, equivalence=equivalence, **kwargs)
        if self.comoving:
            return
        if not self.valid_transform or self.comoving is None:
            raise InvalidConversionError
        # Best to just modify values as otherwise we're just going to have
        # to do a convert_to_units anyway.
        values = self.d
        values /= self.cosmo_factor.a_factor
        self.comoving = True

    def convert_to_physical(
        self,
        units: unyt.Unit | str | None = None,
        *,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Convert the internal data in-place to be in physical units.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        Parameters
        ----------
        units : ~unyt.Unit or str, optional
            The desired units for the converted array. If omitted the units are preserved.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            This array in the requested physical units, transformed in place.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value
        """
        if units is not None:
            self.convert_to_units(units, equivalence=equivalence, **kwargs)
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

    def to_physical(
        self,
        units: unyt.Unit | str | None = None,
        *,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> "cosmo_array":
        """
        Create a copy of the data in physical units.

        Parameters
        ----------
        units : ~unyt.Unit or str, optional
            The desired units for the new array. If omitted the units are preserved.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            Copy of this array in the requested physical units.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value
        """
        if not self.valid_transform and self.comoving:
            raise InvalidConversionError
        copied_data = self.in_units(
            self.units if units is None else units, equivalence=equivalence, **kwargs
        )
        copied_data.convert_to_physical()

        return copied_data

    def to_comoving(
        self,
        units: unyt.Unit | str | None = None,
        *,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> "cosmo_array":
        """
        Create a copy of the data in comoving units.

        Parameters
        ----------
        units : ~unyt.Unit or str, optional
            The desired units for the new array. If omitted the units are preserved.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            Copy of this array in the requested comoving units.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value
        """
        if not self.valid_transform and self.comoving is not False:
            raise InvalidConversionError
        copied_data = self.in_units(
            self.units if units is None else units, equivalence=equivalence, **kwargs
        )
        copied_data.convert_to_comoving()

        return copied_data

    def to(
        self,
        units: unyt.Unit | str | None = None,
        *,
        equivalence: str | None = None,
        comoving: bool | None = None,
        **kwargs: Any,
    ) -> "cosmo_array":
        """
        Create a copy of the data in specified comoving or physical units.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        .. note::

            All additional keyword arguments are passed to the equivalency, which should
            be used if that particular equivalency requires them.

        Parameters
        ----------
        units : ~unyt.Unit or str, optional
            The desired units for the new array.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        comoving : bool, optional
            If ``True``, the result is comoving, if ``False`` it is physical. By default
            the ``comoving`` status of the array is preserved.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            Copy of this array in comoving or physical units.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.
        
        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value

        Examples
        --------
        .. code-block:: python

           >>> import unyt as u
           >>> d = cosmo_quantity(
           ...     1,
           ...     u.Mpc,
           ...     comoving=True,
           ...     scale_factor=0.5,
           ...     scale_exponent=1
           ... )
           >>> d.to(u.kpc, comoving=False)
           cosmo_quantity(500., 'kpc', comoving='False', cosmo_factor='a at a=0.5', \
           valid_transform='True')
        """
        if (
            (comoving is not None)
            and (not self.valid_transform)
            and (comoving != self.comoving)
        ):
            raise InvalidConversionError
        if units is None:
            arr_copy = self.copy()
        else:
            arr_copy = _copy_cosmo_array_attributes(
                self,
                _ensure_result_is_cosmo_array_or_quantity(super().to)(
                    units, equivalence=equivalence, **kwargs
                ),
            )
        # do comoving conversion in-place since we already made a copy.
        # if comoving is not None and self.comoving is None this will (and should) raise.
        if comoving is True:
            arr_copy.convert_to_comoving()
        elif comoving is False:  # can't use else in case comoving is None
            arr_copy.convert_to_physical()
        else:  # comoving is None - keep whatever self had
            pass  # attribute was already copied above
        return arr_copy

    def to_value(
        self,
        units: unyt.Unit | str | None = None,
        *,
        equivalence: str | None = None,
        comoving: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Create a copy of the data in specified comoving or physical units.

        The copy is then returned with the values as a bare :mod:`~numpy` array.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        .. note::

            All additional keyword arguments are passed to the equivalency, which should
            be used if that particular equivalency requires them.

        Parameters
        ----------
        units : ~unyt.Unit or str
            The desired units for the new array.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        comoving : bool, optional
            If ``True``, the result is comoving, if ``False`` it is physical. By default
            the ``comoving`` status of the array is preserved.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            Copy of this array in comoving or physical units.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_comoving_value

        Examples
        --------
        .. code-block:: python

           >>> import unyt as u
           >>> d = cosmo_quantity(
           ...     1,
           ...     u.Mpc,
           ...     comoving=True,
           ...     scale_factor=0.5,
           ...     scale_exponent=1
           ... )
           >>> d.to_value(u.kpc, comoving=False)
           500.0
        """
        return np.array(
            self.to(units, equivalence=equivalence, comoving=comoving, **kwargs)
        )

    def to_physical_value(
        self,
        units: Unit | str,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Return a copy of the array values in the specified physical units.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        .. note::

            All additional keyword arguments are passed to the equivalency, which should
            be used if that particular equivalency requires them.

        Parameters
        ----------
        units : ~unyt.Unit or str
            The desired units for the new array.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            Copy of the array values in the specified physical units.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_comoving_value
        swiftsimio.objects.cosmo_array.to_value
        """
        return self.to_value(units, equivalence=equivalence, comoving=False, **kwargs)

    def to_comoving_value(
        self,
        units: Unit | str,
        equivalence: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Return a copy of the array values in the specified comoving units.

        Optionally, an equivalence can be specified to convert to an equivalent quantity
        which is not in the same dimensions.

        .. note::

            All additional keyword arguments are passed to the equivalency, which should
            be used if that particular equivalency requires them.

        Parameters
        ----------
        units : ~unyt.Unit or str
            The desired units for the new array.

        equivalence : str, optional
            The equivalence to use. To see which equivalences are supported for this
            quantity, try the :meth:`~unyt.array.unyt_array.list_equivalencies` method.

        **kwargs : Any
            Any additional keyword arguments are supplied to the equivalence.

        Returns
        -------
        np.ndarray
            Copy of the array values in the specified comoving units.

        Raises
        ------
        UnitConversionError
            If the provided ``units`` does not have the same dimensions as the array and
            cannot be converted via a provided ``equivalence``.

        See Also
        --------
        swiftsimio.objects.cosmo_array.convert_to_physical
        swiftsimio.objects.cosmo_array.convert_to_comoving
        swiftsimio.objects.cosmo_array.convert_to
        swiftsimio.objects.cosmo_array.to_physical
        swiftsimio.objects.cosmo_array.to_comoving
        swiftsimio.objects.cosmo_array.to
        swiftsimio.objects.cosmo_array.to_physical_value
        swiftsimio.objects.cosmo_array.to_value
        """
        return self.to_value(units, equivalence=equivalence, comoving=True, **kwargs)

    def compatible_with_comoving(self) -> bool:
        """
        Check whether array is compatible with comoving units.

        This is the case if the :class:`~swiftsimio.objects.cosmo_array` is comoving, or
        if the scale factor exponent is 0, or the scale factor is 1
        (either case satisfies ``cosmo_factor.a_factor() == 1``).

        Returns
        -------
        bool
            ``True`` if compatible, ``False`` otherwise.
        """
        return self.comoving or (self.cosmo_factor.a_factor == 1.0)

    def compatible_with_physical(self) -> bool:
        """
        Check whether array is compatible with physical units.

        This is the case if the :class:`~swiftsimio.objects.cosmo_array` is physical, or
        if the scale factor exponent is 0, or the scale factor is 1
        (either case satisfies ``cosmo_factor.a_factor() == 1``).

        Returns
        -------
        bool
            ``True`` if compatible, ``False`` otherwise.
        """
        return (not self.comoving) or (self.cosmo_factor.a_factor == 1.0)

    @classmethod
    def from_astropy(
        cls,
        arr: "astropy.units.quantity.Quantity",
        unit_registry: unyt.unit_registry.UnitRegistry = None,
        comoving: bool | None = None,
        cosmo_factor: cosmo_factor = NULL_CF,
        compression: str | None = None,
        valid_transform: bool = True,
    ) -> "cosmo_array":
        """
        Convert :mod:`astropy` arrays to our cosmology array class.

        Convert an :class:`astropy.units.quantity.Quantity` to a
        :class:`~swiftsimio.objects.cosmo_array`.

        Parameters
        ----------
        arr : astropy.units.quantity.Quantity
            The quantity to convert from.

        unit_registry : unyt.unit_registry.UnitRegistry, optional
            A unyt registry to use in the conversion. If one is not supplied, the
            default one will be used.

        comoving : bool
            Flag to indicate whether using comoving coordinates.

        cosmo_factor : swiftsimio.objects.cosmo_factor
            Object to store conversion data between comoving and physical coordinates.

        compression : str
            Description of the compression filters that were applied to that array in the
            hdf5 file.

        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving. If
            ``False``, then ``comoving`` must be ``False`` (or ``None``).

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmology-aware array.

        Examples
        --------
        .. code-block:: python

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
        arr: "pint.registry.Quantity",
        unit_registry: unyt.unit_registry.UnitRegistry = None,
        comoving: bool | None = None,
        cosmo_factor: cosmo_factor = NULL_CF,
        compression: str | None = None,
        valid_transform: bool = True,
    ) -> "cosmo_array":
        """
        Convert :mod:`pint` arrays to our cosmology array class.

        Convert a :class:`pint.registry.Quantity` to a
        :class:`~swiftsimio.objects.cosmo_array`.

        Parameters
        ----------
        arr : pint.registry.Quantity
            The quantity to convert from.
        unit_registry : unyt.unit_registry.UnitRegistry, optional
            A unyt registry to use in the conversion. If one is not supplied, the
            default one will be used.
        comoving : bool
            Flag to indicate whether using comoving coordinates.
        cosmo_factor : swiftsimio.objects.cosmo_factor
            Object to store conversion data between comoving and physical coordinates.
        compression : str
            Description of the compression filters that were applied to that array in the
            hdf5 file.
        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving. If
            ``False``, then ``comoving`` must be ``False`` (or ``None``).

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmology-aware array.

        Examples
        --------
        .. code-block:: python

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

    @classmethod
    def __unyt_ufunc_prepare__(
        cls, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> tuple[np.ufunc, str, tuple, dict]:
        """
        Prepare arguments for a ufunc call.

        This function gives us the opportunity to pre-process arguments to a ufunc call
        before handing control off to :mod:`unyt`. The arguments and kwargs are checked
        for consistent ``cosmo_factor`` attributes and coerced to a common
        comoving/physical state.

        Parameters
        ----------
        ufunc : ~numpy.ufunc
            The ufunc that is about to be called.

        method : str
            The call method for the ufunc (for example ``"call"`` or ``"reduce"``).

        *inputs : Any
            The ufunc arguments.

        **kwargs : Any
            The ufunc kwargs.

        Returns
        -------
        ~numpy.ufunc
            The ufunc that is about to be called.

        str
            The call method for the ufunc.

        tuple
            The now prepared arguments for the ufunc.

        dict
            The now prepared kwargs for the ufunc.
        """
        helper_result = _prepare_array_func_args(*inputs, **kwargs)
        return ufunc, method, helper_result["args"], helper_result["kwargs"]

    @classmethod
    def __unyt_ufunc_finalize__(
        cls,
        result: tuple | unyt_array,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> "tuple | cosmo_array":
        """
        Finalize results after a ufunc call.

        This function gives us the opportunity to post-process return value(s) from a
        ufunc when we get control back from :mod:`unyt`. We check that the return type is
        consistent with its shape (i.e. a :class:`~swiftsimio.objects.cosmo_array` or
        :class:`~swiftsimio.objects.cosmo_quantity`) and attach our cosmo attributes.

        Parameters
        ----------
        result : ~unyt.array.unyt_array or tuple
            The return value of the called ufunc.

        ufunc : ~numpy.ufunc
            The ufunc that was called.

        method : str
            The call method for the ufunc (for example ``"call"`` or ``"reduce"``).

        *inputs : Any
            The ufunc arguments.

        **kwargs : Any
            The ufunc kwargs.

        Returns
        -------
        tuple or comso_array
            The result of the ufunc call, with the appropriate type and cosmo attributes
            attached.
        """
        # wonder if we could cache helper_result during __unyt_ufunc_prepare__ to use
        # here?
        helper_result = _prepare_array_func_args(*inputs, **kwargs)
        cfs = helper_result["cfs"]
        ret_cf: cosmo_factor | None
        # make sure we evaluate the cosmo_factor_ufunc_registry function:
        # might raise/warn even if we're not returning a cosmo_array
        if ufunc in (multiply, divide) and method == "reduce":
            power_map = POWER_MAPPING[ufunc]
            if "axis" in kwargs and kwargs["axis"] is not None:
                ret_cf = _power_cosmo_factor(
                    cfs[0], None, power=power_map(inputs[0].shape[kwargs["axis"]])
                )
            else:
                ret_cf = _power_cosmo_factor(
                    cfs[0], None, power=power_map(inputs[0].size)
                )
        elif (
            ufunc in (logical_and, logical_or, logical_xor, logical_not)
            and method == "reduce"
        ):
            ret_cf = _return_without_cosmo_factor(cfs[0])
        else:
            ret_cf = cls._cosmo_factor_ufunc_registry[ufunc](*cfs, inputs=inputs)
        # if we get a tuple we have multiple return values to deal with
        if isinstance(result, tuple):
            result = tuple(
                (
                    r.view(cosmo_quantity)
                    if r.shape == ()
                    else (
                        r.view(cosmo_array)
                        if isinstance(r, unyt_array) and not isinstance(r, cosmo_array)
                        else r
                    )
                )
                for r in result
            )
            for r in result:
                if isinstance(r, cosmo_array):  # also recognizes cosmo_quantity
                    r.comoving = helper_result["comoving"]
                    r.cosmo_factor = ret_cf
                    r.valid_transform = helper_result["valid_transform"]
                    r.compression = helper_result["compression"]
        elif isinstance(result, unyt_array):  # also recognizes cosmo_quantity
            if not isinstance(result, cosmo_array):
                result = (
                    result.view(cosmo_quantity)
                    if result.shape == ()
                    else result.view(cosmo_array)
                )
            result.comoving = helper_result["comoving"]
            result.cosmo_factor = ret_cf
            result.valid_transform = helper_result["valid_transform"]
            result.compression = helper_result["compression"]
        if "out" in kwargs:
            out: tuple | unyt_array = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                aout = out[0]
                if isinstance(aout, unyt_array) and not isinstance(aout, cosmo_array):
                    aout = (
                        aout.view(cosmo_quantity)
                        if aout.shape == ()
                        else aout.view(cosmo_array)
                    )
                if isinstance(aout, cosmo_array):  # also recognizes cosmo_quantity
                    aout.comoving = helper_result["comoving"]
                    aout.cosmo_factor = ret_cf
                    aout.valid_transform = helper_result["valid_transform"]
                    aout.compression = helper_result["compression"]
            else:
                out = tuple(
                    (
                        (
                            o.view(cosmo_quantity)
                            if o.shape == ()
                            else o.view(cosmo_array)
                        )
                        if isinstance(o, unyt_array) and not isinstance(o, cosmo_array)
                        else o
                    )
                    for o in out
                )
                for o in out:
                    if isinstance(o, cosmo_array):  # also recognizes cosmo_quantity
                        o.comoving = helper_result["comoving"]
                        o.cosmo_factor = ret_cf
                        o.valid_transform = helper_result["valid_transform"]
                        o.compression = helper_result["compression"]
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> object:
        """
        Handle :mod:`numpy` ufunc calls on :class:`~swiftsimio.objects.cosmo_array` input.

        :mod:`numpy` facilitates wrapping array classes by handing off to this function
        when a function of :class:`numpy.ufunc` type is called with arguments from an
        inheriting array class. Since we inherit from :class:`~unyt.array.unyt_array`,
        we let :mod:`unyt` handle what to do with the units and take care of processing
        the cosmology information via our helper functions.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The numpy function being called.

        method : str, optional
            Some ufuncs have methods accessed as attributes, such as ``"reduce"``.
            If using such a method, this argument receives its name.

        *inputs : Any
            Arguments to the ufunc.

        **kwargs : Any
            Keyword arguments to the ufunc.

        Returns
        -------
        object
            The result of the ufunc call, with our cosmology attribute processing applied.
        """
        helper_result = _prepare_array_func_args(*inputs, **kwargs)
        cfs = helper_result["cfs"]

        # make sure we evaluate the cosmo_factor_ufunc_registry function:
        # might raise/warn even if we're not returning a cosmo_array
        if ufunc in (multiply, divide) and method == "reduce":
            power_map = POWER_MAPPING[ufunc]
            if "axis" in kwargs and kwargs["axis"] is not None:
                ret_cf = _power_cosmo_factor(
                    cfs[0], None, power=power_map(inputs[0].shape[kwargs["axis"]])
                )
            else:
                ret_cf = _power_cosmo_factor(
                    cfs[0], None, power=power_map(inputs[0].size)
                )
        elif (
            ufunc in (logical_and, logical_or, logical_xor, logical_not)
            and method == "reduce"
        ):
            _return_without_cosmo_factor(cfs[0])  # check validity
            ret_cf = None
        else:
            ret_cf = self._cosmo_factor_ufunc_registry[ufunc](*cfs, inputs=inputs)

        result = _ensure_result_is_cosmo_array_or_quantity(super().__array_ufunc__)(
            ufunc, method, *helper_result["args"], **helper_result["kwargs"]
        )
        # if we get a tuple we have multiple return values to deal with
        if isinstance(result, tuple):
            for r in result:
                if isinstance(r, cosmo_array):  # also recognizes cosmo_quantity
                    r.comoving = helper_result["comoving"]
                    r.cosmo_factor = ret_cf
                    r.valid_transform = helper_result["valid_transform"]
                    r.compression = helper_result["compression"]
        elif isinstance(result, cosmo_array):  # also recognizes cosmo_quantity
            result.comoving = helper_result["comoving"]
            result.cosmo_factor = ret_cf
            result.valid_transform = helper_result["valid_transform"]
            result.compression = helper_result["compression"]
        if "out" in kwargs:
            out: tuple | unyt_array = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                out = out[0]
                if isinstance(out, cosmo_array):  # also recognizes cosmo_quantity
                    out.comoving = helper_result["comoving"]
                    out.cosmo_factor = ret_cf
                    out.valid_transform = helper_result["valid_transform"]
                    out.compression = helper_result["compression"]
            else:
                for o, r in zip(out, result):
                    if isinstance(o, cosmo_array):  # also recognizes cosmo_quantity
                        o.comoving = helper_result["comoving"]
                        o.cosmo_factor = ret_cf
                        o.valid_transform = helper_result["valid_transform"]
                        o.compression = helper_result["compression"]

        return result

    def __array_function__(
        self,
        func: Callable,
        types: Collection,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> object:
        """
        Handle :mod:`numpy` functions for :class:`~swiftsimio.objects.cosmo_array` input.

        :mod:`numpy` facilitates wrapping array classes by handing off to this function
        when a numpy-defined function is called with arguments from an
        inheriting array class. Since we inherit from :class:`~unyt.array.unyt_array`,
        we let :mod:`unyt` handle what to do with the units and take care of processing
        the cosmology information via our helper functions.

        Parameters
        ----------
        func : Callable
            The numpy function being called.

        types : collections.abc.Collection
            A collection of unique argument types from the original :mod:`numpy` function
            call that implement ``__array_function__``.

        args : tuple
            Arguments to the functions.

        kwargs : dict
            Keyword arguments to the function.

        Returns
        -------
        object
            The result of the ufunc call, with our cosmology attribute processing applied.
        """
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

        if not all(issubclass(t, cosmo_array) or t is np.ndarray for t in types):
            # Note: this allows subclasses that don't override
            # __array_function__ to handle cosmo_array objects
            return NotImplemented

        if func in _HANDLED_FUNCTIONS:
            function_to_invoke = _HANDLED_FUNCTIONS[func]
        elif func in _UNYT_HANDLED_FUNCTIONS:
            function_to_invoke = _UNYT_HANDLED_FUNCTIONS[func]
        else:
            # default to numpy's private implementation
            if hasattr(func, "_implementation"):
                function_to_invoke = getattr(func, "_implementation")
            else:
                raise ValueError(f"Unsupported function {func}, please report this.")
        return function_to_invoke(*args, **kwargs)

    def __mul__(
        self,
        b: "numeric_type | np.ndarray | unyt.unit_object.Unit | cosmo_array | _AHelper",
    ) -> "cosmo_array | _AHelper":
        """
        Multiply this :class:`~swiftsimio.objects.cosmo_array`.

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit` and the case where the
        second argument is a :class:`~swiftsimio.objects._AHelper`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit` or :class:`~swiftsimio.objects.cosmo_array` or \
        :class:`~swiftsimio.objects._AHelper`
            The object to multiply with this one.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            The result of the multiplication.
        """
        if getattr(b, "is_Unit", False):
            return _copy_cosmo_array_attributes(
                self,
                _ensure_result_is_cosmo_array_or_quantity(b.__mul__)(
                    self.view(unyt_quantity)
                    if self.shape == ()
                    else self.view(unyt_array)
                ),
            )
        elif isinstance(b, _AHelper):
            return b.__mul__(self)
        else:
            return super().__mul__(b)

    def __rmul__(
        self,
        b: "numeric_type | np.ndarray | unyt.unit_object.Unit | cosmo_array | _AHelper",
    ) -> "cosmo_array | _AHelper":
        """
        Multiply this :class:`~swiftsimio.objects.cosmo_array` (as the right argument).

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit` or :class:`~swiftsimio.objects.cosmo_array` or \
        :class:`~swiftsimio.objects._AHelper`
            The object to multiply with this one.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            The result of the multiplication.
        """
        if getattr(b, "is_Unit", False):
            return self.__mul__(b)
        else:
            return super().__rmul__(b)

    def __imul__(
        self,
        b: "numeric_type | np.ndarray | unyt.unit_object.Unit | cosmo_array | _AHelper",
    ) -> "cosmo_array | _AHelper":
        """
        Multiply this :class:`~swiftsimio.objects.cosmo_array` (in-place).

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit` or :class:`~swiftsimio.objects.cosmo_array` or \
        :class:`~swiftsimio.objects._AHelper`
            The object to multiply with this one.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            The result of the multiplication.
        """
        return self.__mul__(b)

    def __truediv__(
        self,
        b: "numeric_type | np.ndarray | unyt.unit_object.Unit | cosmo_array | _AHelper",
    ) -> "cosmo_array | _AHelper":
        """
        Divide this :class:`~swiftsimio.objects.cosmo_array`.

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit` and the case where the
        second argument is a :class:`~swiftsimio.objects._AHelper`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit` or :class:`~swiftsimio.objects.cosmo_array` or \
        :class:`~swiftsimio.objects._AHelper`
            The object to divide this one by.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            The result of the division.
        """
        if getattr(b, "is_Unit", False):
            return _copy_cosmo_array_attributes(
                self,
                _ensure_result_is_cosmo_array_or_quantity((1 / b).__mul__)(
                    self.view(unyt_quantity)
                    if self.shape == ()
                    else self.view(unyt_array)
                ),
            )
        elif isinstance(b, _AHelper):
            return (1 / b).__mul__(self)
        else:
            return super().__truediv__(b)

    def __rtruediv__(
        self,
        b: "numeric_type | np.ndarray | unyt.unit_object.Unit | cosmo_array | _AHelper",
    ) -> "cosmo_array | _AHelper":
        """
        Divide by this :class:`~swiftsimio.objects.cosmo_array` (as the right argument).

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit` or :class:`~swiftsimio.objects.cosmo_array` or \
        :class:`~swiftsimio.objects._AHelper`
            The object to divide by this one.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            The result of the division.
        """
        if getattr(b, "is_Unit", False):
            return (self.__rtruediv__(1)).__mul__(b)
        else:
            return super().__rtruediv__(b)

    def __itruediv__(
        self,
        b: "numeric_type | np.ndarray | unyt.unit_object.Unit | cosmo_array | _AHelper",
    ) -> "cosmo_array | _AHelper":
        """
        Divide this :class:`~swiftsimio.objects.cosmo_array` (in-place).

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit` or :class:`~swiftsimio.objects.cosmo_array` or \
        :class:`~swiftsimio.objects._AHelper`
            The object to divide this one by.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            The result of the division.
        """
        return self.__truediv__(b)


class cosmo_quantity(cosmo_array, unyt_quantity):
    """
    Cosmology scalar class.

    This inherits from both the :class:`~swiftsimio.objects.cosmo_array` and the
    :class:`~unyt.array.unyt_quantity`, and has the same four attributes as
    :class:`~swiftsimio.objects.cosmo_array`: ``compression``, ``cosmo_factor``,
    ``comoving``, and ``valid_transform``.

    Like :class:`unyt.array.unyt_quantity`, it is intended to hold a scalar value.
    Values of this type will be returned by :mod:`numpy` functions that return
    scalar values.

    Other than containing a scalar, functionality is identical to
    :class:`~swiftsimio.objects.cosmo_array`. Refer to that class's documentation.

    Attributes
    ----------
    comoving : bool
        if True then the array is in comoving co-ordinates, and if
        False then it is in physical units.

    cosmo_factor : float
        Object to store conversion data between comoving and physical coordinates

    compression : str
        String describing any compression that was applied to this array in the
        hdf5 file.

    valid_transform: bool
       if True then the array can be converted from physical to comoving units
    """

    def __new__(
        cls,
        input_scalar: numeric_type | unyt.unyt_quantity,
        units: "str | unyt.unit_object.Unit | astropy.units.core.Unit | None" = None,
        # *,
        registry: unyt.unit_registry.UnitRegistry | None = None,
        dtype: np.dtype | str | None = None,
        bypass_validation: bool = False,
        name: str | None = None,
        cosmo_factor: cosmo_factor | None = None,
        scale_factor: numeric_type | None = None,
        scale_exponent: numeric_type | None = None,
        comoving: bool | None = None,
        valid_transform: bool = True,
        compression: str | None = None,
    ) -> "cosmo_quantity":
        """
        Construct a cosmo_quantity instance.

        Parameters
        ----------
        input_scalar : int or float or unyt.array.unyt_quantity
            A tuple, list, or array to attach units and cosmology information to.

        units : str, unyt.unit_object.Unit or astropy.units.core.Unit, optional
            The units of the array. When using strings, powers must be specified using
            python syntax (``cm**3``, not ``cm^3``).

        registry : unyt.unit_registry.UnitRegistry, optional
            The registry to create units from. If ``units`` is already associated
            with a unit registry and this is specified, this will be used instead of the
            registry associated with the unit object.

        dtype : np.dtype or str, optional
            The dtype of the array data. Defaults to the dtype of the input data, or, if
            none is found, uses ``np.float64``.

        bypass_validation : bool, optional
            If ``True``, all input validation is skipped. Using this option may produce
            corrupted or invalid data, but can lead to significant speedups
            in the input validation logic adds significant overhead. If set, minimally
            pass valid values for units, comoving and cosmo_factor. Defaults to ``False``.

        name : str, optional
            The name of the array. Defaults to ``None``. This attribute does not propagate
            through mathematical operations, but is preserved under indexing and unit
            conversions.

        cosmo_factor : swiftsimio.objects.cosmo_factor
            Object to store conversion data between comoving and physical coordinates.

        scale_factor : float
            The scale factor associated to the data. Also provide a value for
            ``scale_exponent``.

        scale_exponent : int or float
            The exponent for the scale factor giving the scaling for conversion to/from
            comoving units. Also provide a value for ``scale_factor``.

        comoving : bool
            Flag to indicate whether using comoving coordinates.

        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving. If
            ``False``, then ``comoving`` must be ``False`` (or ``None``).

        compression : str
            Description of the compression filters that were applied to that array in the
            hdf5 file.

        Returns
        -------
        cosmo_quantity
            The constructed object.
        """
        if bypass_validation is True:
            result = super().__new__(
                cls,
                np.asarray(input_scalar),
                units=units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                name=name,
                cosmo_factor=cosmo_factor,
                scale_factor=scale_factor,
                scale_exponent=scale_exponent,
                comoving=comoving,
                valid_transform=valid_transform,
                compression=compression,
            )

        if not isinstance(input_scalar, (numeric_type, np.ndarray)):
            raise RuntimeError("cosmo_quantity values must be numeric")

        # Use values from kwargs, if None use values from input_scalar
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
            else (valid_transform if comoving is not None else False)
        )
        compression = (
            getattr(input_scalar, "compression", None)
            if compression is None
            else compression
        )
        result = super().__new__(
            cls,
            np.asarray(input_scalar),
            units=units,
            registry=registry,
            dtype=dtype,
            bypass_validation=bypass_validation,
            name=name,
            cosmo_factor=cosmo_factor,
            scale_factor=scale_factor,
            scale_exponent=scale_exponent,
            comoving=comoving,
            valid_transform=valid_transform,
            compression=compression,
        )
        if result.size > 1:
            raise RuntimeError("cosmo_quantity instances must be scalars")
        return result

    __round__ = _propagate_cosmo_array_attributes_to_result(
        _ensure_result_is_cosmo_array_or_quantity(unyt_quantity.__round__)
    )


class _AHelper:
    """
    Offer an easy way to initialize cosmo scalars and arrays.

    Using the full :class:`~swiftsimio.objects.cosmo_array` or
    :class:`~swiftsimio.objects.cosmo_quantity` constructors can be tedious. This
    helper offers a way to write things like ``10 * u.Mpc * a.comoving**1`` or
    ``np.array([1, 2]) * u.g / u.cm**3 * a.physical**-3``. It is initialized with the
    rest of the mask/dataset metadata and then stored as ``metadata.a`` so that it
    can be easily retrieved. It also allows defining "cosmo units" such as
    ``cMpc = u.Mpc * a.comoving``.

    In most cases this class should return new instances of itself with modified
    attributes: we should not be modifying the state of the helper assigned to
    ``metadata.a``!

    There is an important design choice around data copying.
    :class:`~unyt.array.unyt_array` does copy data on construction:

    .. code-block:: python

        >>> import unyt as u
        >>> x = np.array([1, 2])
        >>> y = x * u.kpc
        >>> x[0] = 999  # notice units are not checked/enforced...
        >>> x
        array([999, 2])
        >>> y
        unyt_array([1, 2], 'kpc')

    But :class:`~swiftsimio.objects.cosmo_array` does not copy data, instead using a view:
    
    .. code-block:: python

        >>> from swiftsimio import cosmo_array
        >>> x = np.array([1, 2]) * u.kpc
        >>> y = x * a.comoving
        >>> x[0] = 999
        >>> x
        unyt_array([999, 2], 'kpc')
        >>> y
        cosmo_array([999, 2], 'kpc', comoving='True', cosmo_factor='a at a=1', \
        valid_transform='True')

    This helper therefore uses views to stay consistent with the
    :class:`~swiftsimio.objects.cosmo_array` behaviour.

    Parameters
    ----------
    scale_factor : float
        The scale factor used to create/modify :class:`~swiftsimio.objects.cosmo_factor`
        objects as needed.

    scale_exponent : int, optional
        The exponent that the scale factor scales with, default of ``1``.

    units : ~unyt.Unit, optional
        The units to attach to objects.

    comoving : bool, optional
        Whether to create comoving or physical objects.
    """

    _scale_factor: numeric_type
    scale_exponent: numeric_type
    units: unyt.Unit
    _comoving_state: bool | None

    def __init__(
        self,
        scale_factor: numeric_type,
        scale_exponent: numeric_type = 1,
        units: unyt.Unit = unyt.Unit("1"),  # auto-simplifies unlike unyt.dimensionless
        comoving: bool | None = None,
    ) -> None:
        self._scale_factor = scale_factor
        self.scale_exponent = scale_exponent
        self.units = units
        self._comoving_state = comoving

    @property
    def scale_factor(self) -> numeric_type:
        """
        Get the scale factor.

        The scale factor should always be accessed through this property. This ensures
        that the :class:`~swiftsimio.objects._AHelper` cannot be used when its
        ``_comoving`` attribute is ``None``, i.e. ``10 * metadata.a * u.kpc`` is invalid
        but ``10 * metadata.a.physical * u.kpc`` is valid (the ``comoving`` and
        ``physical`` properties return a copy with ``_comoving is not None``).

        The exception is when creating new :class:`~swiftsimio.objects._AHelper` objects
        from old ones, then ``_AHelper(scale_factor=self._scale_factor, ...)`` is
        needed.

        Returns
        -------
        float
            The scale factor.

        Raises
        ------
        InvalidCosmoUnit
            If access is attempted when physical or comoving has not been specified.
        """
        if self._comoving_state is None:
            raise InvalidCosmoUnit(
                "Cannot use scale factor helper as `a` alone, use `a.comoving` or "
                "`a.physical`. For the scale factor as a number, use "
                "`metadata.scale_factor` instead of `metadata.a`."
            )
        return self._scale_factor

    @property
    def _comoving(self) -> bool:
        """
        Get the comoving status of the helper.

        The comoving status should always be accessed through this property. This ensures
        that the :class:`~swiftsimio.objects._AHelper` cannot be used when its
        ``_comovin`` attriubte is ``None``, i.e. ``10 * metadata.a * u.kpc`` is invalid
        but ``10 * metadata.a.physical * u.kpc`` is valid (the ``comoving`` and
        ``physical`` properties return a copy with ``_comoving is not None``).

        The exception is when creating new :class:`~swiftsimio.objects._AHelper` objects
        from old ones, then ``_AHelper(comoving=self._comoving_state, ...)`` is
        needed.

        Returns
        -------
        float
            The comoving status.

        Raises
        ------
        InvalidCosmoUnit
            If access is attempted when physical or comoving has not been specified.
        """
        if self._comoving_state is None:
            raise InvalidCosmoUnit(
                "Cannot use scale factor helper as `a` alone, use `a.comoving` or "
                "`a.physical`. For the scale factor as a number, use "
                "`metadata.scale_factor` instead of `metadata.a`."
            )
        return self._comoving_state

    @property
    def _comoving_str(self) -> str:
        """
        Get the comoving status as a string for use in messages.

        Returns
        -------
        str
            The state, either ``"comoving"``, ``"physical"`` or ``"None"``.
        """
        if self._comoving:
            return "comoving"
        elif self._comoving is False:
            return "physical"
        else:
            return "None"

    @classmethod
    def __unyt_ufunc_prepare__(
        cls, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> tuple[np.ufunc, str, tuple, dict]:
        """
        Prepare arguments for a ufunc call.

        This function gives us the opportunity to pre-process arguments to a ufunc call
        before handing control off to :mod:`unyt`. We strip away any cosmo attributes
        to be restored in :meth:`~swiftsimio.objects._AHelper.__unyt_ufunc_finalize__`.

        Parameters
        ----------
        ufunc : ~numpy.ufunc
            The ufunc that is about to be called.

        method : str
            The call method for the ufunc (for example ``"call"`` or ``"reduce"``).

        *inputs : Any
            The ufunc arguments.

        **kwargs : Any
            The ufunc kwargs.

        Returns
        -------
        ~numpy.ufunc
            The ufunc that is about to be called.

        str
            The call method for the ufunc.

        tuple
            The now prepared arguments for the ufunc.

        dict
            The now prepared kwargs for the ufunc.
        """
        if ufunc not in (np.multiply, np, divide):
            return NotImplemented
        prepared_inputs = tuple(
            unyt_quantity(
                1,
                inp.units,
            )
            if isinstance(inp, _AHelper)
            else inp
            for inp in inputs
        )
        return (ufunc, method, prepared_inputs, kwargs)

    @classmethod
    def __unyt_ufunc_finalize__(
        cls,
        result: unyt_array,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> cosmo_array:
        """
        Finalize results after a ufunc call.

        This function gives us the opportunity to post-process return value(s) from a
        ufunc when we get control back from :mod:`unyt`. We turn it into a
        :class:`~swiftsimio.objects.cosmo_array` or
        :class:`~swiftsimio.objects.cosmo_quantity` and set its attributes.

        We only support ``multiply`` and ``divide`` ufuncs so result is a
        :class:`~unyt.array.unyt_array` (never a ``tuple``).

        Parameters
        ----------
        result : ~unyt.array.unyt_array
            The return value of the called ufunc.

        ufunc : ~numpy.ufunc
            The ufunc that was called.

        method : str
            The call method for the ufunc (for example ``"call"`` or ``"reduce"``).

        *inputs : Any
            The ufunc arguments.

        **kwargs : Any
            The ufunc kwargs.

        Returns
        -------
        tuple or comso_array
            The result of the ufunc call, with the appropriate type and cosmo attributes
            attached.
        """
        for inp in inputs:
            if isinstance(inp, _AHelper):
                a_helper_input: _AHelper = inp

        return (cosmo_array if np.asarray(result).ndim else cosmo_quantity)(
            result,
            comoving=a_helper_input._comoving,
            scale_factor=a_helper_input.scale_factor,
            scale_exponent=(-1 if ufunc is np.divide else 1)
            * a_helper_input.scale_exponent,
        )

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> cosmo_array:
        """
        Handle :mod:`numpy` ufunc calls on :class:`~swiftsimio.objects.cosmo_array` input.

        :mod:`numpy` facilitates wrapping array classes by handing off to this function
        when a function of :class:`numpy.ufunc` type is called with arguments from an
        inheriting array class. Since we inherit from :class:`~unyt.array.unyt_array`,
        we let :mod:`unyt` handle what to do with the units and take care of processing
        the cosmology information via our helper functions.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The numpy function being called.

        method : str, optional
            Some ufuncs have methods accessed as attributes, such as ``"reduce"``.
            If using such a method, this argument receives its name.

        *inputs : Any
            Arguments to the ufunc.

        **kwargs : Any
            Keyword arguments to the ufunc.

        Returns
        -------
        object
            The result of the ufunc call, with our cosmology attribute processing applied.
        """
        prepared_ufunc, prepared_method, prepared_inputs, prepared_kwargs = (
            self.__unyt_ufunc_prepare__(ufunc, method, *inputs, **kwargs)
        )
        result = getattr(prepared_ufunc, prepared_method)(
            *prepared_inputs, **prepared_kwargs
        )
        return self.__unyt_ufunc_finalize__(result, ufunc, method, *inputs, **kwargs)

    @singledispatchmethod
    def __mul__(self, other: object) -> "cosmo_array | _AHelper":
        """
        Default implementation for invalid multiplications.

        Parameters
        ----------
        other : Any
            The object to multiply with.

        Returns
        -------
        NotImplemented
            This operation is not defined.
        """
        return NotImplemented

    @__mul__.register
    def _(self, other: int) -> cosmo_quantity:
        """
        Multiply with a number.

        Parameters
        ----------
        other : int
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_quantity
            A :class:`~swiftsimio.objects.cosmo_quantity` based on the content of this
            helper and the ``other`` number.
        """
        return self._mul_numeric_type(other)

    @__mul__.register
    def _(self, other: float) -> cosmo_quantity:
        """
        Multiply with a number.

        Parameters
        ----------
        other : float
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_quantity
            A :class:`~swiftsimio.objects.cosmo_quantity` based on the content of this
            helper and the ``other`` number.
        """
        return self._mul_numeric_type(other)

    @__mul__.register
    def _(self, other: np.number) -> cosmo_quantity:
        """
        Multiply with a number.

        Parameters
        ----------
        other : np.number
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_quantity
            A :class:`~swiftsimio.objects.cosmo_quantity` based on the content of this
            helper and the ``other`` number.
        """
        return self._mul_numeric_type(other)

    @__mul__.register
    def _(self, other: complex) -> cosmo_quantity:
        """
        Multiply with a number.

        Parameters
        ----------
        other : complex
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_quantity
            A :class:`~swiftsimio.objects.cosmo_quantity` based on the content of this
            helper and the ``other`` number.
        """
        return self._mul_numeric_type(other)

    def _mul_numeric_type(self, other: numeric_type) -> cosmo_quantity:
        """
        Multiply with a number.

        Parameters
        ----------
        other : int or float or np.number or complex
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_quantity
            A :class:`~swiftsimio.objects.cosmo_quantity` based on the content of this
            helper and the ``other`` number.
        """
        # the four registered functions calling this can be merged into one with
        # type hint `other: numeric_type` once python3.10 support is
        # dropped
        return cosmo_quantity(
            other,
            units=self.units,
            comoving=self._comoving,
            scale_factor=self.scale_factor,
            scale_exponent=self.scale_exponent,
        )

    @__mul__.register
    def _(self, other: np.ndarray) -> cosmo_array:
        """
        Multiply with a :mod:`numpy` array.

        Parameters
        ----------
        other : np.ndarray
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmo array or quantity based on the content of this helper and the
            ``other`` array.
        """
        return (cosmo_array if other.ndim else cosmo_quantity)(
            other,
            units=self.units,
            comoving=self._comoving,
            scale_factor=self.scale_factor,
            scale_exponent=self.scale_exponent,
        )

    def _tuple_or_list_mul(self, other: tuple | list) -> cosmo_array:
        """
        Multiply with a :obj:`list` or :obj:`tuple`.

        Parameters
        ----------
        other : tuple or list
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmo array or quantity based on the content of this helper and the
            ``other`` :obj:`tuple` or :obj:`list`.
        """
        # the two registered functions calling this can be merged into one with
        # type hint `other: tuple | list` once python3.10 support is
        # dropped

        # leave other arguments implicit, could pick up values from content of other:
        ret = cosmo_array(other)
        # now modify attributes:
        if ret.comoving is not None:
            ret.convert_to(ret.units, comoving=self._comoving)
        else:
            ret.comoving = self._comoving
        ret.units = ret.units * self.units
        if ret.cosmo_factor == NULL_CF:
            ret.cosmo_factor = cosmo_factor.create(
                self.scale_factor, self.scale_exponent
            )
        else:
            ret.cosmo_factor = ret.cosmo_factor * cosmo_factor.create(
                self.scale_factor, self.scale_exponent
            )
        return ret

    @__mul__.register
    def _(self, other: tuple) -> cosmo_array:
        """
        Multiply with a :obj:`tuple`.

        Parameters
        ----------
        other : tuple
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmo array or quantity based on the content of this helper and the
            ``other`` :obj:`tuple`.
        """
        return self._tuple_or_list_mul(other)

    @__mul__.register
    def _(self, other: list) -> cosmo_array:
        """
        Multiply with a :obj:`list` or :obj:`tuple`.

        Parameters
        ----------
        other : tuple or list
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmo array or quantity based on the content of this helper and the
            ``other`` :obj:`tuple` or :obj:`list`.
        """
        return self._tuple_or_list_mul(other)

    @__mul__.register
    def _(self, other: unyt.unit_object.Unit) -> Self:
        """
        Multiply with a :class:`unyt.unit_object.Unit`.

        These are multplied with the existing units on the helper (could be dimensionless)
        and a new helper with these new units is returned.

        Parameters
        ----------
        other : ~unyt.unit_object.Unit
            The unit to multiply with.

        Returns
        -------
        ~swiftsimio.objects._AHelper
            A helper updated with the provided units.
        """
        # Ideally want to set the return type to `"_AHelper"` or even `_AHelper` but this
        # isn't possible in python<3.15, see github.com/python/cpython/issues/86153.
        # Best workaround found is to set return type to `Self`, but this forces us
        # to `return self.__class__(...)` instead of `return _AHelper(...)` (a subclass
        # could otherwise invalidate the return type).
        return self.__class__(
            scale_factor=self._scale_factor,
            scale_exponent=self.scale_exponent,
            units=other.units * self.units,
            comoving=self._comoving_state,
        )

    @__mul__.register
    def _(self, other: cosmo_array) -> cosmo_array:
        """
        Multiply with a :class:`~swiftsimio.objects.cosmo_array`.

        Parameters
        ----------
        other : ~swiftsimio.objects.cosmo_array
            The quantity to multiply with.

        Returns
        -------
        ~swiftsimio.objects.cosmo_array
            A cosmo array or quantity based on the content of this helper and the
            ``other`` :class:`~swiftsimio.objects.cosmo_array`.
        """
        if self._comoving:
            other.convert_to_comoving()  # no-op if already comoving
        elif self._comoving is False:
            other.convert_to_physical()  # no-op if already physical
        return other.__class__(
            other.ndview,
            units=other.units * self.units,
            comoving=self._comoving,
            cosmo_factor=other.cosmo_factor
            * cosmo_factor.create(self.scale_factor, self.scale_exponent),
        )

    def __rmul__(
        self, other: unyt.Unit | unyt.unyt_array | cosmo_array
    ) -> "_AHelper | cosmo_array":
        """
        Multiply with argument on the right.

        Just pass the operation to :meth:`~swiftsimio.objects._AHelper.__mul__` to
        handle.

        Parameters
        ----------
        other : ~unyt.unit_object.Unit, ~unyt.array.unyt_array or \
        ~swiftsimio.objects.cosmo_array
            The object to multiply with.

        Returns
        -------
        ~swiftsimio.objects._AHelper or ~swiftsimio.objects.cosmo_aray
            The result of applying the helper to the other operand.
        """
        return self.__mul__(other)

    def __truediv__(
        self, other: unyt.Unit | unyt.unyt_array | cosmo_array
    ) -> "_AHelper | cosmo_array":
        """
        Divide this helper by a unit, number or array.

        Just delegates the operation to :meth:`~swiftsimio.objects._AHelper.__mul__` to
        handle.

        Parameters
        ----------
        other : ~unyt.unit_object.Unit, ~unyt.array.unyt_array or \
        ~swiftsimio.objects.cosmo_array
            The object to divide by.

        Returns
        -------
        ~swiftsimio.objects._AHelper or ~swiftsimio.objects.cosmo_aray
            The result of applying the helper to the other operand.
        """
        # avoid using other ** -1, e.g. integers raise on this
        inv: "_AHelper | cosmo_array" = other * _AHelper(
            scale_factor=self._scale_factor,
            scale_exponent=-self.scale_exponent,
            units=self.units,
            comoving=self._comoving_state,
        )
        return 1 / inv

    def __rtruediv__(
        self, other: unyt.Unit | unyt.unyt_array | cosmo_array
    ) -> "_AHelper | cosmo_array":
        """
        Divide a unit, number or array by this helper.

        Just delegates the operation to :meth:`~swiftsimio.objects._AHelper.__mul__` to
        handle.

        Parameters
        ----------
        other : ~unyt.unit_object.Unit, ~unyt.array.unyt_array or \
        ~swiftsimio.objects.cosmo_array
            The object to divide by this.

        Returns
        -------
        ~swiftsimio.objects._AHelper or ~swiftsimio.objects.cosmo_aray
            The result of applying the helper to the other operand.
        """
        return other * _AHelper(
            scale_factor=self._scale_factor,
            scale_exponent=-self.scale_exponent,
            units=self.units,
            comoving=self._comoving_state,
        )

    def __pow__(self, exponent: numeric_type) -> "_AHelper":
        """
        Raise this helper to a power.

        This modifies the exponent of the scale factor, so that for density scaling as
        the inverse cube we can write e.g. (for a helper called ``a``):

        .. code-block:: python

           raw_density * a.comoving ** -3 * u.g * u.cm**-3

        Parameters
        ----------
        exponent : int or float
            The exponent to raise the scale factor to (cumulative with current exponent).

        Returns
        -------
        ~swiftsimio.objects._AHelper
            A new helper with the modified exponent.
        """
        return _AHelper(
            scale_factor=self._scale_factor,
            scale_exponent=self.scale_exponent * exponent,
            units=self.units**exponent,
            comoving=self._comoving_state,
        )

    @singledispatchmethod
    def __imul__(self, other: object) -> None:
        """
        Do not support in-place multiplication as left operand.

        Parameters
        ----------
        other : int or float or tuple or list or np.ndarray or ~unyt.array.unyt_array or \
        ~swiftsimio.objects.cosmo_array
            The other object to multiply with this one.

        Raises
        ------
        TypeError
            In-place operations with :class:`~swiftsimio.objects._AHelper` as left
            argument are not supported.
        """
        raise TypeError(
            "In-place operations with :class:`~swiftsimio.objects._AHelper` as left "
            "argument are not supported."
        )

    def __ipow__(self, exponent: numeric_type) -> None:
        """
        Do not support in-place exponentiation as left operand.

        Parameters
        ----------
        exponent : int or float
            The exponent to raise this to.

        Raises
        ------
        TypeError
            In-place operations with :class:`~swiftsimio.objects._AHelper` as left
            argument are not supported.
        """
        raise TypeError(
            "In-place operations with :class:`~swiftsimio.objects._AHelper` as left "
            "argument are not supported."
        )

    def __itruediv__(
        self,
        other: numeric_type | tuple | list | np.ndarray | unyt_array | cosmo_array,
    ) -> None:
        """
        Do not support in-place division as left operand.

        Parameters
        ----------
        other : int or float or tuple or list or np.ndarray or ~unyt.array.unyt_array or \
        ~swiftsimio.objects.cosmo_array
            The other object to divide this one by.

        Raises
        ------
        TypeError
            In-place operations with :class:`~swiftsimio.objects._AHelper` as left
            argument are not supported.
        """
        raise TypeError(
            "In-place operations with :class:`~swiftsimio.objects._AHelper` as left "
            "argument are not supported."
        )

    @property
    def comoving(self) -> "_AHelper":
        """
        Indicate that this helper is for a comoving quantity.

        This helper makes it easy to attach the cosmological scaling of quantities to
        arrays. To do this, whether the quantity is physical or comoving must be
        specified. This is done by accessing the ``comoving`` or ``physical`` attributes.

        Returns
        -------
        ~swiftsimio.objects._AHelper
            A new helper object flagged for comoving quantities.

        Examples
        --------
        .. code-block:: python

           >>> from swiftsimio import load
           >>> dat = load("snap.hdf5")
           >>> a = dat.metadata.a
           >>> cMpc = a.comoving * u.Mpc
           >>> comoving_distances = np.arange(3) * cMpc
        """
        return _AHelper(
            scale_factor=self._scale_factor,
            scale_exponent=self.scale_exponent,
            units=self.units,
            comoving=True,
        )

    @property
    def physical(self) -> "_AHelper":
        """
        Indicate that this helper is for a physical quantity.

        This helper makes it easy to attach the cosmological scaling of quantities to
        arrays. To do this, whether the quantity is physical or comoving must be
        specified. This is done by accessing the ``comoving`` or ``physical`` attributes.

        Returns
        -------
        ~swiftsimio.objects._AHelper
            A new helper object flagged for comoving quantities.

        Examples
        --------
        .. code-block:: python

           >>> from swiftsimio import load
           >>> dat = load("snap.hdf5")
           >>> a = dat.metadata.a
           >>> pMpc = a.physical * u.Mpc
           >>> physical_distances = np.arange(3) * pMpc
        """
        return _AHelper(
            scale_factor=self._scale_factor,
            scale_exponent=self.scale_exponent,
            units=self.units,
            comoving=False,
        )

    # provide aliases:
    com = comoving
    phys = physical
    c = comoving
    p = physical
