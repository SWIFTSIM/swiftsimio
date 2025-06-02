"""
Contains classes for our custom :class:`~swiftsimio.objects.cosmo_array`,
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
from numbers import Number as numeric_type
from typing import Iterable, Union, Tuple, Callable, Optional
from collections.abc import Collection

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
    vecdot,
)
from numpy._core.umath import _ones_like, clip
from ._array_functions import (
    _propagate_cosmo_array_attributes_to_result,
    _ensure_result_is_cosmo_array_or_quantity,
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


def _verify_valid_transform_validity(obj: "cosmo_array") -> None:
    """
    Checks that ``comoving`` and ``valid_transform`` attributes are compatible.

    Comoving arrays must be able to transform, while arrays that don't transform must
    be physical. This function raises if this is not the case.

    Parameters
    ----------
    obj : swiftsimio.objects.cosmo_array
        The array whose validity is to be checked.

    Raises
    ------
    AssertionError
        When an invalid combination of ``comoving`` and ``valid_transform`` is found.
    """
    if not obj.valid_transform:
        assert (
            not obj.comoving
        ), "Cosmo arrays without a valid transform to comoving units must be physical"
    if obj.comoving:
        assert (
            obj.valid_transform
        ), "Comoving cosmo_arrays must be able to be transformed to physical"


class InvalidConversionError(Exception):
    """
    Raised when converting from comoving from physical to comoving is not allowed.

    Parameters
    ----------
    message : str, optional
        Message to print in case of invalid conversion.
    """

    def __init__(
        self, message: str = "Could not convert to comoving coordinates."
    ) -> None:
        """
        Constructor for warning of invalid conversion.

        Parameters
        ----------
        message : str, optional
            Message to print in case of invalid conversion.
        """
        self.message = message


class InvalidScaleFactor(Exception):
    """
    Raised when a scale factor is invalid, such as when adding
    two cosmo_factors with inconsistent scale factors.

    Parameters
    ----------
    message : str, optional
        Message to print in case of invalid scale factor.
    """

    def __init__(self, message: str = None, *args) -> None:
        """
        Constructor for warning of invalid scale factor.

        Parameters
        ----------
        message : str, optional
            Message to print in case of invalid scale factor.
        """
        self.message = message

    def __str__(self):
        """
        Print warning message for invalid scale factor.

        Returns
        -------
        out : str
            The error message.
        """
        return f"InvalidScaleFactor: {self.message}"


class InvalidSnapshot(Exception):
    """
    Generated when a snapshot is invalid (e.g. you are trying to partially load a
    sub-snapshot).

    Parameters
    ----------
    message : str, optional
        Message to print in case of invalid snapshot.
    """

    def __init__(self, message: str = None, *args) -> None:
        """
        Constructor for warning of invalid snapshot

        Parameters
        ----------
        message : str, optional
            Message to print in case of invalid snapshot
        """
        self.message = message

    def __str__(self) -> str:
        """
        Print warning message of invalid snapshot
        """
        return f"InvalidSnapshot: {self.message}"


class cosmo_factor(object):
    """
    Cosmology factor class for storing and computing conversion between
    comoving and physical coordinates.

    This takes the expected exponent of the array that can be parsed
    by :mod:`sympy`, and the current value of the cosmological scale factor ``a``.

    This should be given as the conversion from comoving to physical, i.e.
    :math:`r = a^f \times r` where :math:`a` is the scale factor,
    :math:`r` is a physical quantity and :math`r'` a comoving quantity.

    Parameters
    ----------
    expr : sympy.Expr
        Expression used to convert between comoving and physical coordinates.
    scale_factor : float
        The scale factor (a).

    Attributes
    ----------
    expr : sympy.Expr
        Expression used to convert between comoving and physical coordinates.

    scale_factor : float
        The scale factor (a).

    Examples
    --------
    Mass density transforms as :math:`a^3`. To set up a ``cosmo_factor``, supposing
    a current ``scale_factor=0.97``, we import the scale factor ``a`` and initialize
    as:

    ::

        from swiftsimio.objects import a  # the scale factor (a sympy symbol object)
        density_cosmo_factor = cosmo_factor(a**3, scale_factor=0.97)

    :class:`~swiftsimio.objects.cosmo_factor` supports arithmetic, for example:

    ::

        >>> cosmo_factor(a**2, scale_factor=0.5) * cosmo_factor(a**-1, scale_factor=0.5)
        cosmo_factor(expr=a, scale_factor=0.5)

    See Also
    --------
    swiftsimio.objects.cosmo_factor.create
    """

    def __init__(self, expr: sympy.Expr, scale_factor: float) -> None:
        """
        Constructor for cosmology factor class.

        Parameters
        ----------
        expr : sympy.expr
            Expression used to convert between comoving and physical coordinates.

        scale_factor : float
            The scale factor (a).

        See Also
        --------
        swiftsimio.objects.cosmo_factor.create
        """
        self.expr = expr
        self.scale_factor = scale_factor
        pass

    @classmethod
    def create(cls, scale_factor: float, exponent: numeric_type) -> "cosmo_factor":
        """
        Create a :class:`~swiftsimio.objects.cosmo_factor` from a scale factor and
        exponent.

        Parameters
        ----------
        scale_factor : :obj:`float`
            The scale factor.

        exponent : :obj:`int` or :obj:`float`
            The exponent defining the scaling with the scale factor.

        Examples
        --------
        ::

            >>> cosmo_factor.create(0.5, 2)
            cosmo_factor(expr=a**2, scale_factor=0.5)
        """

        obj = cls(a ** exponent, scale_factor)

        return obj

    def __str__(self) -> str:
        """
        Print exponent and current scale factor.

        Returns
        -------
        out : str
            String with exponent and current scale factor.
        """
        return str(self.expr) + f" at a={self.scale_factor}"

    @property
    def a_factor(self) -> float:
        """
        The multiplicative factor for conversion from comoving to physical.

        For example, for density this is :math:`a^{-3}`.

        Returns
        -------
        out : float
            The multiplicative factor for conversion from comoving to physical.
        """
        if (self.expr is None) or (self.scale_factor is None):
            return None
        return float(self.expr.subs(a, self.scale_factor))

    @property
    def redshift(self) -> float:
        """
        The redshift computed from the scale factor.

        Returns the redshift :math:`z = \\frac{1}{a} - 1`, where :math:`a` is the scale
        factor.

        Returns
        -------
        out : float
            The redshift.
        """
        if self.scale_factor is None:
            return None
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
        out : swiftsimio.objects.cosmo_factor
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
        out : swiftsimio.objects.cosmo_factor
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
        out : swiftsimio.objects.cosmo_factor
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

        if ((self.expr is None) and (b.expr is not None)) or (
            (self.expr is not None) and (b.expr is None)
        ):
            raise InvalidScaleFactor(
                "Attempting to multiply an initialized cosmo_factor with an "
                f"uninitialized cosmo_factor {self} and {b}."
            )
        if (self.expr is None) and (b.expr is None):
            # let's be permissive and allow two uninitialized cosmo_factors through
            return cosmo_factor(expr=None, scale_factor=self.scale_factor)

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
        out : swiftsimio.objects.cosmo_factor
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

        if ((self.expr is None) and (b.expr is not None)) or (
            (self.expr is not None) and (b.expr is None)
        ):
            raise InvalidScaleFactor(
                "Attempting to divide an initialized cosmo_factor with an "
                f"uninitialized cosmo_factor {self} and {b}."
            )
        if (self.expr is None) and (b.expr is None):
            # let's be permissive and allow two uninitialized cosmo_factors through
            return cosmo_factor(expr=None, scale_factor=self.scale_factor)

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
        out : swiftsimio.objects.cosmo_factor
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
        out : swiftsimio.objects.cosmo_factor
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
        out : swiftsimio.objects.cosmo_factor
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
        out : swiftsimio.objects.cosmo_factor
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
        out : swiftsimio.objects.cosmo_factor
            The exponentiated :class:`~swiftsimio.objects.cosmo_factor`s.
        """
        if self.expr is None:
            return cosmo_factor(expr=None, scale_factor=self.scale_factor)
        return cosmo_factor(expr=self.expr ** p, scale_factor=self.scale_factor)

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
        out : bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only compare cosmo_factor with another cosmo_factor.")
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
        out : bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only compare cosmo_factor with another cosmo_factor.")
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
        out : bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only compare cosmo_factor with another cosmo_factor.")
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
        out : bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only compare cosmo_factor with another cosmo_factor.")
        return self.a_factor >= b.a_factor

    def __eq__(self, b: "cosmo_factor") -> bool:
        """
        Compare the expressions and values of two
        :meth:`~swiftsimio.objects.cosmo_factor.a_factor`s.

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
        out : bool
            The result of the comparison.

        Raises
        ------
        ValueError
            If the object to compare is not a :class:`~swiftsimio.objects.cosmo_factor`.
        """
        if not isinstance(b, cosmo_factor):
            raise ValueError("Can only compare cosmo_factor with another cosmo_factor.")
        scale_factor_match = self.scale_factor == b.scale_factor
        if self.a_factor is None and b.a_factor is None:
            # guards passing None to isclose
            a_factor_match = True
        elif self.a_factor is None or b.a_factor is None:
            # we know they're not both None from previous case
            a_factor_match = False
        elif np.isclose(self.a_factor, b.a_factor, rtol=1e-9):
            a_factor_match = True
        else:
            a_factor_match = False
        return scale_factor_match and a_factor_match

    def __ne__(self, b: "cosmo_factor") -> bool:
        """
        Compare the expressions and values of two
        :meth:`~swiftsimio.objects.cosmo_factor.a_factor`s.

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
        out : bool
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
        out : str
            String representation of the scaling with the scale factor.
        """
        return f"cosmo_factor(expr={self.expr}, scale_factor={self.scale_factor})"


NULL_CF = cosmo_factor(None, None)  # helps avoid name collisions with kwargs below


def _parse_cosmo_factor_args(
    cf: cosmo_factor = None,
    scale_factor: float = None,
    scale_exponent: numeric_type = None,
) -> cosmo_factor:
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
    scale_factor : numeric_type
        The scale factor passed as a kwarg.
    scale_exponent : float
        The exponent for the scale factor to convert to/from comoving passed as a kwarg.

    Returns
    -------
    out : cosmo_factor or None
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
    comoving : bool
        Flag to indicate whether using comoving coordinates.
    valid_transform : bool
        Flag to indicate whether this array can be converted to comoving. If ``False``,
        then ``comoving`` must be ``False``.
    compression : string
        Description of the compression filters that were applied to that array in the
        hdf5 file.

    Attributes
    ----------
    comoving : bool
        If ``True`` then the array is in comoving coordinates, if``False`` then it is in
        physical units.

    cosmo_factor : swiftsimio.objects.cosmo_factor
        Object to store conversion data between comoving and physical coordinates.

    compression : string
        String describing any compression that was applied to this array in the
        hdf5 file.

    valid_transform: bool
       If ``True`` then the array can be converted from physical to comoving units.

    Notes
    -----
    This class will generally try to make sense of input and initialize an array-like
    object consistent with the input, and warn or raise if this cannot be done
    consistently. However, the way that :class:`~unyt.array.unyt_array` handles input
    imposes some limits to this. In particular, nested non-numpy containers given in
    input are not traversed recursively, but only one level deep. This means that
    while with this input the attributes are detected by the new array correctly:

    ::

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

    ::

        >>> cosmo_array([[x, x],[x, x]])
        cosmo_array([[[0, 1, 2],[0, 1, 2]],[[0, 1, 2],[0, 1, 2]]],
               '(dimensionless)', comoving='None', cosmo_factor='None at a=None',
               valid_transform='True')

    See Also
    --------
    swiftsimio.objects.cosmo_quantity
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
        vecdot: _multiply_cosmo_factor,
    }

    def __new__(
        cls,
        input_array: Iterable,
        units: Union[str, unyt.unit_object.Unit, "astropy.units.core.Unit"] = None,
        *,
        registry: unyt.unit_registry.UnitRegistry = None,
        dtype: Union[np.dtype, str] = None,
        bypass_validation: bool = False,
        name: str = None,
        cosmo_factor: cosmo_factor = None,
        scale_factor: Optional[float] = None,
        scale_exponent: Optional[float] = None,
        comoving: bool = None,
        valid_transform: bool = True,
        compression: str = None,
    ) -> "cosmo_array":
        """
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
            ``False``, then ``comoving`` must be ``False``.
        compression : string
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
                obj.cosmo_factor = cosmo_factor.create(scale_factor, scale_exponent)
            obj.valid_transform = valid_transform
            obj.compression = compression

            return obj

        if isinstance(input_array, cosmo_array):

            obj = input_array.view(cls)

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
            # else is already copied from input_array

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

        elif _iterable(input_array) and input_array:
            # if _prepare_array_func_args finds cosmo_array input it will convert to:
            default_cm = comoving if comoving is not None else True

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
                if cosmo_factor is None
                else cosmo_factor
            )
            compression = (
                helper_result["compression"] if compression is None else compression
            )
            # valid_transform has a non-None default, so we have to decide to always
            # respect it

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
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.cosmo_factor = getattr(obj, "cosmo_factor", NULL_CF)
        self.comoving = getattr(obj, "comoving", None)
        self.compression = getattr(obj, "compression", None)
        self.valid_transform = getattr(obj, "valid_transform", True)

    def __str__(self) -> str:
        if self.comoving:
            comoving_str = "(Comoving)"
        elif self.comoving is None:
            comoving_str = "(Physical/comoving not set)"
        else:
            comoving_str = "(Physical)"

        return super().__str__() + " " + comoving_str

    def __repr__(self) -> str:
        return super().__repr__()

    def __reduce__(self) -> tuple:
        """
        Pickle reduction method.

        Here we add an extra element at the start of the :class:`~unyt.array.unyt_array`
        state tuple to store the cosmology info.

        Returns
        -------
        out : tuple
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

    def __setstate__(self, state: Tuple) -> None:
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
        self.cosmo_factor, self.comoving, self.compression, self.valid_transform = state[
            0
        ]

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
    __deepcopy__ = _propagate_cosmo_array_attributes_to_result(unyt_array.__deepcopy__)
    in_cgs = _propagate_cosmo_array_attributes_to_result(unyt_array.in_cgs)
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

    # Also wrap some array "properties":
    T = property(_propagate_cosmo_array_attributes_to_result(unyt_array.transpose))
    ua = property(_propagate_cosmo_array_attributes_to_result(np.ones_like))
    unit_array = property(_propagate_cosmo_array_attributes_to_result(np.ones_like))

    def convert_to_comoving(self) -> None:
        """
        Convert the internal data in-place to be in comoving units.
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
        Convert the internal data in-place to be in physical units.
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

    def to_physical(self) -> "cosmo_array":
        """
        Creates a copy of the data in physical units.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            Copy of this array in physical units.
        """
        copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
        copied_data.convert_to_physical()

        return copied_data

    def to_comoving(self) -> "cosmo_array":
        """
        Creates a copy of the data in comoving units.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            Copy of this array in comoving units
        """
        if not self.valid_transform:
            raise InvalidConversionError
        copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
        copied_data.convert_to_comoving()

        return copied_data

    def to_physical_value(self, units: Unit) -> np.ndarray:
        """
        Returns a copy of the array values in the specified physical units.

        Parameters
        ----------
        units : unyt.unit_object.Unit

        Returns
        -------
        out : np.ndarray
            Copy of the array values in the specified physical units.
        """
        return self.to_physical().to_value(units)

    def to_comoving_value(self, units: Unit) -> np.ndarray:
        """
        Returns a copy of the array values in the specified comoving units.

        Parameters
        ----------
        units : unyt.unit_object.Unit

        Returns
        -------
        out : np.ndarray
            Copy of the array values in the specified comoving units.
        """
        return self.to_comoving().to_value(units)

    def compatible_with_comoving(self) -> bool:
        """
        Is this :class:`~swiftsimio.objects.cosmo_array` compatible with a comoving
        :class:`~swiftsimio.objects.cosmo_array`?

        This is the case if the :class:`~swiftsimio.objects.cosmo_array` is comoving, or
        if the scale factor exponent is 0, or the scale factor is 1
        (either case satisfies ``cosmo_factor.a_factor() == 1``).

        Returns
        -------
        out : bool
            ``True`` if compatible, ``False`` otherwise.
        """
        return self.comoving or (self.cosmo_factor.a_factor == 1.0)

    def compatible_with_physical(self) -> bool:
        """
        Is this :class:`~swiftsimio.objects.cosmo_array` compatible with a physical
        :class:`~swiftsimio.objects.cosmo_array`?

        This is the case if the :class:`~swiftsimio.objects.cosmo_array` is physical, or
        if the scale factor exponent is 0, or the scale factor is 1
        (either case satisfies ``cosmo_factor.a_factor() == 1``).

        Returns
        -------
        out : bool
            ``True`` if compatible, ``False`` otherwise.
        """
        return (not self.comoving) or (self.cosmo_factor.a_factor == 1.0)

    @classmethod
    def from_astropy(
        cls,
        arr: "astropy.units.quantity.Quantity",
        unit_registry: unyt.unit_registry.UnitRegistry = None,
        comoving: bool = None,
        cosmo_factor: cosmo_factor = cosmo_factor(None, None),
        compression: str = None,
        valid_transform: bool = True,
    ) -> "cosmo_array":
        """
        Convert an :class:`astropy.units.quantity.Quantity` to a
        :class:`~swiftsimio.objects.cosmo_array`.

        Parameters
        ----------
        arr: astropy.units.quantity.Quantity
            The quantity to convert from.
        unit_registry : unyt.unit_registry.UnitRegistry, optional
            A unyt registry to use in the conversion. If one is not supplied, the
            default one will be used.
        comoving : bool
            Flag to indicate whether using comoving coordinates.
        cosmo_factor : swiftsimio.objects.cosmo_factor
            Object to store conversion data between comoving and physical coordinates.
        compression : string
            Description of the compression filters that were applied to that array in the
            hdf5 file.
        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving. If
            ``False``, then ``comoving`` must be ``False``.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            A cosmology-aware array.

        Example
        -------
        ::

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
        comoving: bool = None,
        cosmo_factor: cosmo_factor = cosmo_factor(None, None),
        compression: str = None,
        valid_transform: bool = True,
    ) -> "cosmo_array":
        """
        Convert a :class:`pint.registry.Quantity` to a
        :class:`~swiftsimio.objects.cosmo_array`.

        Parameters
        ----------
        arr: pint.registry.Quantity
            The quantity to convert from.
        unit_registry : unyt.unit_registry.UnitRegistry, optional
            A unyt registry to use in the conversion. If one is not supplied, the
            default one will be used.
        comoving : bool
            Flag to indicate whether using comoving coordinates.
        cosmo_factor : swiftsimio.objects.cosmo_factor
            Object to store conversion data between comoving and physical coordinates.
        compression : string
            Description of the compression filters that were applied to that array in the
            hdf5 file.
        valid_transform : bool
            Flag to indicate whether this array can be converted to comoving. If
            ``False``, then ``comoving`` must be ``False``.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            A cosmology-aware array.

        Examples
        --------
        ::

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

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> object:
        """
        Handles :mod:`numpy` ufunc calls on :class:`~swiftsimio.objects.cosmo_array`
        input.

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

        inputs : tuple
            Arguments to the ufunc.

        kwargs : dict
            Keyword arguments to the ufunc.

        Returns
        -------
        out : object
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
            ret_cf = _return_without_cosmo_factor(cfs[0])
        else:
            ret_cf = self._cosmo_factor_ufunc_registry[ufunc](*cfs, inputs=inputs)

        ret = _ensure_result_is_cosmo_array_or_quantity(super().__array_ufunc__)(
            ufunc, method, *helper_result["args"], **helper_result["kwargs"]
        )
        # if we get a tuple we have multiple return values to deal with
        if isinstance(ret, tuple):
            for r in ret:
                if isinstance(r, cosmo_array):  # also recognizes cosmo_quantity
                    r.comoving = helper_result["comoving"]
                    r.cosmo_factor = ret_cf
                    r.compression = helper_result["compression"]
        elif isinstance(ret, cosmo_array):  # also recognizes cosmo_quantity
            ret.comoving = helper_result["comoving"]
            ret.cosmo_factor = ret_cf
            ret.compression = helper_result["compression"]
        if "out" in kwargs:
            out = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                out = out[0]
                if isinstance(out, cosmo_array):  # also recognizes cosmo_quantity
                    out.comoving = helper_result["comoving"]
                    out.cosmo_factor = ret_cf
                    out.compression = helper_result["compression"]
            else:
                for o in out:
                    if isinstance(o, cosmo_array):  # also recognizes cosmo_quantity
                        o.comoving = helper_result["comoving"]
                        o.cosmo_factor = ret_cf
                        o.compression = helper_result["compression"]

        return ret

    def __array_function__(
        self, func: Callable, types: Collection, args: tuple, kwargs: dict
    ):
        """
        Handles :mod:`numpy` function calls on :class:`~swiftsimio.objects.cosmo_array`
        input.

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
        out : object
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
            function_to_invoke = func._implementation
        return function_to_invoke(*args, **kwargs)

    def __mul__(
        self, b: Union[int, float, np.ndarray, unyt.unit_object.Unit]
    ) -> "cosmo_array":
        """
        Multiply this :class:`~swiftsimio.objects.cosmo_array`.

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit`
            The object to multiply with this one.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            The result of the multiplication.
        """
        if isinstance(b, unyt.unit_object.Unit):
            retval = self.__copy__()
            retval.units = retval.units * b
            return retval
        else:
            return super().__mul__(b)

    def __rmul__(
        self, b: Union[int, float, np.ndarray, unyt.unit_object.Unit]
    ) -> "cosmo_array":
        """
        Multiply this :class:`~swiftsimio.objects.cosmo_array` (as the right argument).

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit`.

        .. note::

            This function is never called when `b` is a :class:`unyt.unit_object.Unit`
            because :mod:`unyt` handles the operation. This results in a silent demotion
            to a :class:`unyt.array.unyt_array`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit`
            The object to multiply with this one.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            The result of the multiplication.
        """
        if isinstance(b, unyt.unit_object.Unit):
            return self.__mul__(b)
        else:
            return super().__rmul__(b)

    def __truediv__(
        self, b: Union[int, float, np.ndarray, unyt.unit_object.Unit]
    ) -> "cosmo_array":
        """
        Divide this :class:`~swiftsimio.objects.cosmo_array`.

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit`
            The object to divide this one by.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            The result of the division.
        """
        if isinstance(b, unyt.unit_object.Unit):
            return self.__mul__(1 / b)
        else:
            return super().__truediv__(b)

    def __rtruediv__(
        self, b: Union[int, float, np.ndarray, unyt.unit_object.Unit]
    ) -> "cosmo_array":
        """
        Divide this :class:`~swiftsimio.objects.cosmo_array` (as the right argument).

        We delegate most cases to :mod:`unyt`, but we need to handle the case where the
        second argument is a :class:`~unyt.unit_object.Unit`.

        .. note::

            This function is never called when `b` is a :class:`unyt.unit_object.Unit`
            because :mod:`unyt` handles the operation. This results in a silent demotion
            to a :class:`unyt.array.unyt_array`.

        Parameters
        ----------
        b : :class:`~numpy.ndarray`, :obj:`int`, :obj:`float` or \
        :class:`~unyt.unit_object.Unit`
            The object to divide by this one.

        Returns
        -------
        out : swiftsimio.objects.cosmo_array
            The result of the division.
        """
        if isinstance(b, unyt.unit_object.Unit):
            return (1 / self).__mul__(b)
        else:
            return super().__rtruediv__(b)


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

    Parameters
    ----------
    input_scalar : float or unyt.array.unyt_quantity
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
    comoving : bool
        Flag to indicate whether using comoving coordinates.
    valid_transform : bool
        Flag to indicate whether this array can be converted to comoving. If
        ``False``, then ``comoving`` must be ``False``.
    compression : string
        Description of the compression filters that were applied to that array in the
        hdf5 file.

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
        input_scalar: numeric_type,
        units: Optional[
            Union[str, unyt.unit_object.Unit, "astropy.units.core.Unit"]
        ] = None,
        *,
        registry: Optional[unyt.unit_registry.UnitRegistry] = None,
        dtype: Optional[Union[np.dtype, str]] = None,
        bypass_validation: bool = False,
        name: Optional[str] = None,
        cosmo_factor: Optional[cosmo_factor] = None,
        scale_factor: Optional[float] = None,
        scale_exponent: Optional[float] = None,
        comoving: Optional[bool] = None,
        valid_transform: bool = True,
        compression: Optional[str] = None,
    ) -> "cosmo_quantity":
        """
        Closely inspired by the :meth:`unyt.array.unyt_quantity.__new__` constructor.

        Parameters
        ----------
        input_scalar : float or unyt.array.unyt_quantity
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
            The same information can be provided using the ``scale_factor`` and
            ``scale_exponent`` arguments, instead.
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
            ``False``, then ``comoving`` must be ``False``.
        compression : string
            Description of the compression filters that were applied to that array in the
            hdf5 file.
        """
        if bypass_validation is True:
            ret = super().__new__(
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

        if not isinstance(input_scalar, (numeric_type, np.number, np.ndarray)):
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
        if ret.size > 1:
            raise RuntimeError("cosmo_quantity instances must be scalars")
        return ret

    __round__ = _propagate_cosmo_array_attributes_to_result(
        _ensure_result_is_cosmo_array_or_quantity(unyt_quantity.__round__)
    )
