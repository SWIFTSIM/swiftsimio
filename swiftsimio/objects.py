"""
Contains global objects, e.g. the superclass version of the
unyt_array that we use, called cosmo_array.
"""

from unyt import unyt_array, unyt_quantity
from unyt.array import multiple_output_operators, _iterable, POWER_MAPPING
from numbers import Number as numeric_type

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
    _propagate_cosmo_array_attributes,
    _ensure_cosmo_array_or_quantity,
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
)

# The scale factor!
a = sympy.symbols("a")


class InvalidConversionError(Exception):
    def __init__(self, message="Could not convert to comoving coordinates"):
        self.message = message


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

    take = _propagate_cosmo_array_attributes(
        _ensure_cosmo_array_or_quantity(unyt_array.take)
    )
    reshape = _propagate_cosmo_array_attributes(
        _ensure_cosmo_array_or_quantity(unyt_array.reshape)
    )
    __getitem__ = _propagate_cosmo_array_attributes(
        _ensure_cosmo_array_or_quantity(unyt_array.__getitem__)
    )

    # Also wrap some array "properties":
    T = property(_propagate_cosmo_array_attributes(unyt_array.transpose))
    ua = property(_propagate_cosmo_array_attributes(np.ones_like))
    unit_array = property(_propagate_cosmo_array_attributes(np.ones_like))

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

        if not all(issubclass(t, cosmo_array) or t is np.ndarray for t in types):
            # Note: this allows subclasses that don't override
            # __array_function__ to handle cosmo_array objects
            return NotImplemented

        if func in _HANDLED_FUNCTIONS:
            ret = _HANDLED_FUNCTIONS[func](*args, **kwargs)
        elif func in _UNYT_HANDLED_FUNCTIONS:
            ret = _UNYT_HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            # default to numpy's private implementation
            ret = func._implementation(*args, **kwargs)
        if (
            isinstance(ret, cosmo_array)
            and ret.shape == ()
            and not isinstance(ret, cosmo_quantity)
        ):
            return cosmo_quantity(ret)
        elif isinstance(ret, cosmo_quantity) and ret.shape != ():
            return cosmo_array(ret)
        else:
            return ret


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
