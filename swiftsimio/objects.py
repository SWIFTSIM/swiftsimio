"""
Contains global objects, e.g. the superclass version of the
unyt_array that we use, called cosmo_array.
"""

from unyt import unyt_array

from typing import Union

import sympy

# The scale factor!
a = sympy.symbols("a")


class InvalidScaleFactor(Exception):
    """
    Raised when a scale factor is invalid, such as when adding
    two cosmo_factors with inconsistent scale factors.
    """

    def __init__(self, message=None, *args):
        self.message = message

    def __str__(self):
        return f"InvalidScaleFactor: {self.message}"


class cosmo_factor:
    """
    Cosmology factor. This takes two arguments, one which takes the expected
    exponent of the array that can be parsed by sympy, and the current
    value of the cosmological scale factor a.

    This should be given as the conversion from comoving to physical, i.e.

    r = cosmo_factor * r' with r in physical and r' comoving

    Typicall this would make cosmo_factor = a for the conversion between
    comoving positions r' and physical co-ordinates r.

    To do this, use the a imported from objects multiplied as you'd like:

    density_cosmo_factor = cosmo_factor(a**3, scale_factor=0.97)
    """

    def __init__(self, expr, scale_factor):
        self.expr = expr
        self.scale_factor = scale_factor
        pass

    def __str__(self):
        return str(self.expr) + f" at a={self.scale_factor}"

    @property
    def a_factor(self):
        """
        The a-factor for the unit, e.g. for density this is 1 / a**3.
        """
        return float(self.expr.subs(a, self.scale_factor))

    @property
    def redshift(self):
        """
        Compute the redshift from the scale factor.
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
                "Attempting to subtract two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        return cosmo_factor(expr=self.expr * b.expr, scale_factor=self.scale_factor)

    def __div__(self, b):
        if not self.scale_factor == b.scale_factor:
            raise InvalidScaleFactor(
                "Attempting to subtract two cosmo_factors with different scale factors "
                f"{self.scale_factor} and {b.scale_factor}"
            )

        return cosmo_factor(expr=self.expr / b.expr, scale_factor=self.scale_factor)

    def __radd__(self, b):
        return self.__add__(b)

    def __rsub__(self, b):
        return self.__sub__(b)

    def __rmul__(self, b):
        return self.__mul__(b)

    def __rdiv__(self, b):
        return b.__div__(self)

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
        return self.a_factor == b.a_factor

    def __ne__(self, b):
        return self.a_factor != b.a_factor


class cosmo_array(unyt_array):
    """
    Cosmology array. This inherits from the unyt.unyt_array, and adds
    a two variables; cosmo_factor, and comoving. Data is assumed to be
    comoving when passed to the object but you can override this by setting
    the latter flag to be False.

    It provides four new methods:

    + convert_to_physical() (in-place)
    + convert_to_comoving() (in-pace)
    + to_physical() (returns copy)
    + to_comoving() (returns copy)

    and provides a state variable:

    + comoving, if True then the array is in comoving co-ordinates, and if
        False then it is in physical units.

    """

    def __new__(
        self,
        input_array,
        units=None,
        registry=None,
        dtype=None,
        bypass_validation=False,
        input_units=None,
        name=None,
        cosmo_factor=None,
        comoving=True,
    ):
        """
        Essentially a copy of the __new__ constructor.
        """
        obj = super().__new__(
            self,
            input_array,
            units=units,
            registry=registry,
            dtype=dtype,
            bypass_validation=bypass_validation,
            input_units=input_units,
            name=None,
        )

        obj.cosmo_factor = cosmo_factor
        obj.comoving = comoving

        return obj

    def __str__(self):
        if self.comoving:
            comoving_str = "(Comoving)"
        else:
            comoving_str = "(Physical)"

        return super().__str__() + " " + comoving_str

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
            values *= self.cosmo_factor.a_factor
            self.comoving = True

    def convert_to_physical(self) -> None:
        """
        Convert the internal data to be in physical units.
        """
        if self.comoving:
            # Best to just modify values as otherwise we're just going to have
            # to do a convert_to_units anyway.
            values = self.d
            values /= self.cosmo_factor.a_factor
            self.comoving = False
        else:
            return

    def to_physical(self):
        """
        Returns a copy of the data in physical units.
        """
        copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
        copied_data.convert_to_physical()

        return copied_data

    def to_comoving(self):
        """
        Returns a copy of the data in comoving units.
        """
        copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
        copied_data.convert_to_comoving()

        return copied_data
