"""
Contains global objects, e.g. the superclass version of the
unyt_array that we use, called cosmo_array.
"""

from unyt import unyt_array

from typing import Union

import sympy

# The scale factor!
a = sympy.symbols("a")


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

    density_cosmo_factor = cosmo_factor(
        a**3,
        scale_factor=0.97
    )
    """

    def __init__(self, expr, scale_factor):
        self.expr = expr
        self.scale_factor = scale_factor
        pass

    def __str__(self):
        return str(self.expr)

    @property
    def a_factor(self):
        """
        The a-factor for the unit, e.g. for density this is 1 / a**3.
        """
        return self.expr.subs(a, self.scale_factor)

    @property
    def redshift(self):
        """
        Compute the redshift from the scale factor.
        """
        return (1.0 / self.scale_factor) - 1.0


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
        units,
        registry=None,
        dtype=None,
        bypass_validation=False,
        *args,
        **kwargs
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
        )
        return obj

    def __init__(
        self,
        input_array,
        units,
        registry=None,
        dtype=None,
        bypass_validation=False,
        cosmo_factor: Union[cosmo_factor, None] = None,
        description: str = "",
        comoving: bool = True,
    ):
        """
        Our version of the __init__. We also call the super().
        """
        super().__init__()

        self.cosmo_factor = cosmo_factor
        self.comoving = comoving
        self.description = description

    def __str__(self):
        if self.comoving:
            comoving_str = "(Comoving)"
        else:
            comoving_str = "(Physical)"

        return super().__str__() + " " + comoving_str

    # TODO: Fix this API. At the moment, it doesn't work.

    # def convert_to_comoving(self) -> None:
    #     """
    #     Convert the internal data to be in comoving units.
    #     """
    #     if self.comoving:
    #         return
    #     else:
    #         self.units *= self.cosmo_factor.a_factor
    #         self.comoving = True

    # def convert_to_physical(self) -> None:
    #     """
    #     Convert the internal data to be in physical units.
    #     """
    #     if self.comoving:
    #         self.units /= self.cosmo_factor.a_factor
    #         self.comoving = False
    #     else:
    #         return

    # def to_physical(self):
    #     """
    #     Returns a copy of the data in physical units.
    #     """
    #     copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
    #     copied_data.convert_to_physical()

    #     return copied_data

    # def to_comoving(self):
    #     """
    #     Returns a copy of the data in comoving units.
    #     """
    #     copied_data = self.in_units(self.units, cosmo_factor=self.cosmo_factor)
    #     copied_data.convert_to_comoving()

    #     return copied_data
