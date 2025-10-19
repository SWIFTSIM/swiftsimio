"""
Contains the names and types of the units as read from the file.
"""

import unyt
from numpy import log

unit_names_to_unyt = {
    "Unit mass in cgs (U_M)": unyt.g,
    "Unit length in cgs (U_L)": unyt.cm,
    "Unit time in cgs (U_t)": unyt.s,
    "Unit current in cgs (U_I)": unyt.A,
    "Unit temperature in cgs (U_T)": unyt.K,
}


possible_base_units = {
    "mass": [unyt.g, unyt.kg, unyt.Mearth, unyt.Solar_Mass],
    "length": [
        unyt.cm,
        unyt.m,
        unyt.km,
        unyt.Rearth,
        unyt.pc,
        unyt.kpc,
        unyt.Mpc,
        unyt.Gpc,
    ],
    "time": [unyt.s, unyt.year, unyt.Myr, unyt.Gyr],
    "current": [unyt.A],
    "temperature": [unyt.K],
}


def find_nearest_base_unit(unit: unyt.unyt_quantity, dimension: str):
    """
    Uses the possible_base_units dictionary to find the closest
    base unit to your internal units, and returns that. This assumes
    that internal units and unyt units should line up to within
    1e-5 relative precision (i.e. to 5 significant figures), as
    this is what is usually specified in parameter files.

    Parameters
    ----------
    unit: unyt_quantity
        Quantity to convert to a nearby unit

    dimension: str
        Dimension. Supports ``length``, ``mass``, ``time``,
        ``current`` and ``temperature``.


    Returns
    -------
    unyt_quantity
        Output quantity corresponding to ``unit`` converted to the
        closest unit.

    Example
    -------

    .. code-block::python

        find_nearest_base_unit(1e43 * unyt.g, "mass")
        >>> 1e10 * unyt.Solar_Mass
    """
    possible_bases = possible_base_units[dimension]

    closest_unit = min(possible_bases, key=lambda x: abs(log((1.0 * x).to(unit))))

    return unyt.unyt_quantity(
        float("%.5g" % float(unit.to(closest_unit))), units=closest_unit
    )
